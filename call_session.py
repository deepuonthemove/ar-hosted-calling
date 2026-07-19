"""CallSession — per-call voice pipeline over Twilio Media Streams.

Port of the Cloudflare Durable Object (src/do.ts):
  Twilio WS ↔ VAD → Faster-Whisper → vLLM (markers) → Piper TTS → Twilio
  with DTMF, hold detection, barge-in, silence watchdog, result persistence.
"""
import asyncio
import base64
import json
import logging
import os
import time

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from audio import VAD, twilio_to_whisper, piper_to_twilio, rms
from prompts import build_system_prompt, build_greeting, parse_markers

log = logging.getLogger("call-session")

MAX_SILENCE_MS = 19 * 60 * 1000   # ported: 19-minute hard cap
HOLD_NUDGE_MS = 4000              # inject [Hold music detected] after 4s silence
PIPER_RATE = int(os.getenv("PIPER_SAMPLE_RATE", "22050"))


class CallSession:
    def __init__(self, ws: WebSocket, call_sid: str, deps: dict):
        self.ws = ws
        self.call_sid = call_sid
        self.redis = deps["redis"]
        self.stt_model = deps["stt_model"]
        self.stt_lock = deps["stt_lock"]
        self.llm_client = deps["llm_client"]
        self.llm_model = deps["llm_model"]
        self.tts_stream_fn = deps["tts_stream_fn"]

        self.stream_sid: str | None = None
        self.account_uid = ""
        self.account: dict | None = None
        self.conversation: list[dict] = []
        self.vad = VAD()
        self.is_bot_speaking = False
        self.is_call_ended = False
        self.last_activity = time.time()
        self.last_hold_nudge = 0.0
        self.tts_task: asyncio.Task | None = None
        self.watchdog_task: asyncio.Task | None = None

    # ── Entry ────────────────────────────────────────────────────────────
    async def run(self):
        await self.ws.accept()
        log.info("[%s] Media stream connected", self.call_sid)

        # Load call + account context from Redis (same keys as Cloudflare)
        call_data = await self.redis.hgetall(f"call:{self.call_sid}")
        self.account_uid = call_data.get("account_uid", "")
        if self.account_uid:
            self.account = await self.redis.hgetall(f"account:{self.account_uid}") or None

        # Mistral doesn't support system role — store separately, prepend to first user message
        self.system_prompt = build_system_prompt(self.account)
        self.conversation: list[dict] = []
        self.has_system = False
        self.watchdog_task = asyncio.create_task(self._silence_watchdog())

        try:
            async for message in self.ws.iter_text():
                data = json.loads(message)
                event = data.get("event")
                if event == "start":
                    await self._on_start(data)
                elif event == "media":
                    await self._on_media(data)
                elif event == "stop":
                    await self._finalize("disconnected")
                    break
        except WebSocketDisconnect:
            log.info("[%s] Twilio WS disconnected", self.call_sid)
            await self._finalize("disconnected")
        except Exception as e:
            log.error("[%s] Session error: %s", self.call_sid, e)
            await self._finalize("failed", error=str(e))
        finally:
            if self.watchdog_task:
                self.watchdog_task.cancel()
            await self._cancel_tts()

    # ── Twilio events ────────────────────────────────────────────────────
    async def _on_start(self, data: dict):
        self.stream_sid = data["start"]["streamSid"]
        params = data["start"].get("customParameters", {})
        if not self.account_uid:
            self.account_uid = params.get("account_uid", "")
            if self.account_uid:
                self.account = await self.redis.hgetall(f"account:{self.account_uid}") or None
                self.system_prompt = build_system_prompt(self.account)

        log.info("[%s] Stream started: %s", self.call_sid, self.stream_sid)
        await self.redis.hset(f"call:{self.call_sid}", mapping={"status": "in-progress"})

        # Speak greeting (patient context injected), don't add to conversation
        greeting = build_greeting(self.account)
        await self._speak(greeting)

    async def _on_media(self, data: dict):
        payload = base64.b64decode(data["media"]["payload"])
        audio = twilio_to_whisper(payload)
        now = time.time()
        self.last_activity = now

        segment = self.vad.add(audio, now)
        if segment is None:
            # Barge-in: user started speaking while bot talks
            if self.is_bot_speaking and rms(audio) > 0.02:
                await self._barge_in()
            return

        await self._cancel_tts()
        text = await self._transcribe(segment)
        if not text or len(text.strip()) < 3:
            return

        log.info("[%s] Heard: %s", self.call_sid, text)
        await self._run_llm(text)

    # ── STT ──────────────────────────────────────────────────────────────
    async def _transcribe(self, audio: np.ndarray) -> str:
        async with self.stt_lock:
            loop = asyncio.get_running_loop()
            segments, _ = await loop.run_in_executor(
                None,
                lambda: self.stt_model.transcribe(
                    audio, beam_size=1, vad_filter=True, language="en"
                ),
            )
            return " ".join(s.text for s in segments).strip()

    # ── LLM with markers (ported from runLLM) ────────────────────────────
    async def _run_llm(self, user_text: str):
        if self.is_call_ended:
            return

        # End-of-call detection — check if user wants to hang up
        if self._detect_call_end(user_text):
            log.info("[%s] Detected end of call from user", self.call_sid)
            result = {
                "payer": (self.account or {}).get("Responsible Payer", "unknown"),
                "claim_id": (self.account or {}).get("Account Number", "unknown"),
                "status": "completed",
                "next_action": "Call completed by user",
            }
            await self._speak("Thank you. Have a great day!")
            await asyncio.sleep(1.5)
            await self._finalize("completed", result=result)
            return

        # Mistran doesn't support system role — prepend to first user message
        msg = f"{self.system_prompt}\n\n{user_text}" if not self.conversation else user_text
        self.conversation.append({"role": "user", "content": msg})

        try:
            stream = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=self.conversation,
                max_tokens=150,
                temperature=0,
                stream=True,
            )
            bot_text = ""
            sentence_buffer = ""
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                bot_text += token
                sentence_buffer += token
                # Stream TTS at sentence boundaries for low latency
                if token in (".", "!", "?", "\n") and len(sentence_buffer.strip()) > 2:
                    parsed = parse_markers(sentence_buffer)
                    if parsed["spoken"]:
                        await self._speak(parsed["spoken"])
                    sentence_buffer = ""

            # Handle any trailing tokens
            if sentence_buffer.strip():
                parsed_tail = parse_markers(sentence_buffer)
                if parsed_tail["spoken"]:
                    await self._speak(parsed_tail["spoken"])

            log.info("[%s] LLM: %s", self.call_sid, bot_text)
            self.conversation.append({"role": "assistant", "content": bot_text})
            await self.redis.hset(f"call:{self.call_sid}", mapping={"last_llm_response": bot_text})

            parsed = parse_markers(bot_text)

            if parsed["dtmf"]:
                log.info("[%s] DTMF: %s", self.call_sid, parsed["dtmf"])
                await self._send_dtmf(parsed["dtmf"])

            if parsed["waiting"]:
                log.info("[%s] Waiting on hold", self.call_sid)

            if parsed["end"]:
                await self._finalize("completed", result=parsed["end"])

        except Exception as e:
            log.error("[%s] LLM error: %s", self.call_sid, e)
            await self.redis.hset(f"call:{self.call_sid}", mapping={"last_error": f"LLM: {e}"})

    # ── End-of-call detection ──────────────────────────────────────────
    _END_PHRASES = [
        "bye", "goodbye", "see you", "that's all", "i'm done", "cut the call",
        "end the call", "hang up", "that is all", "no more", "i'm finished",
        "all done", "thank you goodbye",
    ]

    def _detect_call_end(self, text: str) -> bool:
        if len(self.conversation) < 2:
            return False
        lower = text.lower().strip()
        if any(phrase in lower for phrase in self._END_PHRASES):
            return True
        # Assistant already said goodbye + user acknowledges
        if self.conversation and any(
            "have a great day" in (m.get("content", "")).lower()
            or "goodbye" in (m.get("content", "")).lower()
            for m in self.conversation[-3:]
        ):
            if lower in ("okay", "ok", "thanks", "thank you", "bye", "sure", "yes"):
                return True
        return False

    # ── TTS → Twilio μ-law ───────────────────────────────────────────────
    async def _speak(self, text: str):
        if self.is_call_ended or not self.stream_sid:
            return
        self.is_bot_speaking = True
        self.tts_task = asyncio.create_task(self._stream_tts(text))

    async def _stream_tts(self, text: str):
        try:
            frame = bytearray()
            async for pcm_chunk in self.tts_stream_fn(text):
                mulaw = piper_to_twilio(
                    np.frombuffer(pcm_chunk, dtype=np.int16), PIPER_RATE
                )
                frame.extend(mulaw)
                # Twilio expects ~20ms frames = 160 bytes at 8kHz μ-law
                while len(frame) >= 160:
                    chunk = bytes(frame[:160])
                    del frame[:160]
                    await self.ws.send_text(json.dumps({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": base64.b64encode(chunk).decode()},
                    }))
            if frame:
                await self.ws.send_text(json.dumps({
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": base64.b64encode(bytes(frame)).decode()},
                }))
            self.is_bot_speaking = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.warning("[%s] TTS stream error: %s", self.call_sid, e)
            self.is_bot_speaking = False

    async def _cancel_tts(self):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()
            try:
                await self.tts_task
            except asyncio.CancelledError:
                pass
        self.tts_task = None
        self.is_bot_speaking = False

    async def _barge_in(self):
        """Ported: clear Twilio buffer + cancel TTS when caller speaks."""
        await self._cancel_tts()
        if self.stream_sid:
            try:
                await self.ws.send_text(json.dumps({
                    "event": "clear", "streamSid": self.stream_sid
                }))
            except Exception:
                pass

    async def _send_dtmf(self, digit: str):
        """Ported: Twilio Media Streams DTMF message."""
        try:
            await self.ws.send_text(json.dumps({
                "event": "dtmf",
                "streamSid": self.stream_sid,
                "dtmf": {"digit": digit},
            }))
        except Exception as e:
            log.error("[%s] DTMF send failed: %s", self.call_sid, e)

    # ── Hold music + silence watchdog (ported) ───────────────────────────
    async def _silence_watchdog(self):
        try:
            while not self.is_call_ended:
                await asyncio.sleep(0.5)
                silence_ms = (time.time() - self.last_activity) * 1000

                # Hold music nudge (ported: VAD events → [Hold music detected])
                if (silence_ms > HOLD_NUDGE_MS
                        and time.time() - self.last_hold_nudge > 10
                        and not self.is_bot_speaking):
                    self.last_hold_nudge = time.time()
                    await self._run_llm("[Hold music detected]")

                # 19-minute hard cap (ported from do.ts silenceTimer)
                if silence_ms > MAX_SILENCE_MS:
                    log.warning("[%s] Max silence reached", self.call_sid)
                    await self._finalize("failed", error="Max silence reached")
                    break
        except asyncio.CancelledError:
            pass

    # ── Finalization (ported from closeCall + finalizeCall) ──────────────
    async def _finalize(self, status: str, result: dict | None = None, error: str = ""):
        if self.is_call_ended:
            return
        self.is_call_ended = True
        await self._cancel_tts()

        try:
            call_data = await self.redis.hgetall(f"call:{self.call_sid}")
            started_at = float(call_data.get("started_at", time.time()))
            duration_ms = int((time.time() - started_at) * 1000)

            update = {
                "status": (result or {}).get("status", status),
                "ended_at": str(time.time()),
                "duration_ms": str(duration_ms),
            }
            if result:
                update.update({
                    "payer": result.get("payer", "unknown"),
                    "claim_id": result.get("claim_id", "unknown"),
                    "amount": result.get("amount") or "",
                    "next_action": result.get("next_action", "none"),
                })
            if error:
                update["last_error"] = error

            await self.redis.hset(f"call:{self.call_sid}", mapping=update)
            await self.redis.publish("call-updates", json.dumps({
                "callSid": self.call_sid, **update
            }))

            # Update Excel account row (ported)
            if self.account_uid:
                today = time.strftime("%m/%d/%Y")
                if result:
                    account_update = {
                        "Call Comments": result.get("next_action") or "Call completed",
                        "Call Date": today,
                        "Call Status": "Calls Done",
                    }
                else:
                    account_update = {
                        "Call Comments": "Call disconnected" if status == "disconnected" else "Call failed",
                        "Call Date": today,
                        "Call Status": "Disconnected" if status == "disconnected" else "Failed",
                    }
                await self.redis.hset(f"account:{self.account_uid}", mapping=account_update)

        except Exception as e:
            log.error("[%s] Finalize error: %s", self.call_sid, e)

        try:
            await self.ws.close()
        except Exception:
            pass
