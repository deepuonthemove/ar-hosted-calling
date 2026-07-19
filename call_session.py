"""CallSession — per-call voice pipeline with state machine and knowledge base.

Port of src/do.ts with extensions:
  - JSON [CALL_RESULT] instead of positional [END:...]
  - State machine (GREETING → ... → CLOSE)
  - Layered prompts (base + claim context + payer IVR map + denial codes)
  - IVR drift detection (verify phrase matching)
  - Hold polling (30s interval, configurable timeout)
  - Human handoff triggers
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
from prompts import (
    build_call_prompt, build_greeting, parse_markers,
    parse_call_result, load_payer, load_denial_codes,
)

log = logging.getLogger("call-session")

MAX_SILENCE_MS = 19 * 60 * 1000
HOLD_NUDGE_MS = 10000
HOLD_POLL_MS = 5000
MAX_HOLD_SEC = int(os.getenv("MAX_HOLD_SEC", "1800"))
PIPER_RATE = int(os.getenv("PIPER_SAMPLE_RATE", "22050"))

STATES = ["GREETING", "IVR_NAV", "CLAIM_VERIFY", "STATUS_GATHER",
          "DENIAL_HANDLE", "APPROVED_HANDLE", "CLOSE"]


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
        self.state = "GREETING"
        self.payer_data: dict | None = None
        self.last_verify_phrase: str | None = None
        self.drift_logged = False
        self.hold_start: float | None = None
        self.call_result_retries = 0

        self.vad = VAD()
        self.is_bot_speaking = False
        self.is_call_ended = False
        self.last_activity = time.time()
        self.last_hold_nudge = 0.0
        self.conversation: list[dict] = []
        self.system_prompt = build_call_prompt("GREETING", None, None, None)
        self.tts_task: asyncio.Task | None = None
        self.watchdog_task: asyncio.Task | None = None

    # ── Entry ────────────────────────────────────────────────────────────
    async def run(self):
        await self.ws.accept()
        log.info("[%s] Media stream connected", self.call_sid)

        call_data = await self.redis.hgetall(f"call:{self.call_sid}")
        self.account_uid = call_data.get("account_uid", "")
        if self.account_uid:
            self.account = await self.redis.hgetall(f"account:{self.account_uid}") or None

        payer_name = (
            self.account.get("Responsible Payer") if self.account
            else call_data.get("payer")
        )
        if payer_name:
            self.payer_data = load_payer(payer_name)
            codes = load_denial_codes()

        denial_subset = None
        if self.payer_data:
            denial_subset = self.payer_data.get("common_denials")

        self.system_prompt = build_call_prompt(
            "GREETING", self.payer_data, denial_subset, self.account
        )
        self.conversation = [{"role": "system", "content": self.system_prompt}]
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
                payer_name = self.account.get("Responsible Payer") if self.account else None
                if payer_name:
                    self.payer_data = load_payer(payer_name)
                denial_subset = self.payer_data.get("common_denials") if self.payer_data else None
                self.system_prompt = build_call_prompt(
                    "GREETING", self.payer_data, denial_subset, self.account
                )

        log.info("[%s] Stream started: %s | state=%s", self.call_sid, self.stream_sid, self.state)
        await self.redis.hset(f"call:{self.call_sid}", mapping={"status": "in-progress"})

        greeting = build_greeting(self.account)
        self.conversation.append({"role": "assistant", "content": greeting})
        await self._speak(greeting)
        self.state = "IVR_NAV"

    async def _on_media(self, data: dict):
        payload = base64.b64decode(data["media"]["payload"])
        audio = twilio_to_whisper(payload)
        now = time.time()
        self.last_activity = now

        # Barge-in: user spoke while bot is talking
        if self.is_bot_speaking and rms(audio) > 0.02:
            await self._barge_in()

        segment = self.vad.add(audio, now)
        if segment is None:
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

    # ── LLM with state machine ───────────────────────────────────────────
    async def _run_llm(self, user_text: str):
        if self.is_call_ended:
            return

        if self._detect_call_end(user_text):
            log.info("[%s] End-of-call detected from user", self.call_sid)
            await self._speak("Thank you. Have a great day!")
            await asyncio.sleep(1.5)
            await self._finalize("completed")
            return

        # IVR drift: check if the expected verify phrase was in this transcript
        await self._check_verify_on_transcript(user_text)

        # Qwen supports system role natively. Prepend state info to user message.
        state_tag = f"[STATE: {self.state}]"
        self.conversation.append({"role": "user", "content": f"{state_tag} {user_text}"})

        try:
            stream = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=self.conversation,
                max_tokens=200,
                temperature=0,
                stream=True,
            )
            bot_text = ""
            sentence_buffer = ""
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                bot_text += token
                sentence_buffer += token
                if token in (".", "!", "?", "\n") and len(sentence_buffer.strip()) > 2:
                    parsed = parse_markers(sentence_buffer)
                    if parsed["spoken"]:
                        await self._speak(parsed["spoken"])
                    sentence_buffer = ""

            if sentence_buffer.strip():
                parsed_tail = parse_markers(sentence_buffer)
                if parsed_tail["spoken"]:
                    await self._speak(parsed_tail["spoken"])

            log.info("[%s] LLM (state=%s): %s", self.call_sid, self.state, bot_text)
            self.conversation.append({"role": "assistant", "content": bot_text})
            await self.redis.hset(f"call:{self.call_sid}", mapping={"last_llm_response": bot_text})

            parsed = parse_markers(bot_text)

            # ── DTMF handling + IVR drift detection ─────────────────────
            if parsed["dtmf"]:
                log.info("[%s] DTMF: %s", self.call_sid, parsed["dtmf"])
                await self._send_dtmf(parsed["dtmf"])
                # IVR drift: detect if the verify phrase is missing
                if self.payer_data and not self.drift_logged:
                    await self._check_ivr_drift(parsed["dtmf"])

            # ── Waiting on hold ──────────────────────────────────────────
            if parsed["waiting"]:
                log.info("[%s] Waiting on hold", self.call_sid)
                if self.hold_start is None:
                    self.hold_start = time.time()

            # ── End call with [CALL_RESULT] JSON ─────────────────────────
            call_result = parse_call_result(bot_text)
            if call_result:
                await self._finalize("completed", result=call_result)
                return

            # ── State machine transition ─────────────────────────────────
            self._advance_state(bot_text)

        except Exception as e:
            log.error("[%s] LLM error: %s", self.call_sid, e)
            await self.redis.hset(f"call:{self.call_sid}", mapping={"last_error": f"LLM: {e}"})

    # ── State Machine ────────────────────────────────────────────────────
    def _advance_state(self, bot_text: str):
        lower = bot_text.lower()
        if self.state == "IVR_NAV" and ("connected" in lower or "agent" in lower or "representative" in lower):
            self.state = "CLAIM_VERIFY"
            log.info("[%s] state: IVR_NAV → CLAIM_VERIFY", self.call_sid)
        elif self.state == "CLAIM_VERIFY" and any(w in lower for w in ["paid", "denied", "pending", "status"]):
            if "denied" in lower or "denial" in lower:
                self.state = "DENIAL_HANDLE"
            elif "paid" in lower or "approved" in lower:
                self.state = "APPROVED_HANDLE"
            else:
                self.state = "STATUS_GATHER"
            log.info("[%s] state: → %s", self.call_sid, self.state)
        elif self.state == "STATUS_GATHER" and any(w in lower for w in ["denied", "denial"]):
            self.state = "DENIAL_HANDLE"
            log.info("[%s] state: → DENIAL_HANDLE", self.call_sid)
        elif self.state == "STATUS_GATHER" and any(w in lower for w in ["paid", "approved"]):
            self.state = "APPROVED_HANDLE"
            log.info("[%s] state: → APPROVED_HANDLE", self.call_sid)

    # ── IVR Drift Detection ─────────────────────────────────────────────
    async def _check_ivr_drift(self, dtmf_digit: str):
        """After sending DTMF, if the system expected a verify_phrase
        but the next transcription doesn't match, log an anomaly."""
        if not self.payer_data:
            return
        ivr = self.payer_data.get("ivr_tree", {})
        expected_phrase = None
        for key, nodes in ivr.items():
            for n in nodes:
                if n.get("dtmf") == dtmf_digit:
                    expected_phrase = n.get("verify_phrase", "")
                    break

        if not expected_phrase:
            return

        # Listen for the next transcription (non-blocking: check next user input)
        # We store the expected phrase; on the next _run_llm call, check the transcript
        self.last_verify_phrase = expected_phrase

    async def _check_verify_on_transcript(self, text: str):
        """Called from _run_llm after hearing the next utterance."""
        if not self.last_verify_phrase or self.drift_logged:
            return
        if self.last_verify_phrase.lower() not in text.lower():
            log.warning("[%s] IVR DRIFT: expected '%s' but heard '%s'",
                        self.call_sid, self.last_verify_phrase, text)
            self.drift_logged = True
            await self.redis.hset(f"call:{self.call_sid}", mapping={
                "ivr_drift": json.dumps({
                    "expected": self.last_verify_phrase,
                    "heard": text,
                    "payer": (
                        self.account.get("Responsible Payer", "unknown")
                        if self.account else "unknown"
                    ),
                })
            })
            # Log anomaly to the payer drift set
            payer_name = self.account.get("Responsible Payer", "unknown") if self.account else "unknown"
            anomaly = json.dumps({
                "ts": time.time(),
                "call_sid": self.call_sid,
                "expected": self.last_verify_phrase,
                "heard": text,
            })
            await self.redis.zadd(f"ivr_drift:{payer_name}", {anomaly: time.time()})
        self.last_verify_phrase = None

    # ── TTS → Twilio ────────────────────────────────────────────────────
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
            log.warning("[%s] TTS error: %s", self.call_sid, e)
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
        await self._cancel_tts()
        if self.stream_sid:
            try:
                await self.ws.send_text(json.dumps({
                    "event": "clear", "streamSid": self.stream_sid
                }))
            except Exception:
                pass

    async def _send_dtmf(self, digit: str):
        try:
            await self.ws.send_text(json.dumps({
                "event": "dtmf",
                "streamSid": self.stream_sid,
                "dtmf": {"digit": digit},
            }))
        except Exception as e:
            log.error("[%s] DTMF send failed: %s", self.call_sid, e)

    # ── Hold + Silence Watchdog ─────────────────────────────────────────
    async def _silence_watchdog(self):
        try:
            while not self.is_call_ended:
                silence_ms = (time.time() - self.last_activity) * 1000
                is_holding = self.hold_start is not None

                if is_holding:
                    await asyncio.sleep(30)
                    if (time.time() - self.hold_start) > MAX_HOLD_SEC:
                        log.warning("[%s] Hold timeout exceeded", self.call_sid)
                        await self._speak("I've been unable to reach a representative. I'll call back later.")
                        await asyncio.sleep(2)
                        await self._finalize("hold_timeout")
                        break
                else:
                    await asyncio.sleep(0.5)
                    if (silence_ms > HOLD_NUDGE_MS
                            and time.time() - self.last_hold_nudge > 10
                            and not self.is_bot_speaking):
                        self.last_hold_nudge = time.time()
                        await self._run_llm("[Hold music detected]")
                    if silence_ms > MAX_SILENCE_MS:
                        log.warning("[%s] Max silence", self.call_sid)
                        await self._finalize("failed", error="Max silence")
                        break
        except asyncio.CancelledError:
            pass

    # ── End-of-call detection ────────────────────────────────────────────
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
        if any(
            "have a great day" in (m.get("content", "")).lower()
            or "goodbye" in (m.get("content", "")).lower()
            for m in self.conversation[-3:]
        ):
            if lower in ("okay", "ok", "thanks", "thank you", "bye", "sure", "yes"):
                return True
        return False

    # ── Finalization ─────────────────────────────────────────────────────
    async def _finalize(self, status: str, result: dict | None = None, error: str = ""):
        if self.is_call_ended:
            return
        self.is_call_ended = True
        await self._cancel_tts()

        # If a [CALL_RESULT] was provided, use it.
        # Otherwise, if we have [CALL_RESULT] in the last LLM output, parse it.
        if not result:
            last_msg = self.conversation[-1]["content"] if self.conversation else ""
            result = parse_call_result(last_msg)

        # Retry with LLM repair if parsing failed
        if not result and self.call_result_retries < 2:
            self.call_result_retries += 1
            last_msg = self.conversation[-1]["content"] if self.conversation else ""
            try:
                import prompts as pmod
                result = pmod.attempt_repair(last_msg, self.llm_client, self.llm_model)
            except Exception:
                pass

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
                for key in ["payer", "claim_id", "next_action", "denial_code",
                            "paid_amount", "billed_amount", "appeal_deadline",
                            "call_summary", "satisfaction"]:
                    if key in result:
                        update[key] = str(result[key])
                update["payer"] = result.get("payer", call_data.get("payer", "unknown"))
                update["claim_id"] = result.get("claim_id", call_data.get("claim_id", "unknown"))
            if error:
                update["last_error"] = error

            await self.redis.hset(f"call:{self.call_sid}", mapping=update)
            await self.redis.publish("call-updates", json.dumps({
                "callSid": self.call_sid, **update
            }))

            if self.account_uid:
                today = time.strftime("%m/%d/%Y")
                if result:
                    account_update = {
                        "Call Comments": result.get("call_summary", "Call completed"),
                        "Call Date": today,
                        "Call Status": "Calls Done",
                    }
                    if result.get("denial_code"):
                        account_update["Denial Code"] = result["denial_code"]
                    if result.get("paid_amount"):
                        account_update["Amount Paid"] = result["paid_amount"]
                    if result.get("next_action"):
                        account_update["Next Action"] = result["next_action"]
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
