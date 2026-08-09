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
import datetime
import json
import logging
import os
import re
import time

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from audio import VAD, twilio_to_whisper, piper_to_twilio, rms, transcribe_live, transcribe_offline
from prompts import (
    build_call_prompt, build_greeting, parse_markers,
    parse_call_result, load_payer, load_denial_codes, strip_markers,
)

log = logging.getLogger("call-session")

_bg_tasks: set = set()  # strong refs for background tasks (avoids GC cancellation)


def _opik_span(trace, name: str, span_type: str, input_data, output_data,
               start_ts: float, end_ts: float, model: str | None = None):
    """Create a COMPLETE span in a single call (no separate .end()).

    Opik batches span messages; .end() shortly after creation drops the create
    payload (name/type/start_time lost → epoch-0 timestamps). Passing measured
    start/end + input/output up front avoids the race and yields real durations.
    """
    if trace is None:
        return
    try:
        trace.span(
            name=name, type=span_type, input=input_data, output=output_data,
            start_time=datetime.datetime.fromtimestamp(start_ts, tz=datetime.timezone.utc),
            end_time=datetime.datetime.fromtimestamp(end_ts, tz=datetime.timezone.utc),
            model=model, provider="vllm" if span_type == "llm" else None,
        )
    except Exception:
        pass

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
        self.use_silero = deps.get("use_silero", True)
        self.tts_engine = deps.get("tts_engine", "piper")
        self.vad_mode = deps.get("vad_mode", "silero" if self.use_silero else "rms")
        self.stt_model_name = deps.get("stt_model_name", "unknown")
        self.opik = deps.get("opik")
        self._current_trace = None
        self.peak_prompt_tokens = 0
        self.total_completion_tokens = 0

        self.stream_sid: str | None = None
        self.full_audio = bytearray()
        self.account_uid = ""
        self.account: dict | None = None
        self.state = "GREETING"
        self.payer_data: dict | None = None
        self.last_verify_phrase: str | None = None
        self.drift_logged = False
        self.hold_start: float | None = None
        self.call_result_retries = 0

        self.vad = VAD(use_silero=self.use_silero)
        self.is_bot_speaking = False
        self.is_call_ended = False
        self.last_activity = time.time()
        self.last_hold_nudge = 0.0
        self.conversation: list[dict] = []
        self.system_prompt = build_call_prompt("GREETING", None, None, None)
        self.tts_task: asyncio.Task | None = None
        self.watchdog_task: asyncio.Task | None = None
        self.last_tts_duration: float = 0.0

        # TTR timing accumulators (ms)
        self.stt_total_ms = 0.0
        self.stt_count = 0
        self.llm_total_ms = 0.0
        self.llm_count = 0
        self.tts_total_ms = 0.0
        self.tts_count = 0

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

        self.system_prompt = await self._resolve_prompt(self.account)
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
                self.system_prompt = await self._resolve_prompt(self.account)

        log.info("[%s] Stream started: %s | state=%s", self.call_sid, self.stream_sid, self.state)
        await self.redis.hset(f"call:{self.call_sid}", mapping={"status": "in-progress"})

        greeting = build_greeting(self.account)
        self.conversation.append({"role": "assistant", "content": greeting})
        await self._live_append({"role": "assistant", "text": greeting})
        await self._speak(greeting)
        self.state = "IVR_NAV"

    async def _on_media(self, data: dict):
        payload = base64.b64decode(data["media"]["payload"])
        self.full_audio.extend(payload)  # raw μ-law (8kHz) for offline re-transcription
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

        # Start the turn trace (thread = call) before STT
        self._current_trace = None
        if self.opik:
            try:
                self._current_trace = self.opik.trace(
                    name="twilio.turn",
                    input={"audio_sec": round(len(segment) / 16000, 2),
                           "state": self.state},
                    metadata={"type": "twilio_call", "call_id": self.call_sid},
                    thread_id=self.call_sid, project_name="ar-voice-agent",
                )
            except Exception:
                self._current_trace = None

        stt_t0 = time.time()
        text = await self._transcribe(segment)
        _opik_span(self._current_trace, "twilio.stt", "general",
                   {"audio_sec": round(len(segment) / 16000, 2)}, {"text": text},
                   stt_t0, time.time())
        if not text or len(text.strip()) < 3:
            self._end_trace(output={"text": text, "status": "empty"})
            return

        log.info("[%s] Heard: %s", self.call_sid, text)
        await self._live_append({"role": "user", "text": text})
        await self._run_llm(text)

    def _end_trace(self, output=None, error: str | None = None):
        if self._current_trace is None:
            return
        try:
            if error:
                self._current_trace.end(output={"error": error},
                                        error_info={"error_type": "error", "message": error})
            else:
                self._current_trace.end(output=output)
        except Exception:
            pass
        self._current_trace = None

    # ── STT ──────────────────────────────────────────────────────────────
    async def _transcribe(self, audio: np.ndarray) -> str:
        async with self.stt_lock:
            loop = asyncio.get_running_loop()
            t0 = time.time()
            text = await loop.run_in_executor(
                None, lambda: transcribe_live(self.stt_model, audio)
            )
            self.stt_total_ms += (time.time() - t0) * 1000
            self.stt_count += 1
            return text

    async def _transcribe_full(self) -> str:
        """Offline STT on the full μ-law recording (no VAD)."""
        if len(self.full_audio) < 3200:
            return ""
        from audio import mulaw_to_pcm16, resample
        pcm = mulaw_to_pcm16(bytes(self.full_audio))               # int16 @8kHz
        f32 = pcm.astype(np.float32) / 32768.0
        f32 = resample(f32, 8000, 16000)                           # to Whisper 16kHz
        async with self.stt_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: transcribe_offline(self.stt_model, f32)
            )

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
        self.conversation.append({"role": "user", "content": f"[INSURANCE REP] [STATE: {self.state}] {user_text}"})

        try:
            llm_t0 = time.time()
            stream = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=self.conversation,
                max_tokens=200,
                temperature=0,
                stream=True,
                stream_options={"include_usage": True},
            )
            bot_text = ""
            usage = None
            sentence_buffer = ""
            seen_call_result = False
            asked_question = False
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                    continue
                token = chunk.choices[0].delta.content or ""
                bot_text += token
                # Once the CALL_RESULT marker appears, stop speaking entirely
                # (JSON can be split across sentence buffers by newlines).
                if not seen_call_result and re.search(r"CALL_RESULT", bot_text, re.IGNORECASE):
                    seen_call_result = True
                    marker_idx = re.search(r"CALL_RESULT", bot_text, re.IGNORECASE).start()
                    before = bot_text[:marker_idx]
                    if before.strip():
                        _pre = strip_markers(before).rstrip().rstrip("[").strip()
                        if "?" in _pre:
                            _pre = _pre.split("?")[0] + "?"
                            asked_question = True
                        await self._speak(_pre)
                    sentence_buffer = ""
                    if asked_question:
                        break
                    continue
                if seen_call_result:
                    continue
                sentence_buffer += token
                if token[-1:] in (".", "!", "?", "\n") and len(sentence_buffer.strip()) > 2:
                    parsed = parse_markers(sentence_buffer)
                    if parsed["spoken"]:
                        _pre = parsed["spoken"]
                        if "?" in _pre:
                            _pre = _pre.split("?")[0] + "?"
                            asked_question = True
                        await self._speak(_pre)
                    sentence_buffer = ""
                    if asked_question:
                        break

            if sentence_buffer.strip() and not seen_call_result and not asked_question:
                parsed_tail = parse_markers(sentence_buffer)
                if parsed_tail["spoken"]:
                    _pre = parsed_tail["spoken"]
                    if "?" in _pre:
                        _pre = _pre.split("?")[0] + "?"
                        asked_question = True
                    await self._speak(_pre)

            # ONE question per turn: stop after the first question so the agent
            # never bundles questions or finalizes before the rep answers.
            if asked_question:
                bot_text = strip_markers(bot_text.split("?")[0] + "?")
                seen_call_result = False

            self.llm_total_ms += (time.time() - llm_t0) * 1000
            self.llm_count += 1
            if usage is not None:
                pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                ct = int(getattr(usage, "completion_tokens", 0) or 0)
                self.peak_prompt_tokens = max(self.peak_prompt_tokens, pt)
                self.total_completion_tokens += ct
            log.info("[%s] LLM (state=%s): %s", self.call_sid, self.state, bot_text)
            self.conversation.append({"role": "assistant", "content": bot_text})
            await self.redis.hset(f"call:{self.call_sid}", mapping={"last_llm_response": bot_text})
            await self._live_append({"role": "assistant", "text": strip_markers(bot_text)})
            _opik_span(self._current_trace, "twilio.llm", "llm",
                       {"messages": self.conversation[-2:]}, bot_text,
                       llm_t0, time.time(), model=self.llm_model)

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
            # The marker itself is the end-of-call signal — finalize even if
            # the JSON is missing/malformed (repair is attempted in _finalize).
            call_result = parse_call_result(bot_text)
            has_result_marker = bool(call_result or re.search(r"CALL_RESULT", bot_text, re.IGNORECASE))

            # SAFETY NET: if the response BOTH asks a question (e.g. a
            # confirmation "...correct?") AND emits [CALL_RESULT], do NOT end
            # the call — discard the result and wait for the representative's
            # answer instead of proceeding on our own.
            if has_result_marker:
                spoken_before = strip_markers(bot_text.split("[CALL_RESULT]")[0]).strip()
                if "?" in spoken_before:
                    log.info("[%s] CALL_RESULT emitted with a pending question — waiting for answer", self.call_sid)
                    has_result_marker = False
                    call_result = None

            if has_result_marker:
                if not call_result:
                    log.info("[%s] CALL_RESULT marker without valid JSON — ending call", self.call_sid)
                # Say a farewell so the call doesn't end in silence
                log.info("[%s] Speaking farewell (ended=%s sid=%s)", self.call_sid, self.is_call_ended, self.stream_sid)
                await self._speak("Okay, thank you. Goodbye!")
                if self.tts_task:
                    try:
                        await self.tts_task
                        log.info("[%s] Farewell TTS completed", self.call_sid)
                    except asyncio.CancelledError:
                        log.warning("[%s] Farewell TTS cancelled", self.call_sid)
                    except Exception as e:
                        log.warning("[%s] Farewell TTS error: %s", self.call_sid, e)
                # Give Twilio time to flush/play the farewell audio before closing.
                # Wait the full audio duration + a margin for Twilio's playback buffer.
                await asyncio.sleep(max(1.5, self.last_tts_duration + 1.0))
                self._end_trace(output={"call_result": call_result or {}, "status": "completed"})
                await self._finalize("completed", result=call_result)
                return

            # ── State machine transition ─────────────────────────────────
            self._advance_state(bot_text)
            self._end_trace(output={"reply": bot_text, "state": self.state})

        except Exception as e:
            log.error("[%s] LLM error: %s", self.call_sid, e)
            self._end_trace(error=str(e))
            await self.redis.hset(f"call:{self.call_sid}", mapping={"last_error": f"LLM: {e}"})

    # ── State Machine ────────────────────────────────────────────────────
    def _advance_state(self, bot_text: str):
        lower = bot_text.lower()
        if self.state == "IVR_NAV":
            if "denied" in lower or "denial" in lower:
                self.state = "DENIAL_HANDLE"
            elif "paid" in lower or "approved" in lower:
                self.state = "APPROVED_HANDLE"
            elif "connected" in lower or "agent" in lower or "representative" in lower:
                self.state = "CLAIM_VERIFY"
            if self.state != "IVR_NAV":
                log.info("[%s] state: IVR_NAV → %s", self.call_sid, self.state)
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

    # ── Custom LLM context (per-account override) ──────────────────────
    async def _resolve_prompt(self, account: dict | None) -> str:
        if not account:
            return build_call_prompt("GREETING", None, None, None)
        uid = account.get("UID", "")
        custom = None
        if uid:
            custom = await self.redis.get(f"account:{uid}:llm_context")
        notes = account.get("Notes", "")
        if custom:
            prompt = custom
            if notes:
                prompt += f"\n\n[PRIOR CALL NOTES]\n{notes}"
            return prompt
        return build_call_prompt("GREETING", None, None, account)

    # ── Live transcript (for UI polling during a call) ──────────────────
    async def _live_append(self, turn: dict):
        try:
            await self.redis.rpush(f"call:{self.call_sid}:live", json.dumps(turn))
        except Exception:
            pass

    async def _live_flush(self):
        try:
            await self.redis.delete(f"call:{self.call_sid}:live")
        except Exception:
            pass

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
        if not text or not text.strip():
            return
        if self.is_call_ended or not self.stream_sid:
            log.warning("[%s] _speak skipped (ended=%s sid=%s): %r", self.call_sid, self.is_call_ended, self.stream_sid, text[:40])
            return
        self.tts_task = asyncio.create_task(self._stream_tts(text))

    async def _stream_tts(self, text: str):
        try:
            log.info("[%s] TTS start: %r", self.call_sid, text[:40])
            tts_t0 = time.time()
            frame = bytearray()
            sent_samples = 0
            audio_started = False
            async for pcm_chunk in self.tts_stream_fn(text):
                mulaw = piper_to_twilio(
                    np.frombuffer(pcm_chunk, dtype=np.int16), PIPER_RATE
                )
                frame.extend(mulaw)
                while len(frame) >= 160:
                    chunk = bytes(frame[:160])
                    del frame[:160]
                    sent_samples += 160
                    # Mark as speaking only once audio is actually flowing,
                    # so a caller's early speech can't barge-in an unheard greeting
                    if not audio_started:
                        self.is_bot_speaking = True
                        audio_started = True
                        self.tts_total_ms += (time.time() - tts_t0) * 1000
                        self.tts_count += 1
                        _opik_span(self._current_trace, "twilio.tts", "general",
                                   {"text": text[:120]},
                                   {"audio_bytes": sent_samples * 2,
                                    "latency_ms": round((time.time() - tts_t0) * 1000, 1)},
                                   tts_t0, time.time())
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
                sent_samples += len(frame)
            # μ-law is 8kHz: 8000 samples = 1 second of audio
            self.last_tts_duration = sent_samples / 8000.0
            self.is_bot_speaking = False
            log.info("[%s] TTS done (%.2fs audio)", self.call_sid, self.last_tts_duration)
        except asyncio.CancelledError:
            log.warning("[%s] TTS cancelled", self.call_sid)
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
        await self._live_flush()

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

            # Only mark "completed" when a CALL_RESULT produced a real outcome.
            # Otherwise (no result) treat a "completed"-grade end as failed.
            if result:
                final_status = result.get("status", status)
            else:
                final_status = "failed" if status in ("completed", "disconnected") else status

            update = {
                "status": final_status,
                "ended_at": str(time.time()),
                "duration_ms": str(duration_ms),
            }
            if result:
                for key in ["payer", "claim_id", "next_action", "denial_code",
                            "paid_amount", "billed_amount", "appeal_deadline",
                            "call_summary", "satisfaction"]:
                    if key in result and result[key] is not None:
                        update[key] = str(result[key])
                update["payer"] = result.get("payer") or call_data.get("payer", "unknown")
                update["claim_id"] = result.get("claim_id") or call_data.get("claim_id", "unknown")
            # TTR metrics
            stt_avg = self.stt_total_ms / max(1, self.stt_count)
            llm_avg = self.llm_total_ms / max(1, self.llm_count)
            tts_avg = self.tts_total_ms / max(1, self.tts_count)
            update["stt_avg_ms"] = str(int(stt_avg))
            update["llm_avg_ms"] = str(int(llm_avg))
            update["tts_avg_ms"] = str(int(tts_avg))
            update["ttr_avg_ms"] = str(int(stt_avg + llm_avg + tts_avg))
            update["peak_prompt_tokens"] = str(self.peak_prompt_tokens)
            update["total_completion_tokens"] = str(self.total_completion_tokens)
            update["context_limit"] = "4096"
            if error:
                update["last_error"] = error

            await self.redis.hset(f"call:{self.call_sid}", mapping=update)
            await self.redis.publish("call-updates", json.dumps({
                "callSid": self.call_sid, **update
            }))

            # Save transcript for the review tab + full interleaved transcript + call config
            try:
                real_time = []
                ai_responses = []
                transcript = []
                for msg in self.conversation:
                    content = msg.get("content", "")
                    role = msg.get("role", "")
                    if role == "user":
                        cleaned = content.replace("[INSURANCE REP] ", "", 1)
                        cleaned = re.sub(r"\[STATE: [^\]]*\]\s*", "", cleaned)
                        real_time.append(cleaned)
                        transcript.append({"role": "user", "text": cleaned})
                    elif role == "assistant":
                        ai_responses.append(content)
                        transcript.append({"role": "assistant", "text": strip_markers(content)})
                review = {
                    "real_time": real_time,
                    "full_audio": "",
                    "ai_responses": ai_responses,
                    "duration_sec": int(duration_ms / 1000),
                    "audio_size_bytes": len(self.full_audio),
                    "call_result": result,
                }
                await self.redis.set(f"call:{self.call_sid}:review", json.dumps(review))

                # Offline STT on the full μ-law recording (background, no VAD)
                audio_bytes = bytes(self.full_audio)
                if len(audio_bytes) >= 3200:
                    async def _offline_full():
                        try:
                            text = await self._transcribe_full()
                            raw = await self.redis.get(f"call:{self.call_sid}:review")
                            if raw:
                                rv = json.loads(raw)
                                rv["full_audio"] = text
                                await self.redis.set(f"call:{self.call_sid}:review", json.dumps(rv))
                            log.info("[%s] Offline full-recording STT done (%d chars)", self.call_sid, len(text))
                        except Exception as e:
                            log.error("[%s] Offline full STT error: %s", self.call_sid, e)
                    task = asyncio.create_task(_offline_full())
                    _bg_tasks.add(task)
                    task.add_done_callback(_bg_tasks.discard)
                await self.redis.set(f"call:{self.call_sid}:transcript", json.dumps(transcript))
                await self.redis.hset(f"call:{self.call_sid}:meta", mapping={
                    "stt_model": self.stt_model_name,
                    "tts_engine": self.tts_engine,
                    "vad_mode": self.vad_mode,
                    "llm_model": self.llm_model,
                    "prompt": self.system_prompt,
                    "call_sid": self.call_sid,
                })
            except Exception as e:
                log.error("[%s] Review save error: %s", self.call_sid, e)

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
