"""Direct browser voice test (16kHz PCM) with barge-in, plus the TTS reply streamer."""
import asyncio
import os
import time

import numpy as np
from fastapi import APIRouter, WebSocket

from audio import VAD, rms, transcribe_live
from prompts import build_greeting, parse_call_result, strip_markers

from ..app_config import get_config as _get_config
from ..call_flow import _is_end_of_call, _save_review
from ..config import PIPER_RATE, log
from ..state import opik_end_trace, opik_span, opik_start_trace, state
from ..tts import get_tts_fn, tts_stream
from .accounts import _effective_prompt, _resolve_account

router = APIRouter()

# Browser-path barge-in: rolling RMS window over ~6 frames (120ms) of mic
# audio, compared to an ADAPTIVE threshold. The fixed floor (BROWSER_BARGE_RMS)
# is the minimum; the real threshold is max(floor, echo_baseline * MARGIN),
# where echo_baseline is the rolling average mic energy while the bot is
# speaking (i.e. the speaker-echo floor). Real speech is a sudden jump well
# above that floor, so a margin trigger catches it regardless of how loud the
# caller's speakers are — no fragile absolute value. Tune via env if needed.
BROWSER_BARGE_RMS = float(os.getenv("BROWSER_BARGE_RMS", "0.02"))
BROWSER_BARGE_FRAMES = int(os.getenv("BROWSER_BARGE_FRAMES", "6"))
BROWSER_BARGE_MARGIN = float(os.getenv("BROWSER_BARGE_MARGIN", "2.0"))
BROWSER_BASELINE_FRAMES = int(os.getenv("BROWSER_BASELINE_FRAMES", "25"))


@router.websocket("/ws/{session_id}")
async def browser_voice_loop(ws: WebSocket, session_id: str):
    """Direct browser voice test (16kHz PCM) with barge-in."""
    await ws.accept()
    await ws.send_json({"type": "config", "sample_rate": PIPER_RATE})

    # Parse query params from WebSocket URL (override app-config defaults)
    account_uid = ""
    project_id = ""
    row_num = ""
    _cfg = await _get_config()
    tts_engine = _cfg["tts_engine"]
    vad_mode = _cfg["vad_mode"]
    try:
        qs = str(ws.url).split("?")[1] if "?" in str(ws.url) else ""
        for part in qs.split("&"):
            k, _, v = part.partition("=")
            if k == "account_uid" and v:
                account_uid = v
            elif k == "project_id" and v:
                project_id = v
            elif k == "row_num" and v:
                row_num = v
            elif k == "tts" and v:
                tts_engine = v
            elif k == "vad" and v:
                vad_mode = v
    except Exception:
        pass

    # Resolve account context from project_id + row_num if provided
    if project_id and row_num:
        _uid, _acct = await _resolve_account(project_id, int(row_num))
        if _acct:
            account_uid = _uid or ""
        else:
            log.warning("[%s] No account for project %s row %s", session_id, project_id, row_num)

    tts_fn = get_tts_fn(tts_engine)
    use_silero = vad_mode == "silero"

    account = None
    if account_uid:
        account = await state["redis"].hgetall(f"account:{account_uid}") or None

    system_prompt = await _effective_prompt(account)
    conversation: list[dict] = [{"role": "system", "content": system_prompt}]

    # TTR timing accumulators (ms) — must exist before greeting TTS
    _tts_times = []
    stt_total_ms = 0.0
    stt_count = 0
    llm_total_ms = 0.0
    llm_count = 0
    peak_prompt_tokens = 0
    total_completion_tokens = 0
    tts_total_ms = 0.0
    tts_count = 0

    tts_task = None
    # Set synchronously the instant we want to stop — the generator checks
    # this at every segment/character boundary, so it stops queuing further
    # audio immediately rather than waiting on task-cancellation timing
    # (which can be deferred until an in-flight synth call finishes).
    tts_stop_event: asyncio.Event | None = None

    def speak(text: str, trace=None):
        nonlocal tts_task, tts_stop_event
        tts_stop_event = asyncio.Event()
        tts_task = asyncio.create_task(
            _stream_tts_reply(ws, text, tts_fn, _tts_times, trace=trace, stop_event=tts_stop_event))
        return tts_task

    # Speak greeting with claim context right away
    greeting = build_greeting(account)
    conversation.append({"role": "assistant", "content": greeting})
    await ws.send_json({"type": "llm_text", "text": greeting})
    tts_task = speak(greeting)
    greeting_task = tts_task

    vad = VAD(use_silero=use_silero)
    barge_in = False
    _barge_frames: list[float] = []
    _barge_baseline: list[float] = []
    full_audio = bytearray()
    call_start = time.time()
    review_saved = False

    async def cancel_tts():
        nonlocal tts_task, tts_stop_event
        if tts_stop_event is not None:
            tts_stop_event.set()
        if tts_task and not tts_task.done():
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass
            tts_task = None
        # Clear the browser's playback buffer too — otherwise audio already
        # sent is still queued/playing on the client even though the server
        # stopped producing it, so the current utterance "keeps talking"
        # while the next reply starts.
        try:
            await ws.send_bytes(b"\x02")
        except Exception:
            pass

    async def _finalize_review():
        nonlocal review_saved
        if review_saved:
            return
        review_saved = True
        await _save_review(session_id, conversation, bytes(full_audio),
                           time.time() - call_start, account_uid, call_start,
                           tts_engine, vad_mode, system_prompt,
                           stt_total_ms, stt_count, llm_total_ms, llm_count,
                           sum(_tts_times), len(_tts_times),
                           peak_prompt_tokens, total_completion_tokens)

    try:
        while True:
            raw = await ws.receive_bytes()
            full_audio.extend(raw)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            now = time.time()

            # NOTE: no raw-energy barge-in. The speaker echo of the bot's own
            # TTS frequently exceeded the energy threshold and cancelled
            # replies before they were heard. Interrupts happen only through
            # the VAD segment flush below (Silero distinguishes real speech
            # from the bot's own voice), which still cancels the TTS so the
            # caller can interrupt after finishing their utterance.
            # Barge-in while the bot is speaking: a rolling window of mic RMS
            # energy compared to an ADAPTIVE echo floor. The VAD flush alone
            # can't interrupt a long spelled run, because the bot's own audio
            # echoing into the mic keeps Silero in continuous "speech" so it
            # never sees the 700ms silence needed to flush. Real speech is a
            # sudden jump above the measured echo baseline, so triggering on
            # max(floor, baseline*margin) works no matter the speaker volume.
            # The greeting is protected so echo can't cancel the first message.
            if tts_task and not tts_task.done() and tts_task is not greeting_task:
                _energy = rms(audio)
                _barge_frames.append(_energy)
                if len(_barge_frames) > BROWSER_BARGE_FRAMES:
                    _barge_frames.pop(0)
                _barge_baseline.append(_energy)
                if len(_barge_baseline) > BROWSER_BASELINE_FRAMES:
                    _barge_baseline.pop(0)
                _floor = sum(_barge_baseline) / max(1, len(_barge_baseline))
                _thr = max(BROWSER_BARGE_RMS, _floor * BROWSER_BARGE_MARGIN)
                # Only act once the echo floor is established (~half the
                # baseline window) so the reply's own opening echo can't
                # trigger an immediate false barge-in.
                if (len(_barge_baseline) >= BROWSER_BASELINE_FRAMES // 2
                        and len(_barge_frames) == BROWSER_BARGE_FRAMES
                        and sum(_barge_frames) / BROWSER_BARGE_FRAMES > _thr):
                    await cancel_tts()
                    _barge_frames = []
                    await ws.send_bytes(b"\x02")
                    # Reset VAD so the echo-accumulated buffer is dropped; the
                    # caller's speech after this point starts a clean segment.
                    vad = VAD(use_silero=use_silero)
                    continue
            else:
                _barge_frames = []
                _barge_baseline = []

            segment = vad.add(audio, now)
            if segment is None:
                continue

            # The greeting is non-interruptible: let it fully stream before
            # acting on the VAD segment captured underneath it (protects the
            # very first message from echo/noise cancelling it). Do NOT cancel
            # its playback: tts_task finishing only means the audio was sent
            # to the browser, whose queue may still be playing it — a \x02
            # here wipes that queue mid-word (audible static click and a
            # cut-off tail). Keep processing the captured segment below
            # instead of dropping it: a real first utterance must produce a
            # transcript turn + Opik trace, while a pure greeting-echo segment
            # simply transcribes to <3 chars and is filtered.
            barge_in = False

            _trace = opik_start_trace("browser.turn", session_id,
                                      {"audio_sec": round(len(segment) / 16000, 2)},
                                      {"type": "browser_call", "session_id": session_id,
                                       "account_uid": account_uid, "state": "browser"})

            _stt_t0 = time.time()
            async with state["stt_lock"]:
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(
                    None, lambda: transcribe_live(state["stt_model"], segment))
                stt_latency_ms = (time.time() - _stt_t0) * 1000
                stt_total_ms += stt_latency_ms
                stt_count += 1
            opik_span(_trace, "browser.stt", "general",
                      {"audio_sec": round(len(segment) / 16000, 2)}, {"text": text},
                      _stt_t0, time.time())

            # Only interrupt the bot for a REAL utterance. Transcribing before
            # cancelling means a <3-char noise/echo segment (cough, throat
            # clear, keyboard click) no longer cuts a multi-sentence reply off
            # mid-stream — previously the flush cancelled TTS up front and then
            # dropped the segment anyway, so a reply like a two-part question
            # would be reduced to its first sentence.
            if len(text) < 3:
                opik_end_trace(_trace, output={"text": text, "status": "empty"})
                continue
            if tts_task is greeting_task:
                try:
                    await tts_task
                except Exception:
                    pass
                tts_task = None
            else:
                await cancel_tts()

            if _is_end_of_call(text, conversation):
                await cancel_tts()
                opik_end_trace(_trace, output={"status": "end_of_call"})
                await ws.send_json({"type": "llm_text", "text": "Call ended. Have a great day!"})
                await _finalize_review()
                await asyncio.sleep(1)
                await ws.close()
                return

            conversation.append({"role": "user", "content": f"[INSURANCE REP] {text}"})
            await ws.send_json({"type": "transcript", "text": text})
            conv = conversation if len(conversation) <= 15 else [conversation[0]] + conversation[-14:]
            _llm_t0 = time.time()
            bot_text = ""
            usage = None
            try:
                stream = await state["llm_client"].chat.completions.create(
                    model=state["llm_model"], messages=conv, max_tokens=150, temperature=0,
                    stream=True, stream_options={"include_usage": True})
                async for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                        continue
                    bot_text += chunk.choices[0].delta.content or ""
            except Exception as _e:
                opik_end_trace(_trace, error=str(_e))
                raise
            llm_total_ms += (time.time() - _llm_t0) * 1000
            llm_count += 1
            if usage is not None:
                peak_prompt_tokens = max(peak_prompt_tokens, int(getattr(usage, "prompt_tokens", 0) or 0))
                total_completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

            # Speak the whole response as ONE TTS input (no per-sentence splits,
            # no truncation): even if the model asks two questions, they are
            # spoken together in a single stream.
            cr_idx = bot_text.upper().find("CALL_RESULT")
            cut = bot_text[:cr_idx] if cr_idx != -1 else bot_text
            spoken = strip_markers(cut).rstrip().rstrip("[").strip()
            reply = bot_text.strip()
            result = parse_call_result(reply)
            has_result_marker = bool(result or cr_idx != -1)

            # SAFETY NET: if the response asks a question AND emits [CALL_RESULT],
            # the agent must not finalize before the representative answers —
            # speak the question(s) and wait instead of ending the call.
            if has_result_marker and "?" in spoken:
                log.info("[%s] CALL_RESULT emitted with a pending question — waiting for answer", session_id)
                result = None
                has_result_marker = False

            conversation.append({"role": "assistant", "content": reply})
            opik_span(_trace, "browser.llm", "llm", {"messages": conv[-2:]}, reply,
                      _llm_t0, time.time(), model=state["llm_model"],
                      metadata={"usage": {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                                          "completion_tokens": getattr(usage, "completion_tokens", None),
                                          "total_tokens": getattr(usage, "total_tokens", None)}} if usage else None)

            if result:
                await state["redis"].hset(f"call:{session_id}", mapping={
                    "claim_id": result.get("claim_id") or "browser",
                    "payer": result.get("payer") or "browser",
                    "status": result.get("status") or "completed",
                    "next_action": result.get("next_action") or "",
                    "denial_code": result.get("denial_code") or "",
                    "paid_amount": str(result.get("paid_amount") or ""),
                    "call_summary": result.get("call_summary") or "",
                    "reference_number": result.get("reference_number") or "",
                    "call_result": reply,
                })

            # Speak the whole turn as ONE TTS input.
            if spoken:
                await ws.send_json({"type": "llm_text", "text": spoken})
                tts_task = speak(spoken, trace=_trace)
                if not has_result_marker:
                    # Normal turn: end the trace once this TTS completes.
                    # (the farewell path below ends the trace itself)
                    asyncio.create_task(_end_browser_trace_after_tts(tts_task, _trace, {"reply": spoken}))

            if has_result_marker:
                await cancel_tts()
                if tts_task:
                    try:
                        await tts_task
                    except Exception:
                        pass
                farewell = "Okay, thank you. Goodbye!"
                await ws.send_json({"type": "llm_text", "text": farewell})
                tts_task = speak(farewell, trace=_trace)
                if tts_task:
                    try:
                        await asyncio.wait_for(tts_task, timeout=10)
                    except asyncio.CancelledError:
                        pass
                    except asyncio.TimeoutError:
                        log.warning("[%s] Farewell TTS timed out — ending anyway", session_id)
                    except Exception:
                        pass
                opik_end_trace(_trace, output={"status": "completed", "call_result": result})
                await _finalize_review()
                await asyncio.sleep(1)
                try:
                    await ws.close()
                except Exception:
                    pass
                return

    except Exception as e:
        log.warning("[%s] browser call ended: %s", session_id, e)
    finally:
        await _finalize_review()
        if tts_stop_event is not None:
            tts_stop_event.set()
        if tts_task and not tts_task.done():
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass


async def _end_browser_trace_after_tts(tts_task, trace, output):
    """End the turn trace after the background TTS (and its span) completes."""
    if trace is None:
        return
    try:
        await tts_task
    except Exception:
        pass
    opik_end_trace(trace, output=output)


async def _stream_tts_reply(ws: WebSocket, text: str, tts_fn=tts_stream, tts_times: list | None = None,
                            trace=None, stop_event: "asyncio.Event | None" = None):
    t0 = time.time()
    first = True
    total_bytes = 0
    sent_log: list = []
    try:
        async for pcm_bytes in tts_fn(text, stop_event=stop_event, sent_log=sent_log):
            if first and tts_times is not None:
                tts_times.append((time.time() - t0) * 1000)
                first = False
            total_bytes += len(pcm_bytes)
            await ws.send_bytes(b"\x01" + pcm_bytes)
        if trace is not None:
            # Single span AFTER streaming so `sent_log` holds every call that
            # was actually made to the TTS engine (verbatim, punctuation and
            # all) — not just the LLM's pre-transform text.
            opik_span(trace, "browser.tts", "general",
                      {"text": text[:2000], "sent": sent_log},
                      {"audio_bytes": total_bytes, "latency_ms": round((time.time() - t0) * 1000, 1)},
                      t0, time.time())
    except asyncio.CancelledError:
        pass
    except Exception as e:
        log.warning("[tts] reply stream error: %s", e)
