"""End-of-call detection, offline STT, and post-call review persistence."""
import asyncio
import json
import time

import numpy as np

from audio import transcribe_offline
from prompts import strip_markers

from .config import TTS_ENGINE, VAD_MODE, WHISPER_MODEL_SIZE, log
from .state import _bg_tasks, state

_END_PHRASES = [
    "bye", "goodbye", "see you", "that's all", "i'm done", "cut the call",
    "end the call", "hang up", "that is all", "no more", "i'm finished",
    "all done", "thank you goodbye",
]


def _is_end_of_call(user_text: str, conversation: list[dict]) -> bool:
    if len(conversation) < 2:
        return False
    lower = user_text.lower().strip()
    if any(p in lower for p in _END_PHRASES):
        return True
    if conversation and any(
        "have a great day" in (m.get("content", "")).lower()
        or "goodbye" in (m.get("content", "")).lower()
        for m in conversation[-3:]
    ):
        if lower in ("okay", "ok", "thanks", "thank you", "bye", "sure", "yes"):
            return True
    return False


# ── Offline STT (full audio, no VAD) ────────────────────────────────────
async def transcribe_full_audio(audio_bytes: bytes) -> str:
    if len(audio_bytes) < 320:
        return ""
    pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    async with state["stt_lock"]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: transcribe_offline(state["stt_model"], pcm)
        )


async def _save_review(session_id: str, conversation: list[dict], full_audio: bytes,
                       call_duration: float, account_uid: str = "", started_at: float = 0.0,
                       tts_engine: str = "", vad_mode: str = "", system_prompt: str = "",
                       stt_total_ms: float = 0.0, stt_count: int = 0,
                       llm_total_ms: float = 0.0, llm_count: int = 0,
                       tts_total_ms: float = 0.0, tts_count: int = 0,
                       peak_prompt_tokens: int = 0, total_completion_tokens: int = 0):
    """Save audio buffer, spawn offline STT, store review data."""
    r = state["redis"]
    audio_key = f"call:{session_id}:audio"
    review_key = f"call:{session_id}:review"

    # Create a call record so browser calls appear in the feed/review list.
    # Use the account context for payer/claim when CALL_RESULT didn't store them.
    existing = await r.hgetall(f"call:{session_id}")
    acct = await r.hgetall(f"account:{account_uid}") if account_uid else {}
    await r.hset(f"call:{session_id}", mapping={
        "claim_id": (existing.get("claim_id") or acct.get("Claim ID")
                     or acct.get("Account Number") or "unknown"),
        "payer": existing.get("payer") or acct.get("Responsible Payer") or "unknown",
        "account_uid": account_uid,
        # Only "completed" if a CALL_RESULT produced a real outcome; otherwise failed
        "status": existing.get("status") or "failed",
        "started_at": str(started_at or time.time()),
        "ended_at": str(time.time()),
        "duration_ms": str(int(call_duration * 1000)),
        "next_action": existing.get("next_action", ""),
        # TTR metrics
        "stt_avg_ms": str(int(stt_total_ms / max(1, stt_count))),
        "llm_avg_ms": str(int(llm_total_ms / max(1, llm_count))),
        "tts_avg_ms": str(int(tts_total_ms / max(1, tts_count))),
        "ttr_avg_ms": str(int((stt_total_ms / max(1, stt_count))
                              + (llm_total_ms / max(1, llm_count))
                              + (tts_total_ms / max(1, tts_count)))),
        "peak_prompt_tokens": str(peak_prompt_tokens),
        "total_completion_tokens": str(total_completion_tokens),
        "context_limit": "8192",
    })

    # Save audio to Redis (24h TTL)
    if full_audio:
        await r.setex(audio_key, 86400, full_audio)

    # Extract real-time user utterances, AI responses, and interleaved transcript
    real_time = []
    ai_responses = []
    transcript = []
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            cleaned = content.replace("[INSURANCE REP] ", "", 1)
            real_time.append(cleaned)
            transcript.append({"role": "user", "text": cleaned})
        elif role == "assistant":
            ai_responses.append(content)
            transcript.append({"role": "assistant", "text": strip_markers(content)})

    # Store call config/meta + full interleaved transcript
    try:
        await r.set(f"call:{session_id}:transcript", json.dumps(transcript))
        await r.hset(f"call:{session_id}:meta", mapping={
            "stt_model": WHISPER_MODEL_SIZE,
            "tts_engine": tts_engine or TTS_ENGINE,
            "vad_mode": vad_mode or VAD_MODE,
            "llm_model": state["llm_model"],
            "prompt": system_prompt,
            "call_sid": session_id,
        })
    except Exception as e:
        log.error("[%s] Meta save error: %s", session_id, e)

    # Run offline STT in background
    async def _offline_stt():
        try:
            if not full_audio or len(full_audio) < 320:
                review = {"real_time": real_time, "full_audio": "", "ai_responses": ai_responses,
                          "duration_sec": int(call_duration), "audio_size_bytes": len(full_audio)}
                await r.set(review_key, json.dumps(review))
                return
            log.info("[%s] Offline STT on %d bytes...", session_id, len(full_audio))
            transcript = await transcribe_full_audio(full_audio)
            review = {"real_time": real_time, "full_audio": transcript, "ai_responses": ai_responses,
                      "duration_sec": int(call_duration), "audio_size_bytes": len(full_audio)}
            await r.set(review_key, json.dumps(review))
            log.info("[%s] Offline STT done (%d chars)", session_id, len(transcript))
        except Exception as e:
            log.error("[%s] Offline STT error: %s", session_id, e)
            review = {"real_time": real_time, "full_audio": "", "ai_responses": ai_responses,
                      "duration_sec": int(call_duration), "audio_size_bytes": len(full_audio), "error": str(e)}
            await r.set(review_key, json.dumps(review))

    # Keep a strong reference so the task isn't garbage collected mid-run
    task = asyncio.create_task(_offline_stt())
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
