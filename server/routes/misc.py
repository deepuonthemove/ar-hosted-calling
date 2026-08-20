"""Health, LLM switching, live/review/detail lookups, and app config API."""
import asyncio
import json
import subprocess
import tempfile

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from .. import app_config, chatterbox_voices
from ..config import (
    KOKORO_VOICE_OPTIONS, OPIK_BASE_URL, OPIK_ENABLED,
    SPELL_PUNCT_OPTIONS, SPELL_SILENCE_OPTIONS, TTS_PAUSE_MODE_OPTIONS,
    WHISPER_MODEL_SIZE, log,
)
from ..state import state

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "whisper": WHISPER_MODEL_SIZE, "llm": state["llm_model"]}


@router.get("/api/current-llm")
async def current_llm():
    return {
        "model": state["llm_model"],
        "options": state["llm_options"],
        "switching": state.get("llm_switching", False),
    }


@router.post("/api/switch-llm")
async def switch_llm(request: Request):
    data = await request.json()
    model_name = data.get("model", "")
    # Find the full model ID from options
    model_id = None
    for label, mid in state["llm_options"].items():
        if model_name in (label, mid):
            model_id = mid
            break
    if not model_id:
        return JSONResponse({"error": f"Unknown model: {model_name}"}, 400)
    if model_id == state["llm_model"]:
        return {"ok": True, "model": model_id, "switching": False}

    state["llm_switching"] = True
    # Write to .env on host
    try:
        env_path = "/project/.env"
        with open(env_path) as f:
            lines = f.readlines()
        found = False
        for i, line in enumerate(lines):
            if line.startswith("LLM_MODEL="):
                lines[i] = f"LLM_MODEL={model_id}\n"
                found = True
                break
        if not found:
            lines.append(f"LLM_MODEL={model_id}\n")
        with open(env_path, "w") as f:
            f.writelines(lines)
    except Exception as e:
        log.warning("Could not update .env: %s", e)

    # Update in-memory model
    old_model = state["llm_model"]
    state["llm_model"] = model_id

    # Restart vLLM in background
    async def _restart_vllm():
        try:
            import subprocess
            subprocess.run(["docker", "restart", "vllm"], timeout=120)
            log.info("vLLM restarted with model %s", model_id)
        except Exception as e:
            log.error("vLLM restart failed: %s", e)
            state["llm_model"] = old_model
        finally:
            state["llm_switching"] = False

    asyncio.create_task(_restart_vllm())
    return {"ok": True, "model": model_id, "switching": True}


@router.get("/api/calls/{session_id}/live")
async def get_live_transcript(session_id: str):
    """Live transcript for an in-progress call (UI polls this)."""
    raw = await state["redis"].lrange(f"call:{session_id}:live", 0, -1)
    turns = [json.loads(x) for x in raw] if raw else []
    record = await state["redis"].hgetall(f"call:{session_id}")
    active = bool(record) and "ended_at" not in record
    return {"call_id": session_id, "active": active, "transcript": turns}


@router.get("/api/calls/{session_id}/review")
async def get_review(session_id: str):
    review_key = f"call:{session_id}:review"
    data = await state["redis"].get(review_key)
    if not data:
        return JSONResponse({"error": "Review not found"}, 404)
    return JSONResponse(json.loads(data))


@router.get("/api/calls/{call_id}/detail")
async def get_call_detail(call_id: str):
    """Full detail for a call: config (STT/TTS/LLM/prompt), interleaved transcript, review."""
    r = state["redis"]
    record = await r.hgetall(f"call:{call_id}")
    if not record:
        return JSONResponse({"error": "Call not found"}, 404)

    meta = await r.hgetall(f"call:{call_id}:meta")

    transcript = []
    raw_t = await r.get(f"call:{call_id}:transcript")
    if raw_t:
        try:
            transcript = json.loads(raw_t)
        except json.JSONDecodeError:
            transcript = []

    review = None
    raw_r = await r.get(f"call:{call_id}:review")
    if raw_r:
        try:
            review = json.loads(raw_r)
        except json.JSONDecodeError:
            review = None

    # Old calls (pre-transcript feature) — reconstruct interleaved transcript
    # from the review arrays if no stored transcript exists.
    if not transcript and review:
        rt = review.get("real_time", []) or []
        ai = review.get("ai_responses", []) or []
        if ai:
            transcript.append({"role": "assistant", "text": ai[0]})
        for i, u in enumerate(rt):
            transcript.append({"role": "user", "text": u})
            if i + 1 < len(ai):
                transcript.append({"role": "assistant", "text": ai[i + 1]})
        if transcript:
            try:
                await r.set(f"call:{call_id}:transcript", json.dumps(transcript))
            except Exception:
                pass

    return {
        "call_id": call_id,
        "twilio_sid": record.get("twilio_sid", ""),
        "call": {k: v for k, v in record.items()},
        "config": {
            "stt_model": meta.get("stt_model", "unknown"),
            "tts_engine": meta.get("tts_engine", "unknown"),
            "vad_mode": meta.get("vad_mode", "unknown"),
            "llm_model": meta.get("llm_model", "unknown"),
        },
        "prompt": meta.get("prompt", ""),
        "transcript": transcript,
        "review": review,
    }


# ── App config (TTS/VAD defaults, persisted in Redis, editable via API) ─
@router.get("/api/config")
async def get_config():
    cfg = await app_config.get_config()
    return {
        "tts_engine": cfg["tts_engine"],
        "vad_mode": cfg["vad_mode"],
        "stay_awake": cfg["stay_awake"] == "1",
        "kokoro_voice": cfg["kokoro_voice"],
        "spell_engine": cfg["spell_engine"],
        "spell_pause_mode": cfg["spell_pause_mode"],
        "spell_silence_s": float(cfg["spell_silence_s"]),
        "spell_punct": cfg["spell_punct"],
        "chatterbox_voice": cfg["chatterbox_voice"],
        "tts_options": ["piper", "kokoro", "chatterbox"],
        "vad_options": ["silero", "rms"],
        "kokoro_voice_options": KOKORO_VOICE_OPTIONS,
        "spell_engine_options": ["piper", "match"],
        "spell_pause_mode_options": TTS_PAUSE_MODE_OPTIONS.get(
            cfg["tts_engine"], ["processed", "silence", "punctuation"]),
        "spell_silence_options": SPELL_SILENCE_OPTIONS,
        "spell_punct_options": SPELL_PUNCT_OPTIONS,
        "chatterbox_voice_options": chatterbox_voices.list_voices(),
        "llm_model": state["llm_model"],
        "llm_options": state["llm_options"],
        "opik_enabled": OPIK_ENABLED,
        "opik_url": OPIK_BASE_URL,
    }


@router.post("/api/config")
async def post_config(request: Request):
    data = await request.json()
    updates = {}
    cfg = await app_config.get_config()
    if "tts_engine" in data:
        if data["tts_engine"] not in ("piper", "kokoro", "chatterbox"):
            return JSONResponse({"error": "tts_engine must be piper, kokoro, or chatterbox"}, 400)
        updates["tts_engine"] = data["tts_engine"]
        # "processed" pause mode is not valid for Chatterbox (see
        # TTS_PAUSE_MODE_OPTIONS) — fall back to punctuation so switching to
        # it never leaves a now-unsupported mode active.
        if data["tts_engine"] == "chatterbox" and cfg.get("spell_pause_mode") == "processed":
            updates["spell_pause_mode"] = "punctuation"
    if "vad_mode" in data:
        if data["vad_mode"] not in ("silero", "rms"):
            return JSONResponse({"error": "vad_mode must be silero or rms"}, 400)
        updates["vad_mode"] = data["vad_mode"]
    if "stay_awake" in data:
        updates["stay_awake"] = "1" if data["stay_awake"] else "0"
    if "kokoro_voice" in data:
        if data["kokoro_voice"] not in KOKORO_VOICE_OPTIONS:
            return JSONResponse({"error": f"kokoro_voice must be one of {KOKORO_VOICE_OPTIONS}"}, 400)
        updates["kokoro_voice"] = data["kokoro_voice"]
    if "spell_engine" in data:
        if data["spell_engine"] not in ("piper", "match"):
            return JSONResponse({"error": "spell_engine must be piper or match"}, 400)
        updates["spell_engine"] = data["spell_engine"]
    if "spell_pause_mode" in data:
        engine = updates.get("tts_engine", cfg.get("tts_engine"))
        allowed = TTS_PAUSE_MODE_OPTIONS.get(engine, ["processed", "silence", "punctuation"])
        if data["spell_pause_mode"] not in allowed:
            return JSONResponse({"error": f"spell_pause_mode for {engine} must be one of {allowed}"}, 400)
        updates["spell_pause_mode"] = data["spell_pause_mode"]
    if "spell_silence_s" in data:
        if data["spell_silence_s"] not in SPELL_SILENCE_OPTIONS:
            return JSONResponse({"error": f"spell_silence_s must be one of {SPELL_SILENCE_OPTIONS}"}, 400)
        updates["spell_silence_s"] = data["spell_silence_s"]
    if "spell_punct" in data:
        if data["spell_punct"] not in SPELL_PUNCT_OPTIONS:
            return JSONResponse({"error": f"spell_punct must be one of {SPELL_PUNCT_OPTIONS}"}, 400)
        updates["spell_punct"] = data["spell_punct"]
    if "chatterbox_voice" in data:
        voice = data["chatterbox_voice"] or ""
        if voice and not chatterbox_voices.voice_exists(voice):
            return JSONResponse({"error": f"unknown chatterbox_voice {voice!r}"}, 400)
        updates["chatterbox_voice"] = voice
    if updates:
        await app_config.set_config(**updates)
    return {"ok": True, "config": await app_config.get_config()}


# ── Chatterbox cloned-voice reference clips (record in Settings) ────────
@router.get("/api/chatterbox/voices")
async def list_chatterbox_voices():
    return {"voices": chatterbox_voices.list_voices()}


@router.post("/api/chatterbox/voices")
async def upload_chatterbox_voice(request: Request, file: UploadFile = File(...)):
    form = await request.form()
    name = str(form.get("name", "")).strip()
    if not name:
        return JSONResponse({"error": "name is required"}, 400)
    safe_name = chatterbox_voices.sanitize_voice_name(name)
    if not safe_name:
        return JSONResponse({"error": "name must contain at least one letter, digit, - or _"}, 400)

    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "empty upload"}, 400)

    dest = chatterbox_voices.voice_path(safe_name)
    # Browsers record via MediaRecorder as webm/opus (or similar), not WAV —
    # ffmpeg converts whatever container/codec comes in to the mono 24kHz
    # WAV Chatterbox expects as an audio_prompt_path reference.
    with tempfile.NamedTemporaryFile(suffix=".input") as src:
        src.write(raw)
        src.flush()
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", src.name, "-ac", "1", "-ar", "24000", dest,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            log.warning("ffmpeg voice conversion failed for %r: %s", safe_name, stderr.decode(errors="replace")[-2000:])
            return JSONResponse({"error": "could not decode uploaded audio"}, 400)

    return {"ok": True, "name": safe_name, "voices": chatterbox_voices.list_voices()}


@router.delete("/api/chatterbox/voices/{name}")
async def delete_chatterbox_voice(name: str):
    if not chatterbox_voices.delete_voice(name):
        return JSONResponse({"error": "voice not found"}, 404)
    cfg = await app_config.get_config()
    if cfg["chatterbox_voice"] == name:
        await app_config.set_config(chatterbox_voice="")
    return {"ok": True, "voices": chatterbox_voices.list_voices()}
