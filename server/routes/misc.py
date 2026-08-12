"""Health, LLM switching, live/review/detail lookups, and app config API."""
import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import app_config
from ..config import KOKORO_VOICE_OPTIONS, WHISPER_MODEL_SIZE, log
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
        "tts_options": ["piper", "kokoro"],
        "vad_options": ["silero", "rms"],
        "kokoro_voice_options": KOKORO_VOICE_OPTIONS,
        "spell_engine_options": ["piper", "match"],
        "llm_model": state["llm_model"],
        "llm_options": state["llm_options"],
    }


@router.post("/api/config")
async def post_config(request: Request):
    data = await request.json()
    updates = {}
    if "tts_engine" in data:
        if data["tts_engine"] not in ("piper", "kokoro"):
            return JSONResponse({"error": "tts_engine must be piper or kokoro"}, 400)
        updates["tts_engine"] = data["tts_engine"]
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
    if updates:
        await app_config.set_config(**updates)
    return {"ok": True, "config": await app_config.get_config()}
