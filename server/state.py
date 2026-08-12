"""Shared in-process state dict + model loading + Opik tracing helpers."""
import asyncio
import datetime

import redis.asyncio as aioredis
from faster_whisper import WhisperModel
from openai import AsyncOpenAI

from .config import (
    KOKORO_DEVICE, KOKORO_VOICE, KOKORO_SPEED, LLM_MODEL, LLM_MODEL_OPTIONS,
    OPIK_API_KEY, OPIK_BASE_URL, OPIK_ENABLED, PIPER_DATA_DIR, PIPER_VOICE,
    REDIS_URL, VLLM_BASE_URL, WHISPER_COMPUTE, WHISPER_DEVICE, WHISPER_MODEL_SIZE,
    log,
)

state: dict = {}
_bg_tasks: set = set()  # strong refs for background tasks (avoids GC cancellation)


def opik_span(trace, name: str, span_type: str, input_data, output_data,
              start_ts: float | None = None, end_ts: float | None = None,
              model: str | None = None, metadata: dict | None = None):
    """Create a COMPLETE span in a single call (no separate .end()).

    Opik batches span messages; calling span.end() shortly after creation can
    drop the create payload (name/type/start_time lost → epoch-0 timestamps).
    Measuring start/end ourselves and passing them with input+output up front
    avoids that race entirely and yields a real duration.
    """
    if trace is None:
        return None
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        start = (datetime.datetime.fromtimestamp(start_ts, tz=datetime.timezone.utc)
                 if start_ts is not None else now)
        end = (datetime.datetime.fromtimestamp(end_ts, tz=datetime.timezone.utc)
               if end_ts is not None else now)
        return trace.span(name=name, type=span_type, input=input_data, output=output_data,
                          start_time=start, end_time=end,
                          model=model, provider="vllm" if span_type == "llm" else None,
                          metadata=dict(metadata or {}))
    except Exception:
        return None


def opik_start_trace(name: str, thread_id: str, input_data, metadata: dict | None = None):
    """Start a trace grouped into a thread (one per call/chat session)."""
    client = state.get("opik")
    if not client:
        return None
    try:
        return client.trace(name=name, input=input_data, metadata=metadata or {},
                            thread_id=thread_id, project_name="ar-voice-agent")
    except Exception:
        return None


def opik_end_trace(trace, output=None, error: str | None = None):
    if trace is None:
        return
    try:
        if error:
            trace.end(output={"error": error}, error_info={"error_type": "error", "message": error})
        else:
            trace.end(output=output)
    except Exception:
        pass


def load_models():
    log.info("Loading Faster-Whisper %s ...", WHISPER_MODEL_SIZE)
    state["stt_model"] = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    state["stt_lock"] = asyncio.Lock()
    state["llm_client"] = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
    state["llm_model"] = LLM_MODEL
    state["llm_options"] = LLM_MODEL_OPTIONS
    state["redis"] = aioredis.from_url(REDIS_URL, decode_responses=True)

    # Load Piper ONNX model persistently (eliminates ~1.8s subprocess startup per utterance)
    try:
        from piper import PiperVoice
        state["piper_voice"] = PiperVoice.load(f"{PIPER_DATA_DIR}/{PIPER_VOICE}.onnx")
        log.info("PiperVoice loaded persistently (%s)", PIPER_VOICE)
    except Exception as e:
        log.warning("Persistent Piper load failed, falling back to subprocess: %s", e)
        state["piper_voice"] = None

    try:
        from kokoro import KPipeline
        state["kokoro_pipeline"] = KPipeline(lang_code='a', device=KOKORO_DEVICE)
        log.info("Kokoro TTS loaded (voice=%s, device=%s)", KOKORO_VOICE, KOKORO_DEVICE)
        # Warm up the voice + CUDA at startup so the FIRST real synthesis is
        # never delayed by a lazy voice download / GPU init.
        try:
            _ = list(state["kokoro_pipeline"]("Hello.", voice=KOKORO_VOICE, speed=KOKORO_SPEED))
        except Exception as we:
            log.warning("Kokoro warmup failed: %s", we)
    except Exception as e:
        log.warning("Kokoro not available, will fall back to Piper: %s", e)
        state["kokoro_pipeline"] = None

    # Opik LLM evaluation/tracing
    if OPIK_ENABLED:
        try:
            import opik as opik_sdk
            # Batching can silently drop fast traces (create → end within the
            # batch window). Disable it on the client so every trace/span is
            # sent immediately.
            opik_sdk.configure(api_key=OPIK_API_KEY or "local", use_local=True,
                               url_override=OPIK_BASE_URL, project_name="ar-voice-agent")
            opik_client = opik_sdk.Opik(batching=False)
            state["opik"] = opik_client
            log.info("Opik enabled at %s", OPIK_BASE_URL)
        except Exception as e:
            log.warning("Opik init failed (disabled): %s", e)
            state["opik"] = None
    else:
        state["opik"] = None

    # Preload Silero VAD (model downloads on first run)
    try:
        from audio import load_silero_vad
        model, _ = load_silero_vad()
        state["silero_loaded"] = model is not None
        log.info("Silero VAD loaded: %s", state["silero_loaded"])
    except Exception as e:
        log.warning("Silero VAD preload failed: %s", e)
        state["silero_loaded"] = False

    log.info("Models loaded.")
