"""App config (TTS/VAD defaults, persisted in Redis, editable via /api/config).

Kept separate from tts.py and the route modules so both can import it without
a circular dependency: tts.py needs it to resolve the active kokoro_voice/
spell_engine per call, and routes/misc.py exposes it over the API.
"""
from .config import KOKORO_VOICE, SPELL_ENGINE, TTS_ENGINE, VAD_MODE
from .state import state

CONFIG_KEY = "config:app"


async def get_config() -> dict:
    cfg = await state["redis"].hgetall(CONFIG_KEY)
    return {
        "tts_engine": cfg.get("tts_engine", TTS_ENGINE),
        "vad_mode": cfg.get("vad_mode", VAD_MODE),
        "stay_awake": cfg.get("stay_awake", "0"),
        "kokoro_voice": cfg.get("kokoro_voice", KOKORO_VOICE),
        "spell_engine": cfg.get("spell_engine", SPELL_ENGINE),
    }


async def set_config(**kwargs):
    await state["redis"].hset(CONFIG_KEY, mapping={k: str(v) for k, v in kwargs.items()})
