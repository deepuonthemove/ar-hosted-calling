"""App config (TTS/VAD defaults, persisted in Redis, editable via /api/config).

Kept separate from tts.py and the route modules so both can import it without
a circular dependency: tts.py needs it to resolve the active kokoro_voice/
spell_engine per call, and routes/misc.py exposes it over the API.
"""
from .config import (
    CHATTERBOX_VOICE, KOKORO_VOICE, SPELL_ENGINE, SPELL_PAUSE_MODE, SPELL_PUNCT,
    SPELL_SILENCE_S, TTS_ENGINE, VAD_MODE,
)
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
        "spell_pause_mode": cfg.get("spell_pause_mode", SPELL_PAUSE_MODE),
        "spell_silence_s": cfg.get("spell_silence_s", str(SPELL_SILENCE_S)),
        "spell_punct": cfg.get("spell_punct", SPELL_PUNCT),
        "chatterbox_voice": cfg.get("chatterbox_voice", CHATTERBOX_VOICE),
    }


async def set_config(**kwargs):
    await state["redis"].hset(CONFIG_KEY, mapping={k: str(v) for k, v in kwargs.items()})
