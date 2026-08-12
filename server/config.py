"""Environment-driven configuration constants, shared across the server package."""
import logging
import os

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "distil-large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8001/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
LLM_MODEL_OPTIONS = {
    "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "Llama 3.1 8B": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
}
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
PUBLIC_DOMAIN = os.getenv("PUBLIC_DOMAIN", "localhost:8080")
PUBLIC_SCHEME = os.getenv("PUBLIC_SCHEME", "https")

PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
PIPER_RATE = int(os.getenv("PIPER_SAMPLE_RATE", "22050"))
PIPER_DATA_DIR = os.getenv("PIPER_DATA_DIR", "/models/piper")
# >1.0 = slower (length-scale), 1.0 = normal
PIPER_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "1.35"))
# Length-scale for spelled-out characters. Same lesson as Kokoro's speed:
# pushing this well above normal (was 2.0) over-stretches the vowel in each
# character's name ("es" -> a drawled "eeeese", "ay" -> "aaaaye"), hurting
# recognizability instead of helping it. Articulation is clearest at
# close-to-normal speed; the slow, clear "spelling" feel comes from the real
# silence gaps injected between characters (SPELL_CHAR_GAP_MS), not from
# stretching each word out.
PIPER_SPELL_LENGTH_SCALE = float(os.getenv("PIPER_SPELL_LENGTH_SCALE", "1.4"))
# Optional extra tuning knobs for the spelled-character voice only — leave
# unset (None) to keep Piper's library defaults, which is what every voice
# has been tuned against so far. Only touch these when experimenting with a
# specific voice model that sounds flat/robotic in isolation; they do not
# affect normal (non-spelled) narration.
PIPER_SPELL_NOISE_SCALE = os.getenv("PIPER_SPELL_NOISE_SCALE")
PIPER_SPELL_NOISE_SCALE = float(PIPER_SPELL_NOISE_SCALE) if PIPER_SPELL_NOISE_SCALE else None
PIPER_SPELL_NOISE_W_SCALE = os.getenv("PIPER_SPELL_NOISE_W_SCALE")
PIPER_SPELL_NOISE_W_SCALE = float(PIPER_SPELL_NOISE_W_SCALE) if PIPER_SPELL_NOISE_W_SCALE else None

TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_bella")
KOKORO_RATE = 24000
# <1.0 = slower (speed), 1.0 = normal
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "0.81"))
KOKORO_DEVICE = os.getenv("KOKORO_DEVICE", "cuda")
# Speed for Kokoro when it IS used for spelled characters (SPELL_ENGINE=
# "match" below). Same lesson as Piper's spell length-scale: stay close to
# normal speed, since slowing Kokoro below ~1.0 breaks its duration model
# (a bogus near-silent gap inside the word) — pacing comes from the real
# silence gaps (SPELL_CHAR_GAP_MS), not from stretching the render.
KOKORO_SPELL_SPEED = float(os.getenv("KOKORO_SPELL_SPEED", "1.0"))
# Kokoro is NOT used for spelled-out characters by default — measured on
# real audio (voice=af_bella) that Kokoro's isolated single-word renders
# (e.g. "tee") come out both less articulate and quieter than Piper's.
# Gain-boosting fixed the loudness (see _fade_edges' RMS matching) but not
# the clarity, so every spelled character goes through Piper instead, even on
# a call that's otherwise using Kokoro — see _spell_char_audio. Since that
# was only ever measured against af_bella, set SPELL_ENGINE=match to
# re-test it against a different Kokoro voice (e.g. af_heart): this makes
# spelling use whatever engine/voice is doing the narration, so if that
# voice's isolated-character quality holds up, the whole call — narration
# AND spelling — becomes one single, fully consistent voice.
SPELL_ENGINE = os.getenv("SPELL_ENGINE", "piper")  # "piper" or "match"
# American voices worth A/B-ing from the settings UI. af_heart is Kokoro's
# own recommended default and tests warmer/more natural than af_bella in
# blind listening comparisons; the rest are other common built-in options.
KOKORO_VOICE_OPTIONS = [
    "af_bella", "af_heart", "af_sarah", "af_nicole", "af_sky",
    "am_michael", "am_adam",
]

# Real silence injected between spelled-out characters/groups, so pacing
# doesn't depend on how each engine happens to interpret punctuation.
SPELL_CHAR_GAP_MS = int(os.getenv("SPELL_CHAR_GAP_MS", "260"))
SPELL_GROUP_GAP_MS = int(os.getenv("SPELL_GROUP_GAP_MS", "420"))

VAD_MODE = os.getenv("VAD_MODE", "silero")  # "rms" or "silero"

# Opik (LLM evaluation/tracing) — optional
OPIK_ENABLED = os.getenv("OPIK_ENABLED", "0") == "1"
OPIK_BASE_URL = os.getenv("OPIK_BASE_URL", "http://opik-backend:8080")
OPIK_API_KEY = os.getenv("OPIK_API_KEY", "")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ar-voice-agent")
