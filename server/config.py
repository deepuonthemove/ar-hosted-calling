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

TTS_ENGINE = os.getenv("TTS_ENGINE", "chatterbox")
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

# Chatterbox (Resemble AI) — zero-shot voice cloning TTS engine. Selected via
# TTS_ENGINE=chatterbox. Every character (narration AND spelling) is
# synthesized by Chatterbox itself, in the cloned voice — unlike Kokoro,
# there is no fallback to Piper for spelled-out runs, since the whole point
# of cloning is a single consistent voice across an entire response.
CHATTERBOX_DEVICE = os.getenv("CHATTERBOX_DEVICE", "cuda")
# Generation knobs — see Chatterbox's README for what these trade off
# (exaggeration: expressiveness/emotion intensity; cfg_weight: how closely
# the output follows the reference voice vs. the text prompt).
CHATTERBOX_EXAGGERATION = float(os.getenv("CHATTERBOX_EXAGGERATION", "0.5"))
CHATTERBOX_CFG_WEIGHT = float(os.getenv("CHATTERBOX_CFG_WEIGHT", "0.5"))
# Directory holding cloned-voice reference clips, recorded from the browser
# in Settings and saved as WAV (see routes/misc.py's /api/chatterbox/voices).
# Each file's basename (without extension) is the voice's selectable name.
CHATTERBOX_VOICES_DIR = os.getenv("CHATTERBOX_VOICES_DIR", "/models/chatterbox/voices")
# "" = Chatterbox's own built-in default voice (no cloning, no audio_prompt_path).
CHATTERBOX_VOICE = os.getenv("CHATTERBOX_VOICE", "")
# Chatterbox runs in its OWN container (chatterbox-tts service) with a separate
# transformers env so its deps don't clash with the main app (Kokoro needs
# transformers 4.x; chatterbox-tts pins 5.2.0). The main app calls it over HTTP.
CHATTERBOX_SERVICE_URL = os.getenv("CHATTERBOX_SERVICE_URL", "http://chatterbox-tts:8082")

# Real silence injected between spelled-out characters/groups, so pacing
# doesn't depend on how each engine happens to interpret punctuation.
SPELL_CHAR_GAP_MS = int(os.getenv("SPELL_CHAR_GAP_MS", "260"))
SPELL_GROUP_GAP_MS = int(os.getenv("SPELL_GROUP_GAP_MS", "420"))

# How the gap between spelled-out characters is produced:
#  - "processed" (default for piper/kokoro): each character goes through the
#    fade/RMS-match pipeline in _synth_char_bytes before a short fixed gap —
#    this is the original approach and stays the default for those engines.
#  - "silence": each character is spoken once, plainly (no fade, no RMS
#    gain-matching, no timeout/lock dance beyond what synthesis itself
#    needs), followed by a flat block of real silence — "B", <silence>, "U",
#    <silence>, "N". Trades a bit of loudness consistency for zero risk of
#    the fade/gain step introducing static, and a simpler, faster path.
#  - "punctuation": no per-character synth calls at all. The whole spelled
#    run is rebuilt as ONE text string — characters separated by a chosen
#    punctuation mark — and synthesized through the exact same single-stream
#    call used for normal narration. The pause comes from however the
#    engine naturally paces that punctuation mark, not an injected silence
#    array, so it can never carry over a splice/click artifact from
#    isolated single-character rendering. Durations below are typical,
#    approximate — not a guaranteed length, since it's the engine's own
#    prosody:
#      period "."  or semicolon ";"  ~= 0.4s
#      colon  ":"                    ~= 0.3s
#      comma  ","                    ~= 0.2s
#      line break "\n"               ~= 0.4s
#
# Chatterbox is excluded from "processed": its autoregressive model drifts
# on very short isolated renders (duplicates/drops a token), and its CLI
# pacing handles punctuation better than the fade/volume pipeline, so only
# "silence", "punctuation", and "plain" are offered per-engine
# (TTS_PAUSE_MODE_OPTIONS).
#
# "plain" skips per-character synthesis entirely: the whole utterance is
# preprocessed into ordinary spoken words (numbers -> digit names, hyphenated
# spellings -> spaced words) and streamed as one continuous narration. It is
# the natural voice-clone path for Chatterbox — see server/tts_plain.py.
SPELL_PAUSE_MODE = os.getenv("SPELL_PAUSE_MODE", "punctuation")
# Pause mode choices offered in the settings UI, scoped per TTS engine.
TTS_PAUSE_MODE_OPTIONS = {
    "chatterbox": ["silence", "punctuation", "plain"],
    "kokoro": ["processed", "silence", "punctuation", "plain"],
    "piper": ["processed", "silence", "punctuation", "plain"],
}
# Silence duration used between characters when SPELL_PAUSE_MODE=silence.
SPELL_SILENCE_S = float(os.getenv("SPELL_SILENCE_S", "1.0"))
SPELL_SILENCE_OPTIONS = [0.5, 0.8, 1.0]

# Punctuation mark injected between spelled characters when
# SPELL_PAUSE_MODE=punctuation.
SPELL_PUNCT = os.getenv("SPELL_PUNCT", "period")  # period|semicolon|colon|comma|linebreak
SPELL_PUNCT_CHARS = {
    "period": ".",
    "semicolon": ";",
    "colon": ":",
    "comma": ",",
    "linebreak": "\n",
}
SPELL_PUNCT_OPTIONS = list(SPELL_PUNCT_CHARS.keys())

VAD_MODE = os.getenv("VAD_MODE", "silero")  # "rms" or "silero"

# Opik (LLM evaluation/tracing) — optional
OPIK_ENABLED = os.getenv("OPIK_ENABLED", "0") == "1"
OPIK_BASE_URL = os.getenv("OPIK_BASE_URL", "http://opik-backend:8080")
OPIK_API_KEY = os.getenv("OPIK_API_KEY", "")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ar-voice-agent")
