"""Plain-mode single-stream TTS for chatterbox / kokoro / piper.

Unlike the sentinel-based spell modes (SPELL_START/SPELL_END + per-character
synthesis with injected silence), "plain" mode preprocesses the ENTIRE
utterance into clearly spoken words and synthesizes it as ONE continuous
stream, chunked only for latency:

- long numeric runs (NPIs, tax IDs, claim numbers) expand to digit names
  ("48449" -> "four eight four four nine") — no per-character drift,
- hyphenated spellings expand to spaced words ("1-3-1-0", "m-e-z-a"),
- a US "STATE ZIP" pair is spoken as spelled letters + zip digit names,
- dates/abbreviations pass through as written prose.

Because every number is a real word, the autoregressive engine never has to
emit long runs of isolated symbols, which is the source of the garbled-tail
Chatterbox clips. This is the natural voice-clone path for Chatterbox; the
same chunked preprocessed text streams cleanly through Kokoro and Piper.

Stream functions share the (text, stop_event=None, sent_log=None) signature
so they drop into browser_ws._stream_tts_reply / call_session._stream_tts and
report their verbatim payloads to the Opik tts span.
"""
import asyncio
import base64
import re

import httpx
import numpy as np

from . import chatterbox_voices
from .app_config import get_config as _get_config
from .config import (
    CHATTERBOX_CFG_WEIGHT, CHATTERBOX_EXAGGERATION, CHATTERBOX_SERVICE_URL,
    CHATTERBOX_VOICE, KOKORO_RATE, KOKORO_SPEED, KOKORO_VOICE,
    PIPER_DATA_DIR, PIPER_LENGTH_SCALE, PIPER_RATE, log,
)
from .state import state
from tts_common import SPELL_END, SPELL_PAUSE, SPELL_START

# ── Plain-mode text preprocessing ───────────────────────────────────────
_NUM_ID_RE = re.compile(r"\b\d{4,}\b")
_DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
)
_MONEY_RE = re.compile(r"\$([0-9]+(?:,[0-9]{3})*)(?:\.(\d{1,2}))?")
_STATE_ZIP_RE = re.compile(r"\b([A-Z]{2})\s*,?\s*(\d{5})\b")
_HYPHEN_SPELL_RE = re.compile(r"\b([A-Za-z0-9](?:-[A-Za-z0-9])+)\b")
# LLM spell-outs arrive space-separated inside the sentinels
# (e.g. "4 8 4 4 9", "M C W"): condense the digits into a run so _NUM_ID_RE
# expands them, and group a run of letters into small clusters so the
# autoregressive engine never has to pronounce many isolated single letters
# in one generation (the garble source for addresses).
_SPELLED_DIGITS_RE = re.compile(r"(?<=\d) (?=\d)")
_SPELLED_LETTERS_RE = re.compile(r"\b(?:[A-Za-z] ){1,}[A-Za-z]\b")
_SPELLED_LETTER_CHUNK = 4

_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _format_digits(match) -> str:
    """Expand a digit run into explicit spoken names for 100% precision."""
    s = match.group(0)
    return " ".join(_DIGIT_WORDS.get(c, c) for c in s)


def _say_two(n: int) -> str:
    """Speak a 2-digit number 10–99 as words ('26' -> 'twenty six')."""
    if n == 0:
        return "zero"
    tens = {1: "ten", 2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
            6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
    if n < 20:
        return {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
                14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
                18: "eighteen", 19: "nineteen"}[n]
    t, u = divmod(n, 10)
    return f"{tens[t]}{'-' + _DIGIT_WORDS[str(u)] if u else ''}"


def _say_year(year: int) -> str:
    """Speak a 4-digit year naturally ('1986' -> 'nineteen eighty six',
    '2026' -> 'twenty twenty six'); falls back to digit names otherwise."""
    if 1900 <= year <= 2099:
        return f"{_say_two(year // 100)} {_say_two(year % 100)}" \
            if year % 100 else f"{_say_two(year // 100)} hundred"
    return " ".join(_DIGIT_WORDS[d] for d in f"{year:04d}")


def _format_date(match) -> str:
    """Expand a date into a natural spoken phrase (e.g. '14th May 2026').

    Applied BEFORE the generic digit-run expansion, and the year is wordified
    here so '2026' is spoken as 'twenty twenty six' — never spelled out
    digit-by-digit as a generic 4-digit run.
    """
    from prompts import format_dos

    raw = match.group(1)
    spoken = format_dos(raw)
    if spoken == "unknown" or spoken == raw:
        return raw
    m = re.search(r"\b(\d{4})\b", spoken)
    if m:
        year = int(m.group(1))
        return spoken.replace(m.group(1), _say_year(year))
    return spoken


def _format_money(match) -> str:
    """Expand a currency amount into words so the engine always says the
    dollar(s) and cents — '$168.50' -> 'one hundred sixty eight dollars and
    fifty cents' instead of reading the raw digits with no currency unit."""
    dollars = int(match.group(1).replace(",", ""))
    cents = int(match.group(2).lstrip(".")) if match.group(2) else 0
    if dollars == 0 and cents == 0:
        return "zero dollars"
    words = _number_to_words(dollars) if dollars else "zero"
    s = f"{words} {'dollar' if dollars == 1 else 'dollars'}"
    if cents:
        cent_word = _say_two(cents) if cents >= 10 else _DIGIT_WORDS[str(cents)]
        s += f" and {cent_word} {'cent' if cents == 1 else 'cents'}"
    return s


def _number_to_words(n: int) -> str:
    """Spell an integer as words (up to 999,999): 236 -> 'two hundred
    thirty six'."""
    if n == 0:
        return "zero"
    if n < 1000:
        h, r = divmod(n, 100)
        parts = []
        if h:
            parts.append(_DIGIT_WORDS[str(h)])
            parts.append("hundred")
        if r:
            parts.append(_say_two(r) if r >= 10 else _DIGIT_WORDS[str(r)])
        return " ".join(parts)
    if n < 1_000_000:
        t, r = divmod(n, 1000)
        s = f"{_number_to_words(t)} thousand"
        if r:
            s += f" {_number_to_words(r)}"
        return s
    return " ".join(_DIGIT_WORDS[d] for d in str(n))


def _format_hyphen_spell(match) -> str:
    """Expand hyphenated spellings (1-3-1-0, m-e-z-a) into spaced words."""
    parts = match.group(1).split("-")
    return " ".join(_DIGIT_WORDS.get(p.lower(), p) for p in parts)


def _format_state_zip(match) -> str:
    state, zip_code = match.group(1), match.group(2)
    state_spelled = " ".join(list(state))
    zip_words = " ".join(_DIGIT_WORDS.get(c, c) for c in zip_code)
    return f"{state_spelled}, {zip_words}"


def _group_spelled_letters(match) -> str:
    """Split a run of space-separated single letters ("M C W") into small
    comma-separated clusters so no single engine generation has to voice
    many isolated letters. Note the "_SPELLED_LETTERS_RE" can not be
    compiled with a lookahead to cap the run length, so chunking here with
    a comma also gives _split_plain_chunks a natural break point."""
    letters = match.group(0).split()
    clusters = [" ".join(letters[i:i + _SPELLED_LETTER_CHUNK])
                for i in range(0, len(letters), _SPELLED_LETTER_CHUNK)]
    return ", ".join(clusters)


def tts_text(text: str) -> str:
    """Preprocess text for plain-mode TTS: numbers and spellings become
    ordinary spoken words so the engine reads everything in one clean
    stream, without sentinel markers or per-character synthesis."""
    if not text:
        return ""
    # Strip the LLM's spelled-out sentinels: they wrap content that other
    # pause modes synthesize character-by-character; here that content is
    # folded into the ordinary word stream instead.
    text = text.replace(SPELL_PAUSE, " ").replace(SPELL_START, "").replace(SPELL_END, "")
    text = _DATE_RE.sub(_format_date, text)
    text = _MONEY_RE.sub(_format_money, text)
    text = _STATE_ZIP_RE.sub(_format_state_zip, text)
    text = _HYPHEN_SPELL_RE.sub(_format_hyphen_spell, text)
    text = _SPELLED_DIGITS_RE.sub("", text)
    text = _SPELLED_LETTERS_RE.sub(_group_spelled_letters, text)
    text = _NUM_ID_RE.sub(_format_digits, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Chunking + per-engine synthesis ─────────────────────────────────────
def _split_plain_chunks(text: str, max_chars: int = 70) -> list[str]:
    """Split text into sentence/clause chunks for low-latency streaming."""
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]
    if not sentences:
        return [text] if text.strip() else []

    pieces = []
    for s in sentences:
        if len(s) <= max_chars:
            pieces.append(s)
            continue
        parts = re.split(r'(?<=,)\s+', s)
        buf = ""
        for p in parts:
            if buf and len(buf) + len(p) + 1 > max_chars:
                pieces.append(buf)
                buf = ""
                while len(p) > max_chars:
                    cut = p.rfind(" ", 0, max_chars)
                    if cut < max_chars // 2:
                        cut = max_chars
                    pieces.append(p[:cut])
                    p = p[cut:].lstrip()
            buf = f"{buf} {p}".strip()
        if buf:
            pieces.append(buf)
    return pieces or [text]


def _resolve_voice_name(voice_name: str) -> str:
    """Return the sanitized chatterbox voice if it exists, else '' (built-in)."""
    name = chatterbox_voices.sanitize_voice_name(voice_name) if voice_name else ""
    if name and chatterbox_voices.voice_exists(name):
        return name
    return ""


_chatterbox_lock: asyncio.Lock | None = None
_kokoro_lock: asyncio.Lock | None = None
_piper_lock: asyncio.Lock | None = None


async def _get_lock(name: str):
    global _chatterbox_lock, _kokoro_lock, _piper_lock
    locks = {"chatterbox": _chatterbox_lock, "kokoro": _kokoro_lock, "piper": _piper_lock}
    if locks[name] is None:
        lock = asyncio.Lock()
        if name == "chatterbox":
            _chatterbox_lock = lock
        elif name == "kokoro":
            _kokoro_lock = lock
        else:
            _piper_lock = lock
        return lock
    return locks[name]


async def _chatterbox_tts_bytes(text: str, voice_name: str) -> bytes:
    """Call the isolated chatterbox-tts service over HTTP, return int16 PCM
    at PIPER_RATE."""
    if not text.strip():
        return b""
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(f"{CHATTERBOX_SERVICE_URL}/tts", json={
                "text": text, "voice_name": voice_name,
                "exaggeration": CHATTERBOX_EXAGGERATION,
                "cfg_weight": CHATTERBOX_CFG_WEIGHT,
            })
            r.raise_for_status()
            d = r.json()
    except Exception as e:
        log.warning("[chatterbox-plain] tts service error: %s (text=%r, voice=%r)", e, text, voice_name)
        return b""
    pcm = np.frombuffer(base64.b64decode(d["pcm"]), dtype=np.int16)
    sr = int(d.get("sr", 24000))
    if sr != PIPER_RATE:
        from audio import resample
        f32 = pcm.astype(np.float32) / 32768.0
        f32 = resample(f32, sr, PIPER_RATE)
        pcm = np.clip(f32 * 32767, -32768, 32767).astype(np.int16)
    log.info("[chatterbox-plain] SENT text=%r voice=%r -> %d pcm @ %dHz (%.2fs)",
             text, voice_name, len(pcm), PIPER_RATE, len(pcm) / 2 / PIPER_RATE)
    return pcm.tobytes()


def _piper_next(gen):
    try:
        return next(gen)
    except StopIteration:
        return None


async def _piper_synth(text: str):
    """Synthesize one chunk through the in-process persistent PiperVoice."""
    voice = state.get("piper_voice")
    if voice is None:
        log.warning("[piper-plain] persistent PiperVoice not loaded")
        return b""
    from piper.config import SynthesisConfig
    cfg = SynthesisConfig(length_scale=PIPER_LENGTH_SCALE)
    loop = asyncio.get_running_loop()
    gen = voice.synthesize(text, cfg)
    buf = bytearray()
    while True:
        chunk = await loop.run_in_executor(None, _piper_next, gen)
        if chunk is None:
            break
        buf.extend(chunk.audio_int16_bytes)
    return bytes(buf)


async def _kokoro_synth(text: str, kokoro_voice: str = KOKORO_VOICE):
    """Synthesize one chunk through the in-process Kokoro pipeline."""
    from audio import resample
    pipeline = state.get("kokoro_pipeline")
    if not pipeline:
        log.warning("[kokoro-plain] Kokoro not loaded")
        return b""
    loop = asyncio.get_running_loop()
    gen = pipeline(text, voice=kokoro_voice, speed=KOKORO_SPEED)
    buf = bytearray()
    while True:
        result = await loop.run_in_executor(None, _piper_next, gen)
        if result is None:
            break
        if result.audio is None or len(result.audio) == 0:
            continue
        audio = result.audio
        if KOKORO_RATE != PIPER_RATE:
            audio = resample(audio, KOKORO_RATE, PIPER_RATE)
        pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        buf.extend(pcm.tobytes())
    return bytes(buf)


def _notify_sent(sent_log, text: str, label: str):
    if sent_log is not None:
        sent_log.append({"kind": label, "text": text})


async def plain_stream(text: str, engine: str = "chatterbox",
                       stop_event: "asyncio.Event | None" = None,
                       sent_log: list | None = None):
    """Shared plain-mode streamer: preprocess → chunk → synthesize per chunk."""
    cleaned = tts_text(text)
    if not cleaned:
        return

    cfg = await _get_config()
    if engine == "chatterbox":
        voice_name = _resolve_voice_name(cfg.get("chatterbox_voice", CHATTERBOX_VOICE))
    else:
        voice_name = cfg.get("kokoro_voice", KOKORO_VOICE)
    chunks = _split_plain_chunks(cleaned)
    lock = await _get_lock(engine)

    for i, chunk in enumerate(chunks):
        if stop_event is not None and stop_event.is_set():
            return
        async with lock:
            if stop_event is not None and stop_event.is_set():
                return
            if engine == "chatterbox":
                audio = await _chatterbox_tts_bytes(chunk, voice_name)
            elif engine == "kokoro":
                audio = await _kokoro_synth(chunk, kokoro_voice=voice_name)
            else:
                audio = await _piper_synth(chunk)
        _notify_sent(sent_log, chunk, f"plain-chunk:{engine}:{i}")
        if audio:
            yield audio


async def chatterbox_stream(text: str, stop_event: "asyncio.Event | None" = None,
                            sent_log: list | None = None):
    async for chunk in plain_stream(text, "chatterbox", stop_event=stop_event, sent_log=sent_log):
        yield chunk


async def kokoro_stream(text: str, stop_event: "asyncio.Event | None" = None,
                        sent_log: list | None = None):
    async for chunk in plain_stream(text, "kokoro", stop_event=stop_event, sent_log=sent_log):
        yield chunk


async def piper_stream(text: str, stop_event: "asyncio.Event | None" = None,
                       sent_log: list | None = None):
    async for chunk in plain_stream(text, "piper", stop_event=stop_event, sent_log=sent_log):
        yield chunk


def get_tts_fn_plain(engine: str = "chatterbox"):
    """Return the plain-mode stream generator for the requested engine."""
    return {
        "chatterbox": chatterbox_stream,
        "kokoro": kokoro_stream,
        "piper": piper_stream,
    }.get(engine, chatterbox_stream)