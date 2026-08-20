"""TTS text preprocessing + Piper/Kokoro streaming synthesis.

Text transforms are shared (tts_common.py) — spelled-out runs (IDs,
addresses, letter-by-letter names) come back wrapped in SPELL_START/
SPELL_END sentinels. The audio layer below splits on those and synthesizes
spelled segments character-by-character, at a slower rate, with real
silence injected between characters — pacing that doesn't depend on how
either engine happens to interpret punctuation.
"""
import asyncio
import os
import re
import time

import numpy as np

import tts_piper as _tts_piper_mod
import tts_kokoro as _tts_kokoro_mod
import tts_chatterbox as _tts_chatterbox_mod
from tts_common import split_spell_segments, flatten_spell_segments, SPELL_PAUSE

from . import chatterbox_voices
from .app_config import get_config as _get_config
from .config import (
    CHATTERBOX_CFG_WEIGHT, CHATTERBOX_EXAGGERATION, CHATTERBOX_SERVICE_URL,
    CHATTERBOX_VOICE,
    KOKORO_RATE, KOKORO_SPEED, KOKORO_SPELL_SPEED, KOKORO_VOICE,
    PIPER_DATA_DIR, PIPER_LENGTH_SCALE, PIPER_RATE, PIPER_SPELL_LENGTH_SCALE,
    PIPER_SPELL_NOISE_SCALE, PIPER_SPELL_NOISE_W_SCALE, PIPER_VOICE,
    SPELL_CHAR_GAP_MS, SPELL_ENGINE, SPELL_GROUP_GAP_MS, SPELL_PAUSE_MODE,
    SPELL_PUNCT, SPELL_PUNCT_CHARS, SPELL_SILENCE_S, log,
)
from .state import state

_TTS_MODS = {"piper": _tts_piper_mod, "kokoro": _tts_kokoro_mod, "chatterbox": _tts_chatterbox_mod}


def tts_text(text: str, engine: str = "piper") -> str:
    """Preprocess text for TTS using the engine-specific transform module."""
    return _TTS_MODS.get(engine, _tts_piper_mod).tts_text(text)


def _piper_next(gen):
    # Sentinel wrapper: StopIteration must not cross the executor/Future boundary.
    try:
        return next(gen)
    except StopIteration:
        return None


def _silence_bytes(ms: int) -> bytes:
    n = int(PIPER_RATE * ms / 1000)
    return np.zeros(n, dtype=np.int16).tobytes()


# Feeding the engine a bare, context-free symbol ("T", "5") triggers its
# "spell this out" handling, which both engines render with a leading
# percussive "t"/click — happens for digits too, and trimming to remove it
# sometimes ate the real letter instead (short letters vs. a fixed-ms trim
# can't be told apart reliably). That means the artifact is baked into the
# generated content, not a splice/boundary effect — no amount of waveform
# post-processing fixes that. So instead we don't synthesize the bare
# symbol at all: each character is spoken as its ordinary name ("tee",
# "five") — a normal short word, which is squarely inside what these
# engines are trained on and renders cleanly.
_LETTER_NAMES = {
    "A": "ay", "B": "bee", "C": "see", "D": "dee", "E": "ee", "F": "eff",
    "G": "jee", "H": "aitch", "I": "eye", "J": "jay", "K": "kay", "L": "el",
    "M": "em", "N": "en", "O": "oh", "P": "pee", "Q": "cue", "R": "are",
    "S": "es", "T": "tee", "U": "you", "V": "vee", "W": "double you",
    "X": "ex", "Y": "why", "Z": "zee",
}
_DIGIT_NAMES = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _spoken_char(tok: str) -> str:
    return _LETTER_NAMES.get(tok.upper(), _DIGIT_NAMES.get(tok, tok))


# Safety-net fade for a clean splice into the surrounding silence. A LINEAR
# ramp has a sharp corner where it meets the un-faded audio (an instant
# change in slope) — that corner is itself audible as a soft click/tick,
# especially on the fade-out when it lands mid-consonant. A raised-cosine
# (Hann) ramp has zero slope at both ends, so it blends into flat audio and
# into silence with no corner — this is what actually kills the click, not
# just the fade duration.
_SPELL_FADE_MS = int(os.getenv("SPELL_FADE_MS", "12"))

# Measured on real audio: an isolated single-word render (what a spelled
# character becomes, e.g. "tee") comes out noticeably quieter than normal
# continuous speech — peak-matching alone doesn't fix this because peak
# only reflects the single loudest sample, not how loud the character
# *sounds* over its duration (a short sharp transient can peak-match normal
# speech and still sound quiet). RMS (average energy) tracks perceived
# loudness much better, so each character is scaled to match the RMS of the
# narration immediately preceding it in the same response — measured live,
# per call, per voice — rather than a fixed guessed constant. This target is
# the fallback only for a response that opens with a spelled run before any
# narration has played yet.
_SPELL_TARGET_RMS = float(os.getenv("SPELL_TARGET_RMS", "6000"))
# Cap how hard a near-silent render can be amplified — otherwise a mostly-
# silent glitch chunk would get boosted into a loud burst of noise.
_SPELL_MAX_GAIN = float(os.getenv("SPELL_MAX_GAIN", "6.0"))


def _rms(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def _fade_edges(pcm_bytes: bytes, fade_ms: int = _SPELL_FADE_MS,
                 target_rms: float | None = None) -> bytes:
    if not pcm_bytes:
        return pcm_bytes
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    rms = _rms(arr)
    target = target_rms if target_rms else _SPELL_TARGET_RMS
    if rms > 0 and target > 0:
        gain = min(target / rms, _SPELL_MAX_GAIN)
        arr = np.clip(arr * gain, -32767, 32767)
    n = len(arr)
    fade_n = min(int(PIPER_RATE * fade_ms / 1000), n // 2)
    if fade_n > 0:
        # Raised-cosine (Hann) half-window: starts/ends at zero slope.
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade_n, dtype=np.float32))
        arr[:fade_n] *= ramp
        arr[-fade_n:] *= ramp[::-1]
    return arr.astype(np.int16).tobytes()


# Each spelled character is its own fresh synth call (phonemizer + model
# invocation from scratch), which can occasionally take much longer than a
# continuation chunk of normal speech would. asyncio can only deliver a
# cancellation (barge-in) at the next await checkpoint — if that checkpoint
# is buried inside one slow character call, cancellation is deferred until
# it finishes. On a long spelled ID that delay stacks up across characters
# and barge-in can feel like it isn't working. Bound each character to a
# timeout so a slow/stuck call can never block cancellation for long.
SPELL_CHAR_TIMEOUT_S = float(os.getenv("SPELL_CHAR_TIMEOUT_S", "2.5"))


async def _synth_char_bytes(char_gen, label: str = "?", engine: str = "?",
                             timeout: float = SPELL_CHAR_TIMEOUT_S,
                             target_rms: float | None = None) -> bytes:
    """Buffer one spelled character's audio (always short) and fade its
    edges, instead of streaming it raw — removes the click/static pop that a
    fresh isolated synth call produces at onset. Bounded by `timeout` so a
    single character can't stall cancellation indefinitely. Logs how long
    the call took, so it's visible whether per-character synth is the thing
    delaying barge-in during a long spell."""
    async def _collect():
        buf = bytearray()
        async for b in char_gen:
            buf.extend(b)
        return bytes(buf)

    t0 = time.time()
    try:
        buf = await asyncio.wait_for(_collect(), timeout=timeout)
    except asyncio.TimeoutError:
        log.warning("[spell] %s char %r: exceeded %.1fs — skipping it", engine, label, timeout)
        return b""
    elapsed_ms = (time.time() - t0) * 1000
    log.info("[spell] %s char %r: %.0fms", engine, label, elapsed_ms)
    return _fade_edges(buf, target_rms=target_rms)


# Serializes in-process Piper synthesis. The shared ONNX session is not safe
# under concurrency (e.g. a cancelled TTS leaving an executor thread mid-inference).
_tts_lock = None


async def _get_tts_lock():
    global _tts_lock
    if _tts_lock is None:
        _tts_lock = asyncio.Lock()
    return _tts_lock


async def _piper_spell_char_audio(tok: str, stop_event: "asyncio.Event | None" = None,
                                   target_rms: float | None = None) -> bytes:
    voice = state.get("piper_voice")
    if voice is None:
        return b""
    from piper.config import SynthesisConfig
    cfg_kwargs = {"length_scale": PIPER_SPELL_LENGTH_SCALE}
    if PIPER_SPELL_NOISE_SCALE is not None:
        cfg_kwargs["noise_scale"] = PIPER_SPELL_NOISE_SCALE
    if PIPER_SPELL_NOISE_W_SCALE is not None:
        cfg_kwargs["noise_w_scale"] = PIPER_SPELL_NOISE_W_SCALE
    cfg = SynthesisConfig(**cfg_kwargs)
    loop = asyncio.get_running_loop()

    async def synth():
        gen = voice.synthesize(_spoken_char(tok), cfg)
        while True:
            chunk = await loop.run_in_executor(None, _piper_next, gen)
            if chunk is None:
                break
            yield chunk.audio_int16_bytes

    lock = await _get_tts_lock()
    async with lock:
        if stop_event is not None and stop_event.is_set():
            return b""
        return await _synth_char_bytes(synth(), label=tok, engine="piper", target_rms=target_rms)


async def _kokoro_spell_char_audio(tok: str, stop_event: "asyncio.Event | None" = None,
                                    kokoro_voice: str = KOKORO_VOICE,
                                    target_rms: float | None = None) -> bytes:
    """Experimental: spell a character through Kokoro itself (spell_engine=
    match), so narration and spelling use the exact same voice. Only worth
    using if the active kokoro_voice's isolated-word renders hold up — see
    the SPELL_ENGINE comment above KOKORO_SPELL_SPEED."""
    from audio import resample
    pipeline = state.get("kokoro_pipeline")
    if not pipeline:
        return b""
    loop = asyncio.get_running_loop()

    async def synth():
        gen = pipeline(_spoken_char(tok), voice=kokoro_voice, speed=KOKORO_SPELL_SPEED)
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
            yield pcm.tobytes()

    lock = await _get_kokoro_lock()
    async with lock:
        if stop_event is not None and stop_event.is_set():
            return b""
        return await _synth_char_bytes(synth(), label=tok, engine="kokoro", target_rms=target_rms)


# Measured on real audio: Kokoro's isolated single-word renders (what a
# spelled-out character becomes, e.g. "tee") come out both quieter AND less
# articulate than Piper's — gain-boosting the volume didn't fix clarity
# because the underlying render itself is muddier, not just quiet. Piper's
# isolated-word renders measured clean (continuous envelope, near-full-scale
# volume) in the same tests. So by default EVERY spelled character is
# synthesized via Piper, even on a call that's otherwise using Kokoro for
# normal speech. Set SPELL_ENGINE=match to route spelling through whichever
# engine is doing the narration instead (for re-testing Kokoro with a
# different voice, e.g. af_heart).
async def _spell_char_audio(engine: str, tok: str, stop_event: "asyncio.Event | None" = None,
                             kokoro_voice: str = KOKORO_VOICE,
                             target_rms: float | None = None) -> bytes:
    if stop_event is not None and stop_event.is_set():
        return b""
    if engine == "kokoro":
        return await _kokoro_spell_char_audio(tok, stop_event=stop_event, kokoro_voice=kokoro_voice,
                                               target_rms=target_rms)
    return await _piper_spell_char_audio(tok, stop_event=stop_event, target_rms=target_rms)


# "silence" spell-pause mode: speak each character once, plainly — no fade,
# no RMS gain-matching, no timeout wrapper — then a flat block of real
# silence. Simpler and faster than the "processed" path above; trades away
# loudness/click cleanup in exchange for never risking the fade/gain step
# itself introducing static.
#
# Unlike the "processed" path (_spell_char_audio), this synthesizes the bare
# character itself ("A", "1") rather than its spoken name ("ay", "one") —
# intentional per product request, even though the engine's "spell this
# out" handling for a bare symbol can add a leading percussive click.
async def _piper_plain_char_audio(tok: str, stop_event: "asyncio.Event | None" = None) -> bytes:
    voice = state.get("piper_voice")
    if voice is None:
        return b""
    from piper.config import SynthesisConfig
    cfg = SynthesisConfig(length_scale=PIPER_LENGTH_SCALE)
    loop = asyncio.get_running_loop()
    lock = await _get_tts_lock()
    async with lock:
        if stop_event is not None and stop_event.is_set():
            return b""
        buf = bytearray()
        gen = voice.synthesize(tok, cfg)
        while True:
            chunk = await loop.run_in_executor(None, _piper_next, gen)
            if chunk is None:
                break
            buf.extend(chunk.audio_int16_bytes)
        return bytes(buf)


async def _kokoro_plain_char_audio(tok: str, stop_event: "asyncio.Event | None" = None,
                                    kokoro_voice: str = KOKORO_VOICE) -> bytes:
    from audio import resample
    pipeline = state.get("kokoro_pipeline")
    if not pipeline:
        return b""
    loop = asyncio.get_running_loop()
    lock = await _get_kokoro_lock()
    async with lock:
        if stop_event is not None and stop_event.is_set():
            return b""
        buf = bytearray()
        gen = pipeline(tok, voice=kokoro_voice, speed=KOKORO_SPEED)
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


async def _plain_spell_char_audio(engine: str, tok: str, stop_event: "asyncio.Event | None" = None,
                                   kokoro_voice: str = KOKORO_VOICE) -> bytes:
    if stop_event is not None and stop_event.is_set():
        return b""
    if engine == "kokoro":
        return await _kokoro_plain_char_audio(tok, stop_event=stop_event, kokoro_voice=kokoro_voice)
    return await _piper_plain_char_audio(tok, stop_event=stop_event)


# "punctuation" spell-pause mode: rebuild a spelled run as ONE text string —
# characters separated by the configured punctuation mark — so it never goes
# through the isolated per-character synth path at all. It is folded back
# into a plain (is_spell=False) segment, so it synthesizes through the exact
# same single continuous call as ordinary narration: no per-character synth,
# no fade, no RMS gain-matching. See SPELL_PAUSE_MODE in config.py for the
# approximate pause each punctuation mark tends to produce.
def _punct_spelled_text(content, punct_char: str) -> str:
    units = []
    for tok in content:
        if tok == SPELL_PAUSE:
            # Longer break at a natural grouping boundary: double up the
            # mark on the preceding unit instead of a special-cased gap.
            if units:
                units[-1] = units[-1] + punct_char
            continue
        units.append(tok)
    if not units:
        return ""
    if punct_char == "\n":
        return "\n".join(units)
    return (punct_char + " ").join(units) + punct_char


def _apply_punct_spell_mode(segments, punct_char: str):
    out = []
    for is_spell, content in segments:
        if is_spell:
            text = _punct_spelled_text(content, punct_char)
            if text:
                out.append((False, text))
        else:
            out.append((False, content))
    return out


# ── Piper TTS streaming ──────────────────────────────────────────────────
async def _piper_synth_segments(segments, stop_event: "asyncio.Event | None" = None,
                                 spell_pause_mode: str = SPELL_PAUSE_MODE,
                                 spell_silence_s: float = SPELL_SILENCE_S,
                                 sent_log: list | None = None):
    """Synthesize (is_spelled, content) segments from split_spell_segments():
    plain segments at normal speed, spelled segments one character at a time
    at a slower length-scale with real silence between characters.

    `stop_event`, when set, is checked at every segment/character boundary
    so a barge-in stops the NEXT unit of work immediately — it doesn't wait
    on asyncio task-cancellation semantics, which can be deferred until a
    slow in-flight synth call finishes when a spelled run has many
    characters queued up.

    `spell_pause_mode="silence"` swaps the fade/RMS-matched character path
    for a plain character render + flat silence block (see
    _plain_spell_char_audio)."""
    voice = state.get("piper_voice")
    if voice is not None:
        # Persistent in-process model — first audio in ~0.1s (vs ~1.8s subprocess)
        from piper.config import SynthesisConfig
        cfg_normal = SynthesisConfig(length_scale=PIPER_LENGTH_SCALE)
        loop = asyncio.get_running_loop()

        async def synth(t, cfg):
            gen = voice.synthesize(t, cfg)
            while True:
                chunk = await loop.run_in_executor(None, _piper_next, gen)
                if chunk is None:
                    break
                yield chunk.audio_int16_bytes

        # Track the RMS of the most recent plain-narration segment so
        # spelled characters that follow it can be loudness-matched to what
        # this call actually sounds like, instead of a fixed guess.
        last_rms = None
        for is_spell, content in segments:
            if stop_event is not None and stop_event.is_set():
                return
            if not is_spell:
                if sent_log is not None:
                    sent_log.append({"kind": "piper-plain", "text": content})
                sumsq, count = 0.0, 0
                lock = await _get_tts_lock()
                async with lock:
                    async for b in synth(content, cfg_normal):
                        if stop_event is not None and stop_event.is_set():
                            return
                        arr = np.frombuffer(b, dtype=np.int16).astype(np.float64)
                        sumsq += float(np.sum(arr * arr))
                        count += len(arr)
                        yield b
                if count:
                    last_rms = (sumsq / count) ** 0.5
                continue
            silence_ms = int(spell_silence_s * 1000) if spell_pause_mode == "silence" else SPELL_CHAR_GAP_MS
            for tok in content:
                if stop_event is not None and stop_event.is_set():
                    return
                if tok == SPELL_PAUSE:
                    yield _silence_bytes(silence_ms if spell_pause_mode == "silence" else SPELL_GROUP_GAP_MS)
                    continue
                if sent_log is not None:
                    sent_log.append({"kind": "piper-char", "tok": tok, "text": _spoken_char(tok)})
                if spell_pause_mode == "silence":
                    audio = await _plain_spell_char_audio("piper", tok, stop_event=stop_event)
                else:
                    audio = await _spell_char_audio("piper", tok, stop_event=stop_event, target_rms=last_rms)
                if audio:
                    yield audio
                yield _silence_bytes(silence_ms)
        return

    # Fallback: subprocess. Spinning up the Piper CLI per character would be
    # far too slow, so flatten spelled runs into plain, period-separated text.
    flat = flatten_spell_segments(segments)
    proc = await asyncio.create_subprocess_exec(
        "python", "-m", "piper",
        "--model", PIPER_VOICE,
        "--data-dir", PIPER_DATA_DIR,
        "--output-raw",
        "--length-scale", str(PIPER_LENGTH_SCALE),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        proc.stdin.write(flat.encode())
        await proc.stdin.drain()
        proc.stdin.close()
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            yield chunk
        await proc.wait()
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass


def _resolve_spell_mode(segments, cfg: dict):
    """Apply spell_pause_mode=punctuation up front (folds spelled segments
    into plain ones), and return the (possibly rewritten) segments plus the
    mode _synth_segments should still act on for its own per-character loop
    ("processed"/"silence" — punctuation leaves no spelled segments left)."""
    mode = cfg["spell_pause_mode"]
    if mode == "punctuation":
        punct_char = SPELL_PUNCT_CHARS.get(cfg["spell_punct"], SPELL_PUNCT_CHARS[SPELL_PUNCT])
        segments = _apply_punct_spell_mode(segments, punct_char)
        mode = "processed"
    return segments, mode


async def tts_stream(text: str, stop_event: "asyncio.Event | None" = None,
                     sent_log: list | None = None):
    cfg = await _get_config()
    if cfg["spell_pause_mode"] == "plain":
        async for chunk in _delegate_plain("piper", text, stop_event, sent_log):
            yield chunk
        return
    text = tts_text(text, "piper")
    segments = split_spell_segments(text)
    segments, mode = _resolve_spell_mode(segments, cfg)
    async for chunk in _piper_synth_segments(
        segments, stop_event=stop_event,
        spell_pause_mode=mode, spell_silence_s=float(cfg["spell_silence_s"]),
        sent_log=sent_log,
    ):
        yield chunk


# ── Kokoro TTS streaming ────────────────────────────────────────────────
# Serializes in-process Kokoro synthesis (single KPipeline, not thread-safe
# under concurrent calls).
_kokoro_lock = None


async def _get_kokoro_lock():
    global _kokoro_lock
    if _kokoro_lock is None:
        _kokoro_lock = asyncio.Lock()
    return _kokoro_lock


async def _kokoro_synth_segments(segments, stop_event: "asyncio.Event | None" = None,
                                  kokoro_voice: str = KOKORO_VOICE, spell_engine: str = SPELL_ENGINE,
                                  spell_pause_mode: str = SPELL_PAUSE_MODE,
                                  spell_silence_s: float = SPELL_SILENCE_S,
                                  sent_log: list | None = None):
    from audio import resample
    pipeline = state.get("kokoro_pipeline")
    if not pipeline:
        log.warning("Kokoro not loaded, falling back to Piper")
        async for chunk in _piper_synth_segments(segments, stop_event=stop_event):
            yield chunk
        return
    loop = asyncio.get_running_loop()

    async def synth(t, speed):
        gen = pipeline(t, voice=kokoro_voice, speed=speed)
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
            yield pcm.tobytes()

    try:
        # Stream segment-by-segment (do NOT list() the whole pipeline — that
        # forces the full synthesis before any audio, causing long silence).
        # `stop_event` is checked at every boundary so a barge-in stops the
        # NEXT unit of work immediately, rather than waiting on asyncio
        # task-cancellation timing (see _piper_synth_segments for why that
        # matters on a long spelled run).
        # Track the RMS of the most recent plain-narration segment so
        # spelled characters that follow it can be loudness-matched to what
        # this call actually sounds like, instead of a fixed guess.
        last_rms = None
        for is_spell, content in segments:
            if stop_event is not None and stop_event.is_set():
                return
            if not is_spell:
                if sent_log is not None:
                    sent_log.append({"kind": "kokoro-plain", "text": content})
                sumsq, count = 0.0, 0
                lock = await _get_kokoro_lock()
                async with lock:
                    async for b in synth(content, KOKORO_SPEED):
                        if stop_event is not None and stop_event.is_set():
                            return
                        arr = np.frombuffer(b, dtype=np.int16).astype(np.float64)
                        sumsq += float(np.sum(arr * arr))
                        count += len(arr)
                        yield b
                if count:
                    last_rms = (sumsq / count) ** 0.5
                continue
            # Spelled characters go through Piper by default, not Kokoro —
            # see _spell_char_audio for why. spell_engine="match" overrides
            # this to keep spelling on Kokoro too (voice consistency test).
            char_engine = "kokoro" if spell_engine == "match" else "piper"
            silence_ms = int(spell_silence_s * 1000) if spell_pause_mode == "silence" else SPELL_CHAR_GAP_MS
            for tok in content:
                if stop_event is not None and stop_event.is_set():
                    return
                if tok == SPELL_PAUSE:
                    yield _silence_bytes(silence_ms if spell_pause_mode == "silence" else SPELL_GROUP_GAP_MS)
                    continue
                if sent_log is not None:
                    sent_log.append({"kind": f"{char_engine}-char", "tok": tok, "text": _spoken_char(tok)})
                if spell_pause_mode == "silence":
                    audio = await _plain_spell_char_audio(char_engine, tok, stop_event=stop_event,
                                                            kokoro_voice=kokoro_voice)
                else:
                    audio = await _spell_char_audio(char_engine, tok, stop_event=stop_event,
                                                     kokoro_voice=kokoro_voice, target_rms=last_rms)
                if audio:
                    yield audio
                yield _silence_bytes(silence_ms)
    except Exception as e:
        log.error("Kokoro TTS error: %s", e)


async def kokoro_stream(text: str, stop_event: "asyncio.Event | None" = None,
                        sent_log: list | None = None):
    cfg = await _get_config()
    if cfg["spell_pause_mode"] == "plain":
        async for chunk in _delegate_plain("kokoro", text, stop_event, sent_log):
            yield chunk
        return
    text = tts_text(text, "kokoro")
    segments = split_spell_segments(text)
    segments, mode = _resolve_spell_mode(segments, cfg)
    async for chunk in _kokoro_synth_segments(
        segments, stop_event=stop_event,
        kokoro_voice=cfg["kokoro_voice"], spell_engine=cfg["spell_engine"],
        spell_pause_mode=mode, spell_silence_s=float(cfg["spell_silence_s"]),
        sent_log=sent_log,
    ):
        yield chunk


# ── Chatterbox TTS streaming ────────────────────────────────────────────
# Zero-shot voice cloning: given a short reference clip (see
# chatterbox_voices.py), every character — narration AND spelling — is
# synthesized in that cloned voice. Unlike Kokoro, spelling never falls back
# to Piper: the whole point of cloning is one consistent voice end-to-end.
# Serializes in-process synthesis (single model instance, not thread-safe
# under concurrent calls).
_chatterbox_lock = None


async def _get_chatterbox_lock():
    global _chatterbox_lock
    if _chatterbox_lock is None:
        _chatterbox_lock = asyncio.Lock()
    return _chatterbox_lock


def _resolve_chatterbox_voice_name(voice_name: str) -> str:
    """Return the sanitized voice name if it exists, else '' (built-in voice)."""
    name = chatterbox_voices.sanitize_voice_name(voice_name) if voice_name else ""
    if name and chatterbox_voices.voice_exists(name):
        return name
    return ""


async def _chatterbox_tts_bytes(text: str, voice_name: str,
                                label: str = "",
                                sent_log: list | None = None) -> bytes:
    """Call the isolated chatterbox-tts service over HTTP and return int16 PCM
    at PIPER_RATE. Chatterbox has no incremental/streaming API — each call
    returns the whole clip's audio at once.

    Logs the EXACT text sent (and what came back) — the /tts service only
    logs the HTTP status line, so without this the payload that produced a
    garbled clip is invisible from the caller side.
    """
    import base64
    import httpx
    from audio import resample
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
        log.warning("[chatterbox] tts service error: %s (text=%r, voice=%r, label=%r)",
                    e, text, voice_name, label)
        return b""
    pcm = np.frombuffer(base64.b64decode(d["pcm"]), dtype=np.int16)
    sr = int(d.get("sr", 24000))
    if sr != PIPER_RATE:
        f32 = pcm.astype(np.float32) / 32768.0
        f32 = resample(f32, sr, PIPER_RATE)
        pcm = np.clip(f32 * 32767, -32768, 32767).astype(np.int16)
    if sent_log is not None:
        sent_log.append({"kind": label or "chatterbox", "text": text,
                         "pcm": len(pcm)})
    log.info("[chatterbox] SENT text=%r voice=%r label=%r -> %d pcm @ %dHz (%.2fs, %d bytes)",
             text, voice_name, label, len(pcm), PIPER_RATE, len(pcm) / 2 / PIPER_RATE, len(pcm) * 2)
    return pcm.tobytes()


async def _chatterbox_char_gen(text: str, voice_name: str, sent_log: list | None = None):
    b = await _chatterbox_tts_bytes(text, voice_name, label="char", sent_log=sent_log)
    if b:
        yield b


async def _chatterbox_spell_char_audio(tok: str, stop_event: "asyncio.Event | None" = None,
                                        voice_name: str = "",
                                        target_rms: float | None = None,
                                        sent_log: list | None = None) -> bytes:
    lock = await _get_chatterbox_lock()
    async with lock:
        if stop_event is not None and stop_event.is_set():
            return b""
        return await _synth_char_bytes(_chatterbox_char_gen(_spoken_char(tok), voice_name),
                                        label=tok, engine="chatterbox", target_rms=target_rms)


async def _chatterbox_plain_char_audio(tok: str, stop_event: "asyncio.Event | None" = None,
                                        voice_name: str = "",
                                        sent_log: list | None = None) -> bytes:
    lock = await _get_chatterbox_lock()
    async with lock:
        if stop_event is not None and stop_event.is_set():
            return b""
        return await _chatterbox_tts_bytes(tok, voice_name, label="plain-char", sent_log=sent_log)


def _split_plain_chunks(text: str, max_chars: int = 140):
    """Split a plain (non-spell) utterance into sentence chunks so the
    Chatterbox service can synthesize them incrementally. A single long
    generation on MPS is both slow (one blocking /tts call) and unpredictable
    (multi-second stalls), which makes the browser feel dead and can drop the
    WS. Streaming per-sentence gets audio flowing fast; boundary chars are
    preserved on the first chunk so pacing/intonation stay natural. A sentence
    longer than max_chars is hard-wrapped on commas/spaces."""
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]
    if not sentences:
        return [text]
    pieces = []
    for s in sentences:
        if len(s) <= max_chars:
            pieces.append(s)
            continue
        # Hard-wrap an over-long sentence on commas, then spaces.
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


async def _chatterbox_punct_group_audio(group: list[str], punct_char: str, voice_name: str,
                                        sent_log: list | None = None) -> bytes:
    """One Chatterbox call for a whole SPELL_PAUSE-delimited group (e.g. 4
    digits of an NPI/tax ID) instead of one call per character. A single
    call for character-by-character text keeps generations short enough
    that the autoregressive model doesn't drift — see the comment on
    spell_pause_mode="punctuation" handling below for why the whole-ID
    version of this was reverted."""
    text = _punct_spelled_text(group, punct_char)
    if not text:
        return b""
    return await _chatterbox_tts_bytes(text, voice_name, label=f"spell-group:{len(group)}",
                                       sent_log=sent_log)


async def _chatterbox_synth_segments(segments, stop_event: "asyncio.Event | None" = None,
                                      voice_name: str = CHATTERBOX_VOICE,
                                      spell_pause_mode: str = SPELL_PAUSE_MODE,
                                      spell_silence_s: float = SPELL_SILENCE_S,
                                      spell_punct_char: str = SPELL_PUNCT_CHARS[SPELL_PUNCT],
                                      sent_log: list | None = None):
    voice_name = _resolve_chatterbox_voice_name(voice_name)
    lock = await _get_chatterbox_lock()
    last_rms = None
    silence_ms = int(spell_silence_s * 1000) if spell_pause_mode == "silence" else SPELL_CHAR_GAP_MS
    for is_spell, content in segments:
        if stop_event is not None and stop_event.is_set():
            return
        if not is_spell:
            for i, chunk in enumerate(_split_plain_chunks(content)):
                if stop_event is not None and stop_event.is_set():
                    return
                async with lock:
                    if stop_event is not None and stop_event.is_set():
                        return
                    audio = await _chatterbox_tts_bytes(chunk, voice_name, label=f"plain-chunk:{i}",
                                                        sent_log=sent_log)
                if audio:
                    arr = np.frombuffer(audio, dtype=np.int16).astype(np.float64)
                    if len(arr):
                        last_rms = float(np.sqrt(np.mean(arr * arr)))
                    yield audio
            continue
        if spell_pause_mode == "punctuation":
            # Folding the WHOLE spelled run (e.g. a 10-digit NPI) into ONE
            # Chatterbox call avoided the per-character timeout/drop, but
            # traded it for a different failure: Chatterbox is autoregressive,
            # and a long generation of many short isolated "words" (digit
            # names) gives it room to drift and duplicate/drop a token —
            # reproduced live, a Group NPI came back with an extra "5".
            # Splitting on the existing SPELL_PAUSE group boundaries (normally
            # every 4 digits) keeps each call short enough that this doesn't
            # happen, while still using far fewer HTTP calls than one per
            # character.
            group: list[str] = []
            for tok in content:
                if stop_event is not None and stop_event.is_set():
                    return
                if tok == SPELL_PAUSE:
                    if group:
                        async with lock:
                            if stop_event is not None and stop_event.is_set():
                                return
                            audio = await _chatterbox_punct_group_audio(group, spell_punct_char, voice_name,
                                                                        sent_log=sent_log)
                        if audio:
                            yield audio
                        group = []
                    yield _silence_bytes(SPELL_GROUP_GAP_MS)
                    continue
                group.append(tok)
            if group:
                if stop_event is not None and stop_event.is_set():
                    return
                async with lock:
                    if stop_event is not None and stop_event.is_set():
                        return
                    audio = await _chatterbox_punct_group_audio(group, spell_punct_char, voice_name,
                                                                sent_log=sent_log)
                if audio:
                    yield audio
            continue
        for tok in content:
            if stop_event is not None and stop_event.is_set():
                return
            if tok == SPELL_PAUSE:
                yield _silence_bytes(silence_ms if spell_pause_mode == "silence" else SPELL_GROUP_GAP_MS)
                continue
            if spell_pause_mode == "silence":
                audio = await _chatterbox_plain_char_audio(tok, stop_event=stop_event, voice_name=voice_name,
                                                            sent_log=sent_log)
            else:
                audio = await _chatterbox_spell_char_audio(tok, stop_event=stop_event, voice_name=voice_name,
                                                             target_rms=last_rms, sent_log=sent_log)
            if audio:
                yield audio
            yield _silence_bytes(silence_ms)


async def chatterbox_stream(text: str, stop_event: "asyncio.Event | None" = None,
                            sent_log: list | None = None):
    cfg = await _get_config()
    if cfg["spell_pause_mode"] == "plain":
        async for chunk in _delegate_plain("chatterbox", text, stop_event, sent_log):
            yield chunk
        return
    text = tts_text(text, "chatterbox")
    segments = split_spell_segments(text)
    punct_char = SPELL_PUNCT_CHARS.get(cfg["spell_punct"], SPELL_PUNCT_CHARS[SPELL_PUNCT])
    async for chunk in _chatterbox_synth_segments(
        segments, stop_event=stop_event,
        voice_name=cfg["chatterbox_voice"],
        spell_pause_mode=cfg["spell_pause_mode"], spell_silence_s=float(cfg["spell_silence_s"]),
        spell_punct_char=punct_char,
        sent_log=sent_log,
    ):
        yield chunk


async def _delegate_plain(engine: str, text: str,
                          stop_event: "asyncio.Event | None",
                          sent_log: list | None):
    """Route to the plain-mode streamer when spell_pause_mode == "plain".

    Plain mode preprocesses the entire utterance into ordinary spoken words
    (numbers -> digit names, hyphenated spellings -> spaced words) and
    synthesizes it as one continuous stream — see server/tts_plain.py. It
    skips the sentinel parsing / per-character synthesis the other pause
    modes use, so the engine never has to emit long runs of isolated symbols
    (the gy source of the garbled-tail Chatterbox clips).
    """
    from .tts_plain import chatterbox_stream as _chatterbox
    from .tts_plain import kokoro_stream as _kokoro
    from .tts_plain import piper_stream as _piper
    fn = {"chatterbox": _chatterbox, "kokoro": _kokoro, "piper": _piper}[engine]
    async for chunk in fn(text, stop_event=stop_event, sent_log=sent_log):
        yield chunk


def get_tts_fn(engine: str):
    return {"piper": tts_stream, "kokoro": kokoro_stream, "chatterbox": chatterbox_stream}.get(engine, tts_stream)
