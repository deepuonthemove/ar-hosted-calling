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
import time

import numpy as np

import tts_piper as _tts_piper_mod
import tts_kokoro as _tts_kokoro_mod
from tts_common import split_spell_segments, flatten_spell_segments, SPELL_PAUSE

from .app_config import get_config as _get_config
from .config import (
    KOKORO_RATE, KOKORO_SPEED, KOKORO_SPELL_SPEED, KOKORO_VOICE,
    PIPER_DATA_DIR, PIPER_LENGTH_SCALE, PIPER_RATE, PIPER_SPELL_LENGTH_SCALE,
    PIPER_SPELL_NOISE_SCALE, PIPER_SPELL_NOISE_W_SCALE, PIPER_VOICE,
    SPELL_CHAR_GAP_MS, SPELL_ENGINE, SPELL_GROUP_GAP_MS, log,
)
from .state import state

_TTS_MODS = {"piper": _tts_piper_mod, "kokoro": _tts_kokoro_mod}


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


# ── Piper TTS streaming ──────────────────────────────────────────────────
async def _piper_synth_segments(segments, stop_event: "asyncio.Event | None" = None):
    """Synthesize (is_spelled, content) segments from split_spell_segments():
    plain segments at normal speed, spelled segments one character at a time
    at a slower length-scale with real silence between characters.

    `stop_event`, when set, is checked at every segment/character boundary
    so a barge-in stops the NEXT unit of work immediately — it doesn't wait
    on asyncio task-cancellation semantics, which can be deferred until a
    slow in-flight synth call finishes when a spelled run has many
    characters queued up."""
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
            for tok in content:
                if stop_event is not None and stop_event.is_set():
                    return
                if tok == SPELL_PAUSE:
                    yield _silence_bytes(SPELL_GROUP_GAP_MS)
                    continue
                audio = await _spell_char_audio("piper", tok, stop_event=stop_event, target_rms=last_rms)
                if audio:
                    yield audio
                yield _silence_bytes(SPELL_CHAR_GAP_MS)
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


async def tts_stream(text: str, stop_event: "asyncio.Event | None" = None):
    text = tts_text(text, "piper")
    segments = split_spell_segments(text)
    async for chunk in _piper_synth_segments(segments, stop_event=stop_event):
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
                                  kokoro_voice: str = KOKORO_VOICE, spell_engine: str = SPELL_ENGINE):
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
            for tok in content:
                if stop_event is not None and stop_event.is_set():
                    return
                if tok == SPELL_PAUSE:
                    yield _silence_bytes(SPELL_GROUP_GAP_MS)
                    continue
                audio = await _spell_char_audio(char_engine, tok, stop_event=stop_event,
                                                 kokoro_voice=kokoro_voice, target_rms=last_rms)
                if audio:
                    yield audio
                yield _silence_bytes(SPELL_CHAR_GAP_MS)
    except Exception as e:
        log.error("Kokoro TTS error: %s", e)


async def kokoro_stream(text: str, stop_event: "asyncio.Event | None" = None):
    text = tts_text(text, "kokoro")
    segments = split_spell_segments(text)
    cfg = await _get_config()
    async for chunk in _kokoro_synth_segments(
        segments, stop_event=stop_event,
        kokoro_voice=cfg["kokoro_voice"], spell_engine=cfg["spell_engine"],
    ):
        yield chunk


def get_tts_fn(engine: str):
    return {"piper": tts_stream, "kokoro": kokoro_stream}.get(engine, tts_stream)
