"""Audio codec utilities: μ-law encode/decode + resampling.

Twilio Media Streams: 8kHz μ-law mono, base64-encoded JSON payloads.
Whisper expects:      16kHz float32 mono.
Piper TTS outputs:    native sample rate (e.g. 22050Hz) int16 PCM.
"""
import numpy as np

BIAS = 0x84
CLIP = 32635

# ── μ-law decode (G.711) ────────────────────────────────────────────────
_exp_table = np.array([
    0, 132, 396, 924, 1980, 4092, 8316, 16764
], dtype=np.int32)


def mulaw_to_pcm16(mulaw_bytes: bytes) -> np.ndarray:
    """Decode μ-law bytes → int16 PCM."""
    u = np.frombuffer(mulaw_bytes, dtype=np.uint8)
    u = ~u
    sign = (u & 0x80).astype(np.int32)
    exponent = ((u >> 4) & 0x07).astype(np.int32)
    mantissa = (u & 0x0F).astype(np.int32)
    sample = _exp_table[exponent] + (mantissa << (exponent + 3))
    sample = np.where(sign != 0, BIAS - sample, sample - BIAS)
    return sample.astype(np.int16)


def pcm16_to_mulaw(pcm: np.ndarray) -> bytes:
    """Encode int16 PCM → μ-law bytes."""
    pcm = pcm.astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0)
    mag = np.abs(pcm)
    mag = np.clip(mag, 0, CLIP) + BIAS
    # exponent: position of highest set bit above bit 7
    exponent = np.zeros_like(mag)
    for e in range(7, 0, -1):
        mask = (mag >> (e + 7)) > 0
        exponent = np.where((exponent == 0) & mask, e, exponent)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    u = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return u.astype(np.uint8).tobytes()


# ── Resampling (linear interpolation — fine for speech) ─────────────────
def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate or len(audio) == 0:
        return audio
    new_len = int(len(audio) * to_rate / from_rate)
    if new_len == 0:
        return np.array([], dtype=audio.dtype)
    old_idx = np.arange(len(audio))
    new_idx = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(new_idx, old_idx, audio).astype(np.float32)


def twilio_to_whisper(mulaw_b64_payload: bytes) -> np.ndarray:
    """Twilio 8kHz μ-law → 16kHz float32 for Whisper."""
    pcm16 = mulaw_to_pcm16(mulaw_b64_payload)
    f32 = pcm16.astype(np.float32) / 32768.0
    return resample(f32, 8000, 16000)


def piper_to_twilio(pcm_int16: np.ndarray, piper_rate: int) -> bytes:
    """Piper int16 PCM → 8kHz μ-law bytes for Twilio."""
    f32 = pcm_int16.astype(np.float32)
    pcm8k = resample(f32, piper_rate, 8000)
    return pcm16_to_mulaw(pcm8k.astype(np.int16))


# ── Energy-based VAD ─────────────────────────────────────────────────────
def rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


class VAD:
    """Accumulates speech; returns segment when silence ends utterance."""

    def __init__(self, energy_threshold=0.015, min_speech_ms=400,
                 min_silence_ms=700, max_speech_ms=10_000):
        self.buffer: list[np.ndarray] = []
        self.speech_start: float | None = None
        self.last_speech: float = 0.0
        self.energy_threshold = energy_threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.max_speech_ms = max_speech_ms

    def add(self, chunk: np.ndarray, now: float) -> np.ndarray | None:
        is_speech = rms(chunk) > self.energy_threshold
        if is_speech:
            self.last_speech = now
            if self.speech_start is None:
                self.speech_start = now
        if self.speech_start is None:
            return None
        self.buffer.append(chunk)
        dur_ms = (now - self.speech_start) * 1000
        silence_ms = (now - self.last_speech) * 1000
        if silence_ms > self.min_silence_ms and dur_ms > self.min_speech_ms:
            return self._flush()
        if dur_ms > self.max_speech_ms:
            return self._flush()
        return None

    def _flush(self) -> np.ndarray:
        audio = np.concatenate(self.buffer) if self.buffer else np.array([], dtype=np.float32)
        self.buffer.clear()
        self.speech_start = None
        return audio
