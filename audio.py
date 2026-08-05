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


# ── VAD (Silero VAD with RMS Energy Fallback) ──────────────────────────
_silero_model = None
_silero_utils: dict = {}


def _find_vad_iterator(utils) -> type | None:
    """Silero returns utils as a dict (older) or tuple (newer). Find VADIterator class."""
    if isinstance(utils, dict):
        return utils.get("VADIterator")
    if isinstance(utils, (tuple, list)):
        for u in utils:
            if isinstance(u, type) and u.__name__ == "VADIterator":
                return u
    return None


def load_silero_vad():
    global _silero_model, _silero_utils
    if _silero_model is not None:
        return _silero_model, _silero_utils
    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True,
            onnx=False
        )
        _silero_model = model
        _silero_utils = {"VADIterator": _find_vad_iterator(utils)}
        return _silero_model, _silero_utils
    except Exception as e:
        print(f"[VAD] Silero VAD load failed ({e}); falling back to RMS energy VAD")
        return None, {}


def is_speech_silero(chunk: np.ndarray, sample_rate: int = 16000) -> float:
    """Returns speech probability 0.0 - 1.0 using Silero VAD on CPU."""
    model, _ = load_silero_vad()
    if model is None:
        return 0.0
    try:
        import torch
        tensor = torch.from_numpy(chunk).float()
        with torch.no_grad():
            prob = model(tensor, sample_rate).item()
        return float(prob)
    except Exception:
        return 0.0


def rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


class VAD:
    """Accumulates speech; returns segment when silence ends utterance.

    When `use_silero` is True and the Silero model is available, uses Silero's
    VADIterator (frame-based, 512-sample frames) for reliable speech boundaries.
    Otherwise falls back to RMS energy thresholding.
    """

    FRAME = 512
    PREROLL_FRAMES = 8  # ~256ms of pre-roll before speech onset

    def __init__(self, energy_threshold=0.015, speech_threshold=0.5,
                 min_speech_ms=400, min_silence_ms=700, max_speech_ms=10_000, use_silero=True):
        self.buffer: list[np.ndarray] = []
        self.speech_start: float | None = None
        self.last_speech: float = 0.0
        self.energy_threshold = energy_threshold
        self.speech_threshold = speech_threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.max_speech_ms = max_speech_ms
        self.use_silero = use_silero

        # Silero frame-based machinery
        self._silero = None
        self._torch = None
        self._pending = None
        self._ring = None
        self._segment = None
        self._speech_start_s = None
        if use_silero:
            self._init_silero()

    def _init_silero(self):
        try:
            import torch
            model, utils = load_silero_vad()
            if model is None:
                self._silero = None
                return
            self._torch = torch
            # min_silence must match RMS path (default 100ms fires 'end' mid-thought)
            self._silero = utils["VADIterator"](
                model, sampling_rate=16000, min_silence_duration_ms=self.min_silence_ms
            )
            self._pending = np.array([], dtype=np.float32)
            self._ring = np.array([], dtype=np.float32)
            self._segment = np.array([], dtype=np.float32)
            self._speech_start_s = None
        except Exception:
            self._silero = None

    def reset_silero(self):
        if self._silero is None:
            return
        self._pending = np.array([], dtype=np.float32)
        self._ring = np.array([], dtype=np.float32)
        self._segment = np.array([], dtype=np.float32)
        self._speech_start_s = None
        self._silero.reset_states()

    def add(self, chunk: np.ndarray, now: float) -> np.ndarray | None:
        if self._silero is not None:
            return self._add_silero(chunk, now)
        return self._add_rms(chunk, now)

    def _add_rms(self, chunk: np.ndarray, now: float) -> np.ndarray | None:
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

    def _add_silero(self, chunk: np.ndarray, now: float) -> np.ndarray | None:
        self._pending = np.concatenate([self._pending, chunk.astype(np.float32)])
        result = None
        while len(self._pending) >= self.FRAME:
            frame = self._pending[:self.FRAME]
            self._pending = self._pending[self.FRAME:]

            # Rolling pre-roll buffer (kept only during speech gaps, used on onset)
            self._ring = np.concatenate([self._ring, frame])
            max_ring = self.PREROLL_FRAMES * self.FRAME
            if len(self._ring) > max_ring:
                self._ring = self._ring[-max_ring:]

            event = self._silero(self._torch.from_numpy(frame).unsqueeze(0))
            if isinstance(event, dict) and "start" in event:
                self._speech_start_s = now
                self._segment = self._ring.copy()
            elif self._speech_start_s is not None:
                self._segment = np.concatenate([self._segment, frame])

            if isinstance(event, dict) and "end" in event:
                if self._speech_start_s is not None and (now - self._speech_start_s) * 1000 >= self.min_speech_ms:
                    result = self._segment
                self._speech_start_s = None
                self._segment = np.array([], dtype=np.float32)
            elif self._speech_start_s is not None and (now - self._speech_start_s) * 1000 > self.max_speech_ms:
                result = self._segment
                self._speech_start_s = None
                self._segment = np.array([], dtype=np.float32)
        return result

    def _flush(self) -> np.ndarray:
        audio = np.concatenate(self.buffer) if self.buffer else np.array([], dtype=np.float32)
        self.buffer.clear()
        self.speech_start = None
        return audio




