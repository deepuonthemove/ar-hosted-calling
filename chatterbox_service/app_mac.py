"""Chatterbox TTS — native Apple Silicon service (Turbo model on MPS).

Runs ChatterboxTurboTTS directly on the host (no Docker) using Metal (MPS).
The main voice-agent app calls this over HTTP, exactly like the CUDA container
it replaces on the VM. API is identical: /health, /voices, /tts.

Turbo is a 350M-param distilled model; CFG/exaggeration are accepted by the
SDK but ignored (it warns and proceeds). A built-in voice is used when no
reference clip is configured.
"""
import asyncio
import base64
import os
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

VOICES_DIR = os.getenv("CHATTERBOX_VOICES_DIR", os.path.expanduser("~/.cache/ar-voice-agent/chatterbox/voices"))
DEVICE = os.getenv("CHATTERBOX_DEVICE", "mps")
IDLE_UNLOAD_S = float(os.getenv("CHATTERBOX_IDLE_UNLOAD_S", "3600"))
MODEL_REPO = os.getenv("CHATTERBOX_MODEL_REPO", "ResembleAI/chatterbox-turbo")
# Nano is a smaller, faster distilled variant (2.8GB vs 3.8GB weights) — far
# better on MPS where Turbo fp32 throttles. Enable with CHATTERBOX_NANO=1.
USE_NANO = os.getenv("CHATTERBOX_NANO", "0") == "1"

app = FastAPI()
_model = None
_model_sr = None
_last_used = 0.0
_watchdog = None


class TTSRequest(BaseModel):
    text: str
    voice_name: str = ""
    exaggeration: float = 0.5
    cfg_weight: float = 0.5


def _load():
    global _model, _model_sr, DEVICE
    import torch
    # MPS fp32 matmul default is slow; 'high' uses the fast Metal path (~2x).
    torch.set_float32_matmul_precision("high")
    if DEVICE == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable — falling back to CPU", flush=True)
        DEVICE = "cpu"
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    _model = ChatterboxTurboTTS.from_pretrained(device=DEVICE, nano=USE_NANO)
    _model_sr = _model.sr
    _patch_prepare_conditionals(_model, DEVICE)
    _label = "Nano" if USE_NANO else "Turbo"
    print(f"Chatterbox ({_label}) loaded: device={DEVICE} model={MODEL_REPO} sr={_model_sr}", flush=True)
    _warmup()


def _warmup():
    """Run one throwaway generation right after load. MPS pays a one-time
    kernel-compile/graph-warmup cost on the first real inference; without
    this, that cost lands on the first live call of an actual conversation
    (e.g. the greeting) and can push a short spelled-out segment past
    SPELL_CHAR_TIMEOUT_S, dropping audio. Paying it here during startup
    instead is silent to the caller."""
    t0 = time.time()
    try:
        _model.generate("Warming up.", exaggeration=0.5, cfg_weight=0.5)
        print(f"Chatterbox warmup: {(time.time() - t0) * 1000:.0f}ms", flush=True)
    except Exception as e:
        print(f"Chatterbox warmup failed (non-fatal): {e}", flush=True)


def _patch_prepare_conditionals(model, device: str):
    """Replace prepare_conditionals with an MPS-safe float32 version.

    The stock implementation loads the reference clip via librosa (float64)
    and feeds it to the s3tokenizer / voice-encoder, which then move it to the
    device. MPS rejects float64 ('Cannot convert a MPS Tensor to float64'),
    so cast every numpy stage to float32. Only the voice-clone path uses this;
    built-in voice (no clip) is unaffected.
    """
    import librosa
    import torch
    from chatterbox.models.s3tokenizer import S3_SR
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    S3GEN_SR = model.sr
    ENC_COND_LEN = model.ENC_COND_LEN
    DEC_COND_LEN = model.DEC_COND_LEN

    def prepare_conditionals(wav_fpath, exaggeration=0.5, norm_loudness=True):
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR, mono=True)
        s3gen_ref_wav = s3gen_ref_wav.astype(np.float32)
        if norm_loudness:
            s3gen_ref_wav = model.norm_loudness(s3gen_ref_wav, _sr).astype(np.float32)
        ref_16k_wav = librosa.resample(
            s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR).astype(np.float32)

        s3gen_ref_wav = s3gen_ref_wav[:DEC_COND_LEN]
        s3gen_ref_dict = model.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=device)

        if plen := model.t3.hp.speech_cond_prompt_len:
            s3_tokzr = model.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [ref_16k_wav[:ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(device)

        ve_embed = torch.from_numpy(
            model.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR).astype(np.float32)
        )
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=device)
        model.conds = chatterbox_conds_from(model, t3_cond, s3gen_ref_dict)

    model.prepare_conditionals = prepare_conditionals


def chatterbox_conds_from(model, t3_cond, gen):
    """Rebuild the Conditionals object after a fresh voice-clip embed."""
    from chatterbox.tts_turbo import Conditionals
    return Conditionals(t3=t3_cond, gen=gen)


def _unload():
    global _model, _model_sr
    if _model is None:
        return
    _model = None
    _model_sr = None
    print("Chatterbox unloaded (idle)", flush=True)


async def _watchdog_loop():
    while True:
        await asyncio.sleep(30)
        if _model is not None and time.time() - _last_used > IDLE_UNLOAD_S:
            _unload()


@app.on_event("startup")
async def startup():
    _load()
    global _watchdog
    _watchdog = asyncio.create_task(_watchdog_loop())


@app.get("/health")
def health():
    return {"ok": _model is not None, "sr": _model_sr}


@app.get("/voices")
def voices():
    if not os.path.isdir(VOICES_DIR):
        return {"voices": []}
    return {"voices": sorted(
        os.path.splitext(f)[0] for f in os.listdir(VOICES_DIR) if f.endswith(".wav")
    )}


@app.post("/tts")
async def tts(req: TTSRequest):
    global _last_used
    if _model is None:
        _load()
    _last_used = time.time()
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    kwargs = {"exaggeration": req.exaggeration, "cfg_weight": req.cfg_weight}
    if req.voice_name:
        vp = os.path.join(VOICES_DIR, req.voice_name + ".wav")
        if os.path.isfile(vp):
            kwargs["audio_prompt_path"] = vp
    wav = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _model.generate(text, **kwargs))
    audio = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    return {"pcm": base64.b64encode(pcm.tobytes()).decode(), "sr": _model_sr}
