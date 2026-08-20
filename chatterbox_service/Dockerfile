# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Chatterbox's declared pin (transformers==5.2.0) is BROKEN (its code does
# `from transformers import LlamaModel`, which only exists in 4.x). Install
# with --no-deps and pin a working stack instead. torch is re-installed LAST
# (cu124) so any torch upgrade pulled by the deps is corrected back to the
# base image's matching 2.5.1/0.20.1/2.5.1 trio.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install 'transformers==4.46.3' fastapi 'uvicorn[standard]' pydantic
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-deps 'chatterbox-tts==0.1.7'
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install \
    'conformer' 'diffusers==0.29.0' 'librosa' 'omegaconf' 'pykakasi' \
    'pyloudnorm' 'resemble-perth' 's3tokenizer' 'safetensors' 'spacy-pkuseg'
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --index-url https://download.pytorch.org/whl/cu124 \
    'torch==2.5.1+cu124' 'torchvision==0.20.1+cu124' 'torchaudio==2.5.1+cu124'

COPY app.py /app/app.py
RUN mkdir -p /models/chatterbox/voices

EXPOSE 8082
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8082", "--workers", "1"]
