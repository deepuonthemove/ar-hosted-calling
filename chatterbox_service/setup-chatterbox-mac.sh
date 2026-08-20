#!/usr/bin/env bash
# One-time setup: build the native Apple Silicon Chatterbox (Turbo) venv.
# Installs into ./chatterbox-venv (gitignored). PyTorch arm64 (MPS-enabled)
# comes from PyPI — the CUDA wheels in the Docker image don't apply on Mac.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/opt/homebrew/opt/python@3.12/bin/python3.12}"
VENV="$PWD/chatterbox-venv"

if [ -x "$VENV/bin/python" ]; then
  echo "venv already exists at $VENV — deleting to reinstall? (Ctrl-C to abort)"
  sleep 3
  rm -rf "$VENV"
fi

"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip >/dev/null

echo ">> Installing torch (arm64, MPS) + chatterbox-tts deps (this takes a while)..."
# Use the git master (pip 0.1.7 predates the nano arg). --no-deps keeps our
# pinned transformers==4.46.3 (the package's declared 5.2.0 pin is broken).
"$VENV/bin/python" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  torch torchaudio \
  fastapi "uvicorn[standard]" pydantic numpy \
  'transformers==4.46.3' \
  'diffusers==0.29.0' 'librosa==0.11.0' 'omegaconf' 'pykakasi' 'pyloudnorm' \
  'conformer' 's3tokenizer' 'safetensors' 'spacy-pkuseg' 'resemble-perth' \
  'setuptools<81'
"$VENV/bin/python" -m pip install --no-deps --force-reinstall \
  "git+https://github.com/resemble-ai/chatterbox.git@master"

echo ">> Smoke test: import turbo/nano + check MPS..."
"$VENV/bin/python" - <<'EOF'
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
print("torch", torch.__version__, "mps", torch.backends.mps.is_available())
print("turbo/nano import OK")
EOF

echo
echo "Done. Start the service with:  ./run-mac.sh"
echo "  (defaults to Chatterbox NANO — much faster on MPS)"
echo "voice-agent will reach it at  http://host.docker.internal:8084"
