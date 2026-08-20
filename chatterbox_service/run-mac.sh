#!/usr/bin/env bash
# Start the native Apple Silicon Chatterbox (Turbo/MPS) service.
# Requires: deploy/chatterbox-venv/ (created by setup-chatterbox-mac.sh).
# Serves /health /voices /tts on port 8084 — the voice-agent hits it via
# CHATTERBOX_SERVICE_URL=http://host.docker.internal:8084.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${CHATTERBOX_VENV:-$PWD/chatterbox-venv}"
PORT="${CHATTERBOX_PORT:-8084}"
# Nano is far faster on MPS; opt back into Turbo with CHATTERBOX_NANO=0.
export CHATTERBOX_NANO="${CHATTERBOX_NANO:-1}"
# Shared voice-clip dir: the voice-agent container bind-mounts the same path
# (docker-compose.local.yml), so recorded clips are visible to both.
export CHATTERBOX_VOICES_DIR="${CHATTERBOX_VOICES_DIR:-$HOME/.cache/ar-voice-agent/chatterbox/voices}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "ERROR: no venv at $VENV — run ./setup-chatterbox-mac.sh first." >&2
  exit 1
fi

exec "$VENV/bin/python" -m uvicorn app_mac:app \
  --host 127.0.0.1 --port "$PORT" --app-dir "$PWD/chatterbox_service"
