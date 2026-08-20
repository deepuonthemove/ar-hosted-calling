#!/usr/bin/env bash
# Local Mac dev: build + start the full stack (redis, voice-agent, frontend).
# Requires: Docker Desktop running, Ollama running with llama3.1:8b.
set -e
cd "$(dirname "$0")"

echo "==> Creating ar-agent Ollama model (llama3.1:8b, num_ctx 8192) if missing..."
if ! ollama list 2>/dev/null | grep -q '^ar-agent'; then
  printf 'FROM llama3.1:8b\nPARAMETER num_ctx 8192\n' | ollama create ar-agent -
fi

# The tracked .env carries prod LLM_MODEL (vLLM AWQ name); force the local
# Ollama model so the compose interpolation can't leak prod config in here.
export LLM_MODEL="${LLM_MODEL:-ar-agent}"

echo "==> Native Chatterbox (Turbo/MPS) service..."
if [ -x "$PWD/chatterbox-venv/bin/python" ]; then
  if ! curl -sf -m 2 http://127.0.0.1:8084/health >/dev/null 2>&1; then
    "$PWD/chatterbox_service/run-mac.sh" >"$PWD/chatterbox-mac.log" 2>&1 &
    echo "   started (pid $!) — first load takes ~15s; log: chatterbox-mac.log"
  else
    echo "   already running."
  fi
else
  echo "   SKIPPED — run ./chatterbox_service/setup-chatterbox-mac.sh first (voice-agent will use Piper)."
fi

echo "==> Building + starting local stack..."
# Opik (tracing/eval) is a heavy 6-container stack and the biggest
# contributor to slow local startup — opt in with: ./start-local.sh --opik
if [ "${1:-}" = "--opik" ]; then
  docker compose -f docker-compose.local.yml --profile opik up -d --build
else
  docker compose -f docker-compose.local.yml up -d --build
fi

echo
echo "Local stack ready:"
echo "  Admin UI : http://localhost:3000"
echo "  Backend  : http://localhost:8080  (/health, /api/config)"
echo "  Redis    : redis://localhost:6379"
echo "  Chatterbox (native Turbo/MPS): http://127.0.0.1:8084"
if [ "${1:-}" = "--opik" ]; then
  echo "  Opik     : http://localhost:5173"
else
  echo "  Opik     : not started (run with --opik to enable tracing)"
fi
