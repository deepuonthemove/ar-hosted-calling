#!/bin/bash
# Rebuild the app images after code changes, then start the stack.
# (Daily restarts just use start.sh — this only runs when you change code.)
set -e
cd ~/ar-voice-agent

docker compose --profile prod --profile tls up -d --build
