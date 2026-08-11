#!/bin/bash
# Start the AR voice agent stack (modern: prod + TLS via Caddy, no ngrok).
# Use bootstrap-new-vm.sh on a fresh VM; this is the everyday start command.
set -e
cd ~/ar-voice-agent

# --build picks up any code changes; prod+tls brings up voice-agent, vllm,
# redis, frontend and caddy (TLS via Let's Encrypt for PUBLIC_DOMAIN).
docker compose --profile prod --profile tls up -d --build

sleep 8
echo "----------------------------------------"
echo "  API health : curl https://\$(grep ^PUBLIC_DOMAIN .env | cut -d= -f2)/health"
echo "  UI         : https://\$(grep ^PUBLIC_DOMAIN .env | cut -d= -f2)"
echo "  Opik       : https://optik.\$(grep ^PUBLIC_DOMAIN .env | cut -d= -f2)"
echo "  Twilio     : /voice webhook base = https://\$(grep ^PUBLIC_DOMAIN .env | cut -d= -f2)"
echo "----------------------------------------"
