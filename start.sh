#!/bin/bash
# Start the AR voice agent stack instantly (no rebuild — uses existing images).
# Use bootstrap-new-vm.sh on a fresh VM; run rebuild.sh after changing code.
set -e
cd ~/ar-voice-agent

# prod+tls brings up voice-agent, vllm, redis, frontend and caddy. No --build:
# instant start from cached images.
docker compose --profile prod --profile tls up -d

# Opik runs as its own compose at /opt/opik (the built-in compose `opik`
# profile is not the real stack). Start it too if present.
if [ -d /opt/opik/deployment/docker-compose ]; then
  (cd /opt/opik/deployment/docker-compose \
    && sudo docker compose -f docker-compose.yaml -f docker-compose.override.yml \
         --profile opik up -d)
fi

sleep 8
echo "----------------------------------------"
echo "  API health : curl https://\$(grep ^PUBLIC_DOMAIN .env | cut -d= -f2)/health"
echo "  UI         : https://\$(grep ^PUBLIC_DOMAIN .env | cut -d= -f2)"
echo "  Opik       : https://optik.\$(grep ^PUBLIC_DOMAIN .env | cut -d= -f2)"
echo "  Twilio     : /voice webhook base = https://\$(grep ^PUBLIC_DOMAIN .env | cut -d= -f2)"
echo "----------------------------------------"
