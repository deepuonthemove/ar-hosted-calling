#!/bin/bash
set -e
cd ~/ar-voice-agent
docker compose --profile prod up -d
sleep 10
# Start ngrok tunnel (replace with your actual token)
ngrok config add-authtoken 3GhEBHePujHeD35ejbjKNnzNTp6_E1SaCh3cXeN3iL5WbgJk 2>/dev/null
nohup ngrok http 8080 --log=stdout > /tmp/ngrok.log 2>&1 &
sleep 3
echo "Ngrok URL: $(curl -s localhost:4040/api/tunnels | python3 -c 'import sys,json;print(json.load(sys.stdin)["tunnels"][0]["public_url"])')"
