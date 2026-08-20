#!/usr/bin/env bash
# Bootstrap a freshly-provisioned Azure T4 VM into a running AR voice agent.
#
# Run this AFTER the VM is provisioned (setup-vm.sh handles VM creation, ports,
# NVIDIA drivers, and the base Docker install). This script handles everything
# else that must happen on the VM itself:
#   1. Docker/containerd data -> /mnt (big data disk), NVIDIA runtime
#   2. Copy the app files + .env
#   3. Build & start the main stack (prod+tls)
#   4. Stand up the full Opik stack + wire it into the app network
#   5. Print remaining manual steps (NSG ports, DNS)
#
# Usage (from the machine with the repo + a path to the .env):
#   scp deploy/bootstrap-new-vm.sh azureuser@<IP>:~/
#   scp .env azureuser@<IP>:~/.env.sourcer    # your secrets
#   rsync -a --exclude node_modules --exclude .next --exclude __pycache__ \
#         --exclude .git deploy/ azureuser@<IP>:~/ar-voice-agent/
#   ssh azureuser@<IP> 'bash ~/bootstrap-new-vm.sh'
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/ar-voice-agent}"
ENV_SRC="${ENV_SRC:-$HOME/.env.sourcer}"
DOMAIN="${DOMAIN:-call.ar-voice.com}"
DATA_DISK="${DATA_DISK:-/mnt}"

echo "===== [1/5] Docker data -> $DATA_DISK + NVIDIA runtime ====="
sudo mkdir -p "$DATA_DISK/docker" "$DATA_DISK/containerd"
sudo tee /etc/docker/daemon.json >/dev/null <<EOF
{"data-root": "$DATA_DISK/docker"}
EOF
if [ -f /etc/containerd/config.toml ]; then
  sudo sed -i "s|root = '/var/lib/containerd'|root = '$DATA_DISK/containerd'|" /etc/containerd/config.toml
else
  sudo mkdir -p /etc/containerd
  sudo sh -c 'containerd config default > /etc/containerd/config.toml'
  sudo sed -i "s|root = '/var/lib/containerd'|root = '$DATA_DISK/containerd'|" /etc/containerd/config.toml
fi
sudo nvidia-ctk runtime configure --runtime=docker || true
sudo systemctl enable --now docker 2>/dev/null || true
sudo systemctl restart containerd docker
sleep 4
sudo usermod -aG docker "$USER"
sudo chmod 666 /var/run/docker.sock
echo "Docker root: $(docker info --format '{{.DockerRootDir}}')"
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi -L 2>/dev/null \
  && echo "GPU in containers: OK" || echo "GPU in containers: FAILED"

echo "===== [2/5] App files + .env ====="
cd "$APP_DIR"
if [ ! -f .env ] && [ -f "$ENV_SRC" ]; then
  cp "$ENV_SRC" .env && chmod 600 .env
fi
# Set PUBLIC_DOMAIN if not already present (call2.ar-voice.com etc.)
if grep -q '^PUBLIC_DOMAIN=' .env 2>/dev/null; then
  : # keep whatever is set
elif [ -n "$DOMAIN" ]; then
  echo "PUBLIC_DOMAIN=$DOMAIN" >> .env
fi

echo "===== [3/5] Build & start main stack (prod+tls) ====="
docker compose --profile prod --profile tls up -d --build

echo "----- Piper model check (baked-in download can silently fail) -----"
if ! docker exec voice-agent ls /models/piper/en_US-lessac-medium.onnx >/dev/null 2>&1; then
  echo ">> Piper model missing in the piper-models volume — downloading into it ..."
  docker run --rm -v ar-voice-agent_piper-models:/models/piper alpine sh -c \
    'apk add --no-cache wget >/dev/null 2>&1; \
     wget -q -O /models/piper/en_US-lessac-medium.onnx \
       https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx && \
     wget -q -O /models/piper/en_US-lessac-medium.onnx.json \
       https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json' \
    && echo ">> Piper model downloaded." || echo ">> WARNING: Piper download failed — copy en_US-lessac-medium.onnx(+.json) into the piper-models volume."
  docker restart voice-agent >/dev/null 2>&1 || true
fi
docker exec voice-agent ls /models/piper/en_US-lessac-medium.onnx >/dev/null 2>&1 \
  && echo "Piper model: OK" || echo "Piper model: MISSING"

echo "===== [4/5] Full Opik stack ====="
if [ ! -d /opt/opik/deployment/docker-compose ]; then
  echo ">> Cloning Opik (deployment compose) ..."
  rm -rf /tmp/opik-bootstrap
  git clone --depth 1 https://github.com/comet-ml/opik.git /tmp/opik-bootstrap
  sudo mkdir -p /opt/opik/deployment
  sudo cp -r /tmp/opik-bootstrap/deployment/docker-compose /opt/opik/deployment/
  rm -rf /tmp/opik-bootstrap
fi
cd /opt/opik/deployment/docker-compose
# Ensure the override remaps the backend to host port 8083, disables auth,
# and joins the app network so Caddy + the SDK can reach the frontend.
OVERRIDE=""
[ -f docker-compose.override.yaml ] && OVERRIDE="-f docker-compose.override.yaml"
[ -f docker-compose.override.yml ] && OVERRIDE="-f docker-compose.override.yml"
if [ -n "$OVERRIDE" ] && ! grep -q '8083' $OVERRIDE 2>/dev/null; then
  sudo tee docker-compose.override.yml >/dev/null <<EOF
services:
  backend:
    ports:
      - "8083:8080"
    environment:
      TOGGLE_OPIK_AI_ENABLED: "true"
      TOGGLE_AGENTS_ENABLED: "true"
      TOGGLE_GUARDRAILS_ENABLED: "false"
      AUTH_ENABLED: "false"
  demo-data-generator:
    environment:
      CREATE_DEMO_DATA: "false"
  frontend:
    networks:
      - default
      - ar-voice-agent_default

networks:
  ar-voice-agent_default:
    external: true
EOF
fi
sudo docker compose -f docker-compose.yaml $OVERRIDE --profile opik up -d
# Let the app's Caddy reach the Opik frontend (different docker network).
sudo docker network connect ar-voice-agent_default opik-frontend-1 2>/dev/null || true
# Point the app at the Opik FRONTEND via the docker gateway (host port 5173).
# The SDK talks to the frontend, which proxies /api to the backend.
GATEWAY=$(docker network inspect ar-voice-agent_default --format '{{(index .IPAM.Config 0).Gateway}}')
cd "$APP_DIR"
sed -i "s|^OPIK_BASE_URL=.*|OPIK_BASE_URL=http://${GATEWAY}:5173|" .env
docker compose --profile prod up -d --force-recreate voice-agent

echo "===== [5/5] Shutdown timer (auto-deactivate) ====="
# az CLI for the deallocate call (system-assigned identity set up separately)
if ! command -v az >/dev/null 2>&1; then
  curl -sL https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor -o /usr/share/keyrings/microsoft.gpg
  echo 'deb [signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/azure-cli/ jammy main' \
    | sudo tee /etc/apt/sources.list.d/azure-cli.list >/dev/null
  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq azure-cli
fi
# Install the cron (10-min interval, same as the original VM) if not present
if ! crontab -l 2>/dev/null | grep -q auto-deactivate; then
  (crontab -l 2>/dev/null; echo "*/10 * * * * $HOME/ar-voice-agent/auto-deactivate.sh") | crontab -
  echo ">> auto-deactivate cron installed. Enable a system-assigned managed identity +"
  echo "   'Virtual Machine Contributor' role so it can deallocate the VM (see notes)."
fi

echo "===== [6/6] Done — remaining MANUAL steps ====="
IP=$(curl -s ifconfig.me || true)
echo "  - VM IP: $IP"
echo "  - Open NSG inbound TCP 80,443 (Azure portal) if not done by setup-vm.sh"
echo "  - Point DNS: $DOMAIN -> $IP  and  optik.$DOMAIN -> $IP"
echo "  - After DNS propagates, Caddy auto-issues TLS for both."
echo "  - Verify:  curl https://$DOMAIN/health   curl https://optik.$DOMAIN/"
echo "  - Twilio: ensure the /voice webhook base uses https://$DOMAIN"
echo "  - Auto-deactivate: VM -> Identity -> System assigned ON +"
echo "    'Virtual Machine Contributor' role on the VM (or its resource group)."
