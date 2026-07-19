#!/usr/bin/env bash
set -euo pipefail

# Quick start: spin up voice agent on Azure
# Usage: ./deploy.sh [dev|prod]

MODE="${1:-dev}"
INFRA_DIR="$(cd "$(dirname "$0")/infra" && pwd)"

echo "==> Pulumi stack up..."
cd "$INFRA_DIR"

# Init if first run
if [ ! -f "Pulumi.voice-agent.yaml" ]; then
  echo "First run — Pulumi login required (use Azure storage account or pulumi.com)"
  pulumi login
fi

pulumi stack init voice-agent --non-interactive 2>/dev/null || true
pulumi stack select voice-agent

# Set defaults if not configured
pulumi config set azure-native:location "${AZURE_LOCATION:-eastus}" 2>/dev/null || true

# Prompt for SSH key if missing
if ! pulumi config get sshPublicKey &>/dev/null; then
  echo ""
  echo "SSH public key not found. Please set it:"
  echo "  pulumi config set --secret sshPublicKey \"\$(cat ~/.ssh/id_rsa.pub)\""
  exit 1
fi

echo "==> Deploying VM + GPU drivers..."
pulumi up --yes

IP=$(pulumi stack output public_ip)
echo ""
echo "==> VM deployed at: $IP"
echo ""

# Wait for bootstrap to finish
echo "==> Waiting for GPU drivers + Docker install (2-3 min)..."
sleep 120

# Copy deploy files
echo "==> Syncing code to VM..."
rsync -avz --exclude infra/ --exclude .git/ "$(cd "$(dirname "$0")" && pwd)/" "azureuser@$IP:/opt/ar-voice-agent/deploy/"

# Start containers
echo "==> Starting containers ($MODE)..."
ssh "azureuser@$IP" "cd /opt/ar-voice-agent/deploy && docker compose --profile $MODE up -d --build"

echo ""
echo "==> Done!"
echo "    Health:  http://$IP:8080/health"
echo "    Client:  open deploy/client.html in browser → ws://$IP:8080/ws/"
echo ""
echo "==> Dev mode tips:"
echo "    # Watch logs:"
echo "    ssh azureuser@$IP 'cd /opt/ar-voice-agent/deploy && docker compose --profile dev logs -f voice-agent-dev'"
echo ""
echo "    # Re-sync after edits:"
echo "    rsync -avz --exclude infra/ ./ azureuser@$IP:/opt/ar-voice-agent/deploy/"
echo "    # (uvicorn --reload auto-detects changes if using dev profile)"
echo ""
echo "    # VS Code Remote-SSH (best dev experience):"
echo "    code --remote ssh-remote+azureuser@$IP /opt/ar-voice-agent"
