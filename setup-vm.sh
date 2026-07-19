#!/usr/bin/env bash
set -euo pipefail

# Usage: ./setup-vm.sh <resource-group> <vm-name> [region]
# Requires: az CLI logged in with BAA-enabled subscription

RG="${1:-rg-voice-agent}"
VM="${2:-vm-voice-agent}"
REGION="${3:-eastus}"
ADMIN="azureuser"

echo "=== 1. Create Resource Group ==="
az group create --name "$RG" --location "$REGION"

echo "=== 2. Create NC4as_T4_v3 VM (16GB T4, 4 vCPU, 28GB RAM) ==="
az vm create \
  --resource-group "$RG" \
  --name "$VM" \
  --image "ubuntu-2204-lts" \
  --size "Standard_NC4as_T4_v3" \
  --admin-username "$ADMIN" \
  --generate-ssh-keys \
  --public-ip-sku Standard \
  --nic-delete-option Delete \
  --os-disk-size-gb 128 \
  --os-disk-delete-option Delete

echo "=== 3. Open ports for WebSocket (80/443) and health (8080) ==="
az vm open-port --resource-group "$RG" --name "$VM" --port 80,443,8080 --priority 100

echo "=== 4. Install NVIDIA drivers + CUDA ==="
az vm extension set \
  --resource-group "$RG" \
  --vm-name "$VM" \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute \
  --version 1.9

echo "=== 5. Install Docker + NVIDIA Container Toolkit ==="
az vm run-command invoke \
  --resource-group "$RG" \
  --name "$VM" \
  --command-id RunShellScript \
  --scripts "
curl -fsSL https://get.docker.com | bash
sudo usermod -aG docker $ADMIN
distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
"

echo "=== 6. Get public IP ==="
IP=$(az vm show --resource-group "$RG" --name "$VM" -d --query publicIps -o tsv)
echo "VM ready at: $IP"
echo "SSH: ssh $ADMIN@$IP"

echo ""
echo "=== Next steps ==="
echo "  ssh $ADMIN@$IP"
echo "  git clone <your-repo> && cd ar-voice-agent/deploy"
echo "  docker compose up -d"
echo ""
echo "=== HIPAA notes ==="
echo "  - Sign BAA via Azure Portal → Subscription → Microsoft Azure BAA"
echo "  - Enable encryption-at-host: az vm create --enable-encryption-at-host"
echo "  - Audit: enable Azure Monitor + Log Analytics on this VM"
