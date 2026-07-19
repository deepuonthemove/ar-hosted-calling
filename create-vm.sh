#!/usr/bin/env bash
set -euo pipefail

RG="rg-voice-agent"
VM="vm-voice-agent"
LOC="southindia"
ADMIN="azureuser"

echo "=== Creating resource group (if not exists) ==="
az group create --name "$RG" --location "$LOC"

echo "=== VNet + Subnet ==="
az network vnet create \
  --resource-group "$RG" --name "vnet-voice" \
  --address-prefix "10.0.0.0/16" \
  --subnet-name "subnet-voice" \
  --subnet-prefix "10.0.1.0/24"

echo "=== NSG (SSH + HTTPS + WebSocket port) ==="
az network nsg create --resource-group "$RG" --name "nsg-voice"
az network nsg rule create --resource-group "$RG" --nsg-name "nsg-voice" \
  --name ssh --priority 100 --direction Inbound --access Allow \
  --protocol Tcp --destination-port-ranges 22
az network nsg rule create --resource-group "$RG" --nsg-name "nsg-voice" \
  --name https --priority 110 --direction Inbound --access Allow \
  --protocol Tcp --destination-port-ranges 80 443
az network nsg rule create --resource-group "$RG" --nsg-name "nsg-voice" \
  --name voice-ws --priority 120 --direction Inbound --access Allow \
  --protocol Tcp --destination-port-ranges 8080

echo "=== Public IP ==="
az network public-ip create \
  --resource-group "$RG" --name "pip-voice" \
  --sku Standard --allocation-method Static

echo "=== NIC ==="
az network nic create \
  --resource-group "$RG" --name "nic-voice" \
  --vnet-name "vnet-voice" --subnet "subnet-voice" \
  --public-ip-address "pip-voice" \
  --network-security-group "nsg-voice"

echo "=== VM (NC4as_T4_v3 — 4 vCPU, 16GB T4 GPU) ==="
az vm create \
  --resource-group "$RG" --name "$VM" \
  --image "ubuntu-2204-lts" \
  --size "Standard_NC4as_T4_v3" \
  --admin-username "$ADMIN" \
  --generate-ssh-keys \
  --nics "nic-voice" \
  --os-disk-size-gb 128

echo "=== NVIDIA GPU Driver ==="
az vm extension set \
  --resource-group "$RG" --vm-name "$VM" \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute --version 1.9

echo "=== Docker + NVIDIA Container Toolkit ==="
az vm run-command invoke \
  --resource-group "$RG" --name "$VM" \
  --command-id RunShellScript \
  --scripts "
curl -fsSL https://get.docker.com | bash
usermod -aG docker $ADMIN
distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-container-toolkit
systemctl restart docker
"

IP=$(az vm show --resource-group "$RG" --name "$VM" -d --query publicIps -o tsv)
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║ VM READY                                        ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║ IP:      $IP"
echo "║ SSH:     ssh $ADMIN@$IP"
echo "║ Ports:   22 (SSH), 443 (WSS), 8080 (voice WS)   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "=== Next: deploy the app ==="
echo "  rsync -avz --exclude infra/ ./ azureuser@$IP:/opt/ar-voice-agent/deploy/"
echo "  ssh $ADMIN@$IP 'cd /opt/ar-voice-agent/deploy && docker compose --profile prod up -d --build'"
