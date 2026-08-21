#!/bin/bash
# Deallocate the Azure VM after 1 hour of inactivity.
# Only real call activity counts — dashboard auto-refresh (/api/*) does not.
# Requires: az CLI installed + system-assigned managed identity with
# "Virtual Machine Contributor" on the VM (or its resource group).
LOG="/tmp/auto-deactivate.log"

echo "[$(date)] Checking activity..." >> "$LOG"

# Grace period: a freshly-booted VM must stay up for at least 1h before it can
# be deactivated, otherwise it self-shuts during setup with no call activity yet.
UPTIME_S=$(awk '{print int($1)}' /proc/uptime 2>/dev/null)
if [ "${UPTIME_S:-0}" -lt 3600 ]; then
  echo "  Up only ${UPTIME_S}s — 1h grace, skipping" >> "$LOG"
  exit 0
fi

COUNT=$(docker logs voice-agent --since 1h 2>/dev/null   | grep -cE '"(GET|POST) /(voice|media/|make-call|call-result|ws/|health)"')

# 'who' doesn't see Azure SSH sessions — check established TCP on port 22 instead.
SSH=$(ss -tn state established '( dport = :22 or sport = :22 )' 2>/dev/null | tail -n +2 | wc -l)

# User toggle: "stay_awake" in Redis disables auto-deactivation while working
STAY=$(docker exec redis redis-cli hget config:app stay_awake 2>/dev/null)

echo "  activity=$COUNT ssh=$SSH stay_awake=$STAY" >> "$LOG"

[ "$STAY" = "1" ] && echo "  Stay-awake enabled" >> "$LOG" && exit 0
[ "${COUNT:-0}" -gt 0 ] && echo "  Active" >> "$LOG" && exit 0
[ "$SSH" -gt 0 ] && echo "  Active (SSH)" >> "$LOG" && exit 0

# Self-discover the VM's resource group + name (no hardcoding).
META=$(curl -s -H 'Metadata: true' 'http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01')
RG=$(echo "$META" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("resourceGroupName",""))' 2>/dev/null)
VM=$(echo "$META" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("name",""))' 2>/dev/null)

[ -z "$RG" ] || [ -z "$VM" ] && echo "  Could not read VM metadata" >> "$LOG" && exit 1

export PATH="$HOME/.local/bin:/usr/bin:/usr/local/bin:$PATH"
az login --identity > /dev/null 2>&1 || { echo "  az login --identity failed (managed identity not set up?)" >> "$LOG"; exit 1; }
az vm deallocate --resource-group "$RG" --name "$VM" --no-wait >> "$LOG" 2>&1

echo "  Deallocate initiated ($RG/$VM)" >> "$LOG"
