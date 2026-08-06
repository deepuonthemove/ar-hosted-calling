#!/bin/bash
# Deallocate the Azure VM after 1 hour of inactivity.
# Only real call activity counts — dashboard auto-refresh (/api/*) does not.
LOG="/tmp/auto-deactivate.log"

echo "[$(date)] Checking activity..." >> "$LOG"

COUNT=$(docker logs voice-agent --since 1h 2>/dev/null \
  | grep -cE '"(GET|POST) /(voice|media/|make-call|call-result|ws/|health)"')

SSH=$(who 2>/dev/null | wc -l)

# User toggle: "stay_awake" in Redis disables auto-deactivation while working
STAY=$(docker exec redis redis-cli hget config:app stay_awake 2>/dev/null)

echo "  activity=$COUNT ssh=$SSH stay_awake=$STAY" >> "$LOG"

[ "$STAY" = "1" ] && echo "  Stay-awake enabled" >> "$LOG" && exit 0
[ "${COUNT:-0}" -gt 0 ] && echo "  Active" >> "$LOG" && exit 0
[ "$SSH" -gt 0 ] && echo "  Active (SSH)" >> "$LOG" && exit 0

az login --identity --object-id 28648ac4-96c1-4834-b048-b5e02bc991ba > /dev/null 2>&1
az vm deallocate --resource-group rg-voice-agent --name vm-voice-agent --no-wait >> "$LOG" 2>&1

echo "Deallocate initiated" >> "$LOG"
