#!/bin/bash
# Deallocate the Azure VM after 1 hour of inactivity
LOG="/tmp/auto-deactivate.log"

echo "[$(date)] Checking activity..." >> "$LOG"

HTTP_REQS=$(docker logs voice-agent --since 1h 2>/dev/null | grep -cE '(GET|POST|websocket)' || echo 0)
SSH_SESSIONS=$(who | wc -l)

echo "  reqs=$HTTP_REQS ssh=$SSH_SESSIONS" >> "$LOG"

[ "$HTTP_REQS" -gt 0 ] && echo "  Active - recent HTTP" >> "$LOG" && exit 0
[ "$SSH_SESSIONS" -gt 0 ] && echo "  Active - SSH" >> "$LOG" && exit 0

az login --identity --object-id 28648ac4-96c1-4834-b048-b5e02bc991ba > /dev/null 2>&1
az vm deallocate --resource-group rg-voice-agent --name vm-voice-agent --no-wait >> "$LOG" 2>&1

echo "Deallocate initiated" >> "$LOG"
