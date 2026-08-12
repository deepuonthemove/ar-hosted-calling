#!/bin/bash
# Snapshot Redis data to disk (guards against volume loss).
# Keep the last 7 daily snapshots.
set -e
BK=/mnt/backups
mkdir -p "$BK"
docker exec redis redis-cli BGSAVE >/dev/null 2>&1 || true
sleep 1
# Copy the current AOF dir (most recent persistence state)
TS=$(date +%Y%m%d-%H%M)
docker run --rm -v ar-voice-agent_redis-data:/d -v "$BK:/bk" -e TS="$TS" alpine \
  sh -c 'mkdir -p /bk/redis-$TS && cp -r /d/appendonlydir/* /bk/redis-$TS/' \
  >/dev/null 2>&1 || true
# The container copies as root — take ownership so pruning works
sudo chown -R "$(id -u):$(id -g)" "$BK/redis-$TS" 2>/dev/null || true
# Prune to last 7
ls -d "$BK"/redis-* 2>/dev/null | sort | head -n -7 | xargs -r rm -rf
echo "[$(date)] Redis backup -> $BK/redis-$TS" >> /tmp/redis-backup.log
