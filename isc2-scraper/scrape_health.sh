#!/usr/bin/env bash
set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://172.19.0.2:6333}"
OLLAMA_HOST="${OLLAMA_HOST:-http://10.20.10.23:11434}"
SCRAPER_NAME="${SCRAPER_NAME:-smart_scrape.py}"

echo "==[ SCRAPER PROCESS ]=============================="
ps -o pid,etime,pcpu,pmem,cmd -C python3 | grep -E "$SCRAPER_NAME|python3" || echo "No python3 process found."
echo

echo "==[ SYSTEM LOAD / MEM ]============================"
uptime
free -h
echo

echo "==[ QDRANT HEALTH ]================================"
set +e
curl -sS -m 5 "$QDRANT_URL/collections" | head -c 300 && echo
RC=$?
if [ $RC -ne 0 ]; then echo "Qdrant check FAILED (curl rc=$RC) -> $QDRANT_URL/collections"; fi
set -e
echo

echo "==[ OLLAMA TAGS (models) ]========================="
set +e
/usr/bin/time -f "  (curl time: %E)" \
  curl -sS -m 8 "$OLLAMA_HOST/api/tags" | head -c 400 && echo
RC=$?
if [ $RC -ne 0 ]; then echo "Ollama tags FAILED (curl rc=$RC) -> $OLLAMA_HOST/api/tags"; fi
set -e
echo

echo "==[ OLLAMA EMBED HEALTH (1 item) ]================="
set +e
/usr/bin/time -f "  (curl time: %E)" \
  curl -sS -m 20 -H 'Content-Type: application/json' \
    -d '{"model":"nomic-embed-text","input":["healthcheck"]}' \
    "$OLLAMA_HOST/api/embed" | head -c 200 && echo
RC=$?
if [ $RC -ne 0 ]; then echo "Ollama embed FAILED (curl rc=$RC) -> $OLLAMA_HOST/api/embed"; fi
set -e
echo

echo "==[ DOCKER STATUS (if installed) ]================="
if command -v docker >/dev/null 2>&1; then
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  echo
  echo "-- docker stats (one-shot) --"
  docker stats --no-stream --format 'table {{.Name}}\tCPU %\tMem Usage / Limit'
else
  echo "Docker not found (skipping)."
fi
echo

echo "==[ CONNECTIONS TO OLLAMA ]========================"
if command -v ss >/dev/null 2>&1; then
  ss -tanp | grep -E '11434|ollama' || echo "No TCP connections on :11434"
elif command -v netstat >/dev/null 2>&1; then
  netstat -tanp 2>/dev/null | grep -E '11434|ollama' || echo "No TCP connections on :11434"
else
  echo "Neither ss nor netstat available."
fi
echo

echo "==[ TEMPS (read-only, if exposed) ]================"
for z in /sys/class/thermal/thermal_zone*; do
  [ -e "$z/temp" ] || continue
  type=$(cat "$z/type" 2>/dev/null || echo "zone")
  temp=$(awk '{printf("%.1f°C",$1/1000)}' "$z/temp")
  echo "$(basename "$z")  $type  $temp"
done
echo "==================================================="
