#!/bin/bash
# Start a classifier study and guarantee shutdown before the daytime peak.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/opt/ids_revision/rm_dmvt}
CACHE=${CACHE:?Set CACHE to a prepared seed cache}
STOP_AT=${STOP_AT:-09:00}
TRIALS=${TRIALS:-30}

cd "$PROJECT_ROOT"
mkdir -p studies logs

stop_epoch=$(date -d "today $STOP_AT" +%s)
now=$(date +%s)
if (( stop_epoch <= now )); then
  stop_epoch=$(date -d "tomorrow $STOP_AT" +%s)
fi
budget=$((stop_epoch - now - 300))
if (( budget <= 0 )); then
  echo "No safe tuning window remains before $STOP_AT" >&2
  exit 2
fi

echo "Starting RM-DMVT classifier tuning for at most $budget seconds"
timeout --signal=TERM "$budget" .venv/bin/ids-tune classifier \
  --cache "$CACHE" \
  --output studies/classifier-best.json \
  --storage sqlite:///studies/rm-dmvt.db \
  --trials "$TRIALS" \
  --device cuda 2>&1 | tee "logs/tune-$(date +%Y%m%d-%H%M%S).log"
