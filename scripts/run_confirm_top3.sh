#!/bin/bash
# Confirm the top three classifier trials concurrently across seeds 42/123/456.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/opt/ids_revision/rm_dmvt}
STOP_AT=${STOP_AT:-09:00}

cd "$PROJECT_ROOT"
mkdir -p studies/confirm logs

stop_epoch=$(date -d "today $STOP_AT" +%s)
now=$(date +%s)
if (( stop_epoch <= now )); then
  stop_epoch=$(date -d "tomorrow $STOP_AT" +%s)
fi
budget=$((stop_epoch - now - 300))
if (( budget <= 0 )); then
  echo "No safe confirmation window remains before $STOP_AT" >&2
  exit 2
fi

for rank in 1 2 3; do
  timeout --signal=TERM "$budget" .venv/bin/ids-confirm \
    --cache runs/cache/unsw_seed42.npz \
    --cache runs/cache/unsw_seed123.npz \
    --cache runs/cache/unsw_seed456.npz \
    --output "studies/confirm/rank${rank}.json" \
    --storage sqlite:///studies/rm-dmvt.db \
    --study-name rm-dmvt \
    --rank "$rank" \
    --device cuda > "logs/confirm-rank${rank}.log" 2>&1 &
done
wait
