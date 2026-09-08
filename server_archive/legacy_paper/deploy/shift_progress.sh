#!/bin/bash
# Progress of the five shift levels.
OUT=/opt/ids_revision/results/shift_experiment

echo "=== TIME ==="
date '+%F %T'

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

echo
echo "=== RUNNING ==="
n=$(pgrep -fc shift_experiment 2>/dev/null || echo 0)
echo "shift_experiment processes: $n"

echo
echo "=== RESULTS WRITTEN ==="
ls -1 "$OUT"/*.json 2>/dev/null | wc -l | xargs echo "json files:"
ls -1 "$OUT"/*.json 2>/dev/null | sed 's|.*/||'

echo
echo "=== PER LEVEL ==="
for L in keep100 keep050 keep020 keep010 keep005; do
  log="$OUT/${L}.log"
  [ -f "$log" ] || continue
  done_seeds=$(grep -c 'rc=0 ===' "$log" 2>/dev/null || echo 0)
  failed=$(grep -c 'FAILED' "$log" 2>/dev/null || echo 0)
  last=$(grep -E '^\s+\[(DDPM|BAL|SHIFT|DATA)' "$log" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
  printf "  %-9s seeds done %s/3  failed %s  | %s\n" "$L" "$done_seeds" "$failed" "${last:0:70}"
done

echo
echo "=== ERRORS ==="
grep -l -E 'Traceback|Error|FAILED' "$OUT"/*.log 2>/dev/null | sed 's|.*/||' || echo "none"
