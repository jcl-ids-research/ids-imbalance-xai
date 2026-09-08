#!/bin/bash
# Progress check for the NSL-KDD semantic-view run.
# Note: cd first - the previous version relied on the caller's directory.
cd /opt/ids_revision/deploy || exit 1

OUT=/opt/ids_revision/results/nslkdd_semantic

echo "=== TIME ==="
date '+%F %T'

echo
echo "=== ELAPSED ==="
if [ -f "$OUT/run.log" ]; then
  start=$(stat -c %Y "$OUT/run.log")
  now=$(date +%s)
  echo "$(( (now - start) / 60 )) minutes since log created"
fi

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

echo
echo "=== JOB ==="
ps -o pid,etime,cmd -p "$(pgrep -f 'correct_cross_dataset' | head -1)" 2>/dev/null || echo "no job running"

echo
echo "=== RESULTS WRITTEN ==="
ls -l "$OUT"/*.json 2>/dev/null || echo "no json yet"

echo
echo "=== SEED MARKERS ==="
grep -E '^=== seed' "$OUT/run.log" 2>/dev/null || echo "none yet"

echo
echo "=== LAST 12 LOG LINES ==="
tail -12 "$OUT/run.log" 2>/dev/null
