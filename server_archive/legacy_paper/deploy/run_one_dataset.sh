#!/bin/bash
# Run all 3 seeds for ONE dataset. Launched once per dataset so the three
# datasets execute concurrently on the A100 (each job uses ~1-2 GB).
# Usage: run_one_dataset.sh <dataset>
set -u

ds="$1"
OUT_DIR="/opt/ids_revision/results/cross_redo"
SCRIPT="/opt/ids_revision/deploy/correct_cross_dataset.py"
PY="/usr/bin/python3"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/${ds}.DONE"

for seed in 42 123 456; do
    echo "[$(date)] START $ds seed=$seed" >> "$OUT_DIR/master.log"
    "$PY" "$SCRIPT" --dataset "$ds" --seed "$seed" --out-dir "$OUT_DIR" \
        >> "$OUT_DIR/${ds}_seed${seed}.log" 2>&1
    rc=$?
    echo "[$(date)] END $ds seed=$seed exit=$rc" >> "$OUT_DIR/master.log"
    if [ "$rc" -ne 0 ]; then
        echo "[$(date)] FAILED $ds seed=$seed" >> "$OUT_DIR/master.log"
    fi
done

echo "[$(date)] DATASET DONE $ds" | tee "$OUT_DIR/${ds}.DONE" >> "$OUT_DIR/master.log"
