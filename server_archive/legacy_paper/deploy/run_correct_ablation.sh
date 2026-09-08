#!/bin/bash
# Correct UNSW-NB15 ablation: 3 seeds, official split, real DDPM augmentation.
set -u

DATA_DIR="/opt/UNSW-NB15"
OUT_DIR="/opt/ids_revision/results/ablation_repeat_semantic"
SCRIPT="/opt/ids_revision/deploy/correct_unsw_ablation.py"
PY="/usr/bin/python3"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/COMPLETE"

for seed in 42 123 456; do
    echo "[$(date)] START seed=$seed" | tee -a "$OUT_DIR/master.log"
    "$PY" "$SCRIPT" --seed "$seed" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" \
        >> "$OUT_DIR/seed${seed}.log" 2>&1
    rc=$?
    echo "[$(date)] END seed=$seed exit=$rc" | tee -a "$OUT_DIR/master.log"
    if [ "$rc" -ne 0 ]; then
        echo "[$(date)] FAILED seed=$seed" >> "$OUT_DIR/master.log"
        exit "$rc"
    fi
done

echo "[$(date)] ALL DONE" | tee "$OUT_DIR/COMPLETE"
