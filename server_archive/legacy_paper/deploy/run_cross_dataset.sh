#!/bin/bash
# Cross-dataset ablation under the corrected protocol.
# Order: nslkdd (official split, fastest) -> cicids2017 -> cicddos2019.
set -u

OUT_DIR="/opt/ids_revision/results/ablation_cross_correct"
SCRIPT="/opt/ids_revision/deploy/correct_cross_dataset.py"
PY="/usr/bin/python3"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/COMPLETE"

for ds in nslkdd cicids2017 cicddos2019; do
    for seed in 42 123 456; do
        echo "[$(date)] START $ds seed=$seed" | tee -a "$OUT_DIR/master.log"
        "$PY" "$SCRIPT" --dataset "$ds" --seed "$seed" --out-dir "$OUT_DIR" \
            >> "$OUT_DIR/${ds}_seed${seed}.log" 2>&1
        rc=$?
        echo "[$(date)] END $ds seed=$seed exit=$rc" | tee -a "$OUT_DIR/master.log"
        if [ "$rc" -ne 0 ]; then
            echo "[$(date)] FAILED $ds seed=$seed" >> "$OUT_DIR/master.log"
        fi
    done
    echo "[$(date)] DATASET DONE $ds" | tee -a "$OUT_DIR/master.log"
done

echo "[$(date)] ALL DONE" | tee "$OUT_DIR/COMPLETE"
