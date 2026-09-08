#!/bin/bash
# NSL-KDD semantic-view ablation, three seeds, sequential.
#
# Same arm as phase1_corrected: official split, train-only preprocessing,
# class-conditional DDPM, mean-target rebalance(), paired initialisation.
# The only change under test is the view partition: semantic (9/12/19 by
# KDD attribute family) instead of positional (14/13/13 by column index).
#
# Output goes to its own directory; nothing existing is overwritten.

set -u

DEPLOY=/opt/ids_revision/deploy
OUT=/opt/ids_revision/results/nslkdd_semantic
LOG=$OUT/run.log

mkdir -p "$OUT"
cd "$DEPLOY" || exit 1

{
  echo "START $(date '+%F %T')"
  echo "out=$OUT"
  echo
} > "$LOG"

for SEED in 42 123 456; do
  echo "=== seed $SEED start $(date '+%F %T') ===" >> "$LOG"
  /usr/bin/python3 correct_cross_dataset.py \
      --dataset nslkdd \
      --seed "$SEED" \
      --out-dir "$OUT" >> "$LOG" 2>&1
  rc=$?
  echo "=== seed $SEED end $(date '+%F %T') rc=$rc ===" >> "$LOG"
  if [ $rc -ne 0 ]; then
    echo "SEED $SEED FAILED rc=$rc" >> "$LOG"
  fi
done

echo "ALL DONE $(date '+%F %T')" >> "$LOG"
ls -l "$OUT" >> "$LOG"
