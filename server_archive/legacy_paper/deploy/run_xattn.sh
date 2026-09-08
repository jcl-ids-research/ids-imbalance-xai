#!/bin/bash
# UNSW-NB15 with cross-view attention fusion, three seeds, sequential.
#
# Everything else matches the arm already reported in the paper: official
# split, train-only preprocessing, class-conditional DDPM, mean-target
# rebalance, paired initialisation, semantic views. The only variable under
# test is how the three view representations are combined - linear projection
# of a concatenation, versus attention across views.
#
# Output goes to its own directory; nothing existing is touched.

set -u

DEPLOY=/opt/ids_revision/deploy
OUT=/opt/ids_revision/results/unsw_xattn
LOG=$OUT/run.log

mkdir -p "$OUT"
cd "$DEPLOY" || exit 1

{
  echo "START $(date '+%F %T')"
  echo "fusion=attention  out=$OUT"
  echo
} > "$LOG"

for SEED in 42 123 456; do
  echo "=== seed $SEED start $(date '+%F %T') ===" >> "$LOG"
  FUSION_MODE=attention /usr/bin/python3 correct_unsw_ablation.py \
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
