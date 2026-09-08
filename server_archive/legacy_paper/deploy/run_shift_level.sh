#!/bin/bash
# Controlled distribution shift on UNSW-NB15.
#
# One process per shift level, five in parallel. Each process walks its three
# seeds in sequence, so the GPU holds five jobs at a time rather than fifteen.
#
# Everything except the shift level is fixed: same features, same official
# split, same preprocessing, same seeds, same models. The test set is never
# touched, so all five levels are scored on identical data.

set -u

DEPLOY=/opt/ids_revision/deploy
OUT=/opt/ids_revision/results/shift_experiment

mkdir -p "$OUT"
cd "$DEPLOY" || exit 1

KEEP=$1
TAG=$(printf "keep%03d" "$(echo "$KEEP * 100" | bc | cut -d. -f1)")
LOG="$OUT/${TAG}.log"

{
  echo "START $(date '+%F %T')"
  echo "keep_fraction=$KEEP"
  echo
} > "$LOG"

for SEED in 42 123 456; do
  echo "=== $TAG seed $SEED start $(date '+%F %T') ===" >> "$LOG"
  /usr/bin/python3 shift_experiment.py \
      --keep "$KEEP" \
      --seed "$SEED" \
      --out-dir "$OUT" >> "$LOG" 2>&1
  rc=$?
  echo "=== $TAG seed $SEED end $(date '+%F %T') rc=$rc ===" >> "$LOG"
  if [ $rc -ne 0 ]; then
    echo "FAILED $TAG seed $SEED rc=$rc" >> "$LOG"
  fi
done

echo "DONE $TAG $(date '+%F %T')" >> "$LOG"
