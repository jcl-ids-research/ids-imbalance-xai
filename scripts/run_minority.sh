#!/bin/bash
# Launch minority-recall evaluations across shift levels and seeds.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/opt/ids_revision/rm_dmvt}
cd "$PROJECT_ROOT"

mkdir -p runs/minority logs/minority
rm -f logs/minority/*.done logs/minority/*.failed

declare -A LEVELS=( ["100"]="1.0" ["020"]="0.2" ["005"]="0.05" )

for tag in 100 020 005; do
  keep="${LEVELS[$tag]}"
  for seed in 42 123 456; do
    name="keep${tag}_seed${seed}"
    setsid -f bash -c "timeout 14400 .venv/bin/python scripts/minority_shift.py \
      --data-root /opt/UNSW-NB15 \
      --output runs/minority/${name}.json \
      --keep-fraction ${keep} \
      --seed ${seed} \
      --threads 12 > logs/minority/${name}.log 2>&1 \
      && touch logs/minority/${name}.done \
      || touch logs/minority/${name}.failed"
    sleep 2
  done
done

sleep 45
echo "JOBS: $(pgrep -cf 'minority_shift.py' || echo 0)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
