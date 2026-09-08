#!/bin/bash
cd /opt/ids_revision/deploy
mkdir -p results/phase1

echo "[$(date)] Phase 1 started" > results/phase1/run.log

for seed in 42 123 456; do
  echo "[$(date)] Starting seed $seed" >> results/phase1/run.log
  
  python3 scripts/run_instrumented.py --dataset unsw --seed $seed --out results/phase1 >> results/phase1/run.log 2>&1 &
  python3 scripts/correct_cross_dataset.py --dataset nslkdd --seed $seed --out-dir results/phase1 >> results/phase1/run.log 2>&1 &
  python3 scripts/correct_cross_dataset.py --dataset cicids2017 --seed $seed --out-dir results/phase1 >> results/phase1/run.log 2>&1 &
  python3 scripts/correct_cross_dataset.py --dataset cicddos2019 --seed $seed --out-dir results/phase1 >> results/phase1/run.log 2>&1 &
  
  wait
  echo "[$(date)] Seed $seed completed" >> results/phase1/run.log
done

echo "[$(date)] Phase 1 complete!" >> results/phase1/run.log

