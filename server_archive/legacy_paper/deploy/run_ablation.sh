#!/bin/bash
cd /opt/ids_revision/deploy
mkdir -p ../results/ablation_repeat

for seed in 42 123 456; do
    echo "[$(date)] Running seed=$seed"
    /usr/bin/python3 ablation_cross_dataset.py --dataset unsw --seed $seed --output ../results/ablation_repeat/unsw_seed${seed}.json > ../results/ablation_repeat/run_${seed}.log 2>&1
    echo "[$(date)] Finished seed=$seed"
done

echo "[$(date)] All done!" > ../results/ablation_repeat/COMPLETE
