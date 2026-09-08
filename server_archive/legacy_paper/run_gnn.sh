#!/bin/bash
cd /opt/ids_revision
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python3 -u deploy/train_gnn_baseline.py     --data_dir /opt/UNSW-NB15     --output_dir /opt/ids_revision/results     --epochs 50 --hidden 128 --layers 3 --k 10     2>&1 | tee /opt/ids_revision/results/gnn_binary_run.log
