#!/bin/bash
cd /opt/ids_revision
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python3 -u deploy/train_gnn_cic.py --data_dir /opt/CIC-IDS-2017/MachineLearningCVE --dataset CIC-IDS-2017 --output /opt/ids_revision/results --max_samples 100000 --k 5 --epochs 50 2>&1 | tee /opt/ids_revision/results/gnn_cic_ids2017_run.log
