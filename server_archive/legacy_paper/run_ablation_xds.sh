#!/bin/bash
cd /opt/ids_revision
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python3 -u deploy/ablation_cross_dataset.py 2>&1 | tee results/ablation_cross_dataset_run.log
