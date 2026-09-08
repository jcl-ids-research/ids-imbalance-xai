#!/bin/bash
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec python3 -u -W ignore /opt/ids_revision/ablation/train_ablation.py "$@"
