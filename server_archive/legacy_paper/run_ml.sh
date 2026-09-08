#!/bin/bash
cd /opt/ids_revision
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python3 -u deploy/traditional_ml_all.py 2>&1 | tee results/traditional_ml_run.log
