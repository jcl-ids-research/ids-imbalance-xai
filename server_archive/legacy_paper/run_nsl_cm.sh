#!/bin/bash
cd /opt/ids_revision
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python3 -u deploy/nsl_cm_only.py 2>&1 | tee results/nsl_cm_run.log
