#!/bin/bash
cd /opt/ids_revision
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python3 -u deploy/adversarial_full.py 2>&1 | tee results/adversarial_full_run.log
