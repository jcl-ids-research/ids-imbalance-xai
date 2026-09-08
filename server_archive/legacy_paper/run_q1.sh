#!/bin/bash
cd /opt/ids_revision
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python3 -u deploy/adversarial_q1_final.py 2>&1 | tee results/adversarial_q1_run.log
