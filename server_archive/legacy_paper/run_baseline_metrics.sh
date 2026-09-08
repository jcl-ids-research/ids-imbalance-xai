#!/bin/bash
cd /opt/ids_revision
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python3 -u deploy/run_all_baseline_metrics.py 2>&1 | tee /opt/ids_revision/results/baseline_metrics_run.log
