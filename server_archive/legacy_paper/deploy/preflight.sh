#!/bin/bash
# Environment check before launching the shift experiment.
cd /opt/ids_revision/deploy || exit 1

echo "=== xgboost ==="
/usr/bin/python3 - <<'PY'
try:
    import xgboost
    print("available, version", xgboost.__version__)
except ImportError as exc:
    print("MISSING:", exc)
PY

echo
echo "=== imports used by shift_experiment ==="
/usr/bin/python3 - <<'PY'
import sys
sys.path.insert(0, "/opt/ids_revision/deploy")
try:
    from correct_unsw_ablation import (
        load_unsw, rebalance, semantic_view_splits, train_ddpm,
        MLPDDPM, MultiViewEncoder, Classifier, _train_classifier,
        DDPM_EPOCHS, CLASSIFIER_EPOCHS, CLASSIFIER_BATCH, CLASSIFIER_PATIENCE,
    )
    print("all symbols resolved")
    print("DDPM_EPOCHS =", DDPM_EPOCHS)
    print("CLASSIFIER_EPOCHS =", CLASSIFIER_EPOCHS)
except Exception as exc:
    print("IMPORT FAILED:", type(exc).__name__, exc)
PY

echo
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

echo
echo "=== data present ==="
ls -1 /opt/UNSW-NB15/ | head -6
