#!/bin/bash
# Was the tuning symmetric? Compare what each side actually received.
cd /opt/ids_revision/deploy || exit 1

echo "=== XGBoost in baseline_fair.py, the paper's Table 7 ==="
grep -n "XGB\|learning_rate\|n_estimators\|max_depth\|GridSearch\|param" baseline_fair.py 2>/dev/null | head -25

echo
echo "=== any hyperparameter sweep for the baselines? ==="
grep -rn "GridSearchCV\|RandomizedSearch\|param_grid\|lr_sweep\|for lr in" baseline_fair.py baseline_deep.py 2>/dev/null | head -15

echo
echo "=== our own model: fixed or swept? ==="
grep -n "lr\s*=\|learning_rate\|LEARNING\|CLASSIFIER_EPOCHS\|D_MODEL\|NHEAD\|N_LAYERS\|DIM_FF\|DROPOUT" correct_unsw_ablation.py | head -20

echo
echo "=== what the deep baselines got ==="
grep -n "lr_sweep\|learning_rate\|for lr" baseline_deep.py 2>/dev/null | head -10
