"""Controlled distribution shift on UNSW-NB15.

The paper wins on NSL-KDD and loses on the other three. NSL-KDD is the only
one whose official split carries distribution shift: R2L is 0.79% of training
rows but 12.81% of test rows. The hypothesis is that diffusion augmentation
earns its keep specifically under that condition, because synthetic minority
samples widen what the model has seen, whereas a tree can only partition
patterns already present in training.

One supporting dataset makes that an explanation, not a finding. This script
manufactures the same kind of shift inside UNSW-NB15, where none exists, and
varies its severity so the relationship can be traced rather than asserted.

Design
------
Binary task. The training set is subsampled so that attacks make up a chosen
fraction of what they would normally be; the test set is left untouched. That
reproduces NSL-KDD's structure - rare in training, ordinary at test time -
while every other factor is held fixed.

    keep = 1.00   no shift, the control
    keep = 0.50
    keep = 0.20
    keep = 0.10
    keep = 0.05   attacks reduced to a twentieth

At each level three models are compared on identical data:

    full          diffusion augmentation + multi-view classifier
    no-diffusion  same classifier, unbalanced training set
    XGBoost       the baseline that beats us when there is no shift

If the gap to XGBoost narrows as shift deepens, the hypothesis holds. If it
does not, it fails, and that is worth knowing before the claim reaches a
reviewer.

Usage
    /usr/bin/python3 shift_experiment.py --keep 0.2 --seed 42 --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, "/opt/ids_revision/deploy")

from correct_unsw_ablation import (  # noqa: E402
    CLASSIFIER_BATCH,
    CLASSIFIER_EPOCHS,
    CLASSIFIER_PATIENCE,
    DDPM_EPOCHS,
    Classifier,
    MLPDDPM,
    MultiViewEncoder,
    SingleViewEncoder,
    load_unsw,
    rebalance,
    semantic_view_splits,
    train_ddpm,
    _train_classifier,
)

DEFAULT_DATA_DIR = "/opt/UNSW-NB15"


def apply_shift(X: np.ndarray, y: np.ndarray, keep: float,
                seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Thin the attack class in training only, leaving the test side alone.

    Attacks are the majority in UNSW-NB15 (68%), so thinning them creates the
    situation NSL-KDD has by design: a category the model rarely sees during
    training but meets at full strength at test time.
    """
    rng = np.random.default_rng(seed)
    attack_idx = np.flatnonzero(y == 1)
    normal_idx = np.flatnonzero(y == 0)

    n_keep = max(1, int(round(len(attack_idx) * keep)))
    kept_attacks = rng.choice(attack_idx, size=n_keep, replace=False)
    order = np.concatenate([normal_idx, kept_attacks])
    rng.shuffle(order)

    report = {
        "keep_fraction": keep,
        "attacks_before": int(len(attack_idx)),
        "attacks_after": int(n_keep),
        "normals": int(len(normal_idx)),
        "attack_share_before": float(len(attack_idx) / len(y)),
        "attack_share_after": float(n_keep / len(order)),
    }
    return X[order], y[order], report


def evaluate(model, X, y, device, batch: int) -> dict:
    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.FloatTensor(X[i : i + batch]).to(device)
            preds.extend(model(xb).argmax(dim=1).cpu().numpy().tolist())
    acc = accuracy_score(y, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        y, preds, average="weighted", zero_division=0
    )
    _, _, f1m, _ = precision_recall_fscore_support(
        y, preds, average="macro", zero_division=0
    )
    return {"acc": float(acc), "f1": float(f1), "f1_macro": float(f1m),
            "precision": float(p), "recall": float(r)}


def run(keep: float, seed: int, data_dir: str, out_dir: str,
        smoke: bool = False) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    data = load_unsw(data_dir, smoke=smoke)
    print(f"  [DATA] train={len(data.X_train)} test={len(data.X_test)} "
          f"features={data.n_features}", flush=True)

    X_shift, y_shift, shift_report = apply_shift(
        data.X_train, data.y_train, keep, seed
    )
    print(f"  [SHIFT] keep={keep}  attacks {shift_report['attacks_before']} -> "
          f"{shift_report['attacks_after']}  share "
          f"{shift_report['attack_share_before']:.3f} -> "
          f"{shift_report['attack_share_after']:.3f}", flush=True)

    X_sub, X_val, y_sub, y_val = train_test_split(
        X_shift, y_shift, test_size=0.1, random_state=seed, stratify=y_shift
    )

    ddpm_epochs = 2 if smoke else DDPM_EPOCHS
    cls_epochs = 2 if smoke else CLASSIFIER_EPOCHS
    cls_batch = 128 if smoke else CLASSIFIER_BATCH

    qt = QuantileTransformer(output_distribution="normal", random_state=seed,
                             n_quantiles=min(1000, len(X_sub)))
    X_sub_q = qt.fit_transform(X_sub)

    ddpm_models: dict[int, MLPDDPM] = {}
    for cls in np.unique(y_sub):
        m = MLPDDPM(data.n_features)
        train_ddpm(m, X_sub_q[y_sub == cls], device, ddpm_epochs,
                   label=str(int(cls)))
        ddpm_models[int(cls)] = m

    X_bal, y_bal, balance_report = rebalance(
        ddpm_models, X_sub, y_sub, qt, device, seed
    )

    view_splits = semantic_view_splits(data.feature_names)
    import copy as _copy

    # Paired initialisation: the two multi-view variants start from one set of
    # weights, the two single-view variants from another, so a comparison
    # within each pair isolates the balancing module rather than the draw.
    full = Classifier(MultiViewEncoder(data.n_features, view_splits))
    no_diff = _copy.deepcopy(full)
    no_mv = Classifier(SingleViewEncoder(data.n_features))
    neither = _copy.deepcopy(no_mv)

    # Variants carrying the balancing module see X_bal; the others see the
    # untouched, shifted sub-train.
    _train_classifier(full, X_bal, y_bal, X_val, y_val, device,
                      cls_epochs, cls_batch, CLASSIFIER_PATIENCE)
    _train_classifier(no_diff, X_sub, y_sub, X_val, y_val, device,
                      cls_epochs, cls_batch, CLASSIFIER_PATIENCE)
    _train_classifier(no_mv, X_bal, y_bal, X_val, y_val, device,
                      cls_epochs, cls_batch, CLASSIFIER_PATIENCE)
    _train_classifier(neither, X_sub, y_sub, X_val, y_val, device,
                      cls_epochs, cls_batch, CLASSIFIER_PATIENCE)

    variants = {
        "full_model": evaluate(full, data.X_test, data.y_test, device, cls_batch),
        "wo_diffusion": evaluate(no_diff, data.X_test, data.y_test, device, cls_batch),
        "wo_multiview": evaluate(no_mv, data.X_test, data.y_test, device, cls_batch),
        "baseline": evaluate(neither, data.X_test, data.y_test, device, cls_batch),
    }

    # the baseline that beats us when there is no shift
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            random_state=seed, n_jobs=8, eval_metric="logloss",
        )
        xgb.fit(X_bal, y_bal)
        pred = xgb.predict(data.X_test)
        _, _, f1m, _ = precision_recall_fscore_support(
            data.y_test, pred, average="macro", zero_division=0
        )
        variants["xgboost_balanced"] = {
            "f1_macro": float(f1m),
            "acc": float(accuracy_score(data.y_test, pred)),
        }
    except ImportError:
        print("  [WARN] xgboost unavailable, baseline skipped", flush=True)

    result = {
        "dataset": "unsw",
        "seed": seed,
        "keep_fraction": keep,
        "protocol": "official split, train-side attack thinning, test untouched",
        "shift_report": shift_report,
        "balance_report": balance_report,
        "sizes": {
            "train_after_shift": int(len(X_sub)),
            "val": int(len(X_val)),
            "test": int(len(data.X_test)),
            "balanced_train": int(len(X_bal)),
        },
        "variants": variants,
        "minutes": round((time.time() - t0) / 60, 1),
    }

    os.makedirs(out_dir, exist_ok=True)
    tag = f"keep{int(round(keep * 100)):03d}"
    path = Path(out_dir) / f"unsw_{tag}_seed{seed}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[OK] {path}  ({result['minutes']} min)", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", type=float, required=True,
                        help="fraction of attack rows retained in training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    result = run(args.keep, args.seed, args.data_dir, args.out_dir, args.smoke)
    print(json.dumps(result["variants"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
