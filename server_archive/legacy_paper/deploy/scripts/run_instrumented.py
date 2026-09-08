"""Instrumented ablation run: one training pass produces every artifact the
reviewers' figures and tables need, so no figure ever requires a re-run.

Saved per (dataset, seed, variant):
  - metrics.json          weighted/macro F1, acc, precision, recall, per-class
  - predictions.npz       y_true, y_pred, y_prob   -> confusion matrices (R1-C6), ROC
  - model.pt              state_dict               -> attention (R2-2), adversarial (R1-C8)
Saved per (dataset, seed):
  - samples.npz           real + DDPM synthetic    -> synthetic quality (R1-C3)
  - meta.json             split sizes, view partition, timings (R1-C7)

Protocol is identical to correct_unsw_ablation.py (official split, train-only
preprocessing, quantile-space class-conditional DDPM, paired init, epoch shuffle).

--task multiclass switches the target from the binary label to the attack
category (UNSW-NB15: 10 classes, NSL-KDD: 5). Nothing else changes: same split,
same preprocessing, same features, same variants. That keeps a multi-class
number comparable with the binary number reported next to it.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from correct_unsw_ablation import (  # noqa: E402
    AUGMENT_FRAC,
    CLASSIFIER_BATCH,
    CLASSIFIER_EPOCHS,
    CLASSIFIER_PATIENCE,
    DDPM_EPOCHS,
    SEMANTIC_VIEWS,
    Classifier,
    MLPDDPM,
    MultiViewEncoder,
    SingleViewEncoder,
    _train_classifier,
    rebalance,
    load_unsw,
    semantic_view_splits,
    train_ddpm,
)

DEFAULT_OUT = "/opt/ids_revision/results/instrumented"
VARIANTS = ("full_model", "wo_diffusion", "wo_multiview", "baseline")


def positional_view_splits(n_features: int, n_views: int = 3) -> list[list[int]]:
    base, rem = divmod(n_features, n_views)
    dims = [base + (1 if i < rem else 0) for i in range(n_views)]
    splits, start = [], 0
    for d in dims:
        splits.append(list(range(start, start + d)))
        start += d
    return splits


@torch.no_grad()
def predict(model: Classifier, X: np.ndarray, device: torch.device,
            batch: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, probs = [], []
    for i in range(0, len(X), batch):
        xb = torch.FloatTensor(X[i:i + batch]).to(device)
        logits = model(xb)
        p = torch.softmax(logits, dim=1)
        probs.append(p.cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def score(y_true: np.ndarray, y_pred: np.ndarray,
          class_names: list[str] | None = None) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    _, _, f1m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    pc = {}
    for c in labels:
        cp, cr, cf, _ = precision_recall_fscore_support(
            y_true == c, y_pred == c, average="binary", zero_division=0)
        # support matters in multi-class: several UNSW categories hold well
        # under 1% of the test set, and a per-class F1 without its support is
        # not interpretable.
        key = (class_names[int(c)]
               if class_names is not None and int(c) < len(class_names)
               else str(int(c)))
        pc[key] = {"precision": float(cp), "recall": float(cr), "f1": float(cf),
                   "support": int((y_true == c).sum())}
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1),
        "f1_macro": float(f1m),
        "precision": float(p),
        "recall": float(r),
        "per_class": pc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def run(dataset: str, seed: int, out_root: str, smoke: bool,
        task: str = "binary") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(out_root, dataset, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    class_names: list[str] | None = None
    if task == "multiclass":
        from multiclass_data import MULTICLASS_DATASETS
        if dataset not in MULTICLASS_DATASETS:
            raise SystemExit(
                f"--task multiclass supports {sorted(MULTICLASS_DATASETS)}; "
                f"the CIC captures carry only benign/attack labels, so a "
                f"multi-class run on '{dataset}' would be meaningless")
        data, class_names = MULTICLASS_DATASETS[dataset](smoke=smoke)
        if dataset == "unsw":
            view_splits = semantic_view_splits(data.feature_names)
            view_meta = {k: list(v) for k, v in SEMANTIC_VIEWS.items()}
        else:
            view_splits = positional_view_splits(data.n_features)
            view_meta = {f"view{i + 1}": [data.feature_names[j] for j in s]
                         for i, s in enumerate(view_splits)}
    elif dataset == "unsw":
        data = load_unsw("/opt/UNSW-NB15", smoke=smoke)
        view_splits = semantic_view_splits(data.feature_names)
        view_meta = {k: list(v) for k, v in SEMANTIC_VIEWS.items()}
    else:
        from correct_cross_dataset import DATASETS
        data = DATASETS[dataset](seed, smoke)
        view_splits = positional_view_splits(data.n_features)
        view_meta = {f"view{i + 1}": [data.feature_names[j] for j in s]
                     for i, s in enumerate(view_splits)}

    # Size the output layer from the label space, not from the classes that
    # happen to appear in this training split. NSL-KDD U2R holds 52 rows out of
    # 125,973; any subsample can drop it entirely, and deriving n_classes from
    # np.unique would then silently shift every label index above the gap.
    n_classes = (len(class_names) if class_names
                 else int(data.y_train.max()) + 1)
    counts = np.bincount(data.y_train, minlength=n_classes).tolist()
    missing = [i for i, c in enumerate(counts) if c == 0]
    if missing:
        names = ([class_names[i] for i in missing] if class_names else missing)
        print(f"[WARN] classes absent from the training split: {names} - "
              f"the output layer still reserves a slot so label indices stay "
              f"aligned with the test set", flush=True)
    print(f"[TASK] {task}  classes={n_classes}  train distribution={counts}",
          flush=True)
    if class_names:
        print(f"[TASK] class names: {class_names}", flush=True)

    X_sub, X_val, y_sub, y_val = train_test_split(
        data.X_train, data.y_train, test_size=0.1,
        random_state=seed, stratify=data.y_train)

    ddpm_epochs = 2 if smoke else DDPM_EPOCHS
    cls_epochs = 2 if smoke else CLASSIFIER_EPOCHS
    cls_batch = 128 if smoke else CLASSIFIER_BATCH

    t0 = time.time()
    qt = QuantileTransformer(output_distribution="normal", random_state=seed,
                             n_quantiles=min(1000, len(X_sub)))
    X_sub_q = qt.fit_transform(X_sub)
    ddpm: dict[int, MLPDDPM] = {}
    for cls in np.unique(y_sub):
        m = MLPDDPM(data.n_features)
        train_ddpm(m, X_sub_q[y_sub == cls], device, ddpm_epochs, label=str(int(cls)))
        ddpm[int(cls)] = m
    ddpm_sec = time.time() - t0

    # Rebalance training set: majority undersample + minority DDPM augment.
    # Returns balanced data + report (E8 evidence) + separate synthetic arrays (E9/E10 fidelity).
    X_bal, y_bal, balance_report, X_syn, y_syn = rebalance(
        ddpm, X_sub, y_sub, qt, device, seed
    )

    # R1-C3: keep real + synthetic side by side for distribution / semantic checks.
    n_keep = min(20000, len(X_sub))
    np.savez_compressed(
        os.path.join(out_dir, "samples.npz"),
        real=X_sub[:n_keep].astype(np.float32),
        real_labels=y_sub[:n_keep].astype(np.int64),
        synthetic=X_syn[:n_keep].astype(np.float32) if len(X_syn) > 0 else np.empty((0, data.n_features), dtype=np.float32),
        synthetic_labels=y_syn[:n_keep].astype(np.int64) if len(y_syn) > 0 else np.empty(0, dtype=np.int64),
        feature_names=np.array(data.feature_names, dtype=object),
    )

    mv_base = Classifier(MultiViewEncoder(data.n_features, view_splits),
                         num_classes=n_classes)
    sv_base = Classifier(SingleViewEncoder(data.n_features),
                         num_classes=n_classes)
    specs = {
        "full_model":   (copy.deepcopy(mv_base), X_bal, y_bal),
        "wo_diffusion": (copy.deepcopy(mv_base), X_sub, y_sub),
        "wo_multiview": (copy.deepcopy(sv_base), X_bal, y_bal),
        "baseline":     (copy.deepcopy(sv_base), X_sub, y_sub),
    }

    metrics: dict[str, dict] = {}
    timings: dict[str, float] = {}
    for name, (model, Xtr, ytr) in specs.items():
        t1 = time.time()
        _train_classifier(model, Xtr, ytr, X_val, y_val, device,
                          cls_epochs, cls_batch, CLASSIFIER_PATIENCE)
        timings[name] = time.time() - t1

        y_pred, y_prob = predict(model, data.X_test, device, cls_batch)
        metrics[name] = score(data.y_test, y_pred, class_names)

        np.savez_compressed(
            os.path.join(out_dir, f"predictions_{name}.npz"),
            y_true=data.y_test.astype(np.int64),
            y_pred=y_pred.astype(np.int64),
            y_prob=y_prob.astype(np.float32),
        )
        torch.save(
            {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
             "variant": name,
             "input_dim": int(data.n_features),
             "view_splits": view_splits,
             "num_classes": n_classes,
             "task": task,
             "multiview": name in ("full_model", "wo_diffusion")},
            os.path.join(out_dir, f"model_{name}.pt"),
        )
        print(f"  [{name}] F1={metrics[name]['f1'] * 100:.2f} "
              f"macroF1={metrics[name]['f1_macro'] * 100:.2f} "
              f"acc={metrics[name]['acc'] * 100:.2f} ({timings[name] / 60:.1f} min)",
              flush=True)

    meta = {
        "dataset": dataset,
        "seed": seed,
        "task": task,
        "n_classes": n_classes,
        "class_names": class_names,
        "protocol": "official/train-only split + quantile DDPM + mean-target rebalance + paired init + shuffle",
        "balance_report": balance_report,
        "sizes": {"train": int(len(X_sub)), "val": int(len(X_val)),
                  "test": int(len(data.X_test)), "balanced_train": int(len(X_bal)),
                  "features": int(data.n_features)},
        "view_partition": view_meta,
        "timings_sec": {"ddpm": ddpm_sec, **timings},
        "variants": {k: {kk: vv for kk, vv in v.items() if kk != "per_class"}
                     for k, v in metrics.items()},
        "per_class": {k: v["per_class"] for k, v in metrics.items()},
        "confusion_matrix": {k: v["confusion_matrix"] for k, v in metrics.items()},
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] artifacts -> {out_dir}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="unsw",
                    choices=["unsw", "nslkdd", "cicids2017", "cicddos2019"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", default="binary",
                    choices=["binary", "multiclass"])
    args = ap.parse_args()

    t0 = time.time()
    run(args.dataset, args.seed, args.out, args.smoke, args.task)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
