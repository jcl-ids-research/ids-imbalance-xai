"""Adversarial robustness against the models this revision actually trained.

The previous script (adversarial_full.py) cannot be reused by repointing
MODEL_PATH. Three defects make its numbers unusable:

  line 46  df = pd.concat([tr, te])  then train_test_split
           -> merges the official train and test sets and re-splits them,
              which is precisely the leakage Reviewers 1 and 2 objected to
  line 53  scaler.fit_transform(X)   on the merged frame
           -> preprocessing fitted on test data
  line 22  from ablation.models.model_variants import create_variant
           -> a different architecture from the Classifier/MultiViewEncoder
              this revision trains, so the checkpoint would not even load

This runs the same attack battery against the real checkpoints, under the
official split with preprocessing fitted on training data only.

Attacking every variant rather than only the full model answers a question the
paper needs: synthetic augmentation shifts the decision boundary, so does it
also change how the boundary behaves under attack?
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from correct_unsw_ablation import (  # noqa: E402
    Classifier,
    MultiViewEncoder,
    SingleViewEncoder,
    load_unsw,
)

VARIANTS = ("full_model", "wo_diffusion", "wo_multiview", "baseline")
EPS_FGSM = [0.01, 0.03, 0.05, 0.1, 0.2]
PGD_CONFIGS = [
    (0.03, 0.007, 10), (0.05, 0.012, 7), (0.05, 0.012, 10),
    (0.05, 0.012, 20), (0.05, 0.012, 40), (0.10, 0.025, 10),
    (0.10, 0.025, 20), (0.10, 0.025, 40), (0.05, 0.005, 100),
]
MIM_EPS = [0.01, 0.05, 0.1]
CW_C = [0.01, 0.1, 1.0]
CW_STEPS = 50
CW_SUBSET = 2000
EVAL_BATCH = 4096


def load_data(dataset: str, seed: int):
    if dataset == "unsw":
        return load_unsw("/opt/UNSW-NB15", smoke=False)
    from correct_cross_dataset import DATASETS
    return DATASETS[dataset](seed, False)


def build_model(ckpt: dict, device: torch.device) -> Classifier:
    """Rebuild the exact architecture recorded in the checkpoint."""
    dim = int(ckpt["input_dim"])
    n_classes = int(ckpt.get("num_classes", 2))
    enc = (MultiViewEncoder(dim, ckpt["view_splits"]) if ckpt["multiview"]
           else SingleViewEncoder(dim))
    model = Classifier(enc, num_classes=n_classes)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, X: torch.Tensor, y: torch.Tensor, device) -> tuple[float, float]:
    preds = []
    for i in range(0, len(X), EVAL_BATCH):
        preds.append(model(X[i:i + EVAL_BATCH].to(device)).argmax(1).cpu())
    p = torch.cat(preds)
    _, _, f1, _ = precision_recall_fscore_support(
        y.numpy(), p.numpy(), average="weighted", zero_division=0)
    return float(accuracy_score(y.numpy(), p.numpy())), float(f1)


def _grad(model, X: torch.Tensor, y: torch.Tensor, device) -> torch.Tensor:
    """Loss gradient wrt the input, accumulated in batches to bound memory."""
    grads = []
    for i in range(0, len(X), EVAL_BATCH):
        xb = X[i:i + EVAL_BATCH].clone().detach().to(device).requires_grad_(True)
        yb = y[i:i + EVAL_BATCH].to(device)
        loss = nn.CrossEntropyLoss()(model(xb), yb)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads.append(xb.grad.detach().cpu())
    return torch.cat(grads)


def fgsm(model, X, y, eps, lo, hi, device):
    return torch.clamp(X + eps * _grad(model, X, y, device).sign(), lo, hi)


def pgd(model, X, y, eps, alpha, steps, lo, hi, device, seed):
    g = torch.Generator().manual_seed(seed)
    Xa = X + torch.randn(X.shape, generator=g) * eps * 0.1
    for _ in range(steps):
        Xa = Xa + alpha * _grad(model, Xa, y, device).sign()
        Xa = torch.clamp(torch.min(torch.max(Xa, X - eps), X + eps), lo, hi)
    return Xa


def mim(model, X, y, eps, steps, mu, lo, hi, device):
    Xa, mom = X.clone(), torch.zeros_like(X)
    alpha = eps / steps
    for _ in range(steps):
        g = _grad(model, Xa, y, device)
        mom = mu * mom + g / (g.abs().mean(dim=1, keepdim=True) + 1e-12)
        Xa = torch.clamp(torch.min(torch.max(Xa + alpha * mom.sign(),
                                             X - eps), X + eps), lo, hi)
    return Xa


def cw_l2(model, X, y, c, steps, lo, hi, device):
    X0 = X.to(device)
    Xa = X0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([Xa], lr=0.01)
    yb = y.to(device)
    for _ in range(steps):
        # maximise misclassification while paying an L2 penalty for distance
        loss = -nn.CrossEntropyLoss()(model(Xa), yb) + c * torch.norm(Xa - X0, p=2)
        opt.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        Xa.data = torch.clamp(Xa.data, lo, hi)
    return Xa.detach().cpu()


def run_variant(model, X, y, lo, hi, device, seed, quick: bool) -> dict:
    out: dict = {}
    clean_acc, clean_f1 = evaluate(model, X, y, device)
    out["clean"] = {"acc": clean_acc, "f1": clean_f1}
    print(f"    clean            acc={clean_acc:.4f} f1={clean_f1:.4f}", flush=True)

    out["fgsm"] = {}
    for eps in EPS_FGSM:
        a, f = evaluate(model, fgsm(model, X, y, eps, lo, hi, device), y, device)
        out["fgsm"][str(eps)] = {"acc": a, "f1": f}
        print(f"    FGSM eps={eps:<5}    acc={a:.4f} f1={f:.4f}", flush=True)

    cfgs = PGD_CONFIGS[:2] if quick else PGD_CONFIGS
    out["pgd"] = {}
    for eps, alpha, steps in cfgs:
        Xa = pgd(model, X, y, eps, alpha, steps, lo, hi, device, seed)
        a, f = evaluate(model, Xa, y, device)
        out["pgd"][f"eps{eps}_steps{steps}"] = {
            "acc": a, "f1": f, "eps": eps, "alpha": alpha, "steps": steps}
        print(f"    PGD  eps={eps} steps={steps:<4} acc={a:.4f} f1={f:.4f}", flush=True)

    out["mim"] = {}
    for eps in (MIM_EPS[:1] if quick else MIM_EPS):
        a, f = evaluate(model, mim(model, X, y, eps, 10, 0.9, lo, hi, device), y, device)
        out["mim"][str(eps)] = {"acc": a, "f1": f}
        print(f"    MIM  eps={eps:<5}    acc={a:.4f} f1={f:.4f}", flush=True)

    idx = np.random.RandomState(seed).choice(
        len(X), min(CW_SUBSET, len(X)), replace=False)
    Xs, ys = X[idx], y[idx]
    out["cw"] = {"subset_size": int(len(idx))}
    for c in (CW_C[:1] if quick else CW_C):
        a, f = evaluate(model, cw_l2(model, Xs, ys, c, CW_STEPS, lo, hi, device),
                        ys, device)
        out["cw"][str(c)] = {"acc": a, "f1": f}
        print(f"    CW   c={c:<6}     acc={a:.4f} f1={f:.4f}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="unsw",
                    choices=["unsw", "nslkdd", "cicids2017", "cicddos2019"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default="results/phase1_corrected")
    ap.add_argument("--out", default="results/adversarial")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--quick", action="store_true",
                    help="reduced attack grid for a wiring check")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_data(args.dataset, args.seed)
    X = torch.FloatTensor(data.X_test)
    y = torch.LongTensor(data.y_test)
    lo, hi = float(X.min()), float(X.max())
    print(f"[{args.dataset} seed{args.seed}] test={tuple(X.shape)} "
          f"range=[{lo:.3f}, {hi:.3f}]", flush=True)

    results, t0 = {}, time.time()
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        path = os.path.join(args.models, args.dataset, f"seed{args.seed}",
                            f"model_{name}.pt")
        if not os.path.exists(path):
            print(f"  [{name}] checkpoint missing: {path}", flush=True)
            continue
        print(f"  [{name}]", flush=True)
        model = build_model(torch.load(path, map_location=device), device)
        results[name] = run_variant(model, X, y, lo, hi, device,
                                    args.seed, args.quick)

    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, f"adversarial_{args.dataset}_seed{args.seed}.json")
    with open(dst, "w") as f:
        json.dump({"dataset": args.dataset, "seed": args.seed,
                   "protocol": "official/train-only split, preprocessing fitted "
                               "on training data only; attacks in standardised "
                               "feature space clipped to the test-set range",
                   "n_test": int(len(X)), "clip_range": [lo, hi],
                   "quick": args.quick, "variants": results}, f, indent=2)
    print(f"[OK] {dst}   ({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
