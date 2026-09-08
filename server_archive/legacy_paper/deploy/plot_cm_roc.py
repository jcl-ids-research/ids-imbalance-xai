"""
Generate confusion matrices + ROC/PR curves from real predictions_*.npz.

Input : results/phase1/{dataset}/seed{N}/predictions_{variant}.npz
        keys: y_true (int64), y_pred (int64), y_prob (float32)
Output: PDF (vector, for the paper) + PNG (for quick viewing)

No fabricated values. Every number is computed from the npz on disk.
Datasets/variants that have no npz are skipped and reported as missing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

VARIANTS = ["full_model", "wo_diffusion", "wo_multiview", "baseline"]
VARIANT_LABEL = {
    "full_model": "Full Model",
    "wo_diffusion": "w/o Diffusion",
    "wo_multiview": "w/o Multi-view",
    "baseline": "Baseline",
}
DATASET_LABEL = {
    "unsw": "UNSW-NB15",
    "nslkdd": "NSL-KDD",
    "cicids2017": "CIC-IDS-2017",
    "cicddos2019": "CIC-DDoS2019",
}
COLORS = {
    "full_model": "#2E86AB",
    "wo_diffusion": "#F39C12",
    "wo_multiview": "#E74C3C",
    "baseline": "#7F8C8D",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


def save(fig, outdir: Path, stem: str) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext, dpi in (("pdf", None), ("png", 200)):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, format=ext, dpi=dpi, bbox_inches="tight")
        written.append(str(p))
    plt.close(fig)
    return written


def load_pred(root: Path, ds: str, seed: int, variant: str):
    p = root / ds / f"seed{seed}" / f"predictions_{variant}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {
        "y_true": d["y_true"],
        "y_pred": d["y_pred"],
        "y_prob": d["y_prob"],
        "path": str(p),
    }


# ---------------------------------------------------------------- confusion


def plot_cm_grid(root: Path, outdir: Path, ds: str, seed: int, normalize: bool):
    """One figure per dataset: 1x4 grid of confusion matrices."""
    found = [(v, load_pred(root, ds, seed, v)) for v in VARIANTS]
    found = [(v, d) for v, d in found if d is not None]
    if not found:
        return None, []

    n = len(found)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    axes = axes[0]

    for ax, (variant, d) in zip(axes, found):
        cm = confusion_matrix(d["y_true"], d["y_pred"])
        disp = cm.astype(float)
        if normalize:
            row = disp.sum(axis=1, keepdims=True)
            disp = np.divide(disp, row, out=np.zeros_like(disp), where=row != 0)

        im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=disp.max() if disp.max() else 1)
        k = cm.shape[0]
        names = ["Normal", "Attack"] if k == 2 else [str(i) for i in range(k)]
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_title(VARIANT_LABEL[variant], fontsize=10, fontweight="bold")
        ax.grid(False)

        thr = disp.max() / 2 if disp.max() else 0.5
        for i in range(k):
            for j in range(k):
                txt = f"{disp[i, j]:.3f}" if normalize else f"{cm[i, j]:,}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if disp[i, j] > thr else "black")

    kind = "normalized" if normalize else "counts"
    fig.suptitle(f"{DATASET_LABEL.get(ds, ds)} - Confusion Matrices ({kind}, seed {seed})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    stem = f"cm_{ds}_seed{seed}_{kind}"
    return stem, save(fig, outdir, stem)


# ---------------------------------------------------------------- roc / pr


def _pos_score(d):
    """Binary positive-class score from y_prob (handles 1-D and 2-D)."""
    p = d["y_prob"]
    if p.ndim == 1:
        return p
    if p.shape[1] == 2:
        return p[:, 1]
    return None  # multi-class -> skip binary ROC


def plot_roc_pr(root: Path, outdir: Path, ds: str, seed: int):
    found = [(v, load_pred(root, ds, seed, v)) for v in VARIANTS]
    found = [(v, d) for v, d in found if d is not None]
    if not found:
        return None, [], {}

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.6))
    stats: dict[str, dict] = {}

    for variant, d in found:
        s = _pos_score(d)
        if s is None:
            continue
        y = d["y_true"]
        if len(np.unique(y)) != 2:
            continue

        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = roc_auc_score(y, s)
        ax_roc.plot(fpr, tpr, lw=1.8, color=COLORS[variant],
                    label=f"{VARIANT_LABEL[variant]} (AUC={roc_auc:.4f})")

        prec, rec, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax_pr.plot(rec, prec, lw=1.8, color=COLORS[variant],
                   label=f"{VARIANT_LABEL[variant]} (AP={ap:.4f})")

        stats[variant] = {
            "roc_auc": float(roc_auc),
            "average_precision": float(ap),
            "pr_auc": float(auc(rec, prec)),
            "n_test": int(len(y)),
            "source": d["path"],
        }

    if not stats:
        plt.close(fig)
        return None, [], {}

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve", fontsize=11, fontweight="bold")
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)

    base = float(np.mean(found[0][1]["y_true"]))
    ax_pr.axhline(base, ls="--", c="k", lw=1, alpha=0.5,
                  label=f"Baseline ({base:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
    ax_pr.legend(fontsize=8, loc="lower left")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)

    fig.suptitle(f"{DATASET_LABEL.get(ds, ds)} (seed {seed})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    stem = f"roc_pr_{ds}_seed{seed}"
    return stem, save(fig, outdir, stem), stats


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/phase1",
                    help="dir holding {dataset}/seed{N}/predictions_*.npz")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--datasets", default="unsw,nslkdd,cicids2017,cicddos2019")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.out)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    written: list[str] = []
    missing: list[str] = []
    all_stats: dict = {}

    for ds in datasets:
        for seed in seeds:
            have = [v for v in VARIANTS
                    if load_pred(root, ds, seed, v) is not None]
            if not have:
                missing.append(f"{ds}/seed{seed}  (no predictions_*.npz)")
                continue

            print(f"[{ds} seed{seed}] variants found: {', '.join(have)}")

            for norm in (False, True):
                stem, files = plot_cm_grid(root, outdir, ds, seed, norm)
                if stem:
                    written += files
                    print(f"    CM   -> {stem}.pdf/.png")

            stem, files, stats = plot_roc_pr(root, outdir, ds, seed)
            if stem:
                written += files
                all_stats[f"{ds}_seed{seed}"] = stats
                print(f"    ROC  -> {stem}.pdf/.png")
                for v, s in stats.items():
                    print(f"           {VARIANT_LABEL[v]:16s} "
                          f"AUC={s['roc_auc']:.4f}  AP={s['average_precision']:.4f}")
            else:
                print("    ROC  -> skipped (not binary / no y_prob)")

    if all_stats:
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / "roc_pr_stats.json"
        with open(p, "w") as f:
            json.dump(all_stats, f, indent=2)
        written.append(str(p))
        print(f"\n[OK] stats -> {p}")

    print("\n" + "=" * 60)
    print(f"generated : {len(written)} files")
    for w in written:
        print(f"  {w}")
    if missing:
        print(f"\nmissing   : {len(missing)}")
        for m in missing:
            print(f"  {m}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
