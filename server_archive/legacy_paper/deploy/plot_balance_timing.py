"""
Balance-procedure and cost figures.

  bal_distribution   class counts before vs after, per dataset  -> E8
  bal_ratio          imbalance ratio before vs after, cap marked -> E8
  bal_expansion      expansion factor per class, 15x cap line    -> E8
  time_breakdown     DDPM and per-variant training cost          -> scalability

E8 is the editor's request to clarify the balancing procedure, "since the
resulting distribution does not appear balanced". A table alone does not make
the undersample/augment/cap interaction legible; these figures do, including
the one dataset that legitimately stops short of parity.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["unsw", "nslkdd", "cicids2017", "cicddos2019"]
LABEL = {"unsw": "UNSW-NB15", "nslkdd": "NSL-KDD",
         "cicids2017": "CIC-IDS-2017", "cicddos2019": "CIC-DDoS2019"}
VARIANTS = ["full_model", "wo_diffusion", "wo_multiview", "baseline"]
VLABEL = {"full_model": "Full Model", "wo_diffusion": "w/o Diffusion",
          "wo_multiview": "w/o Multi-view", "baseline": "Baseline"}
VCOLOR = {"full_model": "#2E86AB", "wo_diffusion": "#F39C12",
          "wo_multiview": "#E74C3C", "baseline": "#7F8C8D"}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def save(fig, outdir: Path, stem: str) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext, dpi in (("pdf", None), ("png", 180)):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, format=ext, dpi=dpi, bbox_inches="tight")
        out.append(str(p))
    plt.close(fig)
    return out


def load(root: Path, seeds: list[int]) -> dict:
    out = {}
    for ds in DATASETS:
        for seed in seeds:
            p = (root / ds / f"seed{seed}" / "metrics.json" if ds == "unsw"
                 else root / f"{ds}_seed{seed}.json")
            if p.exists():
                out[(ds, seed)] = json.loads(p.read_text(encoding="utf-8"))
    return out


def plot_distribution(data, seeds, outdir):
    ref = {}
    for ds in DATASETS:
        for s in seeds:
            if (ds, s) in data and data[(ds, s)].get("balance_report"):
                ref[ds] = (s, data[(ds, s)]["balance_report"])
                break
    if not ref:
        return []

    n = len(ref)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.8), squeeze=False)
    axes = axes[0]
    for ax, (ds, (seed, br)) in zip(axes, ref.items()):
        classes = sorted(br["before"], key=lambda c: int(c))
        before = [br["before"][c] for c in classes]
        after = [br["after"][c] for c in classes]
        x = np.arange(len(classes))
        w = 0.38
        ax.bar(x - w / 2, before, w, label="before", color="#95A5A6",
               edgecolor="white")
        bars = ax.bar(x + w / 2, after, w, label="after", color="#2E86AB",
                      edgecolor="white")
        target = br.get("mean_target")
        if target:
            ax.axhline(target, ls="--", c="#E74C3C", lw=1.5,
                       label=f"mean target = {target:,}")
        for xi, c in enumerate(classes):
            act = br.get("actions", {}).get(c, "")
            if "CAP HIT" in act:
                bars[xi].set_edgecolor("#C0392B")
                bars[xi].set_linewidth(2.5)
                ax.text(xi + w / 2, after[xi] * 1.03, "CAP", ha="center",
                        fontsize=8, color="#C0392B", fontweight="bold")
        for xi, (b, a) in enumerate(zip(before, after)):
            ax.text(xi - w / 2, b, f"{b:,}", ha="center", va="bottom", fontsize=7)
            ax.text(xi + w / 2, a, f"{a:,}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"class {c}" for c in classes], fontsize=8)
        ax.set_ylabel("training records")
        ax.set_title(f"{LABEL[ds]}  (seed {seed})", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_ylim(0, max(max(before), max(after)) * 1.20)
    fig.suptitle("Balancing: majority undersampled, minority augmented toward the "
                 "mean target\n(red outline = 15x expansion cap reached)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, "bal_distribution")


def plot_ratio(data, seeds, outdir):
    rows = []
    for ds in DATASETS:
        b, a, cap = [], [], False
        for s in seeds:
            br = data.get((ds, s), {}).get("balance_report")
            if not br:
                continue
            b.append(br["imbalance_ratio_before"])
            a.append(br["imbalance_ratio_after"])
            cap = cap or any("CAP HIT" in v for v in br.get("actions", {}).values())
        if b:
            rows.append((ds, st.mean(b), st.mean(a), cap))
    if not rows:
        return []

    labels = [LABEL[d] for d, _, _, _ in rows]
    before = [b for _, b, _, _ in rows]
    after = [a for _, _, a, _ in rows]
    caps = [c for _, _, _, c in rows]
    x = np.arange(len(rows))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - w / 2, before, w, label="before", color="#E74C3C", edgecolor="white")
    bars = ax.bar(x + w / 2, after, w, label="after", color="#2E86AB",
                  edgecolor="white")
    ax.axhline(1.0, ls="--", c="k", lw=1.2, label="perfect balance = 1.00")
    for xi, (b, a, c) in enumerate(zip(before, after, caps)):
        ax.text(xi - w / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold" if c else "normal",
                color="#C0392B" if c else "black")
        if c:
            bars[xi].set_edgecolor("#C0392B")
            bars[xi].set_linewidth(2.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("imbalance ratio (majority / minority)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Imbalance before and after balancing\n"
                 "CIC-DDoS2019 stops at 1.07: only 4,467 real benign records, "
                 "and the 15x cap binds",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, "bal_ratio")


def plot_expansion(data, seeds, outdir):
    rows = []
    for ds in DATASETS:
        for s in seeds:
            br = data.get((ds, s), {}).get("balance_report")
            if not br:
                continue
            for c, r in br.get("expansion_ratio", {}).items():
                act = br.get("actions", {}).get(c, "")
                rows.append((LABEL[ds], c, float(r), "CAP HIT" in act,
                             act.startswith("undersampled")))
            break
    if not rows:
        return []

    names = [f"{d}\nclass {c}" for d, c, _, _, _ in rows]
    vals = [v for _, _, v, _, _ in rows]
    colors = ["#C0392B" if cap else ("#95A5A6" if under else "#2E86AB")
              for _, _, _, cap, under in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 1.15), 5.2))
    ax.bar(x, vals, color=colors, edgecolor="white")
    ax.axhline(1.0, ls="--", c="k", lw=1.2, label="no change")
    ax.axhline(15.0, ls="--", c="#C0392B", lw=1.5, label="expansion cap = 15x")
    for xi, v in enumerate(vals):
        ax.text(xi, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("expansion factor (after / before)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Per-class expansion  "
                 "(grey = undersampled, blue = augmented, red = cap reached)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, "bal_expansion")


def plot_timing(data, seeds, outdir):
    present = []
    for ds in DATASETS:
        t = {}
        for s in seeds:
            ts = data.get((ds, s), {}).get("timings_sec")
            if not ts:
                continue
            for k, v in ts.items():
                t.setdefault(k, []).append(v / 60.0)
        if t:
            present.append((ds, t))
    if not present:
        return []

    keys = ["ddpm"] + VARIANTS
    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(len(present))
    w = 0.8 / len(keys)
    for ki, k in enumerate(keys):
        means = [st.mean(t[k]) if k in t else 0.0 for _, t in present]
        errs = [st.stdev(t[k]) if k in t and len(t[k]) > 1 else 0.0
                for _, t in present]
        off = (ki - (len(keys) - 1) / 2) * w
        color = "#8E44AD" if k == "ddpm" else VCOLOR.get(k, "#7F8C8D")
        ax.bar(x + off, means, w * 0.9, yerr=errs, capsize=3,
               label="DDPM" if k == "ddpm" else VLABEL.get(k, k),
               color=color, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[d] for d, _ in present], rotation=15, ha="right")
    ax.set_ylabel("wall-clock minutes")
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("Training cost by stage, mean ± std over seeds\n"
                 "(four datasets ran concurrently, so absolute values include "
                 "contention)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, "time_breakdown")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/phase1")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--seeds", default="42,123,456")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    data = load(Path(args.root), seeds)
    if not data:
        print("no result files found")
        return 1
    print(f"loaded {len(data)} runs")

    outdir = Path(args.out)
    written = []
    for fn in (plot_distribution, plot_ratio, plot_expansion, plot_timing):
        files = fn(data, seeds, outdir)
        if files:
            print(f"  {Path(files[0]).stem}")
            written += files
        else:
            print(f"  {fn.__name__}: skipped (no data)")
    print(f"\n[OK] {len(written)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
