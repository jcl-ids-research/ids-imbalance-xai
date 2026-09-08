"""
Ablation figures across datasets, variants and seeds.

Reads every finished run and produces:
  abl_f1_bars        per-dataset variant comparison, mean +/- std error bars
  abl_contrib        component contribution per dataset, one bar per seed
  abl_seed_spread    per-seed values, showing whether a gap exceeds seed noise
  abl_heatmap        dataset x variant F1 matrix

Numbers come only from real result files; datasets or seeds without a file are
skipped and listed, never filled with placeholders.
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
SEED_MARKER = {42: "o", 123: "s", 456: "^"}

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


def load_all(root: Path, seeds: list[int]) -> tuple[dict, list[str]]:
    """(dataset, seed, variant) -> f1 percent."""
    data, missing = {}, []
    for ds in DATASETS:
        for seed in seeds:
            p = (root / ds / f"seed{seed}" / "metrics.json" if ds == "unsw"
                 else root / f"{ds}_seed{seed}.json")
            if not p.exists():
                missing.append(f"{ds} seed{seed}")
                continue
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
            for v, m in j.get("variants", {}).items():
                if "f1" in m:
                    data[(ds, seed, v)] = m["f1"] * 100
    return data, missing


def agg(data, ds, v, seeds):
    vals = [data[(ds, s, v)] for s in seeds if (ds, s, v) in data]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], None, 1
    return st.mean(vals), st.stdev(vals), len(vals)


def plot_f1_bars(data, seeds, outdir):
    present = [ds for ds in DATASETS if any((ds, s, v) in data
                                            for s in seeds for v in VARIANTS)]
    if not present:
        return []
    fig, axes = plt.subplots(1, len(present), figsize=(4.6 * len(present), 4.8),
                             squeeze=False)
    axes = axes[0]
    for ax, ds in zip(axes, present):
        means, stds, labels, colors, ns = [], [], [], [], []
        for v in VARIANTS:
            m, s, n = agg(data, ds, v, seeds)
            if m is None:
                continue
            means.append(m)
            stds.append(s if s is not None else 0.0)
            labels.append(VLABEL[v])
            colors.append(VCOLOR[v])
            ns.append(n)
        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                      edgecolor="white", error_kw={"lw": 1.2})
        best = int(np.argmax(means))
        bars[best].set_edgecolor("black")
        bars[best].set_linewidth(2.0)
        for xi, (m, s, n) in enumerate(zip(means, stds, ns)):
            txt = f"{m:.2f}" + (f"\n±{s:.2f}" if n > 1 else "")
            ax.text(xi, m + s + (max(means) - min(means) + 1) * 0.04, txt,
                    ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("F1 weighted (%)")
        lo, hi = min(means) - max(stds) - 1.5, max(means) + max(stds) + 2.0
        ax.set_ylim(lo, hi)
        ax.set_title(f"{LABEL[ds]}\n(best outlined)", fontsize=10, fontweight="bold")
    fig.suptitle(f"Ablation - F1 weighted, mean ± std over seeds {seeds}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, "abl_f1_bars")


def plot_contrib(data, seeds, outdir):
    effects = [("Diffusion (with MV)", "full_model", "wo_diffusion", "#2E86AB"),
               ("Diffusion (w/o MV)", "wo_multiview", "baseline", "#5DADE2"),
               ("Multi-view (with Df)", "full_model", "wo_multiview", "#E74C3C"),
               ("Multi-view (w/o Df)", "wo_diffusion", "baseline", "#F1948A")]
    present = [ds for ds in DATASETS if any((ds, s, "full_model") in data for s in seeds)]
    if not present:
        return []
    fig, axes = plt.subplots(1, len(present), figsize=(4.6 * len(present), 4.8),
                             squeeze=False)
    axes = axes[0]
    for ax, ds in zip(axes, present):
        x = np.arange(len(effects))
        width = 0.8 / max(len(seeds), 1)
        for si, seed in enumerate(seeds):
            vals, xs = [], []
            for ei, (_, a, b, _) in enumerate(effects):
                ka, kb = (ds, seed, a), (ds, seed, b)
                if ka in data and kb in data:
                    vals.append(data[ka] - data[kb])
                    xs.append(ei + (si - (len(seeds) - 1) / 2) * width)
            if vals:
                ax.bar(xs, vals, width=width * 0.9,
                       label=f"seed {seed}", edgecolor="white")
        ax.axhline(0, c="k", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([e[0] for e in effects], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("F1 difference (pp)")
        ax.set_title(LABEL[ds], fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
    fig.suptitle("Component contribution per seed  (positive = component helps)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, "abl_contrib")


def plot_seed_spread(data, seeds, outdir):
    present = [ds for ds in DATASETS if any((ds, s, v) in data
                                            for s in seeds for v in VARIANTS)]
    if not present:
        return []
    fig, axes = plt.subplots(1, len(present), figsize=(4.4 * len(present), 4.6),
                             squeeze=False)
    axes = axes[0]
    for ax, ds in zip(axes, present):
        for vi, v in enumerate(VARIANTS):
            for seed in seeds:
                k = (ds, seed, v)
                if k not in data:
                    continue
                ax.scatter(vi, data[k], marker=SEED_MARKER.get(seed, "o"),
                           s=80, color=VCOLOR[v], zorder=3,
                           edgecolors="white", linewidth=1)
            m, s, n = agg(data, ds, v, seeds)
            if m is not None:
                ax.hlines(m, vi - 0.28, vi + 0.28, colors="k", lw=1.6, zorder=4)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels([VLABEL[v] for v in VARIANTS], rotation=25,
                           ha="right", fontsize=8)
        ax.set_ylabel("F1 weighted (%)")
        ax.set_title(LABEL[ds], fontsize=10, fontweight="bold")
    handles = [plt.Line2D([], [], marker=SEED_MARKER.get(s, "o"), ls="",
                          color="grey", label=f"seed {s}") for s in seeds]
    handles.append(plt.Line2D([], [], color="k", lw=1.6, label="mean"))
    axes[0].legend(handles=handles, fontsize=7, loc="best")
    fig.suptitle("Per-seed spread  (is a gap larger than seed noise?)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, "abl_seed_spread")


def plot_heatmap(data, seeds, outdir):
    present = [ds for ds in DATASETS if any((ds, s, v) in data
                                            for s in seeds for v in VARIANTS)]
    if not present:
        return []
    M = np.full((len(present), len(VARIANTS)), np.nan)
    for i, ds in enumerate(present):
        for j, v in enumerate(VARIANTS):
            m, _, _ = agg(data, ds, v, seeds)
            if m is not None:
                M[i, j] = m
    fig, ax = plt.subplots(figsize=(7.5, 1.1 * len(present) + 2.2))
    im = ax.imshow(M, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([VLABEL[v] for v in VARIANTS], rotation=25, ha="right")
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([LABEL[d] for d in present])
    ax.grid(False)
    for i in range(M.shape[0]):
        row = M[i]
        best = int(np.nanargmax(row)) if not np.all(np.isnan(row)) else -1
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, f"{M[i, j]:.2f}" + (" ★" if j == best else ""),
                    ha="center", va="center", fontsize=9,
                    fontweight="bold" if j == best else "normal")
    plt.colorbar(im, ax=ax, shrink=0.8, label="F1 weighted (%)")
    ax.set_title("Mean F1 by dataset and variant  (★ = best per dataset)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, "abl_heatmap")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/phase1")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--seeds", default="42,123,456")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    data, missing = load_all(Path(args.root), seeds)
    if not data:
        print("no result files found")
        return 1

    have = sorted({(ds, s) for ds, s, _ in data})
    print(f"loaded {len(have)} runs: " + ", ".join(f"{d}/s{s}" for d, s in have))
    if missing:
        print(f"missing ({len(missing)}): " + ", ".join(missing))
    print()

    outdir = Path(args.out)
    written = []
    for fn in (plot_f1_bars, plot_contrib, plot_seed_spread, plot_heatmap):
        files = fn(data, seeds, outdir)
        if files:
            print(f"  {Path(files[0]).stem}")
            written += files

    print(f"\n[OK] {len(written)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
