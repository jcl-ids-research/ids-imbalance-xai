"""
Figures for the synthetic-sample fidelity evaluation (R1-3, R2-3).

Generates, per dataset:
  fid_dist_{ds}          real vs synthetic marginals, worst + best features
  fid_corr_{ds}          correlation heatmaps: real | synthetic | |difference|
  fid_ks_{ds}            per-feature KS statistic, sorted

And across datasets:
  fid_summary            KS pass rate / MMD / correlation vs available real samples

All values come from samples.npz and fidelity_seed{N}.json. PDF (vector, for
the paper) + PNG (for quick viewing).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

DATASET_LABEL = {
    "unsw": "UNSW-NB15",
    "nslkdd": "NSL-KDD",
    "cicids2017": "CIC-IDS-2017",
    "cicddos2019": "CIC-DDoS2019",
}
C_REAL = "#2E86AB"
C_SYN = "#E74C3C"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


def save(fig, outdir: Path, stem: str) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext, dpi in (("pdf", None), ("png", 180)):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, format=ext, dpi=dpi, bbox_inches="tight")
        out.append(str(p))
    plt.close(fig)
    return out


def load_samples(root: Path, ds: str, seed: int):
    p = root / ds / f"seed{seed}" / "samples.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {
        "real": np.nan_to_num(d["real"].astype(np.float64), posinf=0, neginf=0),
        "syn": np.nan_to_num(d["synthetic"].astype(np.float64), posinf=0, neginf=0),
        "real_y": d["real_labels"],
        "syn_y": d["synthetic_labels"],
        "names": [str(x) for x in d["feature_names"]],
    }


# ------------------------------------------------------------ distributions


def plot_distributions(s, fid, ds, outdir, seed, n_show=4):
    """Worst and best features by KS, real vs synthetic, matched on class."""
    ks = np.array(fid["ks"]["per_feature_ks"])
    names = fid["feature_names"]
    order = np.argsort(-ks)
    worst = order[:n_show]
    best = order[-n_show:][::-1]

    # Fair comparison: restrict real to the classes present in synthetic.
    syn_classes = np.unique(s["syn_y"])
    rmask = np.isin(s["real_y"], syn_classes)
    real = s["real"][rmask] if rmask.sum() > 50 else s["real"]
    syn = s["syn"]

    fig, axes = plt.subplots(2, n_show, figsize=(4.0 * n_show, 6.4))
    for row, (idxs, tag) in enumerate([(worst, "Worst"), (best, "Best")]):
        for col, fi in enumerate(idxs):
            ax = axes[row, col]
            r = real[:, fi]
            y = syn[:, fi]
            lo = np.percentile(np.concatenate([r, y]), 0.5)
            hi = np.percentile(np.concatenate([r, y]), 99.5)
            if hi <= lo:
                lo, hi = float(min(r.min(), y.min())), float(max(r.max(), y.max()) + 1e-9)
            bins = np.linspace(lo, hi, 45)
            ax.hist(r, bins=bins, alpha=0.55, color=C_REAL, density=True, label="Real")
            ax.hist(y, bins=bins, alpha=0.55, color=C_SYN, density=True, label="Synthetic")
            ax.set_title(f"{names[fi]}\nKS={ks[fi]:.3f}", fontsize=9,
                         fontweight="bold" if row == 0 else "normal")
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(f"{tag} {n_show}\ndensity", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle(f"{DATASET_LABEL.get(ds, ds)} - Real vs Synthetic Marginals "
                 f"(seed {seed}, class-matched)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, f"fid_dist_{ds}_seed{seed}")


# ------------------------------------------------------------ correlation


def plot_correlation(s, ds, outdir, seed):
    syn_classes = np.unique(s["syn_y"])
    rmask = np.isin(s["real_y"], syn_classes)
    real = s["real"][rmask] if rmask.sum() > 50 else s["real"]
    syn = s["syn"]

    with np.errstate(invalid="ignore", divide="ignore"):
        cr = np.nan_to_num(np.corrcoef(real, rowvar=False))
        cs = np.nan_to_num(np.corrcoef(syn, rowvar=False))
    diff = np.abs(cr - cs)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (m, t, cmap, vmin, vmax) in zip(axes, [
        (cr, "Real", "RdBu_r", -1, 1),
        (cs, "Synthetic", "RdBu_r", -1, 1),
        (diff, "|Difference|", "Reds", 0, 1),
    ]):
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(t, fontsize=11, fontweight="bold")
        ax.set_xlabel("feature index")
        ax.set_ylabel("feature index")
        ax.grid(False)
        plt.colorbar(im, ax=ax, shrink=0.85)

    iu = np.triu_indices_from(cr, k=1)
    d = diff[iu]
    fig.suptitle(f"{DATASET_LABEL.get(ds, ds)} - Correlation Structure "
                 f"(seed {seed})   mean|d|={d.mean():.3f}   "
                 f"within 0.10: {(d <= 0.10).mean() * 100:.1f}%",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, f"fid_corr_{ds}_seed{seed}")


# ------------------------------------------------------------ ks bar


def plot_ks_bars(fid, ds, outdir, seed):
    ks = np.array(fid["ks"]["per_feature_ks"])
    pv = np.array(fid["ks"]["per_feature_p"])
    names = fid["feature_names"]
    order = np.argsort(-ks)

    fig, ax = plt.subplots(figsize=(max(8, len(ks) * 0.22), 5))
    colors = [C_SYN if pv[i] <= 0.05 else "#27AE60" for i in order]
    ax.bar(range(len(ks)), ks[order], color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(0.1, ls="--", c="k", lw=1, alpha=0.6, label="KS = 0.10")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=6)
    ax.set_ylabel("KS statistic")
    npass = int((pv > 0.05).sum())
    ax.set_title(f"{DATASET_LABEL.get(ds, ds)} - Per-feature KS "
                 f"(green = passes p>0.05: {npass}/{len(ks)}) seed {seed}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save(fig, outdir, f"fid_ks_{ds}_seed{seed}")


# ------------------------------------------------------------ summary


def plot_summary(data, outdir, seed):
    rows = []
    for ds, r in data.items():
        if "error" in r:
            continue
        pc = r.get("per_class", {})
        if not pc:
            continue
        c = next(iter(pc))
        m = pc[c]
        rows.append({
            "ds": DATASET_LABEL.get(ds, ds),
            "n_real": m["n_real"],
            "ks_pass": m["ks_pass_rate"] * 100,
            "mmd": m["mmd_rbf"]["mmd2_mean"],
            "corr": m["correlation"]["within_0.10"] * 100,
        })
    if not rows:
        return None, []

    rows.sort(key=lambda x: -x["n_real"])
    labels = [r["ds"] for r in rows]
    nreal = [r["n_real"] for r in rows]
    kspass = [r["ks_pass"] for r in rows]
    mmd = [r["mmd"] for r in rows]
    corr = [r["corr"] for r in rows]
    x = np.arange(len(rows))

    fig = plt.figure(figsize=(15, 8.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    b = ax.bar(x, kspass, color="#2E86AB", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("KS pass rate (%)"); ax.set_ylim(0, 100)
    ax.set_title("Marginal match\n(higher = better)", fontsize=10, fontweight="bold")
    for r, v in zip(b, kspass):
        ax.text(r.get_x() + r.get_width() / 2, v + 2, f"{v:.1f}", ha="center", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    b = ax.bar(x, mmd, color="#E74C3C", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("MMD$^2$ (RBF)")
    ax.set_title("Joint distribution gap\n(lower = better)", fontsize=10, fontweight="bold")
    for r, v in zip(b, mmd):
        ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.4f}", ha="center",
                va="bottom", fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    b = ax.bar(x, corr, color="#F39C12", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("corr pairs within 0.10 (%)"); ax.set_ylim(0, 100)
    ax.set_title("Joint structure preserved\n(higher = better)",
                 fontsize=10, fontweight="bold")
    for r, v in zip(b, corr):
        ax.text(r.get_x() + r.get_width() / 2, v + 2, f"{v:.1f}", ha="center", fontsize=8)

    # The headline relationship: fewer real minority samples -> worse fidelity
    for col, (vals, ylab, color, invert) in enumerate([
        (kspass, "KS pass rate (%)", "#2E86AB", False),
        (mmd, "MMD$^2$", "#E74C3C", True),
        (corr, "corr within 0.10 (%)", "#F39C12", False),
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax.scatter(nreal, vals, s=110, c=color, zorder=3, edgecolors="white")
        for xi, yi, li in zip(nreal, vals, labels):
            ax.annotate(li, (xi, yi), fontsize=7,
                        xytext=(5, 5), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("real minority samples available (log)")
        ax.set_ylabel(ylab)
        ax.set_title(("worse" if invert else "better") + " with more real data",
                     fontsize=9)

    fig.suptitle(f"Synthetic Sample Fidelity - class-matched, seed {seed}\n"
                 f"(addresses R1-3: MMD / KL / distribution tests; "
                 f"R2-3: joint feature distribution)",
                 fontsize=13, fontweight="bold")
    return "fid_summary", save(fig, outdir, f"fid_summary_seed{seed}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/phase1")
    ap.add_argument("--fidelity", default="results/fidelity/fidelity_seed123.json")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.out)
    with open(args.fidelity, encoding="utf-8") as f:
        data = json.load(f)

    written = []
    for ds, fid in data.items():
        if "error" in fid:
            print(f"[{ds}] skipped: {fid['error']}")
            continue
        s = load_samples(root, ds, args.seed)
        if s is None:
            print(f"[{ds}] samples.npz missing")
            continue
        print(f"[{ds}] plotting ...", flush=True)
        written += plot_distributions(s, fid, ds, outdir, args.seed)
        written += plot_correlation(s, ds, outdir, args.seed)
        written += plot_ks_bars(fid, ds, outdir, args.seed)
        print(f"    dist / corr / ks  done")

    stem, files = plot_summary(data, outdir, args.seed)
    if stem:
        written += files
        print(f"[summary] done")

    print(f"\n[OK] {len(written)} files")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
