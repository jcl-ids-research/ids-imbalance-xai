"""
Figures for the discrete-attribute failure and its correction (E9, E10).

  corr_hidden_failure   features that pass KS while losing their off-modal mass
  corr_before_after     recall / KS / MMD / correlation, before vs after
  corr_semantic         the security-semantic attributes, named, before vs after

Inputs are the JSON files already produced:
  discrete_collapse_seed{N}.json        the failure, measured
  correction_validation_seed{N}.json    the effect of the correction
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABEL = {"unsw": "UNSW-NB15", "nslkdd": "NSL-KDD",
         "cicids2017": "CIC-IDS-2017", "cicddos2019": "CIC-DDoS2019"}

# What these attributes mean, so the figure states the cost in security terms.
SEMANTICS = {
    "root_shell": "U2R privilege escalation",
    "num_failed_logins": "R2L brute force",
    "num_file_creations": "U2R payload activity",
    "land": "LAND attack",
    "wrong_fragment": "Teardrop",
    "is_guest_login": "guest access",
    "su_attempted": "privilege attempt",
    "SYN Flag Count": "SYN flood",
    "URG Flag Count": "urgent-flag anomaly",
    "CWE Flag Count": "congestion signalling",
    "RST Flag Count": "connection reset",
    "Bwd Packet Length Min": "backward min length",
}

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


def plot_hidden_failure(collapse, outdir, seed):
    """KS says fine, off-modal mass says otherwise."""
    rows = []
    for ds, r in collapse.items():
        if "error" in r:
            continue
        for h in r.get("hidden_failures", []):
            rows.append((ds, h))
    if not rows:
        return []

    rows.sort(key=lambda t: t[1]["off_default_recall"])
    names = [f"{h['feature']}\n({LABEL.get(ds, ds)})" for ds, h in rows]
    recall = [h["off_default_recall"] * 100 for _, h in rows]
    ksp = [h["ks_p"] for _, h in rows]
    x = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(9, len(rows) * 1.5), 8),
                                   sharex=True)

    b = ax1.bar(x, ksp, color="#27AE60", edgecolor="white")
    ax1.axhline(0.05, ls="--", c="k", lw=1, label="p = 0.05 (pass threshold)")
    ax1.set_ylabel("KS test p-value")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("What the KS test reports: every one of these passes",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    for r, v in zip(b, ksp):
        ax1.text(r.get_x() + r.get_width() / 2, v + 0.03, f"{v:.2f}",
                 ha="center", fontsize=8)

    b = ax2.bar(x, recall, color="#E74C3C", edgecolor="white")
    ax2.set_ylabel("off-modal mass recovered (%)")
    ax2.set_ylim(0, max(max(recall) * 1.35, 25))
    ax2.set_title("What actually happened: the non-default values are gone",
                  fontsize=11, fontweight="bold")
    for r, v in zip(b, recall):
        ax2.text(r.get_x() + r.get_width() / 2, v + max(recall) * 0.05 + 0.4,
                 f"{v:.1f}%", ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=8)

    fig.suptitle(f"Discrete attributes: a KS pass can hide a total loss of signal "
                 f"(seed {seed})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, f"corr_hidden_failure_seed{seed}")


def plot_before_after(valid, outdir, seed):
    order = [d for d in ("nslkdd", "unsw", "cicids2017", "cicddos2019")
             if d in valid and "before" in valid[d]]
    if not order:
        return []
    labels = [LABEL[d] for d in order]
    x = np.arange(len(order))
    w = 0.38

    metrics = [("off_default_recall", "off-modal mass recovered", True, 100),
               ("ks_pass_rate", "KS pass rate (%)", True, 100),
               ("mmd2", "MMD$^2$ (lower better)", False, 1),
               ("corr_within_0.10", "corr pairs within 0.10 (%)", True, 100)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title, higher, scale) in zip(axes.ravel(), metrics):
        b0 = [valid[d]["before"][key] * scale for d in order]
        b1 = [valid[d]["after"][key] * scale for d in order]
        ax.bar(x - w / 2, b0, w, label="before", color="#E74C3C", edgecolor="white")
        ax.bar(x + w / 2, b1, w, label="after", color="#2E86AB", edgecolor="white")
        top = max(max(b0), max(b1))
        for xi, (v0, v1) in enumerate(zip(b0, b1)):
            fmt = "{:.4f}" if scale == 1 else "{:.1f}"
            ax.text(xi - w / 2, v0 + top * 0.02, fmt.format(v0),
                    ha="center", fontsize=7)
            ax.text(xi + w / 2, v1 + top * 0.02, fmt.format(v1),
                    ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
        ax.set_ylabel(title)
        arrow = "higher is better" if higher else "lower is better"
        ax.set_title(f"{title}\n({arrow})", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, top * 1.22)

    fig.suptitle(f"Rank-matched marginal correction, before vs after (seed {seed})\n"
                 f"E9: quantitative fidelity   E10: semantic validity of "
                 f"numerical + categorical combination",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, f"corr_before_after_seed{seed}")


def plot_semantic(collapse, outdir, seed):
    """Named security attributes and how much of their signal survived."""
    rows = []
    for ds, r in collapse.items():
        if "error" in r:
            continue
        for f in r.get("all_features", []):
            if f["feature"] in SEMANTICS and f["discrete_like"] \
                    and f["real_off_default_rate"] > 0 \
                    and f["off_default_recall"] is not None:
                rows.append((ds, f))
    if not rows:
        return []
    rows.sort(key=lambda t: t[1]["off_default_recall"])
    rows = rows[:14]

    names = [f"{f['feature']}" for _, f in rows]
    meaning = [SEMANTICS.get(f["feature"], "") for _, f in rows]
    dss = [LABEL.get(ds, ds) for ds, _ in rows]
    recall = [f["off_default_recall"] * 100 for _, f in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(11, max(4.5, len(rows) * 0.48)))
    colors = ["#C0392B" if v < 25 else "#E67E22" if v < 60 else "#27AE60"
              for v in recall]
    ax.barh(y, recall, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\n{m} · {d}" for n, m, d in zip(names, meaning, dss)],
                       fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("off-modal mass recovered by the generator (%)")
    ax.set_xlim(0, max(max(recall) * 1.3, 25))
    for yi, v in enumerate(recall):
        ax.text(v + max(recall) * 0.02 + 0.3, yi, f"{v:.1f}%",
                va="center", fontsize=8)
    ax.set_title("Attack-defining attributes lost by continuous diffusion\n"
                 "(before correction)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, outdir, f"corr_semantic_seed{seed}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fidelity-dir", default="results/fidelity")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    fdir = Path(args.fidelity_dir)
    outdir = Path(args.out)
    written = []

    cpath = fdir / f"discrete_collapse_seed{args.seed}.json"
    vpath = fdir / f"correction_validation_seed{args.seed}.json"

    if cpath.exists():
        collapse = json.loads(cpath.read_text(encoding="utf-8"))
        for fn in (plot_hidden_failure, plot_semantic):
            files = fn(collapse, outdir, args.seed)
            if files:
                print(f"  {Path(files[0]).stem}")
                written += files
    else:
        print(f"missing {cpath}")

    if vpath.exists():
        valid = json.loads(vpath.read_text(encoding="utf-8"))
        files = plot_before_after(valid, outdir, args.seed)
        if files:
            print(f"  {Path(files[0]).stem}")
            written += files
    else:
        print(f"missing {vpath}")

    print(f"\n[OK] {len(written)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
