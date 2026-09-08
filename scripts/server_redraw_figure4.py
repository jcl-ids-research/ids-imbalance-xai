"""Redraw Figure 4 from the authoritative ablation and baseline records.

The old baseline plot contains the right values, but its plotting script assumes
all phase1_corrected files are flat. UNSW is nested, so the source chain cannot
be reproduced on the server. This version resolves both layouts explicitly and
draws the same comparison from current run records only.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

out = Path("/opt/ids_revision/deploy/results/figures")
results = Path("/opt/ids_revision/deploy/results")
seeds = (42, 123, 456)

datasets = (
    ("unsw", "UNSW-NB15"),
    ("nslkdd", "NSL-KDD"),
    ("cicids2017", "CIC-IDS-2017"),
    ("cicddos2019", "CIC-DDoS2019"),
)
baselines = ("RandomForest", "XGBoost", "MLP", "SVM_RBF")
ours = (
    ("full_model", "Ours (full)"),
    ("wo_diffusion", "Ours (w/o Df)"),
    ("wo_multiview", "Ours (w/o MV)"),
    ("baseline", "Ours (base)"),
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def ablation_path(dataset: str, seed: int) -> Path:
    """Resolve the two phase1_corrected layouts."""
    flat = results / "phase1_corrected" / f"{dataset}_seed{seed}.json"
    nested = results / "phase1_corrected" / dataset / f"seed{seed}" / "metrics.json"
    if flat.is_file():
        return flat
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"missing {dataset} seed {seed}")


def aggregate(paths: list[Path], variant: str) -> tuple[float, float]:
    """Mean and standard deviation of macro F1 as percentages."""
    values = [json.loads(path.read_text())["variants"][variant]["f1_macro"] * 100 for path in paths]
    return statistics.fmean(values), statistics.stdev(values)


def entries(dataset: str) -> list[tuple[str, float, float, bool]]:
    """All current baselines and our four variants."""
    rows: list[tuple[str, float, float, bool]] = []
    base_paths = [
        results / "baseline_fair" / dataset / f"seed{seed}" / "metrics.json" for seed in seeds
    ]
    for model in baselines:
        mean, spread = aggregate(base_paths, f"{model}__balanced")
        rows.append((model, mean, spread, False))

    our_paths = [ablation_path(dataset, seed) for seed in seeds]
    for key, label in ours:
        mean, spread = aggregate(our_paths, key)
        rows.append((label, mean, spread, True))
    return sorted(rows, key=lambda row: row[1])


def main() -> None:
    """Draw four horizontal comparisons and write vector/raster twins."""
    out.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 6.2))
    figure.suptitle(
        "Classical baselines vs proposed model under identical conditions\n"
        "macro F1, mean ± std over seeds 42, 123, 456; blue = ours, orange = best",
        fontsize=10.5,
        fontweight="bold",
        y=0.995,
    )

    print("values drawn:")
    for axis, (dataset, title) in zip(axes.flat, datasets, strict=True):
        data = entries(dataset)
        names = [row[0] for row in data]
        values = [row[1] for row in data]
        spreads = [row[2] for row in data]
        colours = ["#2E7EB8" if row[3] else "#9AA8AA" for row in data]
        colours[-1] = "#E67E22"

        positions = np.arange(len(data))
        axis.barh(
            positions,
            values,
            xerr=spreads,
            color=colours,
            edgecolor="white",
            error_kw={"linewidth": 0.9, "capsize": 2.5, "ecolor": "#34495E"},
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(names)
        span = max(values) - min(values)
        left = min(values) - max(spreads) - span * 0.28 - 0.2
        right = max(values) + max(spreads) + span * 0.44 + 0.5
        axis.set_xlim(left, right)
        pad = (right - left) * 0.012
        for y, (value, spread) in enumerate(zip(values, spreads, strict=True)):
            axis.text(
                value + spread + pad,
                y,
                f"{value:.2f}±{spread:.2f}",
                va="center",
                ha="left",
                fontsize=7.5,
            )
        axis.set_xlabel("macro F1 (%)")
        axis.set_title(title)
        axis.set_axisbelow(True)

        print("  " + title)
        for name, value, spread, _ in reversed(data):
            print(f"    {name:16s} {value:.2f} +/- {spread:.2f}")

    figure.tight_layout(rect=(0, 0, 1, 0.92))
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 320})):
        target = out / f"base_comparison_v2.{suffix}"
        figure.savefig(target, format=suffix, **kwargs)
        print(f"written: {target}  ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
