"""Redraw Figure 7 from the authoritative per-seed ablation records.

The original component-contribution plot was produced by the same stale
phase1 reader that produced the defective Figure 6. This reads
phase1_corrected, including UNSW's nested layout, and subtracts the paired
variants seed by seed. Values are printed above the bars so the figure can be
audited without estimating their height.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

CORRECTED = Path("/opt/ids_revision/deploy/results/phase1_corrected")
OUTPUT = Path("/opt/ids_revision/deploy/results/figures")
SEEDS = (42, 123, 456)

DATASETS = (
    ("unsw", "UNSW-NB15"),
    ("nslkdd", "NSL-KDD"),
    ("cicids2017", "CIC-IDS-2017"),
    ("cicddos2019", "CIC-DDoS2019"),
)
EFFECTS = (
    ("Diffusion\n(with MV)", "full_model", "wo_diffusion"),
    ("Diffusion\n(w/o MV)", "wo_multiview", "baseline"),
    ("Multi-view\n(with Df)", "full_model", "wo_multiview"),
    ("Multi-view\n(w/o Df)", "wo_diffusion", "baseline"),
)
COLOURS = ("#2E7EB8", "#E8A33D", "#4F9D69")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def path_for(dataset: str, seed: int) -> Path:
    """Resolve the flat or nested storage layout."""
    flat = CORRECTED / f"{dataset}_seed{seed}.json"
    nested = CORRECTED / dataset / f"seed{seed}" / "metrics.json"
    if flat.is_file():
        return flat
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"no authoritative result for {dataset} seed {seed}")


def values(dataset: str) -> list[list[float]]:
    """Four component effects, each containing the three seed values."""
    by_seed = {seed: json.loads(path_for(dataset, seed).read_text())["variants"] for seed in SEEDS}
    return [
        [
            (
                by_seed[seed][with_component]["f1_macro"]
                - by_seed[seed][without_component]["f1_macro"]
            )
            * 100
            for seed in SEEDS
        ]
        for _, with_component, without_component in EFFECTS
    ]


def main() -> None:
    """Draw all four panels and save vector/raster twins."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 6.5))
    figure.suptitle(
        "Component contribution per seed (positive = component helps)",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )

    for axis, (dataset, title) in zip(axes.flat, DATASETS, strict=True):
        effects = values(dataset)
        positions = np.arange(len(EFFECTS))
        width = 0.24
        span = max(abs(value) for row in effects for value in row)

        for seed_index, seed in enumerate(SEEDS):
            xs = positions + (seed_index - 1) * width
            ys = [row[seed_index] for row in effects]
            bars = axis.bar(
                xs,
                ys,
                width=width * 0.92,
                color=COLOURS[seed_index],
                edgecolor="white",
                linewidth=0.5,
                label=f"seed {seed}",
            )
            for bar, value in zip(bars, ys, strict=True):
                offset = span * 0.035
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (offset if value >= 0 else -offset),
                    f"{value:+.2f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=6.8,
                    rotation=90,
                )

        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(positions)
        axis.set_xticklabels([entry[0] for entry in EFFECTS])
        axis.set_ylabel("macro F1 difference (pp)")
        axis.set_title(title, pad=5)
        axis.set_axisbelow(True)
        axis.set_ylim(-span * 1.35, span * 1.35)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        frameon=True,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))

    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 320})):
        target = OUTPUT / f"abl_contrib_macro_v2.{suffix}"
        figure.savefig(target, format=suffix, **kwargs)
        print(f"written: {target}  ({target.stat().st_size:,} bytes)")

    print("\nvalues drawn:")
    for dataset, title in DATASETS:
        print(f"  {title}")
        for (label, _, _), row in zip(EFFECTS, values(dataset), strict=True):
            cells = "  ".join(f"{value:+.2f}" for value in row)
            print(f"    {label.replace(chr(10), ' '):26s} {cells}")


if __name__ == "__main__":
    main()
