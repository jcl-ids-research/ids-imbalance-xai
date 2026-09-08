"""Redraw Figure 14 on the same three-seed basis as the manuscript text.

The existing figure shows seed 42, while the paragraph beside it reports the
mean across seeds 42, 123 and 456 (0.928, 0.963 and 0.982 at layer 4). Both are
real results, but they are not the same result. This plots the three-seed mean
with standard-deviation error bars, which is the quantity the text interprets.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path("/opt/ids_revision/deploy/results/attention")
OUTPUT = Path("/opt/ids_revision/deploy/results/figures/attention")
SEEDS = (42, 123, 456)
VIEWS = ("view1", "view2", "view3")
LAYERS = (1, 2, 3, 4)
COLOURS = ("#2E7EB8", "#E07A2D", "#4F9D69")
MARKERS = ("o", "s", "^")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
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


def load(seed: int) -> dict:
    """Read one full-model attention record."""
    path = ROOT / f"attention_unsw_seed{seed}_full_model.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    """Draw mean entropy over depth with the seed spread."""
    records = {seed: load(seed) for seed in SEEDS}
    OUTPUT.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.1, 3.8))
    figure.suptitle(
        "UNSW-NB15 attention concentration by depth (mean \u00b1 std over seeds 42, 123, 456)",
        fontsize=10.5,
        fontweight="bold",
        y=0.99,
    )

    for view, colour, marker in zip(VIEWS, COLOURS, MARKERS, strict=True):
        means: list[float] = []
        spreads: list[float] = []
        for layer in LAYERS:
            values = [records[seed]["entropy"][f"{view}/layer{layer}"] for seed in SEEDS]
            means.append(statistics.fmean(values))
            spreads.append(statistics.stdev(values))

        axis.errorbar(
            LAYERS,
            means,
            yerr=spreads,
            marker=marker,
            markersize=5,
            color=colour,
            linewidth=1.7,
            capsize=3,
            label=view,
        )
        offsets = {"view1": (5, -10), "view2": (5, 0), "view3": (5, 9)}
        axis.annotate(
            f"{means[-1]:.3f}",
            (LAYERS[-1], means[-1]),
            xytext=offsets[view],
            textcoords="offset points",
            fontsize=8,
            color=colour,
        )

    axis.set_xticks(LAYERS)
    axis.set_xlabel("Transformer layer")
    axis.set_ylabel("Normalised attention entropy")
    axis.set_ylim(0.68, 1.03)
    axis.set_axisbelow(True)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        fontsize=8.5,
        frameon=True,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))

    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 320})):
        target = OUTPUT / f"attn_evolution_unsw_mean_3seeds_v2.{suffix}"
        figure.savefig(target, format=suffix, **kwargs)
        print(f"written: {target}  ({target.stat().st_size:,} bytes)")

    print("\nlayer 4:")
    for view in VIEWS:
        values = [records[seed]["entropy"][f"{view}/layer4"] for seed in SEEDS]
        print(
            f"  {view}: mean={statistics.fmean(values):.6f}  "
            f"std={statistics.stdev(values):.6f}  seeds="
            + ", ".join(f"{value:.6f}" for value in values)
        )


if __name__ == "__main__":
    main()
