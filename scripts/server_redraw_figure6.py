"""Redraw the ablation figure from the authoritative run records.

Figure 6 in the manuscript was plotted from a superseded round: it shows 88.36
for the UNSW full model where Table 3 and the phase1_corrected records both give
88.90. Every bar is redrawn here from those records, so the figure and the table
come from one source.

Editor item E13 governs how it must look: vector output, no overlapping labels,
drawn at the width it will be printed. The figure is therefore sized in inches
to the text column, the value labels are placed above the error bars rather than
inside the bars, and both PDF and PNG are written from the same call so the
raster twin cannot drift from the vector.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

CORRECTED = Path("/opt/ids_revision/deploy/results/phase1_corrected")
OUTPUT = Path("/opt/ids_revision/deploy/results/figures")
SEEDS = (42, 123, 456)

VARIANTS = (
    ("full_model", "Full Model"),
    ("wo_diffusion", "w/o Diffusion"),
    ("wo_multiview", "w/o Multi-view"),
    ("baseline", "Baseline"),
)
DATASETS = (
    ("unsw", "UNSW-NB15"),
    ("nslkdd", "NSL-KDD"),
    ("cicids2017", "CIC-IDS-2017"),
    ("cicddos2019", "CIC-DDoS2019"),
)
COLOURS = ("#2E7EB8", "#E8A33D", "#D65F5F", "#7F7F7F")

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


def result_path(dataset: str, seed: int) -> Path | None:
    """Return whichever of the two storage layouts holds this run."""
    flat = CORRECTED / f"{dataset}_seed{seed}.json"
    nested = CORRECTED / dataset / f"seed{seed}" / "metrics.json"
    if flat.is_file():
        return flat
    if nested.is_file():
        return nested
    return None


def statistics_for(dataset: str) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation of macro F1 per variant, as percentages."""
    summary: dict[str, tuple[float, float]] = {}
    for key, _ in VARIANTS:
        values: list[float] = []
        for seed in SEEDS:
            path = result_path(dataset, seed)
            if path is None:
                continue
            node = json.loads(path.read_text()).get("variants", {}).get(key)
            if node and "f1_macro" in node:
                values.append(float(node["f1_macro"]) * 100)
        if values:
            spread = statistics.stdev(values) if len(values) > 1 else 0.0
            summary[key] = (statistics.fmean(values), spread)
    return summary


def main() -> None:
    """Draw the four panels and write the vector and raster outputs."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 6.2))
    figure.suptitle(
        "Ablation \u2014 macro F1, mean \u00b1 std over seeds 42, 123, 456",
        fontsize=11,
        fontweight="bold",
        y=0.985,
    )

    for axis, (dataset, title) in zip(axes.flat, DATASETS, strict=True):
        summary = statistics_for(dataset)
        if not summary:
            axis.set_visible(False)
            continue

        labels = [label for key, label in VARIANTS if key in summary]
        means = [summary[key][0] for key, _ in VARIANTS if key in summary]
        spreads = [summary[key][1] for key, _ in VARIANTS if key in summary]
        best = means.index(max(means))

        positions = range(len(means))
        bars = axis.bar(
            positions,
            means,
            yerr=spreads,
            capsize=3,
            color=COLOURS[: len(means)],
            edgecolor="black",
            linewidth=[1.4 if i == best else 0.6 for i in range(len(means))],
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )

        # headroom for the value labels, so nothing collides with the frame
        low = min(m - s for m, s in zip(means, spreads, strict=True))
        high = max(m + s for m, s in zip(means, spreads, strict=True))
        span = max(high - low, 0.6)
        axis.set_ylim(low - span * 0.35, high + span * 0.60)

        for bar, mean, spread in zip(bars, means, spreads, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                mean + spread + span * 0.10,
                f"{mean:.2f}\n\u00b1{spread:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                linespacing=1.15,
            )

        axis.set_title(f"{title}  (best outlined)", pad=6)
        axis.set_ylabel("macro F1 (%)")
        axis.set_xticks(list(positions))
        axis.set_xticklabels(labels, rotation=18, ha="right")
        axis.set_axisbelow(True)

    figure.tight_layout(rect=(0, 0, 1, 0.96))

    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 320})):
        target = OUTPUT / f"abl_f1_bars_macro_v2.{suffix}"
        figure.savefig(target, format=suffix, **kwargs)
        print(f"written: {target}  ({target.stat().st_size:,} bytes)")

    print("\nvalues drawn:")
    for dataset, title in DATASETS:
        summary = statistics_for(dataset)
        rendered = "  ".join(
            f"{label}={summary[key][0]:.2f}" for key, label in VARIANTS if key in summary
        )
        print(f"  {title:14s} {rendered}")


if __name__ == "__main__":
    main()
