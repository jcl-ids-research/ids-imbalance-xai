"""Redraw Figure 13 with legible feature labels.

The original figure uses three stacked vertical bar charts. At print width the
long feature names overlap and the repeated y-axis labels are clipped. The data
are correct; only the layout changes. Horizontal bars give every feature its
full name, and the five features the text discusses are highlighted and labelled
with their percentages.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SOURCE = Path("/opt/ids_revision/deploy/results/attention/attention_unsw_seed42_full_model.json")
OUTPUT = Path("/opt/ids_revision/deploy/results/figures/attention")
VIEWS = ("view1", "view2", "view3")
COLOUR_TOP = "#2E7EB8"
COLOUR_OTHER = "#A9BAC5"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def main() -> None:
    """Draw three full-width horizontal bar charts."""
    record = json.loads(SOURCE.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 1, figsize=(7.1, 8.0))
    figure.suptitle(
        "UNSW-NB15 feature attention by view, layer 4, seed 42",
        fontsize=10.5,
        fontweight="bold",
        y=0.995,
    )

    print("top five features:")
    for axis, view in zip(axes, VIEWS, strict=True):
        weights = record["received_attention"][f"{view}/layer4"]
        ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        top = {name for name, _ in ordered[:5]}
        display = list(reversed(ordered))
        names = [name for name, _ in display]
        values = [value for _, value in display]
        colours = [COLOUR_TOP if name in top else COLOUR_OTHER for name in names]

        bars = axis.barh(names, values, color=colours, edgecolor="white", linewidth=0.4)
        maximum = max(values)
        axis.set_xlim(0, maximum * 1.20)
        axis.set_xlabel("mean attention received")
        entropy = record["entropy"][f"{view}/layer4"]
        axis.set_title(f"{view} ({len(names)} features), normalised entropy {entropy:.3f}")
        axis.set_axisbelow(True)

        for bar, name, value in zip(bars, names, values, strict=True):
            if name not in top:
                continue
            axis.text(
                value + maximum * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value * 100:.1f}%",
                va="center",
                ha="left",
                fontsize=7.2,
            )

        top_values = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:5]
        print(
            f"  {view}: "
            + ", ".join(f"{name}={value * 100:.1f}%" for name, value in top_values)
            + f"; top-5 sum={sum(value for _, value in top_values) * 100:.1f}%"
        )

    figure.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.1)

    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 320})):
        target = OUTPUT / f"attn_importance_unsw_seed42_layer4_v2.{suffix}"
        figure.savefig(target, format=suffix, **kwargs)
        print(f"written: {target}  ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
