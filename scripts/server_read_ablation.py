"""Recompute the ablation numbers from the authoritative result files.

Figure 6 disagrees with Table 3, and provenance checking cannot say which is
right: it only proves the image came from some run. The result JSONs under
phase1_corrected are what the handover calls authoritative, so this reads them
directly, averages each variant over the three seeds, and prints the figures the
table should contain. Whatever this produces is the answer; the table and the
plot are then judged against it rather than against each other.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

RESULTS = Path("/opt/ids_revision/deploy/results/phase1_corrected")
DATASETS = ("unsw", "nslkdd", "cicids2017", "cicddos2019")
SEEDS = (42, 123, 456)
VARIANTS = ("full_model", "without_diffusion", "without_multiview", "baseline")

LABEL = {
    "full_model": "Full",
    "without_diffusion": "w/o Diffusion",
    "without_multiview": "w/o Multi-View",
    "baseline": "w/o Both",
}


def load(dataset: str, seed: int) -> dict:
    """Read one result file, or return an empty mapping when absent."""
    path = RESULTS / f"{dataset}_seed{seed}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def macro(payload: dict, variant: str) -> float | None:
    """Pull the macro F1 for one variant, whatever nesting the file uses."""
    node = payload.get(variant)
    if isinstance(node, dict):
        for key in ("macro_f1", "macro_F1", "f1_macro"):
            if key in node:
                return float(node[key]) * (100 if float(node[key]) <= 1 else 1)
    return None


def main() -> None:
    """Print each dataset's ablation means with their spread."""
    for dataset in DATASETS:
        print(f"\n=== {dataset} ===")
        available = {seed: load(dataset, seed) for seed in SEEDS}
        present = [seed for seed, payload in available.items() if payload]
        if not present:
            print("  no result files")
            continue
        print(f"  seeds present: {present}")

        if present:
            sample = available[present[0]]
            print(f"  keys: {sorted(sample)[:8]}")

        for variant in VARIANTS:
            values = [
                value for seed in present if (value := macro(available[seed], variant)) is not None
            ]
            if not values:
                print(f"  {LABEL[variant]:16s} not found")
                continue
            mean = statistics.fmean(values)
            spread = statistics.stdev(values) if len(values) > 1 else 0.0
            detail = ", ".join(f"{v:.2f}" for v in values)
            print(f"  {LABEL[variant]:16s} {mean:6.2f} +/- {spread:4.2f}   [{detail}]")


if __name__ == "__main__":
    main()
