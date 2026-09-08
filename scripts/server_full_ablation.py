"""Assemble the full ablation table from the authoritative records.

UNSW stores its results one level deeper than the other three datasets, under
phase1_corrected/unsw/seed<n>/metrics.json, which is why the first sweep missed
it. This reads whichever layout each dataset uses and prints all four variants
for all four datasets, together with the component contributions Table 5
reports, so the manuscript can be checked against the runs in one pass.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

CORRECTED = Path("/opt/ids_revision/deploy/results/phase1_corrected")
SEEDS = (42, 123, 456)
VARIANTS = (
    ("full_model", "Full"),
    ("wo_diffusion", "w/o Diffusion"),
    ("wo_multiview", "w/o Multi-View"),
    ("baseline", "w/o Both"),
)
DATASETS = ("unsw", "nslkdd", "cicids2017", "cicddos2019")


def result_path(dataset: str, seed: int) -> Path | None:
    """Return whichever of the two known layouts exists."""
    flat = CORRECTED / f"{dataset}_seed{seed}.json"
    nested = CORRECTED / dataset / f"seed{seed}" / "metrics.json"
    if flat.is_file():
        return flat
    if nested.is_file():
        return nested
    return None


def per_seed(dataset: str) -> dict[str, list[float]]:
    """Collect each variant's macro F1 across the seeds, as percentages."""
    gathered: dict[str, list[float]] = {key: [] for key, _ in VARIANTS}
    for seed in SEEDS:
        path = result_path(dataset, seed)
        if path is None:
            continue
        variants = json.loads(path.read_text()).get("variants", {})
        for key, _ in VARIANTS:
            node = variants.get(key)
            if node and "f1_macro" in node:
                gathered[key].append(float(node["f1_macro"]) * 100)
    return gathered


def main() -> None:
    """Print the ablation means and the two component contributions."""
    for dataset in DATASETS:
        gathered = per_seed(dataset)
        if not any(gathered.values()):
            print(f"\n=== {dataset} ===\n  no records")
            continue

        source = result_path(dataset, 42)
        print(f"\n=== {dataset} ===")
        print(f"  source: {source.relative_to(CORRECTED.parent) if source else '?'}")
        means: dict[str, float] = {}
        for key, label in VARIANTS:
            values = gathered[key]
            if not values:
                print(f"  {label:16s} missing")
                continue
            mean = statistics.fmean(values)
            spread = statistics.stdev(values) if len(values) > 1 else 0.0
            means[key] = mean
            detail = ", ".join(f"{v:.2f}" for v in values)
            print(f"  {label:16s} {mean:6.2f} +/- {spread:4.2f}   [{detail}]")

        if {"full_model", "wo_diffusion", "wo_multiview"} <= means.keys():
            diffusion = means["full_model"] - means["wo_diffusion"]
            multiview = means["full_model"] - means["wo_multiview"]
            print(f"  {'-> diffusion':16s} {diffusion:+6.2f}")
            print(f"  {'-> multi-view':16s} {multiview:+6.2f}")


if __name__ == "__main__":
    main()
