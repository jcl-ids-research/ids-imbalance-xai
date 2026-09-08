"""Recompute the manuscript's headline averages from the per-seed run files.

Seven figures could not be found by string search, which is expected: they are
means over seeds 42, 123 and 456, so they exist in no single run file. The only
way to check them is to recompute the average from the runs that produced it.

Anything that still fails to reconcile after this is a genuine problem.
"""

from __future__ import annotations

import io
import json
import statistics
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RESULTS = Path("/opt/ids_revision/results")
SEEDS = (42, 123, 456)

# claimed value -> (description, directories to search, filename stem, variant key)
TARGETS = [
    ("82.21", "NSL-KDD full model macro F1", ["ablation_cross_correct"], "nslkdd", "full_model"),
    ("98.20", "CIC-IDS-2017 full model", ["ablation_cross_correct"], "cicids2017", "full_model"),
    ("95.85", "CIC-DDoS2019 full model", ["ablation_cross_correct"], "cicddos2019", "full_model"),
]


def macro(entry: dict) -> float | None:
    """Return macro F1 as a percentage, whichever key the run used."""
    for key in ("f1_macro", "macro_f1"):
        if key in entry:
            value = float(entry[key])
            return value * 100 if value <= 1.0 else value
    return None


def main() -> None:
    """Recompute each average and compare it with the manuscript."""
    print("RECOMPUTED THREE-SEED MEANS\n")

    for claimed, description, folders, stem, variant in TARGETS:
        values: list[float] = []
        sources: list[str] = []

        for folder in folders:
            base = RESULTS / folder
            if not base.is_dir():
                continue
            for seed in SEEDS:
                for path in base.glob(f"*{stem}*seed{seed}*.json"):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    block = payload.get("variants", payload).get(variant)
                    if not isinstance(block, dict):
                        continue
                    score = macro(block)
                    if score is not None:
                        values.append(score)
                        sources.append(path.name)

        if not values:
            print(f"  {claimed:8s} {description:34s} no run files matched")
            continue

        mean = statistics.mean(values)
        spread = statistics.stdev(values) if len(values) > 1 else 0.0
        agrees = abs(mean - float(claimed)) < 0.05
        verdict = "matches" if agrees else "DIFFERS"
        print(f"  {claimed:8s} {description:34s} computed {mean:.2f} ± {spread:.2f}  {verdict}")
        print(f"           from {len(values)} runs: {', '.join(sources)}")

    print("\nDIRECTORY CONTENTS FOR REFERENCE")
    for folder in ("ablation_cross_correct", "ablation_repeat_correct"):
        base = RESULTS / folder
        if base.is_dir():
            names = sorted(p.name for p in base.glob("*.json"))
            print(f"  {folder}: {names[:12]}")


if __name__ == "__main__":
    main()
