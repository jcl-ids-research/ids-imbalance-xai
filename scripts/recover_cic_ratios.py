"""Recover the class ratios for the two CIC datasets.

Table 1 quotes 31.24 : 1 for CIC-DDoS2019 but nothing survives to support it,
and the CIC-IDS-2017 figure was never recorded at all. Both are needed to answer
the editor's objection that the resulting distribution does not look balanced.

Only the class counts are required, so this stops well short of a full run: it
reuses the original loader, applies the same balancing rule, and reports the
ratio before and after. No generator is fitted and no GPU is touched, because
the target count per class is arithmetic - the mean class size, capped at
fifteen times a class's real count - and does not depend on what the generator
would produce.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/opt/ids_revision/deploy")

from correct_cross_dataset import DATASETS

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SEED = 42
EXPANSION_CAP = 15
OUTPUT = Path("/opt/ids_revision/rm_dmvt/runs/cic_ratios.json")


def balanced_counts(labels: np.ndarray) -> dict[int, int]:
    """Apply the mean-target rule with the expansion cap, as rebalance() does."""
    values, counts = np.unique(labels, return_counts=True)
    target = round(float(counts.mean()))
    result: dict[int, int] = {}
    for value, count in zip(values.tolist(), counts.tolist(), strict=True):
        result[int(value)] = min(target, count * EXPANSION_CAP) if count < target else target
    return result


def ratio(counts: dict[int, int]) -> float:
    """Return the majority-to-minority ratio."""
    values = list(counts.values())
    return max(values) / min(values)


def main() -> None:
    """Report before/after ratios for both CIC datasets."""
    report: dict[str, dict[str, object]] = {}

    for name in ("cicids2017", "cicddos2019"):
        print(f"loading {name} ...", flush=True)
        data = DATASETS[name](SEED, smoke=False)

        values, counts = np.unique(data.y_train, return_counts=True)
        before = {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist(), strict=True)}
        after = balanced_counts(data.y_train)

        report[name] = {
            "train_rows": len(data.y_train),
            "test_rows": len(data.y_test),
            "before": before,
            "after": after,
            "ratio_before": ratio(before),
            "ratio_after": ratio(after),
        }

        print(f"  train rows {len(data.y_train):,}   test rows {len(data.y_test):,}")
        print(f"  before {before}   ratio {ratio(before):.2f} : 1")
        print(f"  after  {after}   ratio {ratio(after):.2f} : 1")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSUMMARY")
    for name, entry in report.items():
        print(f"  {name:14s} {entry['ratio_before']:.2f} : 1  ->  {entry['ratio_after']:.2f} : 1")
    print(f"\nwritten to {OUTPUT}")


if __name__ == "__main__":
    main()
