"""Verify the imbalance ratios now printed in Table 1 against the real caches.

Numbers written into a table from memory are exactly the kind of claim a
reviewer can check and we cannot defend. This recomputes the majority-to-
minority ratio directly from the prepared UNSW cache, before and after
rebalancing, so the table entry is backed by the data it describes.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CACHES = {
    "UNSW-NB15": Path("runs/cache/unsw_seed42.npz"),
}


def ratio(labels: np.ndarray) -> tuple[str, dict[int, int]]:
    """Return the majority-to-minority ratio and the per-class counts."""
    values, counts = np.unique(labels, return_counts=True)
    mapping = {int(v): int(c) for v, c in zip(values, counts, strict=True)}
    return f"{counts.max() / counts.min():.2f} : 1", mapping


def main() -> None:
    """Print measured ratios for every cache that exists."""
    for name, path in CACHES.items():
        if not path.is_file():
            print(f"{name}: cache not found at {path}")
            continue

        with np.load(path, allow_pickle=False) as archive:
            raw = np.asarray(archive["raw_train_y"], dtype=np.int64)
            balanced = np.asarray(archive["balanced_train_y"], dtype=np.int64)

        before, before_counts = ratio(raw)
        after, after_counts = ratio(balanced)

        print(f"{name}")
        print(f"  before rebalancing : {before}   counts {before_counts}")
        print(f"  after rebalancing  : {after}   counts {after_counts}")
        print(f"  rows before {len(raw):,}  after {len(balanced):,}")


if __name__ == "__main__":
    main()
