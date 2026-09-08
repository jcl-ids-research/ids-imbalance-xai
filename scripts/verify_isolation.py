"""Verify the frozen holdout cannot reach tuning code."""

from pathlib import Path

import numpy as np

from ids_diffusion.data.cache import load_inner
from ids_diffusion.data.holdout import read_split

print("INNER CACHES (must carry no evaluation matrix)")
for seed in (42, 123, 456):
    with np.load(f"runs/outer/inner_seed{seed}.npz") as archive:
        has_test = "test_x" in archive
        train_rows = archive["raw_train_x"].shape
        val_rows = archive["val_x"].shape
    print(f"  seed{seed}: has_test={has_test} train={train_rows} val={val_rows}")

print("HOLDOUT CACHES")
for seed in (42, 123, 456):
    with np.load(f"runs/outer/holdout_seed{seed}.npz") as archive:
        test_shape = archive["test_x"].shape
        counts = np.bincount(archive["test_y"]).tolist()
    print(f"  seed{seed}: test={test_shape} class_counts={counts}")

print("TUNING LOADER GUARD")
try:
    load_inner(Path("runs/outer/holdout_seed42.npz"))
    print("  FAIL: tuning loader accepted a holdout cache")
except Exception as error:  # noqa: BLE001
    print(f"  OK rejected: {error}")

print("SPLIT INTEGRITY")
split, manifest = read_split(Path("runs/outer/split.npz"))
overlap = np.intersect1d(split.inner, split.holdout).size
print(f"  overlap={overlap} inner={manifest.inner_count} holdout={manifest.holdout_count}")
print(f"  source_sha256={manifest.source_sha256}")
