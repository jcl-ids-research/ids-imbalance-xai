"""What the rank-matched correction restores, and what it costs.

The correction fixes discrete marginals that the diffusion model collapsed. This
module recomputes both sides of that trade from the archived validation runs: the
off-default recall it restores, and the correlation and joint-distribution
structure it disturbs. The manuscript reports the cost as a range and names the
one dataset-seed pair where the joint distance moves the wrong way, so both are
derived here rather than trusted.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

from ids_diffusion.errors import DatasetFileError
from ids_diffusion.reproduction.archive_io import JsonObject, nested_float, read_json

CORRELATION_KEY = "corr_within_0.10"


def correction_blocks(root: Path) -> Iterator[tuple[str, str, JsonObject, Path]]:
    """Yield (dataset, seed, block, path) for every archived correction validation."""
    for path in sorted((root / "fidelity").glob("correction_validation_seed*.json")):
        seed = path.stem.rsplit("seed", 1)[-1]
        for dataset, block in read_json(path).items():
            if isinstance(block, Mapping):
                yield dataset, seed, block, path


def correction_cost(root: Path, dataset: str) -> tuple[float, float]:
    """Return the min and max correlation-preservation cost of the correction.

    The correction restores discrete marginals but perturbs feature
    correlations. The manuscript reports that cost as a range in percentage
    points and discloses where it breaches the pre-declared tolerance.
    """
    costs: list[float] = []
    for name, _seed, block, path in correction_blocks(root):
        if name != dataset:
            continue
        before = nested_float(block, path, "before", CORRELATION_KEY)
        after = nested_float(block, path, "after", CORRELATION_KEY)
        if before is not None and after is not None:
            costs.append((before - after) * 100)
    if not costs:
        raise DatasetFileError(
            path=str(root / "fidelity"),
            detail=f"no correction validation recorded for {dataset}",
        )
    return min(costs), max(costs)


def off_default_recalls(root: Path) -> list[float]:
    """Return post-correction off-default recall for every measured pair."""
    values: list[float] = []
    for _dataset, _seed, block, path in correction_blocks(root):
        recall = nested_float(block, path, "after", "off_default_recall")
        if recall is not None:
            values.append(recall * 100)
    return values


def restored_off_default_recall(root: Path) -> tuple[float, float]:
    """Return the lowest and highest post-correction off-default recall."""
    values = off_default_recalls(root)
    if not values:
        raise DatasetFileError(path=str(root / "fidelity"), detail="no recall recorded")
    return min(values), max(values)


def fidelity_pairs(root: Path) -> int:
    """Return how many dataset-seed pairs the fidelity family actually covers.

    The manuscript qualifies its correction claim by this count rather than by
    "every dataset and seed", because UNSW-NB15 carries three seeds while the
    other three benchmarks carry two. Asserting the count means a missing or
    added run is caught instead of quietly widening the claim.
    """
    return len(off_default_recalls(root))


def mmd_improvement(root: Path) -> tuple[int, int, str]:
    """Return improved pairs, total pairs and the exception the paper names.

    The correction restores discrete marginals but is not free in joint
    structure: on one dataset-seed pair MMD-squared rises. The manuscript states
    eight of nine, so the split is recomputed rather than trusted.
    """
    improved = 0
    total = 0
    worsened: list[str] = []
    for dataset, seed, block, path in correction_blocks(root):
        before = nested_float(block, path, "before", "mmd2")
        after = nested_float(block, path, "after", "mmd2")
        if before is None or after is None:
            continue
        total += 1
        if after < before:
            improved += 1
        else:
            worsened.append(f"{dataset} seed{seed}")
    return improved, total, ", ".join(sorted(worsened))


__all__ = [
    "CORRELATION_KEY",
    "correction_blocks",
    "correction_cost",
    "fidelity_pairs",
    "mmd_improvement",
    "off_default_recalls",
    "restored_off_default_recall",
]
