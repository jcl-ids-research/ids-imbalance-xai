"""Compare freshly produced metrics against the archived server results.

The archived JSON files are the numbers the manuscript reports. A reviewer who
reruns the pipeline needs to see, per variant and per metric, how far a new run
lands from the recorded one rather than being told only that it "matches".
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ids_diffusion.errors import DatasetFileError

DEFAULT_EVIDENCE_ROOT = Path("evidence/paper_results/phase1_corrected")

VARIANT_NAMES: dict[str, str] = {
    "full_model": "full_model",
    "wo_diffusion": "without_diffusion",
    "wo_multiview": "without_multiview",
    "baseline": "baseline",
}
METRIC_NAMES: dict[str, str] = {
    "acc": "accuracy",
    "f1": "weighted_f1",
    "f1_macro": "macro_f1",
    "precision": "precision",
    "recall": "recall",
}


@dataclass(frozen=True, slots=True)
class ReferenceRun:
    """Recorded server metrics for one dataset and seed."""

    dataset: str
    seed: int
    path: Path
    variants: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """One recorded metric next to its freshly produced counterpart."""

    variant: str
    metric: str
    reference: float
    produced: float

    @property
    def delta(self) -> float:
        """Return produced minus recorded, in the metric's own units."""
        return self.produced - self.reference


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Every metric difference for one reproduced run."""

    dataset: str
    seed: int
    tolerance: float
    deltas: tuple[MetricDelta, ...]

    @property
    def largest(self) -> MetricDelta | None:
        """Return the metric that moved the most, if any were compared."""
        return max(self.deltas, key=lambda item: abs(item.delta), default=None)

    @property
    def within_tolerance(self) -> bool:
        """Report whether every compared metric stayed inside the tolerance."""
        return all(abs(item.delta) <= self.tolerance for item in self.deltas)


TASK_FAMILIES: dict[str, str] = {
    "binary": "phase1_corrected",
    "multiclass": "multiclass",
}


def reference_path(root: Path, dataset: str, seed: int, task: str = "binary") -> Path:
    """Resolve either archived layout for one dataset, seed and task.

    `root` may name an evidence root that holds several families, or a family
    directory directly. Both are accepted so that callers holding an already
    resolved path keep working.
    """
    family = TASK_FAMILIES.get(task)
    if family is None:
        raise DatasetFileError(path=str(root), detail=f"unknown task {task!r}")

    candidates = [root / family, root] if (root / family).is_dir() else [root]
    for base in candidates:
        flat = base / f"{dataset}_seed{seed}.json"
        if flat.is_file():
            return flat
        nested = base / dataset / f"seed{seed}" / "metrics.json"
        if nested.is_file():
            return nested
    raise DatasetFileError(
        path=str(root),
        detail=f"no archived {task} result for {dataset} seed {seed}",
    )


def load_reference(root: Path, dataset: str, seed: int, task: str = "binary") -> ReferenceRun:
    """Load one archived run and normalise its variant and metric names."""
    path = reference_path(root, dataset, seed, task)
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("variants")
    if not isinstance(recorded, dict):
        raise DatasetFileError(path=str(path), detail="no variants block found")

    variants: dict[str, dict[str, float]] = {}
    for archived_name, canonical_name in VARIANT_NAMES.items():
        entry = recorded.get(archived_name)
        if not isinstance(entry, dict):
            continue
        variants[canonical_name] = {
            METRIC_NAMES[key]: float(value)
            for key, value in entry.items()
            if key in METRIC_NAMES and isinstance(value, (int, float))
        }
    if not variants:
        raise DatasetFileError(path=str(path), detail="no recognised variants recorded")
    return ReferenceRun(dataset=dataset, seed=int(seed), path=path, variants=variants)


def available_references(root: Path) -> tuple[ReferenceRun, ...]:
    """Load every archived run found under one evidence directory."""
    found: list[ReferenceRun] = []
    for path in sorted(root.rglob("*.json")):
        name = path.stem
        if path.name == "metrics.json" and path.parent.name.startswith("seed"):
            dataset = path.parent.parent.name
            seed = int(path.parent.name.removeprefix("seed"))
        elif "_seed" in name:
            dataset, _, raw_seed = name.partition("_seed")
            seed = int(raw_seed)
        else:
            continue
        found.append(load_reference(root, dataset, seed))
    return tuple(found)


def compare_to_reference(
    produced: Mapping[str, Mapping[str, object]],
    reference: ReferenceRun,
    tolerance: float = 0.02,
) -> ComparisonReport:
    """Difference a fresh metrics payload against one archived run."""
    deltas: list[MetricDelta] = []
    for variant, recorded_metrics in reference.variants.items():
        fresh = produced.get(variant)
        if not isinstance(fresh, Mapping):
            continue
        for metric, recorded_value in recorded_metrics.items():
            fresh_value = fresh.get(metric)
            if not isinstance(fresh_value, (int, float)):
                continue
            deltas.append(
                MetricDelta(
                    variant=variant,
                    metric=metric,
                    reference=recorded_value,
                    produced=float(fresh_value),
                )
            )
    return ComparisonReport(
        dataset=reference.dataset,
        seed=reference.seed,
        tolerance=tolerance,
        deltas=tuple(deltas),
    )


__all__ = [
    "DEFAULT_EVIDENCE_ROOT",
    "ComparisonReport",
    "MetricDelta",
    "ReferenceRun",
    "available_references",
    "compare_to_reference",
    "load_reference",
    "reference_path",
]
