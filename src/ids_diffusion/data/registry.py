"""One entry point for every dataset and view partition used by the paper.

Callers name a dataset instead of importing a loader, so the CLI, the tuning
code and the reproduction driver cannot drift apart on preprocessing, split
policy or view construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeGuard

from ids_diffusion.config import DatasetName, TaskName
from ids_diffusion.data.cic import CicReadLimits, load_cic
from ids_diffusion.data.nslkdd import NSL_SEMANTIC_VIEWS, load_nslkdd
from ids_diffusion.data.nslkdd import TEST_FILE as NSL_TEST_FILE
from ids_diffusion.data.nslkdd import TRAIN_FILE as NSL_TRAIN_FILE
from ids_diffusion.data.unsw import TEST_FILE as UNSW_TEST_FILE
from ids_diffusion.data.unsw import TRAIN_FILE as UNSW_TRAIN_FILE
from ids_diffusion.data.unsw import load_unsw
from ids_diffusion.data.views import (
    UNSW_SEMANTIC_VIEWS,
    named_view_indices,
    positional_view_indices,
)
from ids_diffusion.errors import ConfigurationError, ViewPartitionError
from ids_diffusion.types import LoadedDataset, ViewIndices

DATASET_NAMES: tuple[DatasetName, ...] = ("unsw", "nslkdd", "cicids2017", "cicddos2019")
TASK_NAMES: tuple[TaskName, ...] = ("binary", "multiclass")
CIC_DIRECTORY: dict[DatasetName, str] = {
    "cicids2017": "MachineLearningCVE",
    "cicddos2019": "all",
}
SEMANTIC_VIEWS: dict[DatasetName, tuple[tuple[str, ...], ...]] = {
    "unsw": UNSW_SEMANTIC_VIEWS,
    "nslkdd": NSL_SEMANTIC_VIEWS,
}
MULTICLASS_DATASETS: frozenset[DatasetName] = frozenset({"unsw", "nslkdd"})
REQUIRED_FILES: dict[DatasetName, tuple[str, ...]] = {
    "unsw": (UNSW_TRAIN_FILE, UNSW_TEST_FILE),
    "nslkdd": (NSL_TRAIN_FILE, NSL_TEST_FILE),
}


def is_dataset_name(value: str) -> TypeGuard[DatasetName]:
    """Narrow an arbitrary string to a dataset the paper reports."""
    return value in DATASET_NAMES


def is_task_name(value: str) -> TypeGuard[TaskName]:
    """Narrow an arbitrary string to a task the paper reports."""
    return value in TASK_NAMES


def require_task(task: str) -> TaskName:
    """Reject any task the paper does not report."""
    if not is_task_name(task):
        raise ConfigurationError(
            field="task",
            detail=f"expected one of {list(TASK_NAMES)}, got {task!r}",
        )
    return task


def require_dataset(dataset: str) -> DatasetName:
    """Reject any dataset the paper does not report."""
    if not is_dataset_name(dataset):
        raise ConfigurationError(
            field="dataset",
            detail=f"expected one of {list(DATASET_NAMES)}, got {dataset!r}",
        )
    return dataset


def cic_root(dataset: DatasetName, root: Path) -> Path:
    """Return the CSV directory a CIC release is distributed in."""
    suffix = CIC_DIRECTORY[dataset]
    return root if root.name == suffix else root / suffix


def load_dataset(
    dataset: str,
    root: Path,
    seed: int = 42,
    task: TaskName = "binary",
    sample_cap: int = 200_000,
) -> LoadedDataset:
    """Load one benchmark with its published split and train-only preprocessing."""
    name = require_dataset(dataset)
    if task == "multiclass" and name not in MULTICLASS_DATASETS:
        raise ConfigurationError(
            field="task",
            detail=f"multiclass results are reported for {sorted(MULTICLASS_DATASETS)} only",
        )
    if name == "unsw":
        return load_unsw(root, task=task)
    if name == "nslkdd":
        return load_nslkdd(root, task=task)
    return load_cic(
        cic_root(name, root),
        seed=seed,
        task=task,
        limits=CicReadLimits(sample_cap=sample_cap),
    )


def missing_inputs(dataset: str, root: Path) -> tuple[str, ...]:
    """Report which public files a dataset still needs before a run can start."""
    name = require_dataset(dataset)
    required = REQUIRED_FILES.get(name)
    if required is not None:
        return tuple(str(root / file) for file in required if not (root / file).is_file())
    directory = cic_root(name, root)
    if directory.is_dir() and any(directory.glob("*.csv")):
        return ()
    return (str(directory / "*.csv"),)


def view_indices_for_dataset(
    dataset: str,
    feature_names: tuple[str, ...],
    view_count: int = 3,
) -> ViewIndices:
    """Resolve the view partition, falling back to positional blocks.

    UNSW-NB15 and NSL-KDD have published attribute families, so their views are
    semantic. The CIC releases have no such mapping, and preprocessing may drop
    zero-variance columns, so those cases use equal positional blocks.
    """
    name = require_dataset(dataset)
    groups = SEMANTIC_VIEWS.get(name)
    if groups is not None:
        try:
            return named_view_indices(feature_names, groups)
        except ViewPartitionError:
            pass
    return positional_view_indices(len(feature_names), view_count)


__all__ = [
    "DATASET_NAMES",
    "MULTICLASS_DATASETS",
    "TASK_NAMES",
    "cic_root",
    "is_dataset_name",
    "is_task_name",
    "load_dataset",
    "missing_inputs",
    "require_dataset",
    "require_task",
    "view_indices_for_dataset",
]
