"""Portable array cache used by training and hyperparameter search."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ids_diffusion.errors import DataShapeError
from ids_diffusion.types import (
    DatasetMatrix,
    PreparedExperiment,
    PreparedInner,
    ViewIndices,
)

INNER_ARRAYS = (
    "raw_train_x",
    "raw_train_y",
    "balanced_train_x",
    "balanced_train_y",
    "val_x",
    "val_y",
)


def _views(archive: np.lib.npyio.NpzFile) -> ViewIndices:
    view_count = int(archive["view_count"][0])
    return tuple(
        tuple(int(value) for value in archive[f"view_{index}"]) for index in range(view_count)
    )


def _require(archive: np.lib.npyio.NpzFile, keys: tuple[str, ...], path: Path) -> None:
    missing = tuple(key for key in keys if key not in archive)
    if missing:
        raise DataShapeError(detail=f"cache {path.name} is missing arrays: {missing}")


def save_inner(path: Path, prepared: PreparedInner) -> None:
    """Write tuning-side arrays only, with no evaluation partition present."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "raw_train_x": prepared.raw_training.features,
        "raw_train_y": prepared.raw_training.labels,
        "balanced_train_x": prepared.balanced_training.features,
        "balanced_train_y": prepared.balanced_training.labels,
        "val_x": prepared.validation.features,
        "val_y": prepared.validation.labels,
        "class_count": np.asarray([prepared.class_count], dtype=np.int64),
        "view_count": np.asarray([len(prepared.views)], dtype=np.int64),
    }
    for index, view in enumerate(prepared.views):
        arrays[f"view_{index}"] = np.asarray(view, dtype=np.int64)
    np.savez_compressed(path, **arrays)


def load_inner(path: Path) -> PreparedInner:
    """Parse a tuning cache and reject one that carries an evaluation matrix."""
    with np.load(path, allow_pickle=False) as archive:
        _require(archive, INNER_ARRAYS, path)
        if "test_x" in archive:
            raise DataShapeError(
                detail=f"cache {path.name} carries an evaluation matrix and cannot be tuned on"
            )
        return PreparedInner(
            raw_training=DatasetMatrix(
                features=np.asarray(archive["raw_train_x"], dtype=np.float32),
                labels=np.asarray(archive["raw_train_y"], dtype=np.int64),
            ),
            balanced_training=DatasetMatrix(
                features=np.asarray(archive["balanced_train_x"], dtype=np.float32),
                labels=np.asarray(archive["balanced_train_y"], dtype=np.int64),
            ),
            validation=DatasetMatrix(
                features=np.asarray(archive["val_x"], dtype=np.float32),
                labels=np.asarray(archive["val_y"], dtype=np.int64),
            ),
            views=_views(archive),
            class_count=int(archive["class_count"][0]),
        )


def save_prepared(path: Path, prepared: PreparedExperiment) -> None:
    """Write a classifier-ready dataset without serialising executable objects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "raw_train_x": prepared.raw_training.features,
        "raw_train_y": prepared.raw_training.labels,
        "balanced_train_x": prepared.balanced_training.features,
        "balanced_train_y": prepared.balanced_training.labels,
        "val_x": prepared.validation.features,
        "val_y": prepared.validation.labels,
        "test_x": prepared.test.features,
        "test_y": prepared.test.labels,
        "class_count": np.asarray([prepared.class_count], dtype=np.int64),
        "view_count": np.asarray([len(prepared.views)], dtype=np.int64),
    }
    for index, view in enumerate(prepared.views):
        arrays[f"view_{index}"] = np.asarray(view, dtype=np.int64)
    np.savez_compressed(path, **arrays)


def load_prepared(path: Path) -> PreparedExperiment:
    """Parse a classifier cache and prove its required arrays are present."""
    with np.load(path, allow_pickle=False) as archive:
        _require(archive, (*INNER_ARRAYS, "test_x", "test_y"), path)
        return PreparedExperiment(
            raw_training=DatasetMatrix(
                features=np.asarray(archive["raw_train_x"], dtype=np.float32),
                labels=np.asarray(archive["raw_train_y"], dtype=np.int64),
            ),
            balanced_training=DatasetMatrix(
                features=np.asarray(archive["balanced_train_x"], dtype=np.float32),
                labels=np.asarray(archive["balanced_train_y"], dtype=np.int64),
            ),
            validation=DatasetMatrix(
                features=np.asarray(archive["val_x"], dtype=np.float32),
                labels=np.asarray(archive["val_y"], dtype=np.int64),
            ),
            test=DatasetMatrix(
                features=np.asarray(archive["test_x"], dtype=np.float32),
                labels=np.asarray(archive["test_y"], dtype=np.int64),
            ),
            views=_views(archive),
            class_count=int(archive["class_count"][0]),
        )


__all__ = ["load_inner", "load_prepared", "save_inner", "save_prepared"]
