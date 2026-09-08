"""Load the official UNSW training file under a frozen outer holdout split.

Every statistic - category codebook, median fill, and standardisation - is fit
on the inner rows alone. The outer holdout is transformed with those fitted
values and is never used to fit anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from ids_diffusion.data.holdout import OuterSplitManifest, read_split, verify_source
from ids_diffusion.data.unsw import EXCLUDED_COLUMNS, TRAIN_FILE
from ids_diffusion.errors import DatasetFileError
from ids_diffusion.types import DatasetMatrix, IndexVector, LoadedDataset


def source_path(root: Path) -> Path:
    """Return the official training CSV that the outer split is drawn from."""
    path = root / TRAIN_FILE
    if not path.is_file():
        raise DatasetFileError(path=str(path), detail="required file not found")
    return path


def read_labels(root: Path) -> np.ndarray:
    """Read only the label column, which is all the split drawing needs."""
    frame = pl.read_csv(source_path(root), columns=["label"])
    return np.asarray(frame.get_column("label"), dtype=np.int64)


def _categorical_column(
    values: pl.Series,
    inner: IndexVector,
    holdout: IndexVector,
) -> tuple[np.ndarray, np.ndarray]:
    raw = [str(value) for value in values.to_list()]
    inner_raw = [raw[position] for position in inner.tolist()]
    codebook = {value: index for index, value in enumerate(sorted(set(inner_raw)))}
    inner_values = np.asarray(
        [codebook.get(value, -1) for value in inner_raw],
        dtype=np.float32,
    )
    holdout_values = np.asarray(
        [codebook.get(raw[position], -1) for position in holdout.tolist()],
        dtype=np.float32,
    )
    return inner_values, holdout_values


def _numeric_column(
    values: pl.Series,
    inner: IndexVector,
    holdout: IndexVector,
) -> tuple[np.ndarray, np.ndarray]:
    numeric = np.array(values.cast(pl.Float64, strict=False).to_numpy(), dtype=np.float64)
    numeric[~np.isfinite(numeric)] = np.nan
    inner_values = numeric[inner]
    median = float(np.nanmedian(inner_values))
    return (
        np.nan_to_num(inner_values, nan=median).astype(np.float32),
        np.nan_to_num(numeric[holdout], nan=median).astype(np.float32),
    )


def load_unsw_holdout(root: Path, split_path: Path) -> tuple[LoadedDataset, OuterSplitManifest]:
    """Return inner-fit training rows and the frozen holdout as the test matrix."""
    split, manifest = read_split(split_path)
    path = source_path(root)
    verify_source(path, manifest)

    frame = pl.read_csv(path)
    if "label" not in frame.columns:
        raise DatasetFileError(path=str(path), detail="label column not found")
    if len(frame) != manifest.row_count:
        raise DatasetFileError(path=str(path), detail="row count does not match the manifest")

    feature_names = tuple(name for name in frame.columns if name not in EXCLUDED_COLUMNS)
    inner_columns: list[np.ndarray] = []
    holdout_columns: list[np.ndarray] = []
    for name in feature_names:
        series = frame.get_column(name)
        if series.dtype in (pl.String, pl.Categorical, pl.Enum):
            inner_values, holdout_values = _categorical_column(series, split.inner, split.holdout)
        else:
            inner_values, holdout_values = _numeric_column(series, split.inner, split.holdout)
        inner_columns.append(inner_values)
        holdout_columns.append(holdout_values)

    scaler = StandardScaler()
    inner_features = scaler.fit_transform(np.column_stack(inner_columns)).astype(np.float32)
    holdout_features = scaler.transform(np.column_stack(holdout_columns)).astype(np.float32)
    labels = np.asarray(frame.get_column("label"), dtype=np.int64)

    return (
        LoadedDataset(
            training=DatasetMatrix(features=inner_features, labels=labels[split.inner]),
            test=DatasetMatrix(features=holdout_features, labels=labels[split.holdout]),
            feature_names=feature_names,
        ),
        manifest,
    )


__all__ = ["load_unsw_holdout", "read_labels", "source_path"]
