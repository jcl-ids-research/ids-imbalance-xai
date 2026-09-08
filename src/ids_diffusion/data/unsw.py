"""Train-only preprocessing for the official UNSW-NB15 split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from ids_diffusion.config import TaskName
from ids_diffusion.errors import DatasetFileError
from ids_diffusion.types import DatasetMatrix, LoadedDataset

TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE = "UNSW_NB15_testing-set.csv"
EXCLUDED_COLUMNS = frozenset({"id", "Unnamed: 0", "label", "attack_cat"})
CATEGORY_COLUMN = "attack_cat"


def _categorical_column(
    training: pl.Series,
    test: pl.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a codebook from training only; unseen test categories become -1."""
    categories = sorted(str(value) for value in training.drop_nulls().unique().to_list())
    codebook = {value: index for index, value in enumerate(categories)}
    train_values = np.asarray(
        [codebook.get(str(value), -1) for value in training.to_list()],
        dtype=np.float32,
    )
    test_values = np.asarray(
        [codebook.get(str(value), -1) for value in test.to_list()],
        dtype=np.float32,
    )
    return train_values, test_values


def _numeric_column(
    training: pl.Series,
    test: pl.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill missing values with the training median, never with test statistics."""
    train_values = np.asarray(
        training.cast(pl.Float64, strict=False).to_numpy(),
        dtype=np.float64,
    )
    test_values = np.asarray(
        test.cast(pl.Float64, strict=False).to_numpy(),
        dtype=np.float64,
    )
    train_values[~np.isfinite(train_values)] = np.nan
    test_values[~np.isfinite(test_values)] = np.nan
    median = float(np.nanmedian(train_values))
    return (
        np.nan_to_num(train_values, nan=median).astype(np.float32),
        np.nan_to_num(test_values, nan=median).astype(np.float32),
    )


def _category_labels(
    training: pl.DataFrame,
    test: pl.DataFrame,
    root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode attack categories from the training set only.

    A category that appears only in the test partition would signal a split
    problem, so it raises instead of being folded into an "other" bin.
    """
    if CATEGORY_COLUMN not in training.columns or CATEGORY_COLUMN not in test.columns:
        raise DatasetFileError(path=str(root), detail="attack_cat column not found")
    training_names = [str(value).strip() for value in training.get_column(CATEGORY_COLUMN)]
    test_names = [str(value).strip() for value in test.get_column(CATEGORY_COLUMN)]
    index = {name: position for position, name in enumerate(sorted(set(training_names)))}
    unseen = sorted(set(test_names) - set(index))
    if unseen:
        raise DatasetFileError(
            path=str(root),
            detail=f"attack_cat present in test but not train: {unseen}",
        )
    return (
        np.asarray([index[name] for name in training_names], dtype=np.int64),
        np.asarray([index[name] for name in test_names], dtype=np.int64),
    )


def load_unsw(root: Path, task: TaskName = "binary") -> LoadedDataset:
    """Load the publisher's training/test files without merging or resampling."""
    training_path = root / TRAIN_FILE
    test_path = root / TEST_FILE
    for path in (training_path, test_path):
        if not path.is_file():
            raise DatasetFileError(path=str(path), detail="required file not found")

    training_frame = pl.read_csv(training_path)
    test_frame = pl.read_csv(test_path)
    if "label" not in training_frame.columns or "label" not in test_frame.columns:
        raise DatasetFileError(path=str(root), detail="label column not found")

    feature_names = tuple(name for name in training_frame.columns if name not in EXCLUDED_COLUMNS)
    training_columns: list[np.ndarray] = []
    test_columns: list[np.ndarray] = []
    for name in feature_names:
        training_series = training_frame.get_column(name)
        test_series = test_frame.get_column(name)
        if training_series.dtype in (pl.String, pl.Categorical, pl.Enum):
            train_values, test_values = _categorical_column(training_series, test_series)
        else:
            train_values, test_values = _numeric_column(training_series, test_series)
        training_columns.append(train_values)
        test_columns.append(test_values)

    training_features = np.column_stack(training_columns).astype(np.float32)
    test_features = np.column_stack(test_columns).astype(np.float32)
    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(training_features).astype(np.float32)
    test_scaled = scaler.transform(test_features).astype(np.float32)
    if task == "multiclass":
        training_labels, test_labels = _category_labels(training_frame, test_frame, root)
    else:
        training_labels = np.asarray(training_frame.get_column("label"), dtype=np.int64)
        test_labels = np.asarray(test_frame.get_column("label"), dtype=np.int64)

    return LoadedDataset(
        training=DatasetMatrix(features=training_scaled, labels=training_labels),
        test=DatasetMatrix(features=test_scaled, labels=test_labels),
        feature_names=feature_names,
    )


__all__ = ["load_unsw"]
