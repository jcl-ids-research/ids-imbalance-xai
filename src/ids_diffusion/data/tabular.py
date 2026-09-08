"""Train-only encoding, imputation and scaling shared by dataset loaders.

Every statistic used here is fitted on the training partition alone. Test rows
are transformed with those fitted values, so no evaluation information can
reach a model through preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from ids_diffusion.types import FloatMatrix

ZERO_VARIANCE_TOLERANCE = 1e-8
UNSEEN_CATEGORY_CODE = -1.0


@dataclass(frozen=True, slots=True)
class EncodedColumns:
    """Encoded training and test matrices with the columns that survived."""

    training: FloatMatrix
    test: FloatMatrix
    feature_names: tuple[str, ...]


def _encode_categorical(training: pl.Series, test: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """Build a codebook from training values; unseen test values become -1."""
    categories = sorted(str(value) for value in training.drop_nulls().unique().to_list())
    codebook = {value: float(index) for index, value in enumerate(categories)}
    training_codes = np.asarray(
        [codebook.get(str(value), UNSEEN_CATEGORY_CODE) for value in training.to_list()],
        dtype=np.float64,
    )
    test_codes = np.asarray(
        [codebook.get(str(value), UNSEEN_CATEGORY_CODE) for value in test.to_list()],
        dtype=np.float64,
    )
    return training_codes, test_codes


def _encode_numeric(training: pl.Series, test: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float and replace non-finite values with the training median."""
    training_values = np.array(
        training.cast(pl.Float64, strict=False).to_numpy(),
        dtype=np.float64,
        copy=True,
    )
    test_values = np.array(
        test.cast(pl.Float64, strict=False).to_numpy(),
        dtype=np.float64,
        copy=True,
    )
    training_values[~np.isfinite(training_values)] = np.nan
    test_values[~np.isfinite(test_values)] = np.nan
    median = float(np.nanmedian(training_values)) if np.isfinite(training_values).any() else 0.0
    return (
        np.nan_to_num(training_values, nan=median),
        np.nan_to_num(test_values, nan=median),
    )


def encode_frames(
    training: pl.DataFrame,
    test: pl.DataFrame,
    feature_names: tuple[str, ...],
) -> EncodedColumns:
    """Encode both partitions with training-only codebooks and medians.

    Columns whose training values carry no variance are dropped: they add no
    signal and make a later quantile transform degenerate.
    """
    training_columns: list[np.ndarray] = []
    test_columns: list[np.ndarray] = []
    retained: list[str] = []
    for name in feature_names:
        training_series = training.get_column(name)
        test_series = test.get_column(name)
        if training_series.dtype in (pl.String, pl.Categorical, pl.Enum, pl.Boolean):
            training_values, test_values = _encode_categorical(training_series, test_series)
        else:
            training_values, test_values = _encode_numeric(training_series, test_series)
        if float(training_values.std()) <= ZERO_VARIANCE_TOLERANCE:
            continue
        training_columns.append(training_values)
        test_columns.append(test_values)
        retained.append(name)

    scaler = StandardScaler()
    training_matrix = scaler.fit_transform(np.column_stack(training_columns))
    test_matrix = scaler.transform(np.column_stack(test_columns))
    return EncodedColumns(
        training=np.asarray(training_matrix, dtype=np.float32),
        test=np.asarray(test_matrix, dtype=np.float32),
        feature_names=tuple(retained),
    )


__all__ = ["EncodedColumns", "encode_frames"]
