"""Rank-matched correction of generated discrete marginals."""

from __future__ import annotations

import math

import numpy as np

from ids_diffusion.errors import DataShapeError
from ids_diffusion.types import BooleanVector, FloatMatrix

MINIMUM_DISCRETE_LEVELS = 2
MATRIX_DIMENSIONS = 2


def detect_discrete_columns(
    matrix: FloatMatrix,
    maximum_unique_values: int = 32,
) -> BooleanVector:
    """Identify low-cardinality columns from the real training matrix.

    Low cardinality is the stable property after scaling: a coded feature may
    no longer contain integer values, but it still has only a few distinct
    levels. Continuous features are also required to have more than 3 levels,
    preventing tiny test fixtures from being misclassified solely by size.
    """
    if matrix.ndim != MATRIX_DIMENSIONS:
        raise DataShapeError(detail=f"expected a 2D matrix, got {matrix.ndim}D")

    cardinality_limit = min(maximum_unique_values, max(2, math.isqrt(len(matrix))))
    flags = np.zeros(matrix.shape[1], dtype=np.bool_)
    for column_index in range(matrix.shape[1]):
        unique_count = len(np.unique(matrix[:, column_index]))
        flags[column_index] = MINIMUM_DISCRETE_LEVELS <= unique_count <= cardinality_limit
    return flags


def rank_match_column(real_column: FloatMatrix, synthetic_column: FloatMatrix) -> FloatMatrix:
    """Map synthetic ranks onto the empirical marginal of a real column.

    The ordering learned by the generator is preserved: the smallest generated
    value receives the smallest real quantile, and so on. When sample counts
    match, the corrected column has exactly the real empirical counts.
    """
    real = np.asarray(real_column, dtype=np.float32).reshape(-1)
    synthetic = np.asarray(synthetic_column, dtype=np.float32).reshape(-1)
    if len(real) == 0 or len(synthetic) == 0:
        raise DataShapeError(detail="rank matching requires non-empty columns")

    order = np.argsort(synthetic, kind="stable")
    quantiles = (np.arange(len(synthetic), dtype=np.float64) + 0.5) / len(synthetic)
    real_positions = np.minimum((quantiles * len(real)).astype(np.int64), len(real) - 1)
    matched = np.sort(real)[real_positions]
    corrected = np.empty(len(synthetic), dtype=np.float32)
    corrected[order] = matched
    return corrected


def correct_discrete_marginals(
    real: FloatMatrix,
    synthetic: FloatMatrix,
    discrete: BooleanVector,
) -> FloatMatrix:
    """Apply rank matching to flagged columns and leave continuous columns unchanged."""
    if real.ndim != MATRIX_DIMENSIONS or synthetic.ndim != MATRIX_DIMENSIONS:
        raise DataShapeError(detail="real and synthetic data must be 2D")
    if real.shape[1] != synthetic.shape[1] or len(discrete) != real.shape[1]:
        raise DataShapeError(detail="real, synthetic, and flags disagree on feature count")

    corrected = np.asarray(synthetic, dtype=np.float32).copy()
    for column_index in np.flatnonzero(discrete):
        corrected[:, column_index] = rank_match_column(
            real[:, column_index],
            corrected[:, column_index],
        )
    return corrected


__all__ = [
    "correct_discrete_marginals",
    "detect_discrete_columns",
    "rank_match_column",
]
