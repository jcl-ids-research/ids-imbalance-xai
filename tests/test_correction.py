from __future__ import annotations

import numpy as np

from ids_diffusion.training.correction import detect_discrete_columns, rank_match_column


def test_rank_match_restores_real_support_and_counts() -> None:
    # Given: a real sparse discrete column and continuous synthetic values
    real = np.array([0, 0, 0, 0, 1, 2], dtype=np.float32)
    synthetic = np.array([-2.0, -1.0, -0.2, 0.1, 0.5, 2.0], dtype=np.float32)

    # When: synthetic ranks are mapped onto the real empirical marginal
    corrected = rank_match_column(real, synthetic)

    # Then: the corrected column uses only real values with the same counts
    real_values, real_counts = np.unique(real, return_counts=True)
    corrected_values, corrected_counts = np.unique(corrected, return_counts=True)
    np.testing.assert_array_equal(corrected_values, real_values)
    np.testing.assert_array_equal(corrected_counts, real_counts)


def test_detect_discrete_columns_separates_sparse_codes_from_continuous_values() -> None:
    # Given: one coded feature and one continuous feature
    matrix = np.column_stack(
        [
            np.array([0, 0, 0, 1, 1, 1], dtype=np.float32),
            np.linspace(0.1, 0.6, 6, dtype=np.float32),
        ]
    )

    # When: the feature types are inferred from the training matrix
    flags = detect_discrete_columns(matrix)

    # Then: only the coded feature is classified as discrete
    np.testing.assert_array_equal(flags, np.array([True, False]))
