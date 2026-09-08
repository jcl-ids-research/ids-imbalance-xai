from __future__ import annotations

import numpy as np

from ids_diffusion.data.splits import apply_attack_thinning
from ids_diffusion.types import DatasetMatrix


def test_attack_thinning_changes_training_side_only() -> None:
    # Given: six normal and four attack rows
    features = np.arange(20, dtype=np.float32).reshape(10, 2)
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    dataset = DatasetMatrix(features=features, labels=labels)

    # When: half the attack rows are retained
    shifted, report = apply_attack_thinning(dataset, keep_fraction=0.5, seed=42)

    # Then: every normal row remains and two attack rows remain
    assert int((shifted.labels == 0).sum()) == 6
    assert int((shifted.labels == 1).sum()) == 2
    assert report.attacks_before == 4
    assert report.attacks_after == 2


def test_attack_thinning_is_deterministic_for_a_seed() -> None:
    # Given: a fixed training matrix
    features = np.arange(40, dtype=np.float32).reshape(20, 2)
    labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    dataset = DatasetMatrix(features=features, labels=labels)

    # When: the same shift is applied twice
    first, _ = apply_attack_thinning(dataset, keep_fraction=0.2, seed=123)
    second, _ = apply_attack_thinning(dataset, keep_fraction=0.2, seed=123)

    # Then: both the selected rows and their order are identical
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.labels, second.labels)
