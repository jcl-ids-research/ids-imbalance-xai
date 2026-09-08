from __future__ import annotations

from pathlib import Path

import numpy as np

from ids_diffusion.data.cache import load_prepared, save_prepared
from ids_diffusion.types import DatasetMatrix, PreparedExperiment


def test_prepared_cache_round_trip_preserves_arrays_and_views(tmp_path: Path) -> None:
    # Given: one classifier-ready experiment cache
    raw = DatasetMatrix(
        features=np.arange(12, dtype=np.float32).reshape(6, 2),
        labels=np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
    )
    balanced = DatasetMatrix(
        features=np.arange(16, dtype=np.float32).reshape(8, 2),
        labels=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
    )
    prepared = PreparedExperiment(
        raw_training=raw,
        balanced_training=balanced,
        validation=raw,
        test=raw,
        views=((0,), (1,)),
        class_count=2,
    )
    path = tmp_path / "seed42.npz"

    # When: the cache is written and read back
    save_prepared(path, prepared)
    restored = load_prepared(path)

    # Then: the tuning contract is unchanged
    np.testing.assert_array_equal(restored.raw_training.features, raw.features)
    np.testing.assert_array_equal(restored.balanced_training.labels, balanced.labels)
    assert restored.views == ((0,), (1,))
    assert restored.class_count == 2
