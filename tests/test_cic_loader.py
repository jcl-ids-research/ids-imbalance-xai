from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ids_diffusion.data.cic import CicReadLimits, load_cic


def _write_cic(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for file_index in range(2):
        rows = 60
        labels = ["BENIGN" if index % 3 else "DDoS" for index in range(rows)]
        feature = np.arange(file_index * rows, (file_index + 1) * rows, dtype=np.float64)
        pl.DataFrame(
            {
                "Flow ID": [f"flow-{file_index}-{index}" for index in range(rows)],
                "Feature A": feature,
                "Feature B": np.where(feature == 17, np.inf, feature % 7),
                " Label ": labels,
            }
        ).write_csv(root / f"part-{file_index}.csv")


def test_cic_binary_split_is_deterministic_finite_and_training_fitted(tmp_path: Path) -> None:
    # Given: a CICFlowMeter directory with benign and attack rows
    _write_cic(tmp_path)

    # When: the same seeded split is loaded twice
    limits = CicReadLimits(sample_cap=80)
    first = load_cic(tmp_path, seed=42, task="binary", limits=limits)
    second = load_cic(tmp_path, seed=42, task="binary", limits=limits)

    # Then: sampling and preprocessing are deterministic and finite
    np.testing.assert_array_equal(first.training.features, second.training.features)
    np.testing.assert_array_equal(first.test.labels, second.test.labels)
    assert set(first.feature_names) == {"Feature A", "Feature B"}
    assert len(first.training.labels) == 64
    assert len(first.test.labels) == 16
    assert np.isfinite(first.training.features).all()
    assert np.isfinite(first.test.features).all()
    np.testing.assert_allclose(first.training.features.mean(axis=0), 0.0, atol=1e-6)
