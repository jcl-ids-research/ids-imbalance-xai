from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ids_diffusion.data.nslkdd import NSL_COLUMNS, load_nslkdd


def _nsl_row(*, duration: int, protocol: str, label: str) -> list[object]:
    values: list[object] = [0] * len(NSL_COLUMNS)
    values[NSL_COLUMNS.index("duration")] = duration
    values[NSL_COLUMNS.index("protocol_type")] = protocol
    values[NSL_COLUMNS.index("service")] = "http"
    values[NSL_COLUMNS.index("flag")] = "SF"
    values[NSL_COLUMNS.index("src_bytes")] = duration * 2
    values[NSL_COLUMNS.index("label")] = label
    values[NSL_COLUMNS.index("difficulty")] = 1
    return values


def _write_nsl(root: Path, test_duration_offset: int = 0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    train_labels = ["normal", "neptune", "normal", "ipsweep", "normal", "guess_passwd"]
    test_labels = ["normal", "apache2", "mscan", "sqlattack"]
    with (root / "KDDTrain+.txt").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for index, label in enumerate(train_labels, start=1):
            writer.writerow(
                _nsl_row(
                    duration=index,
                    protocol="tcp" if index % 2 else "udp",
                    label=label,
                )
            )
    with (root / "KDDTest+.txt").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for index, label in enumerate(test_labels, start=1):
            writer.writerow(
                _nsl_row(
                    duration=test_duration_offset + index,
                    protocol="icmp",
                    label=label,
                )
            )


def test_nslkdd_binary_preserves_official_split_and_training_statistics(tmp_path: Path) -> None:
    # Given: two official split pairs whose training files are identical
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_nsl(first)
    _write_nsl(second, test_duration_offset=10_000)

    # When: binary preprocessing is fitted independently on each pair
    loaded = load_nslkdd(first, task="binary")
    shifted_test = load_nslkdd(second, task="binary")

    # Then: the official row counts survive and test values cannot alter training features
    assert len(loaded.training.labels) == 6
    assert len(loaded.test.labels) == 4
    np.testing.assert_array_equal(loaded.training.features, shifted_test.training.features)
    np.testing.assert_array_equal(loaded.training.labels, np.array([0, 1, 0, 1, 0, 1]))
    assert np.isfinite(loaded.training.features).all()
    assert np.isfinite(loaded.test.features).all()


def test_nslkdd_multiclass_maps_test_only_attack_names_to_standard_families(tmp_path: Path) -> None:
    # Given: official train/test files containing attack names unique to the test set
    _write_nsl(tmp_path)

    # When: the five-class task is loaded
    loaded = load_nslkdd(tmp_path, task="multiclass")

    # Then: Normal, DoS, Probe and U2R retain their fixed paper label indices
    np.testing.assert_array_equal(loaded.test.labels, np.array([0, 1, 2, 4]))
