"""Tests that the frozen outer holdout cannot silently leak into tuning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ids_diffusion.data.cache import load_inner, load_prepared, save_inner, save_prepared
from ids_diffusion.data.holdout import (
    build_outer_split,
    create_manifest,
    file_sha256,
    indices_sha256,
    manifest_path,
    read_split,
    verify_source,
    write_split,
)
from ids_diffusion.data.unsw import TRAIN_FILE
from ids_diffusion.data.unsw_holdout import load_unsw_holdout, read_labels
from ids_diffusion.errors import ConfigurationError, DatasetFileError, DataShapeError
from ids_diffusion.types import DatasetMatrix, PreparedExperiment

ROWS = 400
FEATURES = 6


def _labels(rng: np.random.Generator) -> np.ndarray:
    labels = np.zeros(ROWS, dtype=np.int64)
    labels[rng.choice(ROWS, size=140, replace=False)] = 1
    return labels


def _write_csv(path: Path, labels: np.ndarray) -> None:
    rng = np.random.default_rng(7)
    frame = pl.DataFrame(
        {
            "id": np.arange(len(labels), dtype=np.int64),
            "dur": rng.normal(size=len(labels)),
            "proto": ["tcp" if value else "udp" for value in labels],
            "sbytes": rng.integers(0, 5000, size=len(labels)).astype(np.float64),
            "label": labels,
        }
    )
    frame.write_csv(path)


def _freeze(tmp_path: Path) -> tuple[Path, np.ndarray]:
    labels = _labels(np.random.default_rng(3))
    root = tmp_path / "data"
    root.mkdir()
    _write_csv(root / TRAIN_FILE, labels)
    split = build_outer_split(labels, 0.2, seed=11)
    manifest = create_manifest(
        source_path=root / TRAIN_FILE,
        labels=labels,
        split=split,
        holdout_fraction=0.2,
        seed=11,
    )
    split_path = tmp_path / "outer.npz"
    write_split(split_path, split, manifest)
    return split_path, labels


def test_split_is_disjoint_and_stratified() -> None:
    labels = _labels(np.random.default_rng(1))
    split = build_outer_split(labels, 0.2, seed=5)

    assert np.intersect1d(split.inner, split.holdout).size == 0
    assert len(split.inner) + len(split.holdout) == ROWS
    inner_rate = float(labels[split.inner].mean())
    holdout_rate = float(labels[split.holdout].mean())
    assert abs(inner_rate - holdout_rate) < 0.05


def test_split_is_deterministic_for_one_seed() -> None:
    labels = _labels(np.random.default_rng(1))
    first = build_outer_split(labels, 0.2, seed=5)
    second = build_outer_split(labels, 0.2, seed=5)

    assert indices_sha256(first.inner) == indices_sha256(second.inner)
    assert indices_sha256(first.holdout) == indices_sha256(second.holdout)


def test_rejects_invalid_holdout_fraction() -> None:
    labels = _labels(np.random.default_rng(1))
    with pytest.raises(ConfigurationError):
        build_outer_split(labels, 1.0, seed=5)


def test_read_split_detects_tampered_indices(tmp_path: Path) -> None:
    split_path, _ = _freeze(tmp_path)
    with np.load(split_path, allow_pickle=False) as archive:
        inner = np.asarray(archive["inner_indices"], dtype=np.int64)
        holdout = np.asarray(archive["holdout_indices"], dtype=np.int64)
    np.savez_compressed(
        split_path,
        inner_indices=np.concatenate((inner, holdout[:1])),
        holdout_indices=holdout,
    )

    with pytest.raises(DatasetFileError):
        read_split(split_path)


def test_verify_source_detects_edited_csv(tmp_path: Path) -> None:
    split_path, labels = _freeze(tmp_path)
    _, manifest = read_split(split_path)
    source = tmp_path / "data" / TRAIN_FILE
    _write_csv(source, labels[::-1])

    with pytest.raises(DatasetFileError):
        verify_source(source, manifest)


def test_manifest_records_digests_and_counts(tmp_path: Path) -> None:
    split_path, labels = _freeze(tmp_path)
    payload = json.loads(manifest_path(split_path).read_text(encoding="utf-8"))

    assert payload["row_count"] == ROWS
    assert payload["inner_count"] + payload["holdout_count"] == ROWS
    assert payload["source_sha256"] == file_sha256(tmp_path / "data" / TRAIN_FILE)
    assert {label for label, _ in payload["holdout_class_counts"]} == {0, 1}
    assert sum(count for _, count in payload["inner_class_counts"]) == payload["inner_count"]
    assert len(labels) == ROWS


def test_loader_fits_only_on_inner_rows(tmp_path: Path) -> None:
    split_path, _ = _freeze(tmp_path)
    dataset, manifest = load_unsw_holdout(tmp_path / "data", split_path)

    assert len(dataset.training.labels) == manifest.inner_count
    assert len(dataset.test.labels) == manifest.holdout_count
    assert abs(float(dataset.training.features.mean())) < 1e-4
    assert abs(float(dataset.training.features.std()) - 1.0) < 1e-2


def test_loader_reads_labels_without_features(tmp_path: Path) -> None:
    split_path, labels = _freeze(tmp_path)
    assert read_split(split_path)[1].row_count == len(labels)
    assert np.array_equal(read_labels(tmp_path / "data"), labels)


def test_inner_cache_has_no_evaluation_matrix(tmp_path: Path) -> None:
    prepared = PreparedExperiment(
        raw_training=DatasetMatrix(
            features=np.zeros((5, FEATURES), dtype=np.float32),
            labels=np.zeros(5, dtype=np.int64),
        ),
        balanced_training=DatasetMatrix(
            features=np.zeros((6, FEATURES), dtype=np.float32),
            labels=np.zeros(6, dtype=np.int64),
        ),
        validation=DatasetMatrix(
            features=np.zeros((3, FEATURES), dtype=np.float32),
            labels=np.zeros(3, dtype=np.int64),
        ),
        test=DatasetMatrix(
            features=np.ones((4, FEATURES), dtype=np.float32),
            labels=np.ones(4, dtype=np.int64),
        ),
        views=((0, 1), (2, 3), (4, 5)),
        class_count=2,
    )
    path = tmp_path / "inner_seed42.npz"
    save_inner(path, prepared.inner())

    with np.load(path, allow_pickle=False) as archive:
        assert "test_x" not in archive
        assert "test_y" not in archive
    assert len(load_inner(path).validation.labels) == 3


def test_tuning_cache_loader_rejects_a_holdout_cache(tmp_path: Path) -> None:
    prepared = PreparedExperiment(
        raw_training=DatasetMatrix(
            features=np.zeros((5, FEATURES), dtype=np.float32),
            labels=np.zeros(5, dtype=np.int64),
        ),
        balanced_training=DatasetMatrix(
            features=np.zeros((6, FEATURES), dtype=np.float32),
            labels=np.zeros(6, dtype=np.int64),
        ),
        validation=DatasetMatrix(
            features=np.zeros((3, FEATURES), dtype=np.float32),
            labels=np.zeros(3, dtype=np.int64),
        ),
        test=DatasetMatrix(
            features=np.ones((4, FEATURES), dtype=np.float32),
            labels=np.ones(4, dtype=np.int64),
        ),
        views=((0, 1), (2, 3), (4, 5)),
        class_count=2,
    )
    path = tmp_path / "holdout_seed42.npz"
    save_prepared(path, prepared)

    assert len(load_prepared(path).test.labels) == 4
    with pytest.raises(DataShapeError):
        load_inner(path)
