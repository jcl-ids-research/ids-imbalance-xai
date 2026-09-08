"""CICFlowMeter loading for CIC-IDS-2017 and CIC-DDoS2019.

Neither dataset ships an official split, so one stratified split is drawn per
seed and every encoder, imputer and scaler is then fitted on the training
partition alone. The CSVs are time ordered and attacks occupy specific windows,
so a bounded read per file is followed by a stratified subsample rather than a
head truncation, which would return almost pure benign traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from ids_diffusion.config import TaskName
from ids_diffusion.data.tabular import encode_frames
from ids_diffusion.errors import ConfigurationError, DatasetFileError
from ids_diffusion.types import DatasetMatrix, LoadedDataset

ROWS_PER_FILE = 50_000
READ_CAP = 500_000
SAMPLE_CAP = 200_000
TEST_FRACTION = 0.2
MINIMUM_CLASSES = 2
BENIGN_LABEL = "BENIGN"


@dataclass(frozen=True, slots=True)
class CicReadLimits:
    """Bounded read and subsample sizes for a CICFlowMeter release."""

    sample_cap: int = SAMPLE_CAP
    rows_per_file: int = ROWS_PER_FILE
    read_cap: int = READ_CAP


DROPPED_COLUMNS = frozenset(
    {
        "unnamed: 0",
        "flow id",
        "source ip",
        "src ip",
        "source port",
        "src port",
        "destination ip",
        "dst ip",
        "destination port",
        "dst port",
        "protocol",
        "timestamp",
        "simillarhttp",
        "inbound",
        "fwd header length.1",
    }
)


def _read_directory(root: Path, rows_per_file: int, read_cap: int) -> pl.DataFrame:
    """Read a bounded number of rows from each CSV and normalise column names."""
    paths = sorted(root.glob("*.csv"))
    if not paths:
        raise DatasetFileError(path=str(root), detail="no CSV files found")

    frames: list[pl.DataFrame] = []
    total = 0
    for path in paths:
        frame = pl.read_csv(path, n_rows=rows_per_file, infer_schema_length=10_000)
        frame = frame.rename({name: name.strip() for name in frame.columns})
        label_column = next((name for name in frame.columns if name.lower() == "label"), None)
        if label_column is None:
            continue
        frames.append(frame.rename({label_column: "Label"}))
        total += frame.height
        if total >= read_cap:
            break

    if not frames:
        raise DatasetFileError(path=str(root), detail="no CSV file contains a label column")
    return pl.concat(frames, how="vertical_relaxed")


def _binary_labels(frame: pl.DataFrame) -> np.ndarray:
    """Collapse every attack family to the positive class."""
    return np.asarray(
        [
            0 if str(value).strip().upper() == BENIGN_LABEL else 1
            for value in frame.get_column("Label").to_list()
        ],
        dtype=np.int64,
    )


def _subsample(
    frame: pl.DataFrame,
    labels: np.ndarray,
    cap: int,
    seed: int,
) -> tuple[pl.DataFrame, np.ndarray]:
    """Reduce the pool to the configured cap while preserving class shares."""
    if frame.height <= cap or len(np.unique(labels)) < MINIMUM_CLASSES:
        return frame, labels
    keep, _ = train_test_split(
        np.arange(frame.height),
        train_size=cap,
        random_state=seed,
        stratify=labels,
    )
    ordered = np.sort(np.asarray(keep, dtype=np.int64))
    return frame[ordered], labels[ordered]


def load_cic(
    root: Path,
    seed: int,
    task: TaskName = "binary",
    limits: CicReadLimits | None = None,
) -> LoadedDataset:
    """Load one CICFlowMeter directory into a seeded, train-only fitted split."""
    if task != "binary":
        raise ConfigurationError(
            field="task",
            detail="CIC datasets are reported as binary detection only",
        )

    bounds = limits or CicReadLimits()
    frame = _read_directory(root, rows_per_file=bounds.rows_per_file, read_cap=bounds.read_cap)
    labels = _binary_labels(frame)
    frame, labels = _subsample(frame, labels, cap=bounds.sample_cap, seed=seed)

    feature_names = tuple(
        name for name in frame.columns if name != "Label" and name.lower() not in DROPPED_COLUMNS
    )
    if not feature_names:
        raise DatasetFileError(path=str(root), detail="no usable feature columns remain")

    training_index, test_index = train_test_split(
        np.arange(frame.height),
        test_size=TEST_FRACTION,
        random_state=seed,
        stratify=labels,
    )
    training_rows = np.asarray(training_index, dtype=np.int64)
    test_rows = np.asarray(test_index, dtype=np.int64)
    encoded = encode_frames(frame[training_rows], frame[test_rows], feature_names)
    return LoadedDataset(
        training=DatasetMatrix(features=encoded.training, labels=labels[training_rows]),
        test=DatasetMatrix(features=encoded.test, labels=labels[test_rows]),
        feature_names=encoded.feature_names,
    )


__all__ = ["DROPPED_COLUMNS", "READ_CAP", "ROWS_PER_FILE", "CicReadLimits", "load_cic"]
