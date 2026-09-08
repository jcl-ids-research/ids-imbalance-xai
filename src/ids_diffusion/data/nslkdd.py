"""NSL-KDD loading on the publisher's official KDDTrain+/KDDTest+ split.

The two files are never merged. The test set deliberately contains attack names
that are absent from training, which is the property that makes the benchmark
hard, so unseen names are mapped through the standard KDD families rather than
being silently bucketed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ids_diffusion.config import TaskName
from ids_diffusion.data.tabular import encode_frames
from ids_diffusion.errors import DatasetFileError
from ids_diffusion.types import DatasetMatrix, LoadedDataset

TRAIN_FILE = "KDDTrain+.txt"
TEST_FILE = "KDDTest+.txt"

NSL_COLUMNS: tuple[str, ...] = (
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
)

NON_FEATURE_COLUMNS = frozenset({"label", "difficulty"})

NSL_SEMANTIC_VIEWS: tuple[tuple[str, ...], ...] = (
    (
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
    ),
    (
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
    ),
    (
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ),
)

NSL_CLASSES: tuple[str, ...] = ("Normal", "DoS", "Probe", "R2L", "U2R")

NSL_CATEGORY: dict[str, str] = {
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "apache2": "DoS",
    "udpstorm": "DoS",
    "processtable": "DoS",
    "mailbomb": "DoS",
    "satan": "Probe",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    "guess_passwd": "R2L",
    "ftp_write": "R2L",
    "imap": "R2L",
    "phf": "R2L",
    "multihop": "R2L",
    "warezmaster": "R2L",
    "warezclient": "R2L",
    "spy": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "snmpguess": "R2L",
    "snmpgetattack": "R2L",
    "httptunnel": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    "worm": "R2L",
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "rootkit": "U2R",
    "perl": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    "ps": "U2R",
    "normal": "Normal",
}


def _attack_names(frame: pl.DataFrame) -> list[str]:
    """Return normalised attack names for one partition."""
    return [str(value).strip().rstrip(".") for value in frame.get_column("label").to_list()]


def _binary_labels(frame: pl.DataFrame) -> np.ndarray:
    """Map every non-normal record to the attack class."""
    return np.asarray(
        [0 if name == "normal" else 1 for name in _attack_names(frame)],
        dtype=np.int64,
    )


def _multiclass_labels(frame: pl.DataFrame, path: Path) -> np.ndarray:
    """Map attack names onto the five standard families, refusing to guess."""
    names = _attack_names(frame)
    unknown = sorted({name for name in names if name not in NSL_CATEGORY})
    if unknown:
        raise DatasetFileError(path=str(path), detail=f"unmapped attack names: {unknown}")
    index = {family: position for position, family in enumerate(NSL_CLASSES)}
    return np.asarray([index[NSL_CATEGORY[name]] for name in names], dtype=np.int64)


def _read_partition(path: Path) -> pl.DataFrame:
    """Read one headerless NSL-KDD partition with the documented schema."""
    if not path.is_file():
        raise DatasetFileError(path=str(path), detail="required file not found")
    frame = pl.read_csv(path, has_header=False, new_columns=list(NSL_COLUMNS))
    if frame.width != len(NSL_COLUMNS):
        raise DatasetFileError(
            path=str(path),
            detail=f"expected {len(NSL_COLUMNS)} columns, found {frame.width}",
        )
    return frame


def load_nslkdd(root: Path, task: TaskName = "binary") -> LoadedDataset:
    """Load the official split with every statistic fitted on training only."""
    training_path = root / TRAIN_FILE
    test_path = root / TEST_FILE
    training_frame = _read_partition(training_path)
    test_frame = _read_partition(test_path)

    if task == "multiclass":
        training_labels = _multiclass_labels(training_frame, training_path)
        test_labels = _multiclass_labels(test_frame, test_path)
    else:
        training_labels = _binary_labels(training_frame)
        test_labels = _binary_labels(test_frame)

    feature_names = tuple(name for name in NSL_COLUMNS if name not in NON_FEATURE_COLUMNS)
    encoded = encode_frames(training_frame, test_frame, feature_names)
    return LoadedDataset(
        training=DatasetMatrix(features=encoded.training, labels=training_labels),
        test=DatasetMatrix(features=encoded.test, labels=test_labels),
        feature_names=encoded.feature_names,
    )


__all__ = [
    "NSL_CATEGORY",
    "NSL_CLASSES",
    "NSL_COLUMNS",
    "NSL_SEMANTIC_VIEWS",
    "TEST_FILE",
    "TRAIN_FILE",
    "load_nslkdd",
]
