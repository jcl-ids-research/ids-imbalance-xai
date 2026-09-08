"""Frozen outer holdout split with a verifiable manifest.

The outer holdout is drawn once from the official UNSW training file, hashed,
and never read by tuning code. Every consumer re-verifies the recorded digests
before use, so a silently edited split fails loudly instead of leaking.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from ids_diffusion.errors import ConfigurationError, DatasetFileError
from ids_diffusion.types import IndexVector, LabelVector

READ_CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class OuterSplit:
    """The two disjoint index sets produced by one freeze."""

    inner: IndexVector
    holdout: IndexVector


@dataclass(frozen=True, slots=True)
class OuterSplitManifest:
    """Everything needed to prove which rows were frozen out of tuning."""

    seed: int
    holdout_fraction: float
    source_file: str
    source_sha256: str
    row_count: int
    inner_count: int
    holdout_count: int
    inner_class_counts: tuple[tuple[int, int], ...]
    holdout_class_counts: tuple[tuple[int, int], ...]
    inner_indices_sha256: str
    holdout_indices_sha256: str


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file read in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def indices_sha256(indices: IndexVector) -> str:
    """Return the SHA-256 digest of an index vector in canonical int64 form."""
    return hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _class_counts(labels: LabelVector) -> tuple[tuple[int, int], ...]:
    values, counts = np.unique(labels, return_counts=True)
    return tuple(
        (int(value), int(count))
        for value, count in zip(values.tolist(), counts.tolist(), strict=True)
    )


def build_outer_split(
    labels: LabelVector,
    holdout_fraction: float,
    seed: int,
) -> OuterSplit:
    """Draw a stratified outer holdout and return sorted inner/holdout positions."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ConfigurationError(
            field="holdout_fraction",
            detail=f"expected (0, 1), got {holdout_fraction}",
        )
    positions = np.arange(len(labels), dtype=np.int64)
    inner, holdout = train_test_split(
        positions,
        test_size=holdout_fraction,
        random_state=seed,
        stratify=labels,
    )
    return OuterSplit(
        inner=np.sort(np.asarray(inner, dtype=np.int64)),
        holdout=np.sort(np.asarray(holdout, dtype=np.int64)),
    )


def create_manifest(
    source_path: Path,
    labels: LabelVector,
    split: OuterSplit,
    holdout_fraction: float,
    seed: int,
) -> OuterSplitManifest:
    """Describe one frozen split, including dataset and index digests."""
    return OuterSplitManifest(
        seed=seed,
        holdout_fraction=holdout_fraction,
        source_file=source_path.name,
        source_sha256=file_sha256(source_path),
        row_count=len(labels),
        inner_count=len(split.inner),
        holdout_count=len(split.holdout),
        inner_class_counts=_class_counts(labels[split.inner]),
        holdout_class_counts=_class_counts(labels[split.holdout]),
        inner_indices_sha256=indices_sha256(split.inner),
        holdout_indices_sha256=indices_sha256(split.holdout),
    )


def manifest_path(path: Path) -> Path:
    """Return the JSON manifest path that accompanies a split archive."""
    return path.with_suffix(".manifest.json")


def write_split(path: Path, split: OuterSplit, manifest: OuterSplitManifest) -> None:
    """Persist split indices plus a human-readable manifest beside them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        inner_indices=np.asarray(split.inner, dtype=np.int64),
        holdout_indices=np.asarray(split.holdout, dtype=np.int64),
    )
    manifest_path(path).write_text(
        json.dumps(asdict(manifest), indent=2),
        encoding="utf-8",
    )


def _manifest_from_payload(payload: dict[str, object]) -> OuterSplitManifest:
    def counts(key: str) -> tuple[tuple[int, int], ...]:
        raw = payload[key]
        if not isinstance(raw, list):
            raise DatasetFileError(path=key, detail="class counts must be a list")
        return tuple((int(item[0]), int(item[1])) for item in raw)

    return OuterSplitManifest(
        seed=int(str(payload["seed"])),
        holdout_fraction=float(str(payload["holdout_fraction"])),
        source_file=str(payload["source_file"]),
        source_sha256=str(payload["source_sha256"]),
        row_count=int(str(payload["row_count"])),
        inner_count=int(str(payload["inner_count"])),
        holdout_count=int(str(payload["holdout_count"])),
        inner_class_counts=counts("inner_class_counts"),
        holdout_class_counts=counts("holdout_class_counts"),
        inner_indices_sha256=str(payload["inner_indices_sha256"]),
        holdout_indices_sha256=str(payload["holdout_indices_sha256"]),
    )


def read_split(path: Path) -> tuple[OuterSplit, OuterSplitManifest]:
    """Load a frozen split and fail if any recorded digest no longer matches."""
    descriptor = manifest_path(path)
    if not path.is_file() or not descriptor.is_file():
        raise DatasetFileError(path=str(path), detail="split archive or manifest not found")
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    manifest = _manifest_from_payload(payload)
    with np.load(path, allow_pickle=False) as archive:
        inner = np.asarray(archive["inner_indices"], dtype=np.int64)
        holdout = np.asarray(archive["holdout_indices"], dtype=np.int64)
    if indices_sha256(inner) != manifest.inner_indices_sha256:
        raise DatasetFileError(path=str(path), detail="inner indices do not match the manifest")
    if indices_sha256(holdout) != manifest.holdout_indices_sha256:
        raise DatasetFileError(path=str(path), detail="holdout indices do not match the manifest")
    if np.intersect1d(inner, holdout).size:
        raise DatasetFileError(path=str(path), detail="inner and holdout indices overlap")
    if len(inner) + len(holdout) != manifest.row_count:
        raise DatasetFileError(path=str(path), detail="split does not cover every source row")
    return OuterSplit(inner=inner, holdout=holdout), manifest


def verify_source(path: Path, manifest: OuterSplitManifest) -> None:
    """Prove the CSV behind a frozen split has not changed since it was drawn."""
    digest = file_sha256(path)
    if digest != manifest.source_sha256:
        raise DatasetFileError(
            path=str(path),
            detail=f"source digest {digest} does not match manifest {manifest.source_sha256}",
        )


__all__ = [
    "OuterSplit",
    "OuterSplitManifest",
    "build_outer_split",
    "create_manifest",
    "file_sha256",
    "indices_sha256",
    "manifest_path",
    "read_split",
    "verify_source",
    "write_split",
]
