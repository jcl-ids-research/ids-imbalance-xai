"""Typed errors raised by the training pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConfigurationError(Exception):
    """A configuration value violates a model invariant."""

    field: str
    detail: str

    def __str__(self) -> str:
        return f"invalid {self.field}: {self.detail}"


@dataclass(frozen=True, slots=True)
class DataShapeError(Exception):
    """Feature and label arrays cannot describe the same dataset."""

    detail: str

    def __str__(self) -> str:
        return f"invalid dataset shape: {self.detail}"


@dataclass(frozen=True, slots=True)
class ViewPartitionError(Exception):
    """A feature partition is empty, overlapping, or incomplete."""

    detail: str

    def __str__(self) -> str:
        return f"invalid view partition: {self.detail}"


@dataclass(frozen=True, slots=True)
class DatasetFileError(Exception):
    """A required public dataset file is missing or malformed."""

    path: str
    detail: str

    def __str__(self) -> str:
        return f"dataset file {self.path!r}: {self.detail}"


__all__ = [
    "ConfigurationError",
    "DataShapeError",
    "DatasetFileError",
    "ViewPartitionError",
]
