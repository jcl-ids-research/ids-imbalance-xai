"""Domain types shared by data, models, training, and tuning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from ids_diffusion.errors import DataShapeError

FloatMatrix: TypeAlias = npt.NDArray[np.float32]
LabelVector: TypeAlias = npt.NDArray[np.int64]
IndexVector: TypeAlias = npt.NDArray[np.int64]
BooleanVector: TypeAlias = npt.NDArray[np.bool_]
ViewIndices: TypeAlias = tuple[tuple[int, ...], ...]
MATRIX_DIMENSIONS = 2


class FusionMode(str, Enum):
    """How per-view embeddings are combined."""

    LINEAR = "linear"
    ATTENTION = "attention"


class DeviceChoice(str, Enum):
    """Compute device selected at the CLI boundary."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class ViewMode(str, Enum):
    """How feature columns are assigned to views."""

    SEMANTIC = "semantic"
    POSITIONAL = "positional"


class VariantName(str, Enum):
    """The four ablation variants used throughout the paper."""

    FULL = "full_model"
    WITHOUT_DIFFUSION = "wo_diffusion"
    WITHOUT_MULTIVIEW = "wo_multiview"
    BASELINE = "baseline"


@dataclass(frozen=True, slots=True)
class DatasetMatrix:
    """A feature matrix and its labels, with shape agreement guaranteed."""

    features: FloatMatrix
    labels: LabelVector

    def __post_init__(self) -> None:
        if self.features.ndim != MATRIX_DIMENSIONS:
            raise DataShapeError(detail=f"features must be 2D, got {self.features.ndim}D")
        if self.labels.ndim != 1:
            raise DataShapeError(detail=f"labels must be 1D, got {self.labels.ndim}D")
        if len(self.features) != len(self.labels):
            raise DataShapeError(
                detail=f"{len(self.features)} feature rows but {len(self.labels)} labels"
            )

    @property
    def n_features(self) -> int:
        """Return the feature width."""
        return int(self.features.shape[1])


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Train, validation, and untouched real test partitions."""

    train: DatasetMatrix
    validation: DatasetMatrix
    test: DatasetMatrix


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """Official training and real test partitions with stable feature names."""

    training: DatasetMatrix
    test: DatasetMatrix
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.training.n_features != self.test.n_features:
            raise DataShapeError(detail="training and test feature counts differ")
        if len(self.feature_names) != self.training.n_features:
            raise DataShapeError(detail="feature names do not match the matrix width")


@dataclass(frozen=True, slots=True)
class ShiftReport:
    """What training-side thinning changed."""

    keep_fraction: float
    attacks_before: int
    attacks_after: int
    normals: int
    attack_share_before: float
    attack_share_after: float


@dataclass(frozen=True, slots=True)
class ClassBalance:
    """How one class was moved toward the common training target."""

    label: int
    before: int
    after: int
    generated: int
    discarded: int


@dataclass(frozen=True, slots=True)
class BalanceReport:
    """Class counts before and after diffusion-assisted balancing."""

    target_per_class: int
    classes: tuple[ClassBalance, ...]
    imbalance_before: float
    imbalance_after: float


@dataclass(frozen=True, slots=True)
class BalancedDataset:
    """Balanced training rows plus the generated subset for fidelity analysis."""

    training: DatasetMatrix
    synthetic: DatasetMatrix
    report: BalanceReport


@dataclass(frozen=True, slots=True)
class PreparedInner:
    """Tuning-side arrays only.

    This type deliberately has no evaluation partition. Hyperparameter search
    consumes it, so no search code path can reach a held-out matrix.
    """

    raw_training: DatasetMatrix
    balanced_training: DatasetMatrix
    validation: DatasetMatrix
    views: ViewIndices
    class_count: int


@dataclass(frozen=True, slots=True)
class PreparedExperiment:
    """Arrays consumed repeatedly by classifier trials."""

    raw_training: DatasetMatrix
    balanced_training: DatasetMatrix
    validation: DatasetMatrix
    test: DatasetMatrix
    views: ViewIndices
    class_count: int

    def inner(self) -> PreparedInner:
        """Drop the evaluation partition before handing arrays to a search."""
        return PreparedInner(
            raw_training=self.raw_training,
            balanced_training=self.balanced_training,
            validation=self.validation,
            views=self.views,
            class_count=self.class_count,
        )


@dataclass(frozen=True, slots=True)
class SeededPreparedInner:
    """One tuning-side cache paired with the seed that produced it."""

    seed: int
    prepared: PreparedInner


@dataclass(frozen=True, slots=True)
class SeededPreparedExperiment:
    """One cached dataset paired with the seed that produced it."""

    seed: int
    prepared: PreparedExperiment


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Per-class precision, recall, and F1 with the support behind them.

    Averaged scores hide what an intrusion detector is judged on: whether the
    rare attack class is actually caught. These are reported per class so a
    minority-class regression cannot be masked by the majority.
    """

    label: int
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Metrics reported for every model variant."""

    accuracy: float
    precision: float
    recall: float
    weighted_f1: float
    macro_f1: float
    per_class: tuple[ClassMetrics, ...] = ()


@dataclass(frozen=True, slots=True)
class VariantMetrics:
    """Metrics for the four paired ablation variants."""

    full_model: ClassificationMetrics
    without_diffusion: ClassificationMetrics
    without_multiview: ClassificationMetrics
    baseline: ClassificationMetrics


__all__ = [
    "BalanceReport",
    "BalancedDataset",
    "BooleanVector",
    "ClassBalance",
    "ClassMetrics",
    "ClassificationMetrics",
    "DatasetMatrix",
    "DatasetSplit",
    "DeviceChoice",
    "FloatMatrix",
    "FusionMode",
    "IndexVector",
    "LabelVector",
    "LoadedDataset",
    "PreparedExperiment",
    "PreparedInner",
    "SeededPreparedExperiment",
    "SeededPreparedInner",
    "ShiftReport",
    "VariantMetrics",
    "VariantName",
    "ViewIndices",
    "ViewMode",
]
