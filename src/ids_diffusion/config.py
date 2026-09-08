"""Immutable configuration for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from ids_diffusion.errors import ConfigurationError
from ids_diffusion.types import FusionMode, ViewMode

DatasetName = Literal["unsw", "nslkdd", "cicids2017", "cicddos2019"]
TaskName = Literal["binary", "multiclass"]


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Dataset location and split controls."""

    dataset: DatasetName = "unsw"
    task: TaskName = "binary"
    root: Path = Path("/opt/UNSW-NB15")
    validation_fraction: float = 0.1
    sample_cap: int = 200_000
    attack_keep_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ConfigurationError(
                field="validation_fraction",
                detail=f"expected (0, 1), got {self.validation_fraction}",
            )
        if not 0.0 < self.attack_keep_fraction <= 1.0:
            raise ConfigurationError(
                field="attack_keep_fraction",
                detail=f"expected (0, 1], got {self.attack_keep_fraction}",
            )


@dataclass(frozen=True, slots=True)
class DiffusionConfig:
    """Class-conditional DDPM and post-generation correction."""

    timesteps: int = 500
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden: tuple[int, ...] = (256, 256, 256)
    epochs: int = 1000
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    quantile_transform: bool = True
    discrete_correction: bool = True

    def __post_init__(self) -> None:
        if self.timesteps < 1:
            raise ConfigurationError(field="timesteps", detail="must be positive")
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ConfigurationError(
                field="beta schedule",
                detail=(
                    "expected 0 < beta_start < beta_end < 1, "
                    f"got {self.beta_start} and {self.beta_end}"
                ),
            )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Multi-view Transformer architecture."""

    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    n_views: int = 3
    view_mode: ViewMode = ViewMode.SEMANTIC
    fusion_mode: FusionMode = FusionMode.LINEAR
    fusion_heads: int = 4

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ConfigurationError(
                field="n_heads",
                detail=f"{self.n_heads} does not divide d_model={self.d_model}",
            )
        if self.fusion_mode is FusionMode.ATTENTION and self.d_model % self.fusion_heads:
            raise ConfigurationError(
                field="fusion_heads",
                detail=f"{self.fusion_heads} does not divide d_model={self.d_model}",
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigurationError(
                field="dropout",
                detail=f"expected [0, 1), got {self.dropout}",
            )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Classifier optimisation and balancing controls."""

    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 5
    expansion_cap: int = 15
    balance: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ConfigurationError(field="epochs", detail="must be positive")
        if self.patience < 1:
            raise ConfigurationError(field="patience", detail="must be positive")
        if self.expansion_cap < 1:
            raise ConfigurationError(field="expansion_cap", detail="must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Every setting needed to reproduce one run."""

    name: str = "default"
    seed: int = 42
    output_root: Path = Path("runs")
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def run_dir(self) -> Path:
        """Return the seed-specific output directory."""
        return self.output_root / self.name / f"seed{self.seed}"

    def with_seed(self, seed: int) -> ExperimentConfig:
        """Return a copy for another reproducibility seed."""
        return replace(self, seed=seed)

    def with_model(self, model: ModelConfig) -> ExperimentConfig:
        """Return a copy with another classifier architecture."""
        return replace(self, model=model)

    def with_data(self, data: DataConfig) -> ExperimentConfig:
        """Return a copy with another dataset boundary."""
        return replace(self, data=data)

    def with_diffusion(self, diffusion: DiffusionConfig) -> ExperimentConfig:
        """Return a copy with another generator configuration."""
        return replace(self, diffusion=diffusion)

    def with_training(self, training: TrainingConfig) -> ExperimentConfig:
        """Return a copy with another optimisation configuration."""
        return replace(self, training=training)


__all__ = [
    "DataConfig",
    "DatasetName",
    "DiffusionConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TaskName",
    "TrainingConfig",
]
