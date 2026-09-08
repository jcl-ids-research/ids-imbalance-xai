"""Prepare one immutable classifier cache from an official data split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer

from ids_diffusion.config import ExperimentConfig
from ids_diffusion.data.splits import stratified_validation_split
from ids_diffusion.models.diffusion import ClassConditionalDdpm
from ids_diffusion.training.balance import BalancingJob, rebalance
from ids_diffusion.training.diffusion import DiffusionTrainingJob, train_diffusion
from ids_diffusion.types import (
    DatasetMatrix,
    DatasetSplit,
    LoadedDataset,
    PreparedExperiment,
    PreparedInner,
    ViewIndices,
)


@dataclass(frozen=True, slots=True)
class PreparationJob:
    """Inputs required to generate, correct, and cache balanced training data."""

    dataset: LoadedDataset
    views: ViewIndices
    config: ExperimentConfig
    device: torch.device


@dataclass(frozen=True, slots=True)
class InnerPreparationJob:
    """Inputs required to build tuning-side arrays with no evaluation matrix."""

    training: DatasetMatrix
    views: ViewIndices
    config: ExperimentConfig
    device: torch.device


def _balanced_training(
    split: DatasetSplit,
    config: ExperimentConfig,
    device: torch.device,
) -> DatasetMatrix:
    """Fit generators on training rows only, then rebalance those rows."""
    transformer = QuantileTransformer(
        output_distribution="normal",
        random_state=config.seed,
        n_quantiles=min(1000, len(split.train.labels)),
    )
    gaussian = transformer.fit_transform(split.train.features).astype(np.float32)
    generators: dict[int, ClassConditionalDdpm] = {}
    for label in np.unique(split.train.labels).tolist():
        class_label = int(label)
        model = ClassConditionalDdpm(split.train.n_features, config.diffusion)
        generators[class_label] = train_diffusion(
            DiffusionTrainingJob(
                model=model,
                features=gaussian[split.train.labels == class_label],
                config=config.diffusion,
                device=device,
                class_label=class_label,
            )
        )
    balanced = rebalance(
        BalancingJob(
            training=split.train,
            generators=generators,
            transformer=transformer,
            config=config.training,
            device=device,
            seed=config.seed,
        )
    )
    return balanced.training


def prepare_inner(job: InnerPreparationJob) -> PreparedInner:
    """Build tuning arrays from inner rows, never touching a held-out matrix."""
    empty = DatasetMatrix(
        features=np.empty((0, job.training.n_features), dtype=np.float32),
        labels=np.empty(0, dtype=np.int64),
    )
    split = stratified_validation_split(
        training=job.training,
        test=empty,
        validation_fraction=job.config.data.validation_fraction,
        seed=job.config.seed,
    )
    return PreparedInner(
        raw_training=split.train,
        balanced_training=_balanced_training(split, job.config, job.device),
        validation=split.validation,
        views=job.views,
        class_count=len(np.unique(split.train.labels)),
    )


def prepare_experiment(job: PreparationJob) -> PreparedExperiment:
    """Fit train-only generators and return classifier-ready raw/balanced arrays."""
    split = stratified_validation_split(
        training=job.dataset.training,
        test=job.dataset.test,
        validation_fraction=job.config.data.validation_fraction,
        seed=job.config.seed,
    )
    return PreparedExperiment(
        raw_training=split.train,
        balanced_training=_balanced_training(split, job.config, job.device),
        validation=split.validation,
        test=split.test,
        views=job.views,
        class_count=len(np.unique(split.train.labels)),
    )


__all__ = [
    "InnerPreparationJob",
    "PreparationJob",
    "prepare_experiment",
    "prepare_inner",
]
