"""Leakage-safe classifier objective for hyperparameter optimisation."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch

from ids_diffusion.config import ExperimentConfig
from ids_diffusion.models.classifier import IntrusionClassifier
from ids_diffusion.models.multiview import MultiViewEncoder
from ids_diffusion.training.classifier import (
    ClassifierTrainingJob,
    evaluate_classifier,
    train_classifier,
)
from ids_diffusion.training.prepare import PreparationJob, prepare_experiment
from ids_diffusion.tuning.search_space import (
    TrialProtocol,
    suggest_classifier_trial,
    suggest_diffusion_trial,
    suggest_narrow_classifier_trial,
)
from ids_diffusion.types import LoadedDataset, SeededPreparedInner, ViewIndices
from ids_diffusion.utils import set_seed


@dataclass(frozen=True, slots=True)
class ClassifierObjective:
    """Average validation macro F1 over cached seeds.

    The objective accepts inner caches only. Those carry no evaluation
    partition at all, so no search path can reach held-out rows even by
    mistake.
    """

    prepared_runs: tuple[SeededPreparedInner, ...]
    base_config: ExperimentConfig
    device: torch.device
    narrow: bool = False

    def __call__(self, trial: TrialProtocol) -> float:
        """Return mean validation macro F1 for one proposed configuration."""
        suggest = suggest_narrow_classifier_trial if self.narrow else suggest_classifier_trial
        config = suggest(trial, self.base_config)
        scores: list[float] = []
        for run in self.prepared_runs:
            set_seed(run.seed)
            prepared = run.prepared
            model = IntrusionClassifier(
                MultiViewEncoder(config.model, prepared.views),
                config.model,
                prepared.class_count,
            )
            trained = train_classifier(
                ClassifierTrainingJob(
                    model=model,
                    training=prepared.balanced_training,
                    validation=prepared.validation,
                    config=config.training,
                    device=self.device,
                )
            )
            metrics = evaluate_classifier(
                trained,
                prepared.validation,
                config.training,
                self.device,
            )
            scores.append(metrics.macro_f1)
        return statistics.mean(scores)


@dataclass(frozen=True, slots=True)
class DiffusionObjective:
    """Tune the generator on validation macro F1 with a fixed classifier."""

    dataset: LoadedDataset
    views: ViewIndices
    base_config: ExperimentConfig
    device: torch.device
    seeds: tuple[int, ...] = (42,)

    def __call__(self, trial: TrialProtocol) -> float:
        """Generate corrected data, train the full model, and score validation only."""
        proposed = suggest_diffusion_trial(trial, self.base_config)
        scores: list[float] = []
        for seed in self.seeds:
            set_seed(seed)
            config = proposed.with_seed(seed)
            prepared = prepare_experiment(
                PreparationJob(
                    dataset=self.dataset,
                    views=self.views,
                    config=config,
                    device=self.device,
                )
            )
            model = IntrusionClassifier(
                MultiViewEncoder(config.model, prepared.views),
                config.model,
                prepared.class_count,
            )
            trained = train_classifier(
                ClassifierTrainingJob(
                    model=model,
                    training=prepared.balanced_training,
                    validation=prepared.validation,
                    config=config.training,
                    device=self.device,
                )
            )
            metrics = evaluate_classifier(
                trained,
                prepared.validation,
                config.training,
                self.device,
            )
            scores.append(metrics.macro_f1)
        return statistics.mean(scores)


__all__ = ["ClassifierObjective", "DiffusionObjective"]
