"""The unified RM-DMVT classifier pipeline and its paired ablations."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from ids_diffusion.config import ExperimentConfig
from ids_diffusion.models.classifier import IntrusionClassifier
from ids_diffusion.models.multiview import MultiViewEncoder, SingleViewEncoder
from ids_diffusion.training.classifier import (
    ClassifierTrainingJob,
    evaluate_classifier,
    train_classifier,
)
from ids_diffusion.types import PreparedExperiment, VariantMetrics


@dataclass(frozen=True, slots=True)
class VariantModels:
    """Four paired models that isolate the two components."""

    full_model: nn.Module
    without_diffusion: nn.Module
    without_multiview: nn.Module
    baseline: nn.Module


@dataclass(frozen=True, slots=True)
class PipelineJob:
    """Inputs required to train and evaluate one RM-DMVT configuration."""

    prepared: PreparedExperiment
    device: torch.device


class RankMatchedDiffusionMultiViewPipeline:
    """Construct, train, and evaluate RM-DMVT with paired ablations."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def build_models(self, prepared: PreparedExperiment) -> VariantModels:
        """Create paired initialisations for fair component comparisons."""
        multiview = IntrusionClassifier(
            MultiViewEncoder(self.config.model, prepared.views),
            self.config.model,
            prepared.class_count,
        )
        single_view = IntrusionClassifier(
            SingleViewEncoder(self.config.model, prepared.raw_training.n_features),
            self.config.model,
            prepared.class_count,
        )
        return VariantModels(
            full_model=multiview,
            without_diffusion=copy.deepcopy(multiview),
            without_multiview=single_view,
            baseline=copy.deepcopy(single_view),
        )

    def run(self, job: PipelineJob) -> VariantMetrics:
        """Train all four variants and evaluate them on the untouched real test set."""
        models = self.build_models(job.prepared)
        full = train_classifier(
            ClassifierTrainingJob(
                models.full_model,
                job.prepared.balanced_training,
                job.prepared.validation,
                self.config.training,
                job.device,
            )
        )
        without_diffusion = train_classifier(
            ClassifierTrainingJob(
                models.without_diffusion,
                job.prepared.raw_training,
                job.prepared.validation,
                self.config.training,
                job.device,
            )
        )
        without_multiview = train_classifier(
            ClassifierTrainingJob(
                models.without_multiview,
                job.prepared.balanced_training,
                job.prepared.validation,
                self.config.training,
                job.device,
            )
        )
        baseline = train_classifier(
            ClassifierTrainingJob(
                models.baseline,
                job.prepared.raw_training,
                job.prepared.validation,
                self.config.training,
                job.device,
            )
        )
        return VariantMetrics(
            full_model=evaluate_classifier(
                full, job.prepared.test, self.config.training, job.device
            ),
            without_diffusion=evaluate_classifier(
                without_diffusion, job.prepared.test, self.config.training, job.device
            ),
            without_multiview=evaluate_classifier(
                without_multiview, job.prepared.test, self.config.training, job.device
            ),
            baseline=evaluate_classifier(
                baseline, job.prepared.test, self.config.training, job.device
            ),
        )


__all__ = [
    "PipelineJob",
    "RankMatchedDiffusionMultiViewPipeline",
    "VariantModels",
]
