"""Diffusion-assisted class balancing followed by rank-matched correction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer

from ids_diffusion.config import TrainingConfig
from ids_diffusion.models.diffusion import ClassConditionalDdpm
from ids_diffusion.training.correction import (
    correct_discrete_marginals,
    detect_discrete_columns,
)
from ids_diffusion.types import (
    BalancedDataset,
    BalanceReport,
    ClassBalance,
    DatasetMatrix,
)


@dataclass(frozen=True, slots=True)
class BalancingJob:
    """Inputs required to rebalance one training partition."""

    training: DatasetMatrix
    generators: Mapping[int, ClassConditionalDdpm]
    transformer: QuantileTransformer
    config: TrainingConfig
    device: torch.device
    seed: int


def rebalance(job: BalancingJob) -> BalancedDataset:
    """Undersample large classes and generate corrected rows for small classes."""
    generator = np.random.default_rng(job.seed)
    labels, counts = np.unique(job.training.labels, return_counts=True)
    mean_target = round(float(counts.mean()))
    discrete = detect_discrete_columns(job.training.features)
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    synthetic_parts: list[np.ndarray] = []
    synthetic_labels: list[np.ndarray] = []
    class_reports: list[ClassBalance] = []

    for label, count in zip(labels.tolist(), counts.tolist(), strict=True):
        class_label = label
        real = job.training.features[job.training.labels == class_label]
        target = min(mean_target, count * job.config.expansion_cap)
        if count >= target:
            retained = generator.choice(int(count), size=target, replace=False)
            feature_parts.append(real[retained])
            label_parts.append(np.full(target, class_label, dtype=np.int64))
            class_reports.append(
                ClassBalance(class_label, int(count), target, 0, int(count) - target)
            )
            continue

        generated_count = target - int(count)
        generated_normal = (
            job.generators[class_label]
            .sample(generated_count, job.device)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        generated = job.transformer.inverse_transform(generated_normal).astype(np.float32)
        generated = np.clip(generated, real.min(axis=0), real.max(axis=0))
        generated = correct_discrete_marginals(real, generated, discrete)
        feature_parts.append(np.concatenate((real, generated)))
        label_parts.append(np.full(target, class_label, dtype=np.int64))
        synthetic_parts.append(generated)
        synthetic_labels.append(np.full(generated_count, class_label, dtype=np.int64))
        class_reports.append(ClassBalance(class_label, int(count), target, generated_count, 0))

    features = np.concatenate(feature_parts).astype(np.float32)
    balanced_labels = np.concatenate(label_parts).astype(np.int64)
    order = generator.permutation(len(balanced_labels))
    synthetic_features = (
        np.concatenate(synthetic_parts).astype(np.float32)
        if synthetic_parts
        else np.empty((0, job.training.n_features), dtype=np.float32)
    )
    synthetic_y = (
        np.concatenate(synthetic_labels).astype(np.int64)
        if synthetic_labels
        else np.empty(0, dtype=np.int64)
    )
    after = tuple(item.after for item in class_reports)
    return BalancedDataset(
        training=DatasetMatrix(features=features[order], labels=balanced_labels[order]),
        synthetic=DatasetMatrix(features=synthetic_features, labels=synthetic_y),
        report=BalanceReport(
            target_per_class=mean_target,
            classes=tuple(class_reports),
            imbalance_before=float(counts.max() / counts.min()),
            imbalance_after=float(max(after) / min(after)),
        ),
    )


__all__ = ["BalancingJob", "rebalance"]
