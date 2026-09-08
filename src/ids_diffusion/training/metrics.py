"""Metric computation with the exact definitions used in the paper."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ids_diffusion.errors import DataShapeError
from ids_diffusion.types import ClassificationMetrics, ClassMetrics, LabelVector


def _per_class(labels: LabelVector, predictions: LabelVector) -> tuple[ClassMetrics, ...]:
    """Score every class separately so minority behaviour stays visible."""
    present = np.unique(np.concatenate((labels, predictions)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=present,
        average=None,
        zero_division=0,
    )
    return tuple(
        ClassMetrics(
            label=int(label),
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, label in enumerate(present.tolist())
    )


def classification_metrics(
    labels: LabelVector,
    predictions: LabelVector,
) -> ClassificationMetrics:
    """Compute accuracy, weighted/macro averages, and per-class detail."""
    precision, recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return ClassificationMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision),
        recall=float(recall),
        weighted_f1=float(weighted_f1),
        macro_f1=float(macro_f1),
        per_class=_per_class(labels, predictions),
    )


def mean_metrics(metrics: tuple[ClassificationMetrics, ...]) -> ClassificationMetrics:
    """Average repeated runs without hiding the per-seed records."""
    if not metrics:
        raise DataShapeError(detail="at least one metric record is required")
    return ClassificationMetrics(
        accuracy=float(np.mean([item.accuracy for item in metrics])),
        precision=float(np.mean([item.precision for item in metrics])),
        recall=float(np.mean([item.recall for item in metrics])),
        weighted_f1=float(np.mean([item.weighted_f1 for item in metrics])),
        macro_f1=float(np.mean([item.macro_f1 for item in metrics])),
        per_class=_mean_per_class(metrics),
    )


def _mean_per_class(metrics: tuple[ClassificationMetrics, ...]) -> tuple[ClassMetrics, ...]:
    """Average per-class scores across runs that share the same label set."""
    labels = sorted({entry.label for item in metrics for entry in item.per_class})
    averaged: list[ClassMetrics] = []
    for label in labels:
        entries = [entry for item in metrics for entry in item.per_class if entry.label == label]
        if not entries:
            continue
        averaged.append(
            ClassMetrics(
                label=label,
                precision=float(np.mean([entry.precision for entry in entries])),
                recall=float(np.mean([entry.recall for entry in entries])),
                f1=float(np.mean([entry.f1 for entry in entries])),
                support=int(np.mean([entry.support for entry in entries])),
            )
        )
    return tuple(averaged)


__all__ = ["classification_metrics", "mean_metrics"]
