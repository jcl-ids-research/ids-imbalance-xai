"""Classifier optimisation and evaluation."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ids_diffusion.config import TrainingConfig
from ids_diffusion.training.metrics import classification_metrics
from ids_diffusion.types import ClassificationMetrics, DatasetMatrix, LabelVector


@dataclass(frozen=True, slots=True)
class ClassifierTrainingJob:
    """Inputs required to optimise one classifier variant."""

    model: nn.Module
    training: DatasetMatrix
    validation: DatasetMatrix
    config: TrainingConfig
    device: torch.device


def _predict(
    model: nn.Module,
    dataset: DatasetMatrix,
    batch_size: int,
    device: torch.device,
) -> LabelVector:
    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for offset in range(0, len(dataset.labels), batch_size):
            features = torch.from_numpy(dataset.features[offset : offset + batch_size]).to(device)
            predictions.extend(model(features).argmax(dim=1).cpu().numpy().tolist())
    return np.asarray(predictions, dtype=np.int64)


def train_classifier(job: ClassifierTrainingJob) -> nn.Module:
    """Optimise with shuffling, cosine annealing, and validation early stopping."""
    model = job.model.to(job.device)
    criterion = nn.CrossEntropyLoss()
    optimiser = AdamW(
        model.parameters(),
        lr=job.config.learning_rate,
        weight_decay=job.config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=job.config.epochs)
    training_features = torch.from_numpy(job.training.features)
    training_labels = torch.from_numpy(job.training.labels)

    best_score = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for _ in range(job.config.epochs):
        model.train()
        order = torch.randperm(len(training_labels))
        for offset in range(0, len(order), job.config.batch_size):
            batch = order[offset : offset + job.config.batch_size]
            features = training_features[batch].to(job.device)
            labels = training_labels[batch].to(job.device)
            optimiser.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
        scheduler.step()

        predictions = _predict(
            model,
            job.validation,
            job.config.batch_size,
            job.device,
        )
        score = float(f1_score(job.validation.labels, predictions, average="macro"))
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= job.config.patience:
                break

    model.load_state_dict(best_state)
    return model


def evaluate_classifier(
    model: nn.Module,
    dataset: DatasetMatrix,
    config: TrainingConfig,
    device: torch.device,
) -> ClassificationMetrics:
    """Evaluate one trained variant on an untouched real dataset."""
    predictions = _predict(model, dataset, config.batch_size, device)
    return classification_metrics(dataset.labels, predictions)


__all__ = ["ClassifierTrainingJob", "evaluate_classifier", "train_classifier"]
