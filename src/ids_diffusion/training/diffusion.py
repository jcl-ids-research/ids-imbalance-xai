"""Optimisation of class-conditional diffusion generators."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import AdamW

from ids_diffusion.config import DiffusionConfig
from ids_diffusion.models.diffusion import ClassConditionalDdpm
from ids_diffusion.types import FloatMatrix
from ids_diffusion.utils import get_logger

LOGGER = get_logger("training.diffusion")


@dataclass(frozen=True, slots=True)
class DiffusionTrainingJob:
    """Inputs required to fit one class-conditional generator."""

    model: ClassConditionalDdpm
    features: FloatMatrix
    config: DiffusionConfig
    device: torch.device
    class_label: int


def train_diffusion(job: DiffusionTrainingJob) -> ClassConditionalDdpm:
    """Fit the denoiser with seeded shuffling and clipped gradients."""
    model = job.model.to(job.device)
    optimiser = AdamW(
        model.parameters(),
        lr=job.config.learning_rate,
        weight_decay=job.config.weight_decay,
    )
    features = torch.from_numpy(job.features).to(job.device)

    for epoch in range(1, job.config.epochs + 1):
        order = torch.randperm(len(features), device=job.device)
        total_loss = 0.0
        batch_count = 0
        for offset in range(0, len(order), job.config.batch_size):
            batch = features[order[offset : offset + job.config.batch_size]]
            optimiser.zero_grad()
            loss = model(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            total_loss += float(loss.item())
            batch_count += 1
        if epoch == 1 or epoch % 200 == 0 or epoch == job.config.epochs:
            LOGGER.info(
                "diffusion.progress",
                extra={
                    "class_label": job.class_label,
                    "epoch": epoch,
                    "epochs": job.config.epochs,
                    "mean_loss": total_loss / batch_count,
                },
            )
    return model


__all__ = ["DiffusionTrainingJob", "train_diffusion"]
