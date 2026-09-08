"""Classifier head shared by full and ablation variants."""

from __future__ import annotations

import torch
from torch import nn

from ids_diffusion.config import ModelConfig


class IntrusionClassifier(nn.Module):
    """Map an encoder embedding to binary or multi-class logits."""

    def __init__(self, encoder: nn.Module, config: ModelConfig, class_count: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(config.d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, class_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch."""
        return self.head(self.encoder(features))


__all__ = ["IntrusionClassifier"]
