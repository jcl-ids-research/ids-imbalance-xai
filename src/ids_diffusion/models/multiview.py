"""Feature-as-token encoders and fusion across views."""

from __future__ import annotations

import math

import torch
from torch import nn
from typing_extensions import assert_never

from ids_diffusion.config import ModelConfig
from ids_diffusion.errors import ViewPartitionError
from ids_diffusion.types import FusionMode, ViewIndices


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positions for feature tokens."""

    def __init__(self, width: int, maximum_length: int) -> None:
        super().__init__()
        encoding = torch.zeros(maximum_length, width)
        position = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10_000.0) / width)
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Add positions to a feature-token sequence."""
        return tokens + self.encoding[: tokens.size(1)]


class FeatureTransformer(nn.Module):
    """Encode one semantic feature group as a sequence of scalar tokens."""

    def __init__(self, config: ModelConfig, feature_count: int) -> None:
        super().__init__()
        self.project = nn.Linear(1, config.d_model)
        self.position = PositionalEncoding(config.d_model, feature_count)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # norm_first=True is intentional. Disabling nested tensors avoids a
        # PyTorch warning because that fast path cannot be used with pre-norm.
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )
        self.normalise = nn.LayerNorm(config.d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one pooled embedding per record."""
        tokens = self.project(features.unsqueeze(-1))
        encoded = self.encoder(self.position(tokens))
        return self.normalise(encoded).mean(dim=1)


class CrossViewFusion(nn.Module):
    """Exchange context among view embeddings before flattening them."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.fusion_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.normalise_attention = nn.LayerNorm(config.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.normalise_output = nn.LayerNorm(config.d_model)
        self.project = nn.Linear(config.d_model * config.n_views, config.d_model)

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        """Return cross-view attended embeddings."""
        sequence = torch.stack(views, dim=1)
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        sequence = self.normalise_attention(sequence + attended)
        sequence = self.normalise_output(sequence + self.feedforward(sequence))
        return self.project(sequence.flatten(start_dim=1))


def validate_partition(views: ViewIndices) -> int:
    """Return input width after proving the views are non-empty and exhaustive."""
    if not views or any(not view for view in views):
        raise ViewPartitionError(detail="every view must contain at least one feature")
    flattened = tuple(index for view in views for index in view)
    if len(flattened) != len(set(flattened)):
        raise ViewPartitionError(detail="a feature appears in more than one view")
    expected = tuple(range(max(flattened) + 1))
    if tuple(sorted(flattened)) != expected:
        raise ViewPartitionError(detail="views do not cover every input feature")
    return len(expected)


class MultiViewEncoder(nn.Module):
    """Encode feature groups separately, then combine their representations."""

    def __init__(self, config: ModelConfig, views: ViewIndices) -> None:
        super().__init__()
        self.input_dim = validate_partition(views)
        self.views = views
        self.encoders = nn.ModuleList(FeatureTransformer(config, len(indices)) for indices in views)
        self.fusion_mode = config.fusion_mode
        match config.fusion_mode:
            case FusionMode.LINEAR:
                self.fusion = nn.Linear(config.d_model * len(views), config.d_model)
            case FusionMode.ATTENTION:
                self.fusion = CrossViewFusion(config)
            case unreachable:
                assert_never(unreachable)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode and fuse all configured feature groups."""
        encoded = [
            encoder(features[:, indices])
            for encoder, indices in zip(self.encoders, self.views, strict=True)
        ]
        match self.fusion_mode:
            case FusionMode.LINEAR:
                return self.fusion(torch.cat(encoded, dim=-1))
            case FusionMode.ATTENTION:
                return self.fusion(encoded)
            case unreachable:
                assert_never(unreachable)


class SingleViewEncoder(nn.Module):
    """Ablation that treats every feature as one sequence."""

    def __init__(self, config: ModelConfig, input_dim: int) -> None:
        super().__init__()
        self.encoder = FeatureTransformer(config, input_dim)
        self.project = nn.Linear(config.d_model, config.d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode all features as one token sequence."""
        return self.project(self.encoder(features))


__all__ = [
    "CrossViewFusion",
    "FeatureTransformer",
    "MultiViewEncoder",
    "SingleViewEncoder",
    "validate_partition",
]
