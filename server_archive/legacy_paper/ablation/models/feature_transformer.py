"""
Feature-as-Token Transformer Baseline
======================================
A proper Transformer baseline where each feature is treated as an individual token.
Unlike the StandardTransformerEncoder (which takes seq_len=1), this model:
  1. Projects each of N features independently to d_model (seq_len = N)
  2. Adds positional encoding
  3. Applies TransformerEncoder with full self-attention across features
  4. Pools to get global representation → classifier head

This is the standard "TabTransformer" style baseline the reviewer requested.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        x_norm = self.norm2(x)
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout(ffn_out)
        return x


class FeatureTransformer(nn.Module):
    """
    Feature-as-Token Transformer.

    Each feature is projected to d_model independently, forming a sequence
    of length = n_features. Full self-attention is applied across features.

    Args:
        n_features: Number of input features
        d_model: Token/hidden dimension
        nhead: Attention heads
        num_layers: Number of transformer layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        num_classes: Number of output classes
        pooling: 'mean' or 'cls' pooling
    """
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 256,
                 dropout: float = 0.1, num_classes: int = 2,
                 pooling: str = 'mean'):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.pooling = pooling

        # Project each feature independently to d_model
        self.feature_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_features)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # CLS token (optional)
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_features) - flat feature vector
        Returns: (batch, num_classes)
        """
        # Reshape: (batch, n_features) → (batch, n_features, 1)
        x = x.unsqueeze(-1)

        # Project each feature to d_model
        x = self.feature_proj(x)  # (batch, n_features, d_model)

        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1+n_features, d_model)

        x = self.pos_encoder(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Pooling
        if self.pooling == 'cls':
            x = x[:, 0, :]  # CLS token
        else:
            x = x.mean(dim=1)  # Mean pooling

        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("=" * 60)
    print("Feature Transformer - Self Test")
    print("=" * 60)

    n_features = 42
    batch_size = 8
    x = torch.randn(batch_size, n_features)

    model = FeatureTransformer(n_features=n_features, num_classes=2)
    out = model(x)
    n_params = count_parameters(model)
    print(f"Input:  ({batch_size}, {n_features})")
    print(f"Output: {out.shape}")
    print(f"Params: {n_params:,}")
    assert out.shape == (batch_size, 2), f"Unexpected output: {out.shape}"

    # Test CLS pooling
    model_cls = FeatureTransformer(n_features=n_features, num_classes=10, pooling='cls')
    out2 = model_cls(x)
    print(f"CLS pooling output: {out2.shape}")
    assert out2.shape == (batch_size, 10)

    print("\nAll tests passed!")
