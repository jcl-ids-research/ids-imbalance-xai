from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ids_diffusion.config import ModelConfig
from ids_diffusion.models.multiview import MultiViewEncoder, SingleViewEncoder
from ids_diffusion.types import FusionMode


def test_multiview_encoder_returns_one_embedding_per_record() -> None:
    # Given: three views spanning six input features
    config = ModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        fusion_mode=FusionMode.LINEAR,
    )
    encoder = MultiViewEncoder(config, ((0, 1), (2, 3), (4, 5)))
    batch = torch.randn(7, 6)

    # When: the batch is encoded
    result = encoder(batch)

    # Then: each record has one fixed-width embedding
    assert result.shape == (7, 16)


def test_cross_view_attention_has_the_same_output_contract() -> None:
    # Given: an attention-fusion encoder over the same three views
    config = ModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        fusion_mode=FusionMode.ATTENTION,
        fusion_heads=4,
    )
    encoder = MultiViewEncoder(config, ((0, 1), (2, 3), (4, 5)))

    # When: a batch is encoded
    result = encoder(torch.randn(5, 6))

    # Then: fusion mode does not change the classifier contract
    assert result.shape == (5, 16)


def test_single_view_encoder_matches_multiview_embedding_width() -> None:
    # Given: the matching single-view ablation
    config = ModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    encoder = SingleViewEncoder(config, input_dim=6)

    # When: a batch is encoded
    result = encoder(torch.randn(3, 6))

    # Then: the embedding is directly comparable with the multi-view variant
    assert result.shape == (3, 16)
