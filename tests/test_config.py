from __future__ import annotations

import pytest

from ids_diffusion.config import ExperimentConfig, ModelConfig
from ids_diffusion.errors import ConfigurationError
from ids_diffusion.types import FusionMode


def test_model_config_rejects_incompatible_attention_heads() -> None:
    # Given: an embedding width that cannot be split across all heads
    # When / Then: parsing the config rejects the illegal state
    with pytest.raises(ConfigurationError):
        ModelConfig(d_model=130, n_heads=8)


def test_experiment_config_applies_nested_override_immutably() -> None:
    # Given: the baseline experiment configuration
    original = ExperimentConfig()

    # When: one tuning parameter is overridden
    updated = original.with_model(
        ModelConfig(d_model=256, n_heads=8, fusion_mode=FusionMode.LINEAR)
    )

    # Then: the new config changes and the original remains untouched
    assert updated.model.d_model == 256
    assert original.model.d_model == 128
