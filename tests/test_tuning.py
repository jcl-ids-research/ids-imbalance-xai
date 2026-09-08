from __future__ import annotations

from dataclasses import dataclass

from ids_diffusion.config import ExperimentConfig
from ids_diffusion.tuning.search_space import suggest_narrow_classifier_trial, suggest_trial
from ids_diffusion.types import FusionMode


@dataclass(frozen=True, slots=True)
class FixedTrial:
    categorical: dict[str, str]
    floats: dict[str, float]
    integers: dict[str, int]

    def suggest_categorical(self, name: str, choices: tuple[str, ...]) -> str:
        value = self.categorical[name]
        assert value in choices
        return value

    def suggest_float(self, name: str, low: float, high: float, *, log: bool) -> float:
        value = self.floats[name]
        assert low <= value <= high
        if name != "dropout":
            assert log
        return value

    def suggest_int(self, name: str, low: int, high: int) -> int:
        value = self.integers[name]
        assert low <= value <= high
        return value


def test_search_space_returns_valid_joint_configuration() -> None:
    # Given: fixed choices standing in for an Optuna trial
    trial = FixedTrial(
        categorical={
            "architecture": "128x8",
            "fusion_mode": "linear",
            "batch_size": "256",
            "diffusion_hidden": "256,256,256",
        },
        floats={
            "dropout": 0.15,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "diffusion_learning_rate": 8e-4,
        },
        integers={
            "n_layers": 3,
            "feedforward_ratio": 4,
            "diffusion_timesteps": 500,
            "expansion_cap": 15,
        },
    )

    # When: the search space is applied to the baseline configuration
    result = suggest_trial(trial, ExperimentConfig())

    # Then: the generated model is valid and all three concerns are updated
    assert result.model.d_model == 128
    assert result.model.n_heads == 8
    assert result.model.fusion_mode is FusionMode.LINEAR
    assert result.training.batch_size == 256
    assert result.diffusion.timesteps == 500


def test_narrow_space_pins_the_confirmed_architecture() -> None:
    # Given: a trial that only supplies the narrow parameters
    trial = FixedTrial(
        categorical={"batch_size": "128"},
        floats={"dropout": 0.02, "learning_rate": 1.5e-4, "weight_decay": 1.6e-4},
        integers={"n_layers": 2, "feedforward_ratio": 4},
    )

    # When: the narrow space is applied
    result = suggest_narrow_classifier_trial(trial, ExperimentConfig())

    # Then: settled choices are fixed and only the open ones move
    assert result.model.d_model == 256
    assert result.model.n_heads == 8
    assert result.model.fusion_mode is FusionMode.LINEAR
    assert result.model.dim_feedforward == 1024
    assert result.training.batch_size == 128
