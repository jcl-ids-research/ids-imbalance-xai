"""A constrained search space that never proposes an invalid architecture."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from typing_extensions import assert_never

from ids_diffusion.config import DiffusionConfig, ExperimentConfig, ModelConfig, TrainingConfig
from ids_diffusion.types import FusionMode

ARCHITECTURES: tuple[str, ...] = ("64x4", "128x4", "128x8", "256x8")
FUSION_MODES: tuple[str, ...] = tuple(mode.value for mode in FusionMode)
BATCH_SIZES: tuple[str, ...] = ("128", "256", "512")
DIFFUSION_HIDDEN: tuple[str, ...] = ("128,128", "256,256", "256,256,256", "512,256")
NARROW_BATCH_SIZES: tuple[str, ...] = ("128", "256")
NARROW_WIDTH = 256
NARROW_HEADS = 8


class TrialProtocol(Protocol):
    """The subset of an Optuna trial consumed by the search space."""

    def suggest_categorical(self, name: str, choices: tuple[str, ...]) -> str:
        """Return one choice."""
        ...

    def suggest_float(self, name: str, low: float, high: float, *, log: bool) -> float:
        """Return one continuous value."""
        ...

    def suggest_int(self, name: str, low: int, high: int) -> int:
        """Return one integer value."""
        ...


def _architecture(value: str) -> tuple[int, int]:
    match value:
        case "64x4":
            return 64, 4
        case "128x4":
            return 128, 4
        case "128x8":
            return 128, 8
        case "256x8":
            return 256, 8
        case unreachable:
            assert_never(unreachable)


def _fusion(value: str) -> FusionMode:
    match value:
        case "linear":
            return FusionMode.LINEAR
        case "attention":
            return FusionMode.ATTENTION
        case unreachable:
            assert_never(unreachable)


def _hidden(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(","))


def suggest_classifier_trial(
    trial: TrialProtocol,
    base: ExperimentConfig,
) -> ExperimentConfig:
    """Suggest only parameters that affect a fixed classifier cache."""
    width, heads = _architecture(trial.suggest_categorical("architecture", ARCHITECTURES))
    layers = trial.suggest_int("n_layers", 2, 6)
    feedforward_ratio = trial.suggest_int("feedforward_ratio", 2, 4)
    fusion = _fusion(trial.suggest_categorical("fusion_mode", FUSION_MODES))
    dropout = trial.suggest_float("dropout", 0.0, 0.35, log=False)

    model = ModelConfig(
        d_model=width,
        n_heads=heads,
        n_layers=layers,
        dim_feedforward=width * feedforward_ratio,
        dropout=dropout,
        n_views=base.model.n_views,
        view_mode=base.model.view_mode,
        fusion_mode=fusion,
        fusion_heads=4,
    )

    training = TrainingConfig(
        epochs=base.training.epochs,
        batch_size=int(trial.suggest_categorical("batch_size", BATCH_SIZES)),
        learning_rate=trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        patience=base.training.patience,
        expansion_cap=base.training.expansion_cap,
        balance=base.training.balance,
    )

    return replace(base, model=model, training=training)


def suggest_narrow_classifier_trial(
    trial: TrialProtocol,
    base: ExperimentConfig,
) -> ExperimentConfig:
    """Search only the region that held up across seeds in the first study.

    Width, head count, and fusion are fixed to the configuration that survived
    multi-seed confirmation, so the budget is spent on depth, capacity, and
    optimisation rather than on re-exploring settled choices.
    """
    layers = trial.suggest_int("n_layers", 2, 3)
    feedforward_ratio = trial.suggest_int("feedforward_ratio", 3, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.08, log=False)

    model = ModelConfig(
        d_model=NARROW_WIDTH,
        n_heads=NARROW_HEADS,
        n_layers=layers,
        dim_feedforward=NARROW_WIDTH * feedforward_ratio,
        dropout=dropout,
        n_views=base.model.n_views,
        view_mode=base.model.view_mode,
        fusion_mode=FusionMode.LINEAR,
        fusion_heads=base.model.fusion_heads,
    )

    training = TrainingConfig(
        epochs=base.training.epochs,
        batch_size=int(trial.suggest_categorical("batch_size", NARROW_BATCH_SIZES)),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 2.2e-4, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True),
        patience=base.training.patience,
        expansion_cap=base.training.expansion_cap,
        balance=base.training.balance,
    )

    return replace(base, model=model, training=training)


def suggest_diffusion_trial(
    trial: TrialProtocol,
    base: ExperimentConfig,
) -> ExperimentConfig:
    """Suggest generator parameters while holding the classifier fixed."""
    diffusion = DiffusionConfig(
        timesteps=trial.suggest_int("diffusion_timesteps", 300, 700),
        beta_start=base.diffusion.beta_start,
        beta_end=base.diffusion.beta_end,
        hidden=_hidden(trial.suggest_categorical("diffusion_hidden", DIFFUSION_HIDDEN)),
        epochs=base.diffusion.epochs,
        batch_size=base.diffusion.batch_size,
        learning_rate=trial.suggest_float(
            "diffusion_learning_rate",
            3e-4,
            2e-3,
            log=True,
        ),
        weight_decay=base.diffusion.weight_decay,
        quantile_transform=base.diffusion.quantile_transform,
        discrete_correction=base.diffusion.discrete_correction,
    )
    expansion_cap = trial.suggest_int("expansion_cap", 5, 20)
    training = replace(base.training, expansion_cap=expansion_cap)
    return replace(base, diffusion=diffusion, training=training)


def suggest_trial(trial: TrialProtocol, base: ExperimentConfig) -> ExperimentConfig:
    """Suggest a joint configuration for final narrowed-range experiments."""
    classifier = suggest_classifier_trial(trial, base)
    return suggest_diffusion_trial(trial, classifier)


__all__ = [
    "TrialProtocol",
    "suggest_classifier_trial",
    "suggest_diffusion_trial",
    "suggest_narrow_classifier_trial",
    "suggest_trial",
]
