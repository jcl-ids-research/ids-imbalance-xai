"""Optuna study orchestration and durable result output."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import optuna

from ids_diffusion.tuning.objective import ClassifierObjective, DiffusionObjective
from ids_diffusion.tuning.search_space import TrialProtocol


@dataclass(frozen=True, slots=True)
class StudyConfig:
    """Durable study settings."""

    name: str
    trials: int
    storage: str
    output: Path
    sampler_seed: int = 42


@dataclass(frozen=True, slots=True)
class OptunaTrialAdapter(TrialProtocol):
    """Expose Optuna through the narrow typed protocol used by the objective."""

    trial: optuna.Trial | optuna.trial.FixedTrial

    def suggest_categorical(self, name: str, choices: tuple[str, ...]) -> str:
        """Suggest one string-valued option."""
        return str(self.trial.suggest_categorical(name, choices))

    def suggest_float(self, name: str, low: float, high: float, *, log: bool) -> float:
        """Suggest one continuous value."""
        return float(self.trial.suggest_float(name, low, high, log=log))

    def suggest_int(self, name: str, low: int, high: int) -> int:
        """Suggest one integer value."""
        return int(self.trial.suggest_int(name, low, high))


def run_study(
    objective: ClassifierObjective | DiffusionObjective,
    config: StudyConfig,
) -> optuna.Study:
    """Run or resume a maximisation study and persist the best configuration."""
    sampler = optuna.samplers.TPESampler(seed=config.sampler_seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=config.name,
        storage=config.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    def optuna_objective(trial: optuna.Trial) -> float:
        return objective(OptunaTrialAdapter(trial))

    study.optimize(optuna_objective, n_trials=config.trials)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "study": config.name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trial_count": len(study.trials),
        "base_config": asdict(objective.base_config),
    }
    config.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return study


__all__ = ["OptunaTrialAdapter", "StudyConfig", "run_study"]
