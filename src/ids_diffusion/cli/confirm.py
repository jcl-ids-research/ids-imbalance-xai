"""Confirm one ranked Optuna configuration across multiple prepared seeds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import optuna
import typer

from ids_diffusion.cli.common import load_seeded_caches, parse_device
from ids_diffusion.config import ExperimentConfig
from ids_diffusion.errors import ConfigurationError
from ids_diffusion.tuning.objective import ClassifierObjective
from ids_diffusion.tuning.runner import OptunaTrialAdapter
from ids_diffusion.types import DeviceChoice
from ids_diffusion.utils import setup_logging

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def run(
    cache: Annotated[list[Path], typer.Option(help="One prepared cache per seed")],
    output: Annotated[Path, typer.Option(help="Confirmation JSON output")],
    storage: Annotated[str, typer.Option(help="Optuna storage URL")],
    study_name: Annotated[str, typer.Option()] = "rm-dmvt",
    rank: Annotated[int, typer.Option(min=1)] = 1,
    narrow: Annotated[bool, typer.Option(help="Study used the confirmed region")] = False,
    device: Annotated[DeviceChoice, typer.Option()] = DeviceChoice.AUTO,
) -> None:
    """Evaluate a ranked trial on validation data from every supplied seed."""
    logger = setup_logging(log_file=output.with_suffix(".log"))
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = sorted(
        (
            trial
            for trial in study.trials
            if trial.state is optuna.trial.TrialState.COMPLETE and trial.value is not None
        ),
        key=lambda trial: float(trial.value),
        reverse=True,
    )
    if rank > len(completed):
        raise ConfigurationError(
            field="rank",
            detail=f"requested {rank}, but study has {len(completed)} complete trials",
        )
    selected = completed[rank - 1]
    objective = ClassifierObjective(
        prepared_runs=load_seeded_caches(tuple(cache)),
        base_config=ExperimentConfig(),
        device=parse_device(device),
        narrow=narrow,
    )
    mean_validation_macro_f1 = objective(
        OptunaTrialAdapter(optuna.trial.FixedTrial(selected.params))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "rank": rank,
                "trial_number": selected.number,
                "single_seed_value": selected.value,
                "multi_seed_validation_macro_f1": mean_validation_macro_f1,
                "params": selected.params,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "confirmation complete",
        extra={
            "rank": rank,
            "trial_number": selected.number,
            "validation_macro_f1": mean_validation_macro_f1,
        },
    )


if __name__ == "__main__":
    app()
