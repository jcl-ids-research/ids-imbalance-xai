"""Tune RM-DMVT on validation macro F1 without touching the test set."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ids_diffusion.cli.common import load_seeded_caches, parse_device
from ids_diffusion.config import ExperimentConfig
from ids_diffusion.data.unsw import load_unsw
from ids_diffusion.data.views import UNSW_SEMANTIC_VIEWS, named_view_indices
from ids_diffusion.tuning.objective import ClassifierObjective, DiffusionObjective
from ids_diffusion.tuning.runner import StudyConfig, run_study
from ids_diffusion.types import DeviceChoice
from ids_diffusion.utils import setup_logging

app = typer.Typer(add_completion=False, help=__doc__)


@app.command("classifier")
def classifier(
    cache: Annotated[list[Path], typer.Option(help="One cache per seed")],
    output: Annotated[Path, typer.Option(help="Best-trial JSON")],
    storage: Annotated[str, typer.Option(help="Optuna storage URL")],
    study_name: Annotated[str, typer.Option()] = "rm-dmvt",
    trials: Annotated[int, typer.Option()] = 50,
    narrow: Annotated[bool, typer.Option(help="Search only the confirmed region")] = False,
    sampler_seed: Annotated[
        int,
        typer.Option(help="Must differ per parallel worker, or workers repeat each other"),
    ] = 42,
    device: Annotated[DeviceChoice, typer.Option()] = DeviceChoice.AUTO,
) -> None:
    """Run or resume a durable Optuna study."""
    logger = setup_logging(log_file=output.with_suffix(".log"))
    prepared = load_seeded_caches(tuple(cache))
    objective = ClassifierObjective(
        prepared_runs=prepared,
        base_config=ExperimentConfig(),
        device=parse_device(device),
        narrow=narrow,
    )
    study = run_study(
        objective,
        StudyConfig(
            name=study_name,
            trials=trials,
            storage=storage,
            output=output,
            sampler_seed=sampler_seed,
        ),
    )
    logger.info(
        "tuning complete",
        extra={"best_macro_f1": study.best_value, "trial_count": len(study.trials)},
    )


@app.command("diffusion")
def diffusion(
    data_root: Annotated[Path, typer.Option(help="Directory containing UNSW CSV files")],
    output: Annotated[Path, typer.Option(help="Best-trial JSON")],
    storage: Annotated[str, typer.Option(help="Optuna storage URL")],
    study_name: Annotated[str, typer.Option()] = "rm-dmvt-diffusion",
    trials: Annotated[int, typer.Option()] = 15,
    seed: Annotated[int, typer.Option()] = 42,
    device: Annotated[DeviceChoice, typer.Option()] = DeviceChoice.AUTO,
) -> None:
    """Tune the DDPM and expansion cap while holding the classifier fixed."""
    logger = setup_logging(log_file=output.with_suffix(".log"))
    dataset = load_unsw(data_root)
    objective = DiffusionObjective(
        dataset=dataset,
        views=named_view_indices(dataset.feature_names, UNSW_SEMANTIC_VIEWS),
        base_config=ExperimentConfig(seed=seed),
        device=parse_device(device),
        seeds=(seed,),
    )
    study = run_study(
        objective,
        StudyConfig(
            name=study_name,
            trials=trials,
            storage=storage,
            output=output,
        ),
    )
    logger.info(
        "diffusion tuning complete",
        extra={"best_macro_f1": study.best_value, "trial_count": len(study.trials)},
    )


if __name__ == "__main__":
    app()
