"""Generate and cache corrected balanced training data for RM-DMVT."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ids_diffusion.cli.common import parse_device
from ids_diffusion.config import DataConfig, ExperimentConfig
from ids_diffusion.data.cache import save_prepared
from ids_diffusion.data.registry import (
    load_dataset,
    require_dataset,
    require_task,
    view_indices_for_dataset,
)
from ids_diffusion.data.splits import apply_attack_thinning
from ids_diffusion.training.prepare import PreparationJob, prepare_experiment
from ids_diffusion.types import DeviceChoice, LoadedDataset
from ids_diffusion.utils import set_seed, setup_logging

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def run(
    data_root: Annotated[Path, typer.Option(help="Directory containing the dataset files")],
    output: Annotated[Path, typer.Option(help="Classifier-ready .npz cache")],
    dataset: Annotated[str, typer.Option(help="unsw, nslkdd, cicids2017 or cicddos2019")] = "unsw",
    task: Annotated[str, typer.Option(help="binary or multiclass")] = "binary",
    seed: Annotated[int, typer.Option()] = 42,
    attack_keep_fraction: Annotated[float, typer.Option()] = 1.0,
    sample_cap: Annotated[int, typer.Option(help="CIC subsample size")] = 200_000,
    device: Annotated[DeviceChoice, typer.Option()] = DeviceChoice.AUTO,
) -> None:
    """Fit the train-only generator, correct its samples, and write one cache."""
    logger = setup_logging(log_file=output.with_suffix(".log"))
    set_seed(seed)
    dataset_name = require_dataset(dataset)
    task_name = require_task(task)
    loaded = load_dataset(
        dataset_name,
        data_root,
        seed=seed,
        task=task_name,
        sample_cap=sample_cap,
    )
    training, report = apply_attack_thinning(
        loaded.training,
        keep_fraction=attack_keep_fraction,
        seed=seed,
    )
    shifted = LoadedDataset(
        training=training,
        test=loaded.test,
        feature_names=loaded.feature_names,
    )
    config = ExperimentConfig(
        name="rm-dmvt-prepare",
        seed=seed,
        output_root=output.parent,
        data=DataConfig(
            dataset=dataset_name,
            task=task_name,
            root=data_root,
            sample_cap=sample_cap,
            attack_keep_fraction=attack_keep_fraction,
        ),
    )
    prepared = prepare_experiment(
        PreparationJob(
            dataset=shifted,
            views=view_indices_for_dataset(
                dataset_name,
                loaded.feature_names,
                view_count=config.model.n_views,
            ),
            config=config,
            device=parse_device(device),
        )
    )
    save_prepared(output, prepared)
    logger.info(
        "prepared cache written",
        extra={
            "path": str(output),
            "dataset": dataset_name,
            "task": task_name,
            "attacks_before": report.attacks_before,
            "attacks_after": report.attacks_after,
        },
    )


if __name__ == "__main__":
    app()
