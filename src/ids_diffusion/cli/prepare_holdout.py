"""Prepare tuning caches and the sealed holdout cache from a frozen split."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ids_diffusion.cli.common import parse_device
from ids_diffusion.config import DataConfig, ExperimentConfig
from ids_diffusion.data.cache import save_inner, save_prepared
from ids_diffusion.data.unsw_holdout import load_unsw_holdout
from ids_diffusion.data.views import UNSW_SEMANTIC_VIEWS, named_view_indices
from ids_diffusion.training.prepare import (
    InnerPreparationJob,
    PreparationJob,
    prepare_experiment,
    prepare_inner,
)
from ids_diffusion.types import DeviceChoice
from ids_diffusion.utils import set_seed, setup_logging

app = typer.Typer(add_completion=False, help=__doc__)


@app.command("inner")
def inner(
    data_root: Annotated[Path, typer.Option(help="Directory containing UNSW CSV files")],
    split: Annotated[Path, typer.Option(help="Frozen outer split archive")],
    output: Annotated[Path, typer.Option(help="Tuning cache (.npz) to write")],
    seed: Annotated[int, typer.Option()] = 42,
    device: Annotated[DeviceChoice, typer.Option()] = DeviceChoice.AUTO,
) -> None:
    """Write a cache that contains inner train/validation rows only."""
    logger = setup_logging(log_file=output.with_suffix(".log"))
    set_seed(seed)
    dataset, manifest = load_unsw_holdout(data_root, split)
    config = ExperimentConfig(
        name="rm-dmvt-inner",
        seed=seed,
        output_root=output.parent,
        data=DataConfig(dataset="unsw", root=data_root),
    )
    prepared = prepare_inner(
        InnerPreparationJob(
            training=dataset.training,
            views=named_view_indices(dataset.feature_names, UNSW_SEMANTIC_VIEWS),
            config=config,
            device=parse_device(device),
        )
    )
    save_inner(output, prepared)
    logger.info(
        "inner cache written",
        extra={
            "path": str(output),
            "train_rows": len(prepared.raw_training.labels),
            "validation_rows": len(prepared.validation.labels),
            "holdout_rows_excluded": manifest.holdout_count,
        },
    )


@app.command("holdout")
def holdout(
    data_root: Annotated[Path, typer.Option(help="Directory containing UNSW CSV files")],
    split: Annotated[Path, typer.Option(help="Frozen outer split archive")],
    output: Annotated[Path, typer.Option(help="Evaluation cache (.npz) to write")],
    seed: Annotated[int, typer.Option()] = 42,
    device: Annotated[DeviceChoice, typer.Option()] = DeviceChoice.AUTO,
) -> None:
    """Write the sealed cache whose test matrix is the frozen outer holdout."""
    logger = setup_logging(log_file=output.with_suffix(".log"))
    set_seed(seed)
    dataset, manifest = load_unsw_holdout(data_root, split)
    config = ExperimentConfig(
        name="rm-dmvt-holdout",
        seed=seed,
        output_root=output.parent,
        data=DataConfig(dataset="unsw", root=data_root),
    )
    prepared = prepare_experiment(
        PreparationJob(
            dataset=dataset,
            views=named_view_indices(dataset.feature_names, UNSW_SEMANTIC_VIEWS),
            config=config,
            device=parse_device(device),
        )
    )
    save_prepared(output, prepared)
    logger.info(
        "holdout cache written",
        extra={
            "path": str(output),
            "train_rows": len(prepared.raw_training.labels),
            "holdout_rows": len(prepared.test.labels),
            "source_sha256": manifest.source_sha256,
        },
    )


if __name__ == "__main__":
    app()
