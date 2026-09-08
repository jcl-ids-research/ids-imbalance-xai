"""Train the four RM-DMVT ablation variants from a prepared cache."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ids_diffusion.cli.common import (
    experiment_config,
    model_config,
    parse_device,
    write_metrics,
)
from ids_diffusion.config import TrainingConfig
from ids_diffusion.data.cache import load_prepared
from ids_diffusion.training.pipeline import PipelineJob, RankMatchedDiffusionMultiViewPipeline
from ids_diffusion.types import DeviceChoice, FusionMode
from ids_diffusion.utils import set_seed, setup_logging

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def run(
    cache: Annotated[Path, typer.Option(help="Prepared .npz cache")],
    output: Annotated[Path, typer.Option(help="Metrics JSON output")],
    seed: Annotated[int, typer.Option()] = 42,
    device: Annotated[DeviceChoice, typer.Option()] = DeviceChoice.AUTO,
    d_model: Annotated[int, typer.Option()] = 128,
    n_heads: Annotated[int, typer.Option()] = 8,
    n_layers: Annotated[int, typer.Option()] = 4,
    feedforward: Annotated[int, typer.Option()] = 512,
    dropout: Annotated[float, typer.Option()] = 0.1,
    fusion: Annotated[FusionMode, typer.Option()] = FusionMode.LINEAR,
    epochs: Annotated[int, typer.Option()] = 30,
    batch_size: Annotated[int, typer.Option()] = 256,
    learning_rate: Annotated[float, typer.Option()] = 1e-4,
    weight_decay: Annotated[float, typer.Option()] = 1e-5,
) -> None:
    """Train full, no-diffusion, no-multiview, and baseline variants."""
    logger = setup_logging(log_file=output.with_suffix(".log"))
    set_seed(seed)
    prepared = load_prepared(cache)
    training = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    config = experiment_config(
        seed,
        output.parent,
        model_config(d_model, n_heads, n_layers, feedforward, dropout, fusion),
        training,
    )
    compute = parse_device(device)
    logger.info("training RM-DMVT", extra={"seed": seed, "device": str(compute)})
    metrics = RankMatchedDiffusionMultiViewPipeline(config).run(
        PipelineJob(prepared=prepared, device=compute)
    )
    write_metrics(output, metrics)
    logger.info("metrics written", extra={"path": str(output)})


if __name__ == "__main__":
    app()
