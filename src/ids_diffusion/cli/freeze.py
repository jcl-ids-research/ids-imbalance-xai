"""Freeze an outer holdout split drawn from the official UNSW training file."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ids_diffusion.data.holdout import (
    build_outer_split,
    create_manifest,
    manifest_path,
    write_split,
)
from ids_diffusion.data.unsw_holdout import read_labels, source_path
from ids_diffusion.errors import ConfigurationError
from ids_diffusion.utils import setup_logging

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def run(
    data_root: Annotated[Path, typer.Option(help="Directory containing UNSW CSV files")],
    output: Annotated[Path, typer.Option(help="Split archive (.npz) to create")],
    holdout_fraction: Annotated[float, typer.Option()] = 0.2,
    seed: Annotated[int, typer.Option()] = 20260902,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing frozen split")] = False,
) -> None:
    """Draw the split once and record digests that later stages re-verify."""
    logger = setup_logging(log_file=output.with_suffix(".log"))
    if output.exists() and not overwrite:
        raise ConfigurationError(
            field="output",
            detail=f"{output} already exists; refusing to redraw a frozen split",
        )
    labels = read_labels(data_root)
    split = build_outer_split(labels, holdout_fraction, seed)
    manifest = create_manifest(
        source_path=source_path(data_root),
        labels=labels,
        split=split,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    write_split(output, split, manifest)
    logger.info(
        "outer split frozen",
        extra={
            "path": str(output),
            "manifest": str(manifest_path(output)),
            "inner_rows": manifest.inner_count,
            "holdout_rows": manifest.holdout_count,
            "source_sha256": manifest.source_sha256,
        },
    )


if __name__ == "__main__":
    app()
