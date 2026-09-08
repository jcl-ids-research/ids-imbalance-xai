"""Display an RM-DMVT metrics JSON as a compact comparison table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def run(metrics: Annotated[Path, typer.Argument(help="Metrics JSON from ids-train")]) -> None:
    """Print the four ablation variants and their macro F1 scores."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    table = Table(title="RM-DMVT evaluation")
    table.add_column("Variant")
    table.add_column("Accuracy", justify="right")
    table.add_column("Weighted F1", justify="right")
    table.add_column("Macro F1", justify="right")
    for name in ("full_model", "without_diffusion", "without_multiview", "baseline"):
        row = payload[name]
        table.add_row(
            name,
            f"{row['accuracy']:.4f}",
            f"{row['weighted_f1']:.4f}",
            f"{row['macro_f1']:.4f}",
        )
    Console().print(table)


if __name__ == "__main__":
    app()
