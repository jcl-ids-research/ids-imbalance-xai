from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

typer = pytest.importorskip("typer")
pytest.importorskip("rich")

from typer.testing import CliRunner

from ids_diffusion.cli.evaluate import app
from ids_diffusion.cli.freeze import app as freeze_app
from ids_diffusion.data.holdout import manifest_path
from ids_diffusion.data.unsw import TRAIN_FILE


def test_evaluate_cli_prints_all_four_variants(tmp_path: Path) -> None:
    # Given: a metrics file produced by the train command
    row = {
        "accuracy": 0.9,
        "precision": 0.9,
        "recall": 0.9,
        "weighted_f1": 0.9,
        "macro_f1": 0.88,
    }
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "full_model": row,
                "without_diffusion": row,
                "without_multiview": row,
                "baseline": row,
            }
        ),
        encoding="utf-8",
    )

    # When: the user invokes the real CLI surface
    result = CliRunner().invoke(app, [str(path)])

    # Then: every experiment variant is visible
    assert result.exit_code == 0
    assert "full_model" in result.stdout
    assert "without_diffusion" in result.stdout
    assert "without_multiview" in result.stdout
    assert "baseline" in result.stdout


def _write_dataset(root: Path) -> None:
    rng = np.random.default_rng(5)
    labels = np.zeros(300, dtype=np.int64)
    labels[rng.choice(300, size=110, replace=False)] = 1
    root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "dur": rng.normal(size=300),
            "proto": ["tcp" if value else "udp" for value in labels],
            "label": labels,
        }
    ).write_csv(root / TRAIN_FILE)


def test_freeze_cli_refuses_to_redraw_an_existing_split(tmp_path: Path) -> None:
    # Given: a frozen split that already exists on disk
    root = tmp_path / "data"
    _write_dataset(root)
    split = tmp_path / "outer.npz"
    runner = CliRunner()
    first = runner.invoke(
        freeze_app,
        ["--data-root", str(root), "--output", str(split), "--seed", "11"],
    )
    assert first.exit_code == 0
    original = manifest_path(split).read_text(encoding="utf-8")

    # When: a second freeze is attempted without an explicit overwrite
    second = runner.invoke(
        freeze_app,
        ["--data-root", str(root), "--output", str(split), "--seed", "99"],
    )

    # Then: the command fails and the recorded split is untouched
    assert second.exit_code != 0
    assert manifest_path(split).read_text(encoding="utf-8") == original
