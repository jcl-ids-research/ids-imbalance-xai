"""Smoke-test the reproduction command group through its real CLI surface.

The core logic (models, training, tuning, the reproduction runner) has unit
coverage, but the commands a reviewer actually types were unwired from the test
suite. These tests drive `ids-reproduce` with CliRunner so a broken option, a
renamed command or a silently dropped entry point fails here instead of in
front of a reviewer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ids_diffusion.cli.reproduce import app

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "evidence" / "paper_results"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_plan_lists_the_full_reported_matrix(runner: CliRunner) -> None:
    # Given: no filters
    result = runner.invoke(app, ["plan"])

    # Then: the four datasets, three seeds and both tasks are enumerated
    assert result.exit_code == 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 18
    for dataset in ("unsw", "nslkdd", "cicids2017", "cicddos2019"):
        assert any(dataset in line for line in lines)
    assert any("multiclass" in line for line in lines)


def test_plan_accepts_known_filters(runner: CliRunner) -> None:
    # Given: a valid dataset and task restriction
    result = runner.invoke(app, ["plan", "--dataset", "unsw", "--task", "binary"])

    # Then: only that slice is listed, so a narrowed rerun is enumerable
    assert result.exit_code == 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 3  # one seed per run
    assert all("unsw" in line and "binary" in line for line in lines)


def test_plan_rejects_an_unknown_dataset(runner: CliRunner) -> None:
    # Given: a dataset the paper does not report
    result = runner.invoke(app, ["plan", "--dataset", "bogus"])

    # Then: the typo is reported as a bad parameter, not a traceback
    assert result.exit_code != 0
    assert "expected one of" in result.output


def test_plan_rejects_an_unknown_task(runner: CliRunner) -> None:
    # Given: a task the paper does not report
    result = runner.invoke(app, ["plan", "--task", "regression"])

    # Then: the invalid name fails loudly
    assert result.exit_code != 0
    assert "expected one of" in result.output


def test_claims_recomputes_every_number_without_a_gpu(runner: CliRunner) -> None:
    # Given: the archived results committed in the repository
    result = runner.invoke(app, ["claims", "--evidence", str(EVIDENCE)])

    # Then: every headline number is recomputed with no drift
    assert result.exit_code == 0
    assert "claims=39 drifted=0" in result.stdout


def test_audit_confirms_every_family_resolves(runner: CliRunner) -> None:
    # Given: the repository root
    result = runner.invoke(app, ["audit", "--root", str(REPO_ROOT)])

    # Then: every experiment family's sources, evidence and commands exist
    assert result.exit_code == 0
    assert "families=9 failed=0" in result.stdout


def test_families_distinguishes_portable_from_archived(runner: CliRunner) -> None:
    # Given: no arguments
    result = runner.invoke(app, ["families"])

    # Then: the reproducibility surface is stated honestly
    assert result.exit_code == 0
    assert "maintained" in result.stdout
    assert "archived-only" in result.stdout
    assert "ids-reproduce run" in result.stdout
    assert "ids-scarcity run" in result.stdout
