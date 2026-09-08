"""Reproduce the paper's experiment matrix and compare it with archived results."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from ids_diffusion.cli.common import parse_device
from ids_diffusion.config import DatasetName, ExperimentConfig, TaskName
from ids_diffusion.data.registry import DATASET_NAMES
from ids_diffusion.reproduction.claims import DEFAULT_EVIDENCE_ROOT as CLAIMS_ROOT
from ids_diffusion.reproduction.claims import evaluate_claims
from ids_diffusion.reproduction.evidence import (
    DEFAULT_EVIDENCE_ROOT,
    compare_to_reference,
    load_reference,
)
from ids_diffusion.reproduction.jobs import PAPER_SEEDS, ReproductionJob, paper_jobs
from ids_diffusion.reproduction.runner import RunnerSettings, preflight, run_job
from ids_diffusion.types import DeviceChoice
from ids_diffusion.utils import set_seed, setup_logging

app = typer.Typer(add_completion=False, help=__doc__)

DataRootOption = Annotated[list[str], typer.Option("--data-root", help="dataset=path pair")]
DatasetOption = Annotated[
    list[str] | None,
    typer.Option("--dataset", help="Restrict to one dataset"),
]
TaskOption = Annotated[
    list[str] | None,
    typer.Option("--task", help="binary and/or multiclass"),
]
SeedOption = Annotated[list[int] | None, typer.Option("--seed", help="Restrict to one seed")]
OutputOption = Annotated[Path, typer.Option(help="Artefact directory")]


SMOKE_DIFFUSION_EPOCHS = 2
SMOKE_CLASSIFIER_EPOCHS = 2
SMOKE_BATCH_SIZE = 128


def _base_config(smoke: bool) -> ExperimentConfig:
    """Return the full paper configuration, or a fast wiring check.

    The smoke configuration proves the pipeline runs end to end on a laptop; it
    is never a source of reported numbers, so its shortened schedules stay out
    of the default path.
    """
    base = ExperimentConfig()
    if not smoke:
        return base
    return replace(
        base,
        diffusion=replace(base.diffusion, epochs=SMOKE_DIFFUSION_EPOCHS),
        training=replace(
            base.training,
            epochs=SMOKE_CLASSIFIER_EPOCHS,
            batch_size=SMOKE_BATCH_SIZE,
        ),
    )


def _parse_roots(pairs: list[str]) -> dict[str, Path]:
    """Parse repeated dataset=path options into a mapping."""
    roots: dict[str, Path] = {}
    for pair in pairs:
        name, separator, raw_path = pair.partition("=")
        if not separator or name not in DATASET_NAMES:
            message = f"--data-root expects <dataset>=<path>, got {pair!r}"
            raise typer.BadParameter(message)
        roots[name] = Path(raw_path)
    return roots


def _selected_jobs(
    datasets: list[str] | None,
    tasks: list[str] | None,
    seeds: list[int] | None,
) -> tuple[ReproductionJob, ...]:
    """Build the job matrix, optionally narrowed by CLI filters."""
    chosen_datasets: tuple[DatasetName, ...] = tuple(datasets) if datasets else DATASET_NAMES  # type: ignore[assignment]
    chosen_tasks: tuple[TaskName, ...] = tuple(tasks) if tasks else ("binary", "multiclass")  # type: ignore[assignment]
    chosen_seeds = tuple(seeds) if seeds else PAPER_SEEDS
    return paper_jobs(datasets=chosen_datasets, seeds=chosen_seeds, tasks=chosen_tasks)


@app.command()
def plan(
    dataset: DatasetOption = None,
    task: TaskOption = None,
    seed: SeedOption = None,
) -> None:
    """List every run behind the reported tables."""
    for job in _selected_jobs(dataset, task, seed):
        print(f"{job.slug}\t{job.dataset}\t{job.task}\tseed={job.seed}")


@app.command()
def check(
    data_root: DataRootOption,
    output: OutputOption = Path("runs/reproduction"),
    dataset: DatasetOption = None,
    task: TaskOption = None,
    seed: SeedOption = None,
) -> None:
    """Verify dataset inputs before any training starts."""
    jobs = _selected_jobs(dataset, task, seed)
    settings = RunnerSettings(
        output_root=output,
        data_roots=_parse_roots(data_root),
        device=parse_device(DeviceChoice.CPU),
    )
    issues = preflight(jobs, settings)
    for issue in issues:
        print(f"BLOCKED\t{issue.job.slug}\t{issue.detail}")
    print(f"jobs={len(jobs)} blocked={len(issues)}")
    if issues:
        raise typer.Exit(code=1)


@app.command()
def run(
    data_root: DataRootOption,
    output: OutputOption = Path("runs/reproduction"),
    dataset: DatasetOption = None,
    task: TaskOption = None,
    seed: SeedOption = None,
    device: Annotated[DeviceChoice, typer.Option()] = DeviceChoice.AUTO,
    sample_cap: Annotated[int, typer.Option(help="CIC subsample size")] = 200_000,
    overwrite: Annotated[bool, typer.Option(help="Recompute finished jobs")] = False,
    smoke: Annotated[bool, typer.Option(help="Fast wiring check, not paper numbers")] = False,
) -> None:
    """Run the selected jobs, skipping any whose metrics already exist."""
    jobs = _selected_jobs(dataset, task, seed)
    settings = RunnerSettings(
        output_root=output,
        data_roots=_parse_roots(data_root),
        device=parse_device(device),
        sample_cap=sample_cap,
        overwrite=overwrite,
    )
    blocked = preflight(jobs, settings)
    if blocked:
        for issue in blocked:
            print(f"BLOCKED\t{issue.job.slug}\t{issue.detail}")
        raise typer.Exit(code=1)

    logger = setup_logging(log_file=output / "reproduce.log")
    base = _base_config(smoke)
    for job in jobs:
        set_seed(job.seed)
        outcome = run_job(job, settings, base)
        state = "skipped" if outcome.skipped else "done"
        logger.info(
            "reproduction job finished",
            extra={"job": job.slug, "state": state, "seconds": round(outcome.seconds, 1)},
        )
        print(f"{state}\t{job.slug}\t{outcome.metrics_path}")


@app.command()
def claims(
    evidence: Annotated[Path, typer.Option(help="Archived results")] = CLAIMS_ROOT,
) -> None:
    """Recompute the manuscript's headline numbers from the archived results."""
    results = evaluate_claims(evidence)
    for result in results:
        state = "OK" if result.holds else "DRIFT"
        print(
            f"{state}\t{result.name:38s} paper={result.claimed:8.2f} "
            f"archive={result.archived:8.4f} diff={result.difference:+.4f}"
        )
    drifted = sum(1 for result in results if not result.holds)
    print(f"claims={len(results)} drifted={drifted}")
    if drifted:
        raise typer.Exit(code=1)


@app.command()
def verify(
    metrics: Annotated[Path, typer.Option(help="Metrics directory from run")],
    evidence: Annotated[Path, typer.Option(help="Archived results")] = DEFAULT_EVIDENCE_ROOT,
    tolerance: Annotated[float, typer.Option(help="Allowed absolute difference")] = 0.02,
) -> None:
    """Difference reproduced metrics against the archived server results."""
    paths = sorted(metrics.glob("*_binary_seed*.json"))
    if not paths:
        print(f"no reproduced binary metrics under {metrics}")
        raise typer.Exit(code=1)

    failures = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        reference = load_reference(evidence, payload["dataset"], int(payload["seed"]))
        report = compare_to_reference(payload["variants"], reference, tolerance=tolerance)
        largest = report.largest
        summary = (
            "none"
            if largest is None
            else (f"{largest.variant}.{largest.metric} {largest.delta:+.4f}")
        )
        state = "PASS" if report.within_tolerance else "REVIEW"
        failures += 0 if report.within_tolerance else 1
        print(f"{state}\t{path.stem}\tlargest={summary}\tcompared={len(report.deltas)}")
    print(f"runs={len(paths)} outside_tolerance={failures} tolerance={tolerance}")
    if failures:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
