"""Execute reproduction jobs with preflight checks and resumable outputs.

Each job writes its prepared cache and its metrics under a stable name, so an
interrupted reproduction can continue without repeating finished work and a
reviewer can point at the artefact behind any reported number.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from ids_diffusion.config import ExperimentConfig
from ids_diffusion.data.cache import save_prepared
from ids_diffusion.data.registry import load_dataset, missing_inputs, view_indices_for_dataset
from ids_diffusion.reproduction.jobs import ReproductionJob
from ids_diffusion.training.pipeline import PipelineJob, RankMatchedDiffusionMultiViewPipeline
from ids_diffusion.training.prepare import PreparationJob, prepare_experiment
from ids_diffusion.types import VariantMetrics


@dataclass(frozen=True, slots=True)
class RunnerSettings:
    """Where inputs are read from and where artefacts are written."""

    output_root: Path
    data_roots: Mapping[str, Path]
    device: torch.device
    sample_cap: int = 200_000
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    """One reason a job cannot start yet."""

    job: ReproductionJob
    detail: str


@dataclass(frozen=True, slots=True)
class JobOutcome:
    """What happened to a single reproduction job."""

    job: ReproductionJob
    metrics_path: Path
    skipped: bool
    seconds: float


def preflight(
    jobs: tuple[ReproductionJob, ...],
    settings: RunnerSettings,
) -> tuple[PreflightIssue, ...]:
    """Report missing dataset roots and files before any training starts."""
    issues: list[PreflightIssue] = []
    for job in jobs:
        root = settings.data_roots.get(job.dataset)
        if root is None:
            issues.append(PreflightIssue(job=job, detail="no data root configured"))
            continue
        if not root.is_dir():
            issues.append(PreflightIssue(job=job, detail=f"missing directory {root}"))
            continue
        issues.extend(
            PreflightIssue(job=job, detail=f"missing input {item}")
            for item in missing_inputs(job.dataset, root)
        )
    return tuple(issues)


def _job_config(
    job: ReproductionJob,
    settings: RunnerSettings,
    base: ExperimentConfig,
) -> ExperimentConfig:
    """Bind one job's dataset, task and seed to the shared experiment config."""
    data = replace(
        base.data,
        dataset=job.dataset,
        task=job.task,
        root=settings.data_roots[job.dataset],
        sample_cap=settings.sample_cap,
    )
    return replace(
        base,
        name=f"reproduce-{job.dataset}-{job.task}",
        seed=job.seed,
        output_root=settings.output_root,
        data=data,
    )


def run_job(
    job: ReproductionJob,
    settings: RunnerSettings,
    base: ExperimentConfig,
) -> JobOutcome:
    """Prepare corrected data and train the four variants for one job."""
    metrics_path = job.metrics_path(settings.output_root)
    if metrics_path.is_file() and not settings.overwrite:
        return JobOutcome(job=job, metrics_path=metrics_path, skipped=True, seconds=0.0)

    started = time.monotonic()
    config = _job_config(job, settings, base)
    loaded = load_dataset(
        job.dataset,
        settings.data_roots[job.dataset],
        seed=job.seed,
        task=job.task,
        sample_cap=settings.sample_cap,
    )
    views = view_indices_for_dataset(
        job.dataset,
        loaded.feature_names,
        view_count=config.model.n_views,
    )
    prepared = prepare_experiment(
        PreparationJob(dataset=loaded, views=views, config=config, device=settings.device)
    )
    cache_path = job.cache_path(settings.output_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_prepared(cache_path, prepared)

    metrics = RankMatchedDiffusionMultiViewPipeline(config).run(
        PipelineJob(prepared=prepared, device=settings.device)
    )
    write_job_metrics(metrics_path, job, metrics)
    return JobOutcome(
        job=job,
        metrics_path=metrics_path,
        skipped=False,
        seconds=time.monotonic() - started,
    )


def write_job_metrics(path: Path, job: ReproductionJob, metrics: VariantMetrics) -> None:
    """Persist metrics together with the job identity that produced them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": job.dataset,
        "task": job.task,
        "seed": job.seed,
        "variants": asdict(metrics),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "JobOutcome",
    "PreflightIssue",
    "RunnerSettings",
    "preflight",
    "run_job",
    "write_job_metrics",
]
