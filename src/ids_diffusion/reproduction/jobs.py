"""The exact experiment matrix reported in the paper.

Reviewers should not have to reconstruct which dataset, task and seed produced
a table cell. The matrix is defined once here and consumed by the preflight,
run and verification commands alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ids_diffusion.config import DatasetName, TaskName
from ids_diffusion.data.registry import DATASET_NAMES, MULTICLASS_DATASETS

PAPER_SEEDS: tuple[int, ...] = (42, 123, 456)


@dataclass(frozen=True, slots=True)
class ReproductionJob:
    """One dataset, task and seed combination reported in the paper."""

    dataset: DatasetName
    task: TaskName
    seed: int

    @property
    def slug(self) -> str:
        """Return the stable identifier used for caches, metrics and logs."""
        return f"{self.dataset}_{self.task}_seed{self.seed}"

    def cache_path(self, root: Path) -> Path:
        """Return the prepared-cache location for this job."""
        return root / "cache" / f"{self.slug}.npz"

    def metrics_path(self, root: Path) -> Path:
        """Return the metrics location for this job."""
        return root / "metrics" / f"{self.slug}.json"

    def data_root(self, roots: dict[str, Path]) -> Path:
        """Return the dataset directory configured for this job."""
        return roots[self.dataset]


def paper_jobs(
    datasets: tuple[DatasetName, ...] = DATASET_NAMES,
    seeds: tuple[int, ...] = PAPER_SEEDS,
    tasks: tuple[TaskName, ...] = ("binary", "multiclass"),
) -> tuple[ReproductionJob, ...]:
    """Enumerate every run behind the reported ablation tables.

    Multi-class results exist only for the two benchmarks that publish an
    attack-category column, so the matrix omits the CIC releases there instead
    of inventing a task the paper never reported.
    """
    jobs: list[ReproductionJob] = []
    for task in tasks:
        for dataset in datasets:
            if task == "multiclass" and dataset not in MULTICLASS_DATASETS:
                continue
            jobs.extend(ReproductionJob(dataset=dataset, task=task, seed=seed) for seed in seeds)
    return tuple(jobs)


__all__ = ["PAPER_SEEDS", "ReproductionJob", "paper_jobs"]
