"""Reproduction driver: job matrix, execution and comparison against evidence."""

from ids_diffusion.reproduction.evidence import (
    ComparisonReport,
    ReferenceRun,
    compare_to_reference,
    load_reference,
)
from ids_diffusion.reproduction.jobs import ReproductionJob, paper_jobs

__all__ = [
    "ComparisonReport",
    "ReferenceRun",
    "ReproductionJob",
    "compare_to_reference",
    "load_reference",
    "paper_jobs",
]
