"""Reproducibility capabilities for every experiment family in the paper.

Having source and evidence is not the same as having a portable rerun command.
This registry states that distinction explicitly. The main matrix, XGBoost,
scarcity study and tuning workflow use maintained package entry points. Several
secondary analyses still reproduce only through the immutable server snapshot;
their evidence is public and checkable, but their generators are not described
as portable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib.metadata import entry_points
from pathlib import Path


class GenerationLevel(str, Enum):
    """How a reader can regenerate an experiment family."""

    MAINTAINED = "maintained"
    ARCHIVED_ONLY = "archived-only"


@dataclass(frozen=True, slots=True)
class ExperimentFamily:
    """One reported experiment family and its reproducibility surface."""

    slug: str
    description: str
    generation: GenerationLevel
    run_command: str | None
    verify_command: str
    sources: tuple[str, ...]
    artifact_patterns: tuple[str, ...]

    @property
    def entrypoint(self) -> str | None:
        """Return the console command used to run this family, if maintained."""
        return None if self.run_command is None else self.run_command.split()[0]


@dataclass(frozen=True, slots=True)
class FamilyAudit:
    """Presence and command checks for one experiment family."""

    family: ExperimentFamily
    missing_sources: tuple[str, ...]
    artifact_count: int
    entrypoint_available: bool

    @property
    def ok(self) -> bool:
        """Return whether every declared source, artifact and entry point exists."""
        artifacts_ok = self.artifact_count > 0 or not self.family.artifact_patterns
        return not self.missing_sources and artifacts_ok and self.entrypoint_available


EXPERIMENT_FAMILIES: tuple[ExperimentFamily, ...] = (
    ExperimentFamily(
        slug="main-ablation",
        description="four datasets, three seeds, four paired ablations; binary and multiclass",
        generation=GenerationLevel.MAINTAINED,
        run_command="ids-reproduce run",
        verify_command="ids-reproduce verify",
        sources=(
            "src/ids_diffusion/reproduction/runner.py",
            "src/ids_diffusion/training/pipeline.py",
        ),
        artifact_patterns=(
            "evidence/paper_results/phase1_corrected/**/*.json",
            "evidence/paper_results/multiclass/**/*.json",
        ),
    ),
    ExperimentFamily(
        slug="xgboost",
        description="raw and balanced XGBoost on the frozen split",
        generation=GenerationLevel.MAINTAINED,
        run_command="ids-baseline run",
        verify_command="ids-reproduce claims",
        sources=("src/ids_diffusion/cli/baseline.py",),
        artifact_patterns=("evidence/paper_results/baseline_fair/**/*.json",),
    ),
    ExperimentFamily(
        slug="classical-deep-baselines",
        description="RF, SVM, MLP, FTTransformer, CNN1D and GNN comparison families",
        generation=GenerationLevel.ARCHIVED_ONLY,
        run_command=None,
        verify_command="ids-reproduce claims",
        sources=(
            "server_archive/legacy_paper/deploy/traditional_ml_all.py",
            "server_archive/legacy_paper/deploy/cnn_baseline.py",
            "server_archive/legacy_paper/deploy/train_gnn_baseline.py",
            "server_archive/legacy_paper/deploy/scripts/baseline_fair.py",
            "server_archive/legacy_paper/deploy/scripts/baseline_deep.py",
        ),
        artifact_patterns=(
            "evidence/paper_results/baseline_deep/**/*.json",
            "evidence/paper_results/baseline_deep_multiclass/**/*.json",
        ),
    ),
    ExperimentFamily(
        slug="fidelity-correction",
        description="fidelity, discrete collapse and rank-matched correction",
        generation=GenerationLevel.ARCHIVED_ONLY,
        run_command=None,
        verify_command="ids-reproduce claims",
        sources=(
            "server_archive/legacy_paper/deploy/fidelity_evaluation.py",
            "server_archive/legacy_paper/deploy/check_discrete_collapse.py",
            "server_archive/legacy_paper/deploy/verify_correction.py",
        ),
        artifact_patterns=("evidence/paper_results/fidelity/*.json",),
    ),
    ExperimentFamily(
        slug="adversarial",
        description="FGSM and PGD evaluation on trained neural variants",
        generation=GenerationLevel.ARCHIVED_ONLY,
        run_command=None,
        verify_command="ids-reproduce claims",
        sources=("server_archive/legacy_paper/deploy/scripts/adversarial_real.py",),
        artifact_patterns=("evidence/paper_results/adversarial/*.json",),
    ),
    ExperimentFamily(
        slug="attention",
        description="attention extraction and layer-wise concentration analysis",
        generation=GenerationLevel.ARCHIVED_ONLY,
        run_command=None,
        verify_command="ids-reproduce claims",
        sources=("server_archive/legacy_paper/deploy/scripts/attention_real.py",),
        artifact_patterns=("evidence/paper_results/attention/**/*",),
    ),
    ExperimentFamily(
        slug="training-scarcity",
        description="UNSW-NB15 training-side attack thinning with an untouched test set",
        generation=GenerationLevel.MAINTAINED,
        run_command="ids-scarcity run",
        verify_command="ids-reproduce claims",
        sources=("src/ids_diffusion/cli/scarcity.py",),
        artifact_patterns=("evidence/paper_results/scarcity/*.json",),
    ),
    ExperimentFamily(
        slug="tuning-holdout",
        description="Optuna tuning, three-seed confirmation and frozen outer holdout",
        generation=GenerationLevel.MAINTAINED,
        run_command="ids-tune classifier",
        verify_command="ids-reproduce claims",
        sources=(
            "src/ids_diffusion/tuning/objective.py",
            "src/ids_diffusion/tuning/runner.py",
            "src/ids_diffusion/cli/confirm.py",
        ),
        artifact_patterns=("docs/tuning-result-*.md", "docs/outer-holdout-result.md"),
    ),
    ExperimentFamily(
        slug="figures",
        description="publication figures rendered from archived metrics",
        generation=GenerationLevel.ARCHIVED_ONLY,
        run_command=None,
        verify_command="ids-reproduce audit",
        sources=(
            "server_archive/legacy_paper/deploy/plot_ablation.py",
            "server_archive/legacy_paper/deploy/plot_fidelity.py",
        ),
        artifact_patterns=(
            "evidence/paper_results/figures/*.pdf",
            "evidence/paper_results/figures/*.png",
        ),
    ),
)


def audit_families(root: Path) -> tuple[FamilyAudit, ...]:
    """Check every declared source, artifact pattern and maintained command."""
    commands = {item.name for item in entry_points(group="console_scripts")}
    reports: list[FamilyAudit] = []
    for family in EXPERIMENT_FAMILIES:
        missing = tuple(source for source in family.sources if not (root / source).is_file())
        artifacts = {
            path
            for pattern in family.artifact_patterns
            for path in root.glob(pattern)
            if path.is_file()
        }
        entrypoint_available = family.entrypoint is None or family.entrypoint in commands
        reports.append(
            FamilyAudit(
                family=family,
                missing_sources=missing,
                artifact_count=len(artifacts),
                entrypoint_available=entrypoint_available,
            )
        )
    return tuple(reports)


__all__ = [
    "EXPERIMENT_FAMILIES",
    "ExperimentFamily",
    "FamilyAudit",
    "GenerationLevel",
    "audit_families",
]
