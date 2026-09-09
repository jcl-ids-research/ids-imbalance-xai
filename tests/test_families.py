"""The experiment-family registry and the scarcity entry point.

The registry is the single place that states which paper results can be rerun
through a maintained command and which still reproduce only from the immutable
server snapshot. These tests pin that distinction and the scarcity payload
shape, so a source file deletion or a silently dropped entry point fails CI.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

from ids_diffusion.cli.scarcity import ScarcityReport, metrics_to_payload
from ids_diffusion.reproduction.archive_io import as_float, as_object
from ids_diffusion.reproduction.families import (
    EXPERIMENT_FAMILIES,
    GenerationLevel,
    audit_families,
)
from ids_diffusion.types import ClassificationMetrics, ShiftReport

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_every_reported_family_is_registered() -> None:
    # Given: the set of families the paper reports
    slugs = {family.slug for family in EXPERIMENT_FAMILIES}

    # Then: the nine families are all present, including the secondary analyses
    # that a naive "one command reproduces everything" summary would omit
    assert slugs == {
        "main-ablation",
        "xgboost",
        "classical-deep-baselines",
        "fidelity-correction",
        "adversarial",
        "attention",
        "training-scarcity",
        "tuning-holdout",
        "figures",
    }


def test_the_registry_distinguishes_portable_from_archived() -> None:
    # Given: the two generation levels
    maintained = [f.slug for f in EXPERIMENT_FAMILIES if f.generation is GenerationLevel.MAINTAINED]
    archived = [
        f.slug for f in EXPERIMENT_FAMILIES if f.generation is GenerationLevel.ARCHIVED_ONLY
    ]

    # Then: only the four families with package entry points are "maintained",
    # and the archived-only families honestly carry no run command
    assert maintained == ["main-ablation", "xgboost", "training-scarcity", "tuning-holdout"]
    assert archived == [
        "classical-deep-baselines",
        "fidelity-correction",
        "adversarial",
        "attention",
        "figures",
    ]
    for family in EXPERIMENT_FAMILIES:
        if family.generation is GenerationLevel.ARCHIVED_ONLY:
            assert family.run_command is None, family.slug


def test_every_declared_source_and_artifact_exists() -> None:
    # Given: the committed repository root
    reports = audit_families(REPO_ROOT)

    # Then: every family passes, so the registry cannot drift from the files it
    # claims to point at
    for report in reports:
        assert not report.missing_sources, (report.family.slug, report.missing_sources)
        assert report.artifact_count > 0, report.family.slug
        assert report.entrypoint_available, report.family.slug
    assert len(reports) == len(EXPERIMENT_FAMILIES)


def test_the_scarcity_entry_point_is_registered() -> None:
    # Given: the installed console scripts
    commands = {item.name for item in entry_points(group="console_scripts")}

    # Then: the scarcity experiment is reachable as a first-class command, not
    # only as a script the package happens to ship
    assert "ids-scarcity" in commands


def test_scarcity_payload_matches_the_archived_shape() -> None:
    # Given: a ScarcityReport in the same shape the archived JSON used
    def metrics() -> ClassificationMetrics:
        return ClassificationMetrics(
            accuracy=0.9, precision=0.9, recall=0.9, weighted_f1=0.9, macro_f1=0.9
        )

    report = ScarcityReport(
        keep_fraction=0.05,
        seed=42,
        protocol="official split, train-side attack thinning, test untouched",
        shift_report=ShiftReport(
            keep_fraction=0.05,
            attacks_before=1000,
            attacks_after=50,
            normals=9000,
            attack_share_before=0.1,
            attack_share_after=0.005,
        ),
        full_model=metrics(),
        without_diffusion=metrics(),
        without_multiview=metrics(),
        baseline=metrics(),
        xgboost_raw=metrics(),
    )

    # When / Then: the payload keeps the exact variant keys the claim checker
    # and the archived results rely on
    payload = metrics_to_payload(report)
    here = Path("<scarcity-report>")
    variants = as_object(payload["variants"], here, "variants")
    shift = as_object(payload["shift_report"], here, "shift report")
    assert as_float(payload["keep_fraction"], here, "keep_fraction") == 0.05
    assert as_float(payload["seed"], here, "seed") == 42
    assert set(variants) == {
        "full_model",
        "without_diffusion",
        "without_multiview",
        "baseline",
        "xgboost_raw",
    }
    assert as_float(shift["attacks_before"], here, "attacks before") == 1000
    xgboost = as_object(variants["xgboost_raw"], here, "xgboost")
    assert as_float(xgboost["macro_f1"], here, "macro f1") == 0.9
