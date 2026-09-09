from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ids_diffusion.errors import DatasetFileError
from ids_diffusion.reproduction.evidence import compare_to_reference, load_reference
from ids_diffusion.reproduction.jobs import PAPER_SEEDS, ReproductionJob, paper_jobs
from ids_diffusion.reproduction.runner import RunnerSettings, preflight


def test_paper_matrix_covers_binary_everywhere_and_multiclass_only_where_reported() -> None:
    # Given: the complete reported experiment matrix
    jobs = paper_jobs()

    # When: the matrix is grouped by task
    binary = {job.dataset for job in jobs if job.task == "binary"}
    multiclass = {job.dataset for job in jobs if job.task == "multiclass"}

    # Then: every benchmark has binary runs and only labelled ones have multiclass
    assert binary == {"unsw", "nslkdd", "cicids2017", "cicddos2019"}
    assert multiclass == {"unsw", "nslkdd"}
    assert len(jobs) == len(PAPER_SEEDS) * (len(binary) + len(multiclass))


def test_preflight_reports_each_missing_dataset_input(tmp_path: Path) -> None:
    # Given: an NSL-KDD root that only contains the training file
    root = tmp_path / "nsl"
    root.mkdir()
    (root / "KDDTrain+.txt").write_text("", encoding="utf-8")
    job = ReproductionJob(dataset="nslkdd", task="binary", seed=42)
    settings = RunnerSettings(
        output_root=tmp_path / "runs",
        data_roots={"nslkdd": root},
        device=torch.device("cpu"),
    )

    # When: the job is checked before training
    issues = preflight((job,), settings)

    # Then: the missing official test file is named
    assert len(issues) == 1
    assert "KDDTest+.txt" in issues[0].detail


def _write_reference(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "nslkdd",
        "seed": 42,
        "variants": {
            "full_model": {
                "acc": 0.80,
                "f1": 0.80,
                "f1_macro": 0.80,
                "precision": 0.80,
                "recall": 0.80,
            },
            "baseline": {
                "acc": 0.70,
                "f1": 0.70,
                "f1_macro": 0.70,
                "precision": 0.70,
                "recall": 0.70,
            },
        },
    }
    (root / "nslkdd_seed42.json").write_text(json.dumps(payload), encoding="utf-8")


def test_comparison_reports_the_largest_metric_movement(tmp_path: Path) -> None:
    # Given: one archived run and a fresh run that drifted on the full model
    _write_reference(tmp_path)
    reference = load_reference(tmp_path, "nslkdd", 42)
    produced = {
        "full_model": {
            "accuracy": 0.83,
            "weighted_f1": 0.80,
            "macro_f1": 0.80,
            "precision": 0.80,
            "recall": 0.80,
        },
        "baseline": {
            "accuracy": 0.70,
            "weighted_f1": 0.70,
            "macro_f1": 0.70,
            "precision": 0.70,
            "recall": 0.70,
        },
    }

    # When: the fresh run is differenced against the archive
    report = compare_to_reference(produced, reference, tolerance=0.02)

    # Then: the drifted metric is named and flagged as outside tolerance
    largest = report.largest
    assert largest is not None
    assert (largest.variant, largest.metric) == ("full_model", "accuracy")
    assert largest.delta == pytest.approx(0.03)
    assert not report.within_tolerance


def test_missing_archived_run_is_rejected(tmp_path: Path) -> None:
    # Given: an evidence directory without the requested seed
    _write_reference(tmp_path)

    # When / Then: loading a seed that was never archived fails loudly
    with pytest.raises(DatasetFileError):
        load_reference(tmp_path, "nslkdd", 999)


def test_multiclass_runs_resolve_to_their_own_evidence_family(tmp_path: Path) -> None:
    # Given: an evidence root holding a binary and a multi-class run for one seed
    binary_root = tmp_path / "phase1_corrected"
    binary_root.mkdir()
    _write_reference(binary_root)

    multiclass_root = tmp_path / "multiclass" / "nslkdd" / "seed42"
    multiclass_root.mkdir(parents=True)
    payload = {
        "dataset": "nslkdd",
        "seed": 42,
        "task": "multiclass",
        "variants": {
            "full_model": {
                "acc": 0.46,
                "f1": 0.46,
                "f1_macro": 0.46,
                "precision": 0.46,
                "recall": 0.46,
            }
        },
    }
    (multiclass_root / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    # When: each task is loaded for the same dataset and seed
    binary = load_reference(tmp_path, "nslkdd", 42, "binary")
    multiclass = load_reference(tmp_path, "nslkdd", 42, "multiclass")

    # Then: they resolve to different files, so multi-class reruns are not
    # silently compared against binary evidence or skipped altogether
    assert binary.path != multiclass.path
    assert multiclass.variants["full_model"]["macro_f1"] == pytest.approx(0.46)
    assert binary.variants["full_model"]["macro_f1"] == pytest.approx(0.80)


def test_an_unknown_task_is_rejected(tmp_path: Path) -> None:
    # Given: an evidence root that holds only the binary family
    _write_reference(tmp_path)

    # When / Then: an unrecognised task name fails rather than falling back
    with pytest.raises(DatasetFileError):
        load_reference(tmp_path, "nslkdd", 42, "regression")
