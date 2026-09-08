"""End-to-end coverage of the reproduction driver on miniature datasets.

The loader tests check each benchmark in isolation. These tests drive the real
``ids-reproduce`` code path instead: dataset discovery, preflight, generation,
correction, balancing, four-variant training and metric output, for every
dataset and task combination the paper reports. They use tiny synthetic files in
the published on-disk layout, so they need no real data and no GPU.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest
import torch

from ids_diffusion.config import DiffusionConfig, ExperimentConfig, TrainingConfig
from ids_diffusion.reproduction.jobs import ReproductionJob
from ids_diffusion.reproduction.runner import RunnerSettings, preflight, run_job

NSL_COLUMN_COUNT = 43
NSL_TRAIN_LABELS = ("normal", "normal", "normal", "neptune", "ipsweep", "guess_passwd")
NSL_TEST_LABELS = ("normal", "normal", "apache2", "mscan", "sqlattack")
UNSW_CATEGORIES = ("Normal", "Generic", "Exploits", "Fuzzers")
EXPECTED_VARIANTS = {
    "full_model",
    "without_diffusion",
    "without_multiview",
    "baseline",
}


def _write_unsw(root: Path, rng: random.Random) -> None:
    """Write a miniature UNSW-NB15 official split."""
    root.mkdir(parents=True, exist_ok=True)
    header = ["id", "dur", "proto", "state", "spkts", "sbytes", "rate", "attack_cat", "label"]
    for name, rows in (("UNSW_NB15_training-set.csv", 320), ("UNSW_NB15_testing-set.csv", 160)):
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for index in range(rows):
                category = UNSW_CATEGORIES[index % len(UNSW_CATEGORIES)]
                writer.writerow(
                    [
                        index,
                        round(rng.uniform(0, 10), 4),
                        rng.choice(["tcp", "udp", "icmp"]),
                        rng.choice(["FIN", "CON", "INT"]),
                        rng.randint(1, 200),
                        rng.randint(50, 90_000),
                        round(rng.uniform(0, 1000), 3),
                        category,
                        0 if category == "Normal" else 1,
                    ]
                )


def _write_nslkdd(root: Path, rng: random.Random) -> None:
    """Write a miniature NSL-KDD official split."""
    root.mkdir(parents=True, exist_ok=True)
    partitions = (
        ("KDDTrain+.txt", 320, NSL_TRAIN_LABELS),
        ("KDDTest+.txt", 160, NSL_TEST_LABELS),
    )
    for name, count, labels in partitions:
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for index in range(count):
                row: list[object] = [0] * NSL_COLUMN_COUNT
                row[0] = rng.randint(0, 500)
                row[1] = rng.choice(["tcp", "udp", "icmp"])
                row[2] = rng.choice(["http", "smtp", "ftp"])
                row[3] = rng.choice(["SF", "S0", "REJ"])
                row[4] = rng.randint(0, 9000)
                row[22] = rng.randint(0, 400)
                row[24] = round(rng.random(), 2)
                row[41] = labels[index % len(labels)]
                row[42] = rng.randint(0, 21)
                writer.writerow(row)


def _write_cic(root: Path, rng: random.Random, attack: str) -> None:
    """Write a miniature CICFlowMeter directory."""
    root.mkdir(parents=True, exist_ok=True)
    header = ["Flow ID", "Flow Duration", "Total Fwd Packets", "Flow Bytes/s", " Label "]
    for part in range(2):
        with (root / f"part-{part}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for index in range(160):
                writer.writerow(
                    [
                        f"flow-{part}-{index}",
                        rng.randint(1, 100_000),
                        rng.randint(1, 200),
                        round(rng.uniform(0, 5000), 3),
                        "BENIGN" if index % 3 else attack,
                    ]
                )


@pytest.fixture(scope="module")
def data_roots(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build every benchmark once, in the layout the published loaders expect."""
    rng = random.Random(11)  # noqa: S311 - fixture data, not security material
    base = tmp_path_factory.mktemp("benchmarks")
    _write_unsw(base / "unsw", rng)
    _write_nslkdd(base / "nslkdd", rng)
    _write_cic(base / "cicids2017" / "MachineLearningCVE", rng, "DoS Hulk")
    _write_cic(base / "cicddos2019" / "all", rng, "DrDoS_DNS")
    return {
        "unsw": base / "unsw",
        "nslkdd": base / "nslkdd",
        "cicids2017": base / "cicids2017",
        "cicddos2019": base / "cicddos2019",
    }


def _fast_config() -> ExperimentConfig:
    """Return a wiring-check configuration, never a source of reported numbers."""
    base = ExperimentConfig()
    return base.with_diffusion(
        DiffusionConfig(timesteps=8, epochs=1, hidden=(16,), batch_size=64)
    ).with_training(TrainingConfig(epochs=1, batch_size=64, patience=1))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("dataset", "task"),
    [
        ("unsw", "binary"),
        ("unsw", "multiclass"),
        ("nslkdd", "binary"),
        ("nslkdd", "multiclass"),
        ("cicids2017", "binary"),
        ("cicddos2019", "binary"),
    ],
)
def test_every_reported_combination_runs_end_to_end(
    dataset: str,
    task: str,
    data_roots: dict[str, Path],
    tmp_path: Path,
) -> None:
    # Given: one reported dataset/task pair and its miniature benchmark
    job = ReproductionJob(dataset=dataset, task=task, seed=42)  # type: ignore[arg-type]
    settings = RunnerSettings(
        output_root=tmp_path,
        data_roots=data_roots,
        device=torch.device("cpu"),
        sample_cap=240,
    )
    assert preflight((job,), settings) == ()

    # When: the real reproduction driver executes the job
    outcome = run_job(job, settings, _fast_config())

    # Then: it produces the four ablation variants a reviewer expects
    assert not outcome.skipped
    payload = json.loads(outcome.metrics_path.read_text(encoding="utf-8"))
    assert set(payload["variants"]) == EXPECTED_VARIANTS
    assert payload["dataset"] == dataset
    assert payload["task"] == task
    assert job.cache_path(tmp_path).is_file()


@pytest.mark.slow
def test_finished_jobs_are_not_recomputed(
    data_roots: dict[str, Path],
    tmp_path: Path,
) -> None:
    # Given: a job that has already been run once
    job = ReproductionJob(dataset="nslkdd", task="binary", seed=42)
    settings = RunnerSettings(
        output_root=tmp_path,
        data_roots=data_roots,
        device=torch.device("cpu"),
        sample_cap=240,
    )
    first = run_job(job, settings, _fast_config())
    assert not first.skipped

    # When: the same job is requested again without an explicit overwrite
    second = run_job(job, settings, _fast_config())

    # Then: the finished work is reused so an interrupted run can resume
    assert second.skipped
    assert second.metrics_path == first.metrics_path


def test_preflight_blocks_a_missing_benchmark(tmp_path: Path) -> None:
    # Given: a configured root that does not exist on disk
    job = ReproductionJob(dataset="unsw", task="binary", seed=42)
    settings = RunnerSettings(
        output_root=tmp_path / "runs",
        data_roots={"unsw": tmp_path / "absent"},
        device=torch.device("cpu"),
    )

    # When: the job is checked before any training starts
    issues = preflight((job,), settings)

    # Then: the reviewer is told which directory is missing
    assert len(issues) == 1
    assert "absent" in issues[0].detail
