from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import final

import optuna
import pytest
import torch

from ids_diffusion.config import ExperimentConfig, ModelConfig, TrainingConfig
from ids_diffusion.reproduction.archive_io import as_float, as_object, read_json
from ids_diffusion.tuning.runner import OptunaTrialAdapter, StudyConfig, run_study
from ids_diffusion.tuning.search_space import (
    ARCHITECTURES,
    TrialProtocol,
    suggest_classifier_trial,
)

CPU = torch.device("cpu")
optuna.logging.set_verbosity(optuna.logging.WARNING)


@final
@dataclass(frozen=True, slots=True)
class RecordingObjective:
    """Score a trial by reading one parameter, so a study can run in milliseconds.

    The runner's job is orchestration and persistence, not modelling. Substituting
    the objective keeps this test about the thing under test.
    """

    base_config: ExperimentConfig

    def __call__(self, trial: TrialProtocol) -> float:
        """Return a score that rises with the suggested layer count."""
        layers = trial.suggest_int("n_layers", 2, 6)
        _ = trial.suggest_categorical("architecture", ARCHITECTURES)
        return float(layers) / 10.0


def _base() -> ExperimentConfig:
    base = ExperimentConfig()
    return ExperimentConfig(
        data=base.data,
        model=ModelConfig(d_model=16, n_heads=2, n_layers=1, dim_feedforward=32, n_views=3),
        training=TrainingConfig(epochs=1, batch_size=16, patience=1),
        diffusion=base.diffusion,
        seed=base.seed,
    )


def _study_config(tmp_path: Path, name: str, trials: int) -> StudyConfig:
    return StudyConfig(
        name=name,
        trials=trials,
        storage=f"sqlite:///{(tmp_path / 'study.db').as_posix()}",
        output=tmp_path / "best.json",
    )


def test_the_adapter_passes_every_suggestion_through_to_optuna() -> None:
    # Given: a trial with fixed parameters, wrapped in the typed adapter
    fixed = optuna.trial.FixedTrial(
        {"architecture": "128x8", "n_layers": 5, "dropout": 0.2, "learning_rate": 1e-4}
    )
    adapter = OptunaTrialAdapter(fixed)

    # When: each suggestion kind is requested
    categorical = adapter.suggest_categorical("architecture", ARCHITECTURES)
    integer = adapter.suggest_int("n_layers", 2, 6)
    floating = adapter.suggest_float("dropout", 0.0, 0.35, log=False)

    # Then: the recorded values come back with the types the protocol declares,
    # so the search space cannot receive an Optuna object it does not understand
    assert (categorical, integer) == ("128x8", 5)
    assert isinstance(categorical, str)
    assert isinstance(integer, int)
    assert isinstance(floating, float)
    assert floating == pytest.approx(0.2)


def test_the_adapter_drives_the_real_search_space() -> None:
    # Given: a fixed trial covering every parameter the classifier space asks for
    fixed = optuna.trial.FixedTrial(
        {
            "architecture": "256x8",
            "n_layers": 3,
            "feedforward_ratio": 4,
            "fusion_mode": "linear",
            "dropout": 0.1,
            "batch_size": "256",
            "learning_rate": 1.5e-4,
            "weight_decay": 1e-5,
        }
    )

    # When: the space is asked for a configuration through the adapter
    config = suggest_classifier_trial(OptunaTrialAdapter(fixed), _base())

    # Then: the suggestion is carried through faithfully. This is the seam
    # between Optuna and the typed space, and a mismatch here would silently
    # tune something other than what the study recorded.
    assert config.model.d_model == 256
    assert config.model.n_heads == 8
    assert config.model.n_layers == 3
    assert config.model.dim_feedforward == 256 * 4
    assert config.training.batch_size == 256


def test_a_study_persists_the_best_configuration(tmp_path: Path) -> None:
    # Given: an objective that rewards more layers
    config = _study_config(tmp_path, "persist", trials=6)

    # When: the study runs
    study = run_study(RecordingObjective(_base()), config)

    # Then: the best result is written where the tuning protocol expects it,
    # with enough context to reconstruct which configuration produced it
    payload = read_json(config.output)
    assert payload["study"] == "persist"
    assert as_float(payload["trial_count"], config.output, "trial count") == 6
    assert as_float(payload["best_value"], config.output, "best value") == pytest.approx(
        study.best_value
    )
    params = as_object(payload["best_params"], config.output, "best params")
    assert params["n_layers"] == study.best_params["n_layers"]
    assert "base_config" in payload


def test_the_study_maximises_rather_than_minimises(tmp_path: Path) -> None:
    # Given: an objective whose score rises with the layer count
    config = _study_config(tmp_path, "direction", trials=8)

    # When: the study runs
    study = run_study(RecordingObjective(_base()), config)

    # Then: it searches upward. A study minimising validation macro F1 would
    # still complete and still write a file, so the direction is asserted.
    assert study.direction is optuna.study.StudyDirection.MAXIMIZE
    values = [trial.value for trial in study.trials if trial.value is not None]
    assert study.best_value == pytest.approx(max(values))


def test_a_study_resumes_instead_of_starting_over(tmp_path: Path) -> None:
    # Given: a study that has already run some trials
    first = run_study(RecordingObjective(_base()), _study_config(tmp_path, "resume", trials=4))
    assert len(first.trials) == 4

    # When: the same study name and storage are used again
    second = run_study(RecordingObjective(_base()), _study_config(tmp_path, "resume", trials=3))

    # Then: the earlier trials are kept. Tuning runs span nights on the server,
    # so a restart that discarded history would waste the budget silently.
    assert len(second.trials) == 7


def test_the_output_directory_is_created_when_missing(tmp_path: Path) -> None:
    # Given: an output path several levels below an existing directory
    config = StudyConfig(
        name="nested",
        trials=2,
        storage=f"sqlite:///{(tmp_path / 'study.db').as_posix()}",
        output=tmp_path / "studies" / "classifier" / "best.json",
    )

    # When: the study runs
    _ = run_study(RecordingObjective(_base()), config)

    # Then: the result is written rather than lost to a missing directory at the
    # end of a long run
    assert config.output.is_file()
    assert read_json(config.output)["study"] == "nested"


def test_the_sampler_seed_makes_a_study_reproducible(tmp_path: Path) -> None:
    # Given: two studies with the same sampler seed and separate storage
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first = run_study(RecordingObjective(_base()), _study_config(first_dir, "seeded", trials=6))
    second = run_study(RecordingObjective(_base()), _study_config(second_dir, "seeded", trials=6))

    # Then: they explore the same points. Without this the "best configuration"
    # the paper reports could not be re-derived from the recorded seed.
    assert [t.params for t in first.trials] == [t.params for t in second.trials]
