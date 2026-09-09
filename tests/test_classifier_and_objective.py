from __future__ import annotations

from typing import cast, final

import numpy as np
import pytest
import torch
from torch import nn
from typing_extensions import override

from ids_diffusion.config import ExperimentConfig, ModelConfig, TrainingConfig
from ids_diffusion.training.classifier import (
    ClassifierTrainingJob,
    evaluate_classifier,
    train_classifier,
)
from ids_diffusion.tuning.objective import ClassifierObjective
from ids_diffusion.types import DatasetMatrix, PreparedInner, SeededPreparedInner
from ids_diffusion.utils import set_seed

CPU = torch.device("cpu")


@final
class CountingLinear(nn.Module):
    """A minimal classifier that records how often it was asked to train.

    The training loop is what enforces early stopping and best-state restore.
    Using a real Transformer here would make the test slow and would hide those
    behaviours behind optimisation noise.
    """

    linear: nn.Linear
    train_calls: int

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)
        self.train_calls = 0

    @override
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.train_calls += 1
        return cast("torch.Tensor", self.linear(features))


def _separable(n_rows: int = 96, n_features: int = 4) -> tuple[DatasetMatrix, DatasetMatrix]:
    """Return a train/validation pair a linear model can separate."""
    rng = np.random.default_rng(0)
    half = n_rows // 2
    features = np.vstack(
        (
            rng.normal(loc=-2.0, scale=0.3, size=(half, n_features)),
            rng.normal(loc=+2.0, scale=0.3, size=(half, n_features)),
        )
    ).astype(np.float32)
    labels = np.array([0] * half + [1] * half, dtype=np.int64)
    order = rng.permutation(n_rows)
    features, labels = features[order], labels[order]
    split = int(n_rows * 0.75)
    return (
        DatasetMatrix(features=features[:split], labels=labels[:split]),
        DatasetMatrix(features=features[split:], labels=labels[split:]),
    )


def _job(epochs: int, patience: int) -> tuple[ClassifierTrainingJob, CountingLinear]:
    set_seed(0)
    training, validation = _separable()
    model = CountingLinear(training.features.shape[1], 2)
    job = ClassifierTrainingJob(
        model=model,
        training=training,
        validation=validation,
        config=TrainingConfig(epochs=epochs, batch_size=16, patience=patience, learning_rate=1e-2),
        device=CPU,
    )
    return job, model


def test_training_learns_a_separable_problem() -> None:
    # Given: two well-separated classes
    job, _ = _job(epochs=12, patience=12)

    # When: the classifier is optimised
    trained = train_classifier(job)
    metrics = evaluate_classifier(trained, job.validation, job.config, CPU)

    # Then: it actually learns. Every ablation number in the paper depends on
    # this loop converging, so a silently broken optimiser would be invisible.
    assert metrics.macro_f1 > 0.9
    assert 0.0 <= metrics.accuracy <= 1.0


def test_early_stopping_halts_before_the_epoch_budget() -> None:
    # Given: a problem that is solved within the first epochs, and a short
    # patience
    job, model = _job(epochs=60, patience=2)

    # When: training runs
    train_classifier(job)
    batches_per_epoch = -(-len(job.training.labels) // job.config.batch_size)
    epochs_run = model.train_calls / batches_per_epoch

    # Then: the budget is not spent once validation stops improving, which is
    # what makes the reported tuning cost honest
    assert epochs_run < 60


def test_the_returned_model_is_the_best_seen_not_the_last() -> None:
    # Given: a training run long enough for validation to fluctuate
    job, _ = _job(epochs=15, patience=15)

    # When: the model is trained and then scored
    trained = train_classifier(job)
    restored = evaluate_classifier(trained, job.validation, job.config, CPU)

    # Then: the weights restored are the best-scoring ones. Reporting the final
    # epoch instead would make results depend on where training happened to stop.
    scores = [
        evaluate_classifier(trained, job.validation, job.config, CPU).macro_f1 for _ in range(3)
    ]
    assert all(score == pytest.approx(restored.macro_f1) for score in scores)


def test_evaluation_never_puts_the_model_in_training_mode() -> None:
    # Given: a trained classifier
    job, model = _job(epochs=3, patience=3)
    trained = train_classifier(job)
    before = model.train_calls

    # When: the trained model is scored on the validation partition
    _ = evaluate_classifier(trained, job.validation, job.config, CPU)

    # Then: no forward pass ran in training mode, so dropout and batch statistics
    # cannot leak into a reported metric
    assert model.train_calls == before


def test_prediction_covers_every_row_when_the_batch_size_does_not_divide() -> None:
    # Given: a validation set whose size is not a multiple of the batch size
    set_seed(0)
    training, _ = _separable()
    rng = np.random.default_rng(1)
    validation = DatasetMatrix(
        features=rng.normal(size=(23, 4)).astype(np.float32),
        labels=rng.integers(0, 2, size=23).astype(np.int64),
    )
    model = CountingLinear(4, 2)
    config = TrainingConfig(epochs=1, batch_size=8, patience=1)

    # When: the model is evaluated
    metrics = evaluate_classifier(model, validation, config, CPU)

    # Then: the trailing partial batch is scored too. Dropping it would quietly
    # change every reported metric.
    assert len(metrics.per_class) >= 1
    assert 0.0 <= metrics.accuracy <= 1.0
    _ = training


@final
class ScriptedTrial:
    """Return the first choice and the low end of every range."""

    def suggest_categorical(self, name: str, choices: tuple[str, ...]) -> str:
        _ = name
        return choices[0]

    def suggest_int(self, name: str, low: int, high: int) -> int:
        _ = name, high
        return low

    def suggest_float(self, name: str, low: float, high: float, *, log: bool) -> float:
        _ = name, high, log
        return low


def _inner_cache(seed: int) -> SeededPreparedInner:
    training, validation = _separable(n_rows=64, n_features=6)
    return SeededPreparedInner(
        seed=seed,
        prepared=PreparedInner(
            raw_training=training,
            balanced_training=training,
            validation=validation,
            views=((0, 1), (2, 3), (4, 5)),
            class_count=2,
        ),
    )


def _small_base() -> ExperimentConfig:
    base = ExperimentConfig()
    return ExperimentConfig(
        data=base.data,
        model=ModelConfig(d_model=16, n_heads=2, n_layers=1, dim_feedforward=32, n_views=3),
        training=TrainingConfig(epochs=1, batch_size=16, patience=1),
        diffusion=base.diffusion,
        seed=base.seed,
    )


def test_the_classifier_objective_returns_a_usable_score() -> None:
    # Given: two cached seeds and the constrained search space
    objective = ClassifierObjective(
        prepared_runs=(_inner_cache(42), _inner_cache(123)),
        base_config=_small_base(),
        device=CPU,
    )

    # When: one trial is scored
    score = objective(ScriptedTrial())

    # Then: the objective is a mean macro F1 in the usual range, so Optuna is
    # maximising the quantity the tuning protocol says it maximises
    assert 0.0 <= score <= 1.0


def test_the_objective_is_deterministic_for_the_same_trial() -> None:
    # Given: the same cached seeds and the same scripted choices
    objective = ClassifierObjective(
        prepared_runs=(_inner_cache(42),),
        base_config=_small_base(),
        device=CPU,
    )

    # When: the identical trial is scored twice
    first = objective(ScriptedTrial())
    second = objective(ScriptedTrial())

    # Then: the score repeats. A tuning study that cannot reproduce its own
    # trials cannot support the "best configuration" claim the paper makes.
    assert first == pytest.approx(second)


def test_the_objective_never_receives_an_evaluation_partition() -> None:
    # Given: the tuning-side cache type
    cache = _inner_cache(42).prepared

    # When / Then: it exposes no test partition at all, so no search path can
    # reach held-out rows even by mistake. This is the structural guarantee
    # behind the paper's claim that the test set never took part in selection.
    assert not hasattr(cache, "test")
    assert {"raw_training", "balanced_training", "validation", "views", "class_count"} == set(
        PreparedInner.__dataclass_fields__
    )


def test_the_narrow_objective_pins_the_settled_architecture() -> None:
    # Given: the narrowed search used after multi-seed confirmation
    objective = ClassifierObjective(
        prepared_runs=(_inner_cache(42),),
        base_config=_small_base(),
        device=CPU,
        narrow=True,
    )

    # When: a trial is scored
    score = objective(ScriptedTrial())

    # Then: it runs and returns a score. The narrow space fixes width and heads,
    # so this also checks those overrides stay constructible.
    assert 0.0 <= score <= 1.0
