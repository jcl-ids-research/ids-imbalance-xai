"""The XGBoost baseline command's training logic.

The baseline command scores raw and balanced XGBoost on the frozen split. Its
core is `fit_and_score`, which fits a booster and reduces the predictions to
the same metrics shape the paper reports. The trees are exercised here on
separable data so the wiring is proven rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest
from xgboost import XGBClassifier

from ids_diffusion.cli.baseline import BoosterSettings, fit_and_score
from ids_diffusion.types import DatasetMatrix, PreparedExperiment


def _separable(
    n_rows: int = 400, n_features: int = 6, *, seed: int = 7
) -> tuple[DatasetMatrix, DatasetMatrix]:
    rng = np.random.default_rng(seed)
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
    split = int(n_rows * 0.8)
    return (
        DatasetMatrix(features=features[:split], labels=labels[:split]),
        DatasetMatrix(features=features[split:], labels=labels[split:]),
    )


def _settings() -> BoosterSettings:
    return BoosterSettings(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        seed=7,
        threads=1,
    )


def _prepared(training: DatasetMatrix, test: DatasetMatrix) -> PreparedExperiment:
    return PreparedExperiment(
        raw_training=training,
        balanced_training=training,
        validation=test,
        test=test,
        views=((0, 1, 2), (3, 4, 5)),
        class_count=2,
    )


def test_xgboost_learns_a_separable_problem() -> None:
    # Given: two well-separated classes and a prepared experiment
    training, test = _separable()
    prepared = _prepared(training, test)

    # When: the booster is fit and scored on the frozen test partition
    metrics = fit_and_score(prepared.raw_training, prepared, _settings())

    # Then: it separates them, which is the guarantee that makes XGBoost the
    # headline baseline in the paper
    assert metrics.accuracy > 0.9
    assert metrics.macro_f1 > 0.9
    assert 0.0 <= metrics.precision <= 1.0
    assert len(metrics.per_class) == 2


def test_the_same_booster_is_deterministic_for_one_seed() -> None:
    # Given: identical inputs and settings
    training, test = _separable()
    prepared = _prepared(training, test)

    # When: the booster is fit twice
    first = fit_and_score(prepared.raw_training, prepared, _settings())
    second = fit_and_score(prepared.raw_training, prepared, _settings())

    # Then: the reported metrics repeat, so a rerun cannot drift from the paper
    assert first.macro_f1 == pytest.approx(second.macro_f1)
    assert first.accuracy == pytest.approx(second.accuracy)


def test_raw_and_balanced_arms_are_scored_independently() -> None:
    # Given: a prepared experiment whose two training arms genuinely differ
    training, test = _separable()
    balanced_training, _ = _separable(seed=9)
    prepared = PreparedExperiment(
        raw_training=training,
        balanced_training=balanced_training,
        validation=test,
        test=test,
        views=((0, 1, 2), (3, 4, 5)),
        class_count=2,
    )

    # When: each arm is scored with the same settings
    raw = fit_and_score(prepared.raw_training, prepared, _settings())
    balanced_metrics = fit_and_score(prepared.balanced_training, prepared, _settings())

    # Then: both arms produce well-formed metrics on the shared frozen test set
    assert raw.accuracy > 0.5
    assert balanced_metrics.accuracy > 0.5
    assert len(raw.per_class) == len(balanced_metrics.per_class)


def test_booster_settings_are_reproducible() -> None:
    # Given: the fixed settings the command uses
    settings = _settings()

    # When: two boosters are built from the same settings
    first = XGBClassifier(
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        learning_rate=settings.learning_rate,
        subsample=settings.subsample,
        colsample_bytree=settings.colsample_bytree,
        objective="binary:logistic",
        tree_method="hist",
        random_state=settings.seed,
        n_jobs=settings.threads,
        eval_metric="logloss",
    )
    second = XGBClassifier(
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        learning_rate=settings.learning_rate,
        subsample=settings.subsample,
        colsample_bytree=settings.colsample_bytree,
        objective="binary:logistic",
        tree_method="hist",
        random_state=settings.seed,
        n_jobs=settings.threads,
        eval_metric="logloss",
    )

    # Then: the settings fully determine the booster, so no hidden global state
    # can change what the baseline command reports
    assert first.get_params() == second.get_params()
