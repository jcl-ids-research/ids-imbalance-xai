"""Train gradient-boosted trees on the identical split used by RM-DMVT.

Both arms are reported. The raw arm is the fair headline comparison, because
tree ensembles handle imbalance natively and gain nothing from diffusion-based
augmentation; the balanced arm is reported so the augmented inputs are not
silently withheld from the baseline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import typer
from xgboost import XGBClassifier

from ids_diffusion.data.cache import load_prepared
from ids_diffusion.training.metrics import classification_metrics
from ids_diffusion.types import ClassificationMetrics, DatasetMatrix, PreparedExperiment
from ids_diffusion.utils import set_seed, setup_logging

app = typer.Typer(add_completion=False, help=__doc__)

BINARY_CLASS_COUNT = 2


@dataclass(frozen=True, slots=True)
class BoosterSettings:
    """Fixed booster settings shared by both training arms."""

    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    seed: int
    threads: int


def _fit_and_score(
    training: DatasetMatrix,
    prepared: PreparedExperiment,
    settings: BoosterSettings,
) -> ClassificationMetrics:
    model = XGBClassifier(
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        learning_rate=settings.learning_rate,
        subsample=settings.subsample,
        colsample_bytree=settings.colsample_bytree,
        objective=(
            "binary:logistic" if prepared.class_count <= BINARY_CLASS_COUNT else "multi:softprob"
        ),
        tree_method="hist",
        random_state=settings.seed,
        n_jobs=settings.threads,
        eval_metric="logloss",
    )
    model.fit(training.features, training.labels)
    predictions = model.predict(prepared.test.features)
    return classification_metrics(prepared.test.labels, predictions)


@app.command()
def run(
    cache: Annotated[Path, typer.Option(help="Holdout cache produced by ids-prepare-holdout")],
    output: Annotated[Path, typer.Option(help="Metrics JSON output")],
    seed: Annotated[int, typer.Option()] = 42,
    n_estimators: Annotated[int, typer.Option()] = 400,
    max_depth: Annotated[int, typer.Option()] = 8,
    learning_rate: Annotated[float, typer.Option()] = 0.1,
    subsample: Annotated[float, typer.Option()] = 0.9,
    colsample_bytree: Annotated[float, typer.Option()] = 0.9,
    threads: Annotated[int, typer.Option(help="CPU threads for the booster")] = 16,
) -> None:
    """Score both arms on the same frozen holdout as the neural variants."""
    logger = setup_logging(log_file=output.with_suffix(".log"))
    set_seed(seed)
    prepared = load_prepared(cache)
    settings = BoosterSettings(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        seed=seed,
        threads=threads,
    )
    raw = _fit_and_score(prepared.raw_training, prepared, settings)
    balanced = _fit_and_score(prepared.balanced_training, prepared, settings)
    payload = {
        "xgboost_raw": asdict(raw),
        "xgboost_balanced": asdict(balanced),
        "settings": asdict(settings),
        "holdout_rows": len(prepared.test.labels),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "xgboost baseline written",
        extra={
            "path": str(output),
            "raw_macro_f1": payload["xgboost_raw"]["macro_f1"],
            "balanced_macro_f1": payload["xgboost_balanced"]["macro_f1"],
        },
    )


if __name__ == "__main__":
    app()
