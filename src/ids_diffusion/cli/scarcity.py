"""Minority-class recall under controlled training-side attack thinning.

The averaged scores already show this method loses to gradient boosting on
macro F1. The operationally relevant question is different: when the attack
class is rare in training, does diffusion augmentation recover attacks the
unaugmented model misses? Attack-class recall answers that; macro F1 does not.

The test partition is never thinned, so the attack class stays ordinary at
evaluation while becoming rare during training.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import typer
from xgboost import XGBClassifier

from ids_diffusion.cli.common import experiment_config, model_config, parse_device
from ids_diffusion.config import TrainingConfig
from ids_diffusion.data.splits import apply_attack_thinning
from ids_diffusion.data.unsw import load_unsw
from ids_diffusion.data.views import UNSW_SEMANTIC_VIEWS, named_view_indices
from ids_diffusion.reproduction.archive_io import JsonObject
from ids_diffusion.training.metrics import classification_metrics
from ids_diffusion.training.pipeline import PipelineJob, RankMatchedDiffusionMultiViewPipeline
from ids_diffusion.training.prepare import PreparationJob, prepare_experiment
from ids_diffusion.types import (
    ClassificationMetrics,
    DeviceChoice,
    FusionMode,
    LoadedDataset,
    PreparedExperiment,
    ShiftReport,
)
from ids_diffusion.utils import set_seed, setup_logging

app = typer.Typer(add_completion=False, help=__doc__)


@dataclass(frozen=True, slots=True)
class ScarcityReport:
    """One thinning run: the shift, the four variants and XGBoost."""

    keep_fraction: float
    seed: int
    protocol: str
    shift_report: ShiftReport
    full_model: ClassificationMetrics
    without_diffusion: ClassificationMetrics
    without_multiview: ClassificationMetrics
    baseline: ClassificationMetrics
    xgboost_raw: ClassificationMetrics


def _score_xgboost(
    prepared: PreparedExperiment,
    seed: int,
    threads: int,
) -> ClassificationMetrics:
    booster = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        tree_method="hist",
        random_state=seed,
        n_jobs=threads,
        eval_metric="logloss",
    )
    _ = booster.fit(prepared.raw_training.features, prepared.raw_training.labels)
    return classification_metrics(
        prepared.test.labels,
        np.asarray(booster.predict(prepared.test.features), dtype=np.int64),
    )


def metrics_to_payload(report: ScarcityReport) -> JsonObject:
    """Serialise one run to the shape the archived results already use.

    `dataclasses.asdict` is untyped, but every field here is a JSON primitive or
    a nested mapping of primitives, so the casts describe exactly the shape that
    reaches disk and that the claim checker later reads back.
    """
    return {
        "keep_fraction": report.keep_fraction,
        "seed": report.seed,
        "protocol": report.protocol,
        "shift_report": cast("JsonObject", asdict(report.shift_report)),
        "variants": {
            "full_model": cast("JsonObject", asdict(report.full_model)),
            "without_diffusion": cast("JsonObject", asdict(report.without_diffusion)),
            "without_multiview": cast("JsonObject", asdict(report.without_multiview)),
            "baseline": cast("JsonObject", asdict(report.baseline)),
            "xgboost_raw": cast("JsonObject", asdict(report.xgboost_raw)),
        },
    }


@app.command()
def run(
    data_root: Annotated[Path, typer.Option(help="Directory containing UNSW CSV files")],
    output: Annotated[Path, typer.Option(help="Metrics JSON output")],
    keep_fraction: Annotated[float, typer.Option(help="Fraction of attack rows to keep")],
    seed: Annotated[int, typer.Option()] = 42,
    threads: Annotated[int, typer.Option(help="CPU threads for the booster")] = 16,
    device: Annotated[DeviceChoice, typer.Option()] = DeviceChoice.AUTO,
) -> None:
    """Train the four variants plus XGBoost and record per-class scores."""
    logger = setup_logging(log_file=output.with_suffix(".log"))
    set_seed(seed)
    loaded = load_unsw(data_root)
    thinned, shift = apply_attack_thinning(loaded.training, keep_fraction, seed)
    dataset = LoadedDataset(
        training=thinned,
        test=loaded.test,
        feature_names=loaded.feature_names,
    )

    config = experiment_config(
        seed,
        output.parent,
        model_config(256, 8, 3, 768, 0.031162931169110973, FusionMode.LINEAR),
        TrainingConfig(
            epochs=30,
            batch_size=256,
            learning_rate=0.00013207025505264955,
            weight_decay=6.470199087689895e-06,
        ),
    )
    compute = parse_device(device)

    prepared = prepare_experiment(
        PreparationJob(
            dataset=dataset,
            views=named_view_indices(loaded.feature_names, UNSW_SEMANTIC_VIEWS),
            config=config,
            device=compute,
        )
    )
    metrics = RankMatchedDiffusionMultiViewPipeline(config).run(
        PipelineJob(prepared=prepared, device=compute)
    )
    xgboost = _score_xgboost(prepared, seed, threads)

    report = ScarcityReport(
        keep_fraction=keep_fraction,
        seed=seed,
        protocol="official split, train-side attack thinning, test untouched",
        shift_report=shift,
        full_model=metrics.full_model,
        without_diffusion=metrics.without_diffusion,
        without_multiview=metrics.without_multiview,
        baseline=metrics.baseline,
        xgboost_raw=xgboost,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(json.dumps(metrics_to_payload(report), indent=2), encoding="utf-8")
    _ = logger.info(
        "minority evaluation written",
        extra={"path": str(output), "attacks_after": shift.attacks_after},
    )


if __name__ == "__main__":
    app()
