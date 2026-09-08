"""Minority-class recall under controlled training-side shift.

The averaged scores already say this method loses to gradient boosting. The
operationally relevant question is different: when the attack class is rare in
training, does diffusion augmentation recover attacks that the unaugmented
model misses? Recall on the attack class answers that; macro F1 does not.

The test partition is never thinned, so the attack class stays ordinary at
evaluation time while becoming rare during training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from xgboost import XGBClassifier

from ids_diffusion.cli.common import experiment_config, model_config
from ids_diffusion.config import TrainingConfig
from ids_diffusion.data.splits import apply_attack_thinning
from ids_diffusion.data.unsw import load_unsw
from ids_diffusion.data.views import UNSW_SEMANTIC_VIEWS, named_view_indices
from ids_diffusion.training.metrics import classification_metrics
from ids_diffusion.training.pipeline import PipelineJob, RankMatchedDiffusionMultiViewPipeline
from ids_diffusion.training.prepare import PreparationJob, prepare_experiment
from ids_diffusion.types import FusionMode, LoadedDataset
from ids_diffusion.utils import set_seed, setup_logging


def parse_arguments() -> argparse.Namespace:
    """Parse the shift level, seed, and output path at the CLI boundary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    """Train the four variants plus XGBoost and record per-class scores."""
    arguments = parse_arguments()
    logger = setup_logging(log_file=arguments.output.with_suffix(".log"))
    set_seed(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = load_unsw(arguments.data_root)
    thinned, report = apply_attack_thinning(
        loaded.training,
        keep_fraction=arguments.keep_fraction,
        seed=arguments.seed,
    )
    dataset = LoadedDataset(
        training=thinned,
        test=loaded.test,
        feature_names=loaded.feature_names,
    )

    config = experiment_config(
        arguments.seed,
        arguments.output.parent,
        model_config(256, 8, 3, 768, 0.031162931169110973, FusionMode.LINEAR),
        TrainingConfig(
            epochs=30,
            batch_size=256,
            learning_rate=0.00013207025505264955,
            weight_decay=6.470199087689895e-06,
        ),
    )

    prepared = prepare_experiment(
        PreparationJob(
            dataset=dataset,
            views=named_view_indices(loaded.feature_names, UNSW_SEMANTIC_VIEWS),
            config=config,
            device=device,
        )
    )
    metrics = RankMatchedDiffusionMultiViewPipeline(config).run(
        PipelineJob(prepared=prepared, device=device)
    )

    booster = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        tree_method="hist",
        random_state=arguments.seed,
        n_jobs=arguments.threads,
        eval_metric="logloss",
    )
    booster.fit(prepared.raw_training.features, prepared.raw_training.labels)
    xgboost_metrics = classification_metrics(
        prepared.test.labels,
        np.asarray(booster.predict(prepared.test.features), dtype=np.int64),
    )

    payload = {
        "keep_fraction": arguments.keep_fraction,
        "seed": arguments.seed,
        "protocol": "official split, train-side attack thinning, test untouched",
        "shift_report": asdict(report),
        "variants": {
            "full_model": asdict(metrics.full_model),
            "without_diffusion": asdict(metrics.without_diffusion),
            "without_multiview": asdict(metrics.without_multiview),
            "baseline": asdict(metrics.baseline),
            "xgboost_raw": asdict(xgboost_metrics),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "minority evaluation written",
        extra={"path": str(arguments.output), "attacks_after": report.attacks_after},
    )


if __name__ == "__main__":
    main()
