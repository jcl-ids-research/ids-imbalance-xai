"""Short GPU smoke test for RM-DMVT model construction and optimisation.

This is deliberately synthetic and small. It proves the deployed package can
construct both fusion modes, run a backward pass and update parameters without
requiring any public dataset or starting a long job.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(PROJECT_SRC))

from ids_diffusion.config import ModelConfig, TrainingConfig
from ids_diffusion.models.classifier import IntrusionClassifier
from ids_diffusion.models.multiview import MultiViewEncoder
from ids_diffusion.training.classifier import (
    ClassifierTrainingJob,
    evaluate_classifier,
    train_classifier,
)
from ids_diffusion.types import DatasetMatrix, FusionMode
from ids_diffusion.utils import set_seed


def main() -> int:
    """Run two epochs on 96 synthetic records and print the observable result."""
    set_seed(42, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = np.random.default_rng(42)
    features = generator.normal(size=(96, 6)).astype(np.float32)
    labels = (features[:, 0] + features[:, 3] > 0).astype(np.int64)
    training = DatasetMatrix(features=features[:64], labels=labels[:64])
    validation = DatasetMatrix(features=features[64:80], labels=labels[64:80])
    test = DatasetMatrix(features=features[80:], labels=labels[80:])
    training_config = TrainingConfig(
        epochs=2,
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=1e-5,
        patience=2,
    )
    print(f"device={device}")
    print(f"test_rows={len(test.labels)}")
    for mode in FusionMode:
        model_config = ModelConfig(
            d_model=16,
            n_heads=4,
            n_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            n_views=3,
            fusion_mode=mode,
        )
        model = IntrusionClassifier(
            MultiViewEncoder(model_config, ((0, 1), (2, 3), (4, 5))),
            model_config,
            class_count=2,
        )
        trained = train_classifier(
            ClassifierTrainingJob(
                model=model,
                training=training,
                validation=validation,
                config=training_config,
                device=device,
            )
        )
        metrics = evaluate_classifier(trained, test, training_config, device)
        print(f"fusion={mode.value} macro_f1={metrics.macro_f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
