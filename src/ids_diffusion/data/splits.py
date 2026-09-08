"""Pure split operations shared by training and controlled-shift experiments."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from ids_diffusion.errors import ConfigurationError
from ids_diffusion.types import DatasetMatrix, DatasetSplit, ShiftReport


def apply_attack_thinning(
    dataset: DatasetMatrix,
    keep_fraction: float,
    seed: int,
) -> tuple[DatasetMatrix, ShiftReport]:
    """Retain every normal row and a seeded fraction of attack rows.

    This operation changes the training side only. Callers own the untouched
    test matrix, which prevents a shift experiment from contaminating its
    evaluation set.
    """
    if not 0.0 < keep_fraction <= 1.0:
        raise ConfigurationError(
            field="keep_fraction",
            detail=f"expected (0, 1], got {keep_fraction}",
        )

    normal_indices = np.flatnonzero(dataset.labels == 0)
    attack_indices = np.flatnonzero(dataset.labels == 1)
    attack_count = max(1, round(len(attack_indices) * keep_fraction))

    generator = np.random.default_rng(seed)
    retained_attacks = generator.choice(attack_indices, size=attack_count, replace=False)
    retained = np.concatenate((normal_indices, retained_attacks))
    generator.shuffle(retained)

    shifted = DatasetMatrix(
        features=np.asarray(dataset.features[retained], dtype=np.float32),
        labels=np.asarray(dataset.labels[retained], dtype=np.int64),
    )
    report = ShiftReport(
        keep_fraction=keep_fraction,
        attacks_before=len(attack_indices),
        attacks_after=attack_count,
        normals=len(normal_indices),
        attack_share_before=len(attack_indices) / len(dataset.labels),
        attack_share_after=attack_count / len(retained),
    )
    return shifted, report


def stratified_validation_split(
    training: DatasetMatrix,
    test: DatasetMatrix,
    validation_fraction: float,
    seed: int,
) -> DatasetSplit:
    """Split validation rows from training while leaving the real test set intact."""
    if not 0.0 < validation_fraction < 1.0:
        raise ConfigurationError(
            field="validation_fraction",
            detail=f"expected (0, 1), got {validation_fraction}",
        )

    train_x, val_x, train_y, val_y = train_test_split(
        training.features,
        training.labels,
        test_size=validation_fraction,
        random_state=seed,
        stratify=training.labels,
    )
    return DatasetSplit(
        train=DatasetMatrix(
            features=np.asarray(train_x, dtype=np.float32),
            labels=np.asarray(train_y, dtype=np.int64),
        ),
        validation=DatasetMatrix(
            features=np.asarray(val_x, dtype=np.float32),
            labels=np.asarray(val_y, dtype=np.int64),
        ),
        test=test,
    )


__all__ = ["apply_attack_thinning", "stratified_validation_split"]
