"""Shared CLI parsing and output helpers."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import torch
from typing_extensions import assert_never

from ids_diffusion.config import ExperimentConfig, ModelConfig, TrainingConfig
from ids_diffusion.data.cache import load_inner
from ids_diffusion.errors import ConfigurationError
from ids_diffusion.types import (
    DeviceChoice,
    FusionMode,
    SeededPreparedInner,
    VariantMetrics,
)

SEED_PATTERN = re.compile(r"seed(\d+)")


def parse_device(value: DeviceChoice) -> torch.device:
    """Parse cpu/cuda/auto without silently selecting unavailable CUDA."""
    match value:
        case DeviceChoice.AUTO:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        case DeviceChoice.CPU:
            return torch.device("cpu")
        case DeviceChoice.CUDA:
            if not torch.cuda.is_available():
                raise ConfigurationError(field="device", detail="CUDA is unavailable")
            return torch.device("cuda")
        case unreachable:
            assert_never(unreachable)


def load_seeded_caches(paths: tuple[Path, ...]) -> tuple[SeededPreparedInner, ...]:
    """Load tuning caches and infer their seeds from filenames such as seed42.npz."""
    runs: list[SeededPreparedInner] = []
    for path in paths:
        match = SEED_PATTERN.search(path.stem)
        if match is None:
            raise ConfigurationError(
                field="cache filename",
                detail=f"{path.name!r} must contain seed<number>",
            )
        runs.append(SeededPreparedInner(seed=int(match.group(1)), prepared=load_inner(path)))
    return tuple(runs)


def experiment_config(
    seed: int,
    output_root: Path,
    model: ModelConfig,
    training: TrainingConfig,
) -> ExperimentConfig:
    """Build the immutable config shared by CLI commands."""
    return ExperimentConfig(
        name="rm-dmvt",
        seed=seed,
        output_root=output_root,
        model=model,
        training=training,
    )


def write_metrics(path: Path, metrics: VariantMetrics) -> None:
    """Persist JSON metrics in the same four-variant shape used by the paper."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")


def model_config(
    d_model: int,
    n_heads: int,
    n_layers: int,
    feedforward: int,
    dropout: float,
    fusion: FusionMode,
) -> ModelConfig:
    """Parse architecture flags at the CLI boundary."""
    return ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=feedforward,
        dropout=dropout,
        fusion_mode=fusion,
    )


__all__ = [
    "experiment_config",
    "load_seeded_caches",
    "model_config",
    "parse_device",
    "write_metrics",
]
