"""Reproducibility, logging and artefact paths.

Kept deliberately small: seeding and path handling are the two places where a
silent inconsistency invalidates a comparison, so they live in one file that
is easy to audit.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

_LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every generator a run touches.

    ``deterministic`` trades throughput for exact repeatability. Leave it off
    for sweeps, turn it on when a specific result has to be reproduced
    bit-for-bit.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  # sklearn consumes NumPy's global RNG
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the compute device, reporting what was actually chosen."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure the package logger.

    Console output goes to stdout so it interleaves correctly with progress
    printed by long-running loops. A file handler is added when a path is
    given, which is how unattended runs leave a trace.
    """
    logger = logging.getLogger("ids_diffusion")
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Child logger under the package root, so one call configures all."""
    return logging.getLogger(f"ids_diffusion.{name}")


def ensure_dir(path: Path) -> Path:
    """Create a directory tree and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "ensure_dir",
    "get_device",
    "get_logger",
    "set_seed",
    "setup_logging",
]
