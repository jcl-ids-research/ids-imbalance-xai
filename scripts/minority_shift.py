"""Backward-compatible launcher for the training-side scarcity experiment.

The maintained entry point is `ids-scarcity` (see `src/ids_diffusion/cli/scarcity.py`).
This script stays only so existing server wrappers keep working; it forwards its
arguments unchanged and should not grow new logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ids_diffusion.cli.scarcity import run as run_scarcity
from ids_diffusion.types import DeviceChoice


def main() -> None:
    """Parse the legacy flags and forward them to the maintained command."""
    parser = argparse.ArgumentParser(description="Legacy launcher for ids-scarcity")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    arguments = parser.parse_args()

    run_scarcity(
        data_root=arguments.data_root,
        output=arguments.output,
        keep_fraction=arguments.keep_fraction,
        seed=arguments.seed,
        threads=arguments.threads,
        device=DeviceChoice(arguments.device),
    )


if __name__ == "__main__":
    sys.exit(main())
