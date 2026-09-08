"""Reconcile every headline figure against the runs, whichever layout they use.

phase1_corrected stores UNSW as unsw/<seed>/metrics.json but the other three
datasets as <dataset>_<seed>.json at the top level. baseline_fair uses the
nested form throughout. This handles both so that a missing match means a real
discrepancy rather than a path that was not tried.
"""

from __future__ import annotations

import io
import json
import statistics
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path("/opt/ids_revision/deploy/results")
SEEDS = (42, 123, 456)
TOLERANCE = 0.06

CLAIMED_FULL = {
    "unsw": 88.90,
    "nslkdd": 82.21,
    "cicids2017": 98.20,
    "cicddos2019": 95.85,
}

CLAIMED_BASELINE = {
    ("unsw", "XGBoost__balanced"): 90.20,
    ("nslkdd", "XGBoost__balanced"): 81.11,
    ("cicids2017", "RandomForest__balanced"): 99.83,
    ("cicddos2019", "XGBoost__balanced"): 99.55,
}

CLAIMED_MULTICLASS = {
    ("unsw", "full_model"): 45.93,
    ("nslkdd", "full_model"): 64.30,
}


def macro(entry: dict) -> float | None:
    """Return macro F1 as a percentage regardless of key or scale."""
    for key in ("f1_macro", "macro_f1"):
        if key in entry:
            value = float(entry[key])
            return value * 100 if value <= 1.0 else value
    return None


def candidates(experiment: str, dataset: str, seed: int) -> list[Path]:
    """Return both known file layouts for one dataset and seed."""
    base = ROOT / experiment
    return [
        base / dataset / f"seed{seed}" / "metrics.json",
        base / f"{dataset}_seed{seed}.json",
    ]


def collect(experiment: str, dataset: str, variant: str) -> list[float]:
    """Gather a variant's macro F1 across seeds, trying each layout."""
    values: list[float] = []
    for seed in SEEDS:
        for path in candidates(experiment, dataset, seed):
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            block = payload.get("variants", payload).get(variant)
            if isinstance(block, dict):
                score = macro(block)
                if score is not None:
                    values.append(score)
            break
    return values


def report(label: str, claimed: float, values: list[float]) -> bool:
    """Print one reconciliation line; return whether it matched."""
    if not values:
        print(f"  {label:32s} claimed {claimed:6.2f}   no runs found")
        return False
    mean = statistics.mean(values)
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    ok = abs(mean - claimed) < TOLERANCE
    print(
        f"  {label:32s} claimed {claimed:6.2f}   computed {mean:6.2f} ± {spread:4.2f}"
        f"  n={len(values)}  {'matches' if ok else 'DIFFERS'}"
    )
    return ok


def main() -> None:
    """Reconcile binary, baseline and multi-class figures."""
    total = 0
    matched = 0

    print("BINARY FULL MODEL (phase1_corrected)")
    for dataset, claimed in CLAIMED_FULL.items():
        total += 1
        matched += int(report(dataset, claimed, collect("phase1_corrected", dataset, "full_model")))

    print("\nCLASSICAL BASELINES (baseline_fair)")
    for (dataset, model), claimed in CLAIMED_BASELINE.items():
        total += 1
        observed = collect("baseline_fair", dataset, model)
        matched += int(report(f"{dataset} / {model}", claimed, observed))

    print("\nMULTI-CLASS (multiclass)")
    for (dataset, variant), claimed in CLAIMED_MULTICLASS.items():
        total += 1
        observed = collect("multiclass", dataset, variant)
        matched += int(report(f"{dataset} / {variant}", claimed, observed))

    print(f"\nreconciled {matched} of {total}")


if __name__ == "__main__":
    main()
