"""Minority-class recall under extreme training-side scarcity (keep=0.05).

Averaged scores already favour gradient boosting. This asks the operational
question instead: with attacks thinned to a twentieth of the training rows and
the test set untouched, does diffusion augmentation recover attacks that the
unaugmented model misses?
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

RESULTS = Path("runs/minority")
SEEDS = (42, 123, 456)
VARIANTS = ("full_model", "without_diffusion", "without_multiview", "baseline", "xgboost_raw")
ATTACK_LABEL = 1
NORMAL_LABEL = 0


def load(seed: int) -> dict:
    path = RESULTS / f"keep005_seed{seed}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def class_entry(block: dict, label: int) -> dict | None:
    for entry in block.get("per_class", []):
        if entry["label"] == label:
            return entry
    return None


first = load(SEEDS[0])
report = first["shift_report"]
print("SHIFT APPLIED (training side only)")
print(f"  attacks {report['attacks_before']} -> {report['attacks_after']}")
print(f"  attack share {report['attack_share_before']:.3f} -> {report['attack_share_after']:.3f}")
print("  test partition untouched")

print("\nATTACK-CLASS RECALL (the minority at training time)")
print(f"  {'variant':22s} {'mean':>8s} {'sd':>7s}   per seed")
recalls: dict[str, list[float]] = {}
for name in VARIANTS:
    values = []
    for seed in SEEDS:
        entry = class_entry(load(seed)["variants"][name], ATTACK_LABEL)
        if entry is not None:
            values.append(entry["recall"] * 100)
    if not values:
        continue
    recalls[name] = values
    detail = "  ".join(f"{value:.2f}" for value in values)
    print(f"  {name:22s} {statistics.mean(values):8.2f} {statistics.stdev(values):7.2f}   {detail}")

print("\nNORMAL-CLASS RECALL (what is traded away)")
for name in VARIANTS:
    values = []
    for seed in SEEDS:
        entry = class_entry(load(seed)["variants"][name], NORMAL_LABEL)
        if entry is not None:
            values.append(entry["recall"] * 100)
    if values:
        print(f"  {name:22s} {statistics.mean(values):8.2f}")

print("\nDOES DIFFUSION RECOVER MISSED ATTACKS?")
if "full_model" in recalls and "without_diffusion" in recalls:
    deltas = [
        full - without
        for full, without in zip(recalls["full_model"], recalls["without_diffusion"], strict=True)
    ]
    consistent = len({value > 0 for value in deltas}) == 1
    detail = "  ".join(
        f"seed{seed}={value:+.2f}" for seed, value in zip(SEEDS, deltas, strict=True)
    )
    print(f"  full - without_diffusion: {detail}")
    print(f"  mean={statistics.mean(deltas):+.2f} consistent={consistent}")

print("\nVERSUS XGBOOST ON ATTACK RECALL")
if "full_model" in recalls and "xgboost_raw" in recalls:
    gap = statistics.mean(recalls["full_model"]) - statistics.mean(recalls["xgboost_raw"])
    print(f"  full_model - xgboost_raw = {gap:+.2f}")

print("\nMACRO F1 FOR REFERENCE")
for name in VARIANTS:
    values = [load(seed)["variants"][name]["macro_f1"] * 100 for seed in SEEDS]
    print(f"  {name:22s} {statistics.mean(values):8.2f}")
