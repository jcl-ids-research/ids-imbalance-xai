"""Read the completed controlled-shift experiment.

The hypothesis under test: diffusion augmentation earns its advantage under
distribution shift, so the gap to XGBoost should narrow as training-side
attacks are thinned while the test set stays intact.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

RESULTS = Path("/opt/ids_revision/results/shift_experiment")
LEVELS = (
    ("keep100", 1.00),
    ("keep050", 0.50),
    ("keep020", 0.20),
    ("keep010", 0.10),
    ("keep005", 0.05),
)
SEEDS = (42, 123, 456)
VARIANTS = ("full_model", "wo_diffusion", "wo_multiview", "baseline")


def macro(entry: dict) -> float | None:
    for key in ("f1_macro", "macro_f1"):
        if key in entry:
            value = float(entry[key])
            return value * 100 if value <= 1.0 else value
    return None


def load(tag: str, seed: int) -> dict | None:
    path = RESULTS / f"unsw_{tag}_seed{seed}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("variants", {})


print("FULL MODEL vs XGBOOST AS SHIFT DEEPENS")
print(f"  {'keep':>6s} {'seeds':>6s} {'full':>8s} {'xgb':>8s} {'gap':>8s}")
rows: list[tuple[float, float, float, float]] = []
for tag, keep in LEVELS:
    full_values: list[float] = []
    xgb_values: list[float] = []
    for seed in SEEDS:
        variants = load(tag, seed)
        if variants is None:
            continue
        full = macro(variants.get("full_model", {}))
        xgb = macro(variants.get("xgboost_balanced", {}))
        if full is not None:
            full_values.append(full)
        if xgb is not None:
            xgb_values.append(xgb)
    if not full_values or not xgb_values:
        print(f"  {keep:6.2f} {'-':>6s} incomplete")
        continue
    full_mean = statistics.mean(full_values)
    xgb_mean = statistics.mean(xgb_values)
    gap = full_mean - xgb_mean
    rows.append((keep, full_mean, xgb_mean, gap))
    print(f"  {keep:6.2f} {len(full_values):6d} {full_mean:8.2f} {xgb_mean:8.2f} {gap:+8.2f}")

print("\nCOMPONENT CONTRIBUTION BY SHIFT LEVEL")
print(f"  {'keep':>6s} {'diffusion':>12s} {'multi-view':>12s} {'both':>10s} {'consistent':>11s}")
for tag, keep in LEVELS:
    diffusion: list[float] = []
    multiview: list[float] = []
    both: list[float] = []
    for seed in SEEDS:
        variants = load(tag, seed)
        if variants is None:
            continue
        full = macro(variants.get("full_model", {}))
        without_diffusion = macro(variants.get("wo_diffusion", {}))
        without_multiview = macro(variants.get("wo_multiview", {}))
        base = macro(variants.get("baseline", {}))
        if full is not None and without_diffusion is not None:
            diffusion.append(full - without_diffusion)
        if full is not None and without_multiview is not None:
            multiview.append(full - without_multiview)
        if full is not None and base is not None:
            both.append(full - base)
    if not diffusion:
        continue
    consistent = len({value > 0 for value in diffusion}) == 1
    print(
        f"  {keep:6.2f} {statistics.mean(diffusion):+12.2f} "
        f"{statistics.mean(multiview):+12.2f} {statistics.mean(both):+10.2f} "
        f"{consistent!s:>11s}"
    )

print("\nREADING")
if len(rows) >= 2:
    gaps = [row[3] for row in rows]
    print(f"  gap, least shift first: {'  '.join(f'{gap:+.2f}' for gap in gaps)}")
    if gaps[0] < gaps[-1]:
        print("  -> gap narrows as shift deepens: consistent with the hypothesis")
    else:
        print("  -> gap does not narrow: hypothesis not supported")
