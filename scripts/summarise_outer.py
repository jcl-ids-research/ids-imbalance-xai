"""Summarise the outer-holdout experiment: variants, components, and baseline."""

import json
import statistics
from pathlib import Path

VARIANTS = ("full_model", "without_diffusion", "without_multiview", "baseline")
FINAL = Path("runs/outer_final")

per_seed: dict[str, dict[str, float]] = {}
for path in sorted(FINAL.glob("seed*.json")):
    seed = path.stem.replace("seed", "")
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_seed[seed] = {name: payload[name]["macro_f1"] * 100 for name in VARIANTS}

print("PER SEED MACRO F1 (frozen outer holdout)")
for seed, scores in per_seed.items():
    line = "  ".join(f"{name}={scores[name]:.3f}" for name in VARIANTS)
    print(f"  seed{seed}: {line}")

print("\nMEAN +/- SD")
means: dict[str, float] = {}
for name in VARIANTS:
    values = [scores[name] for scores in per_seed.values()]
    mean = statistics.mean(values)
    means[name] = mean
    print(f"  {name:20s} {mean:.3f} +/- {statistics.stdev(values):.3f}")

print("\nCOMPONENT CONTRIBUTION")
pairs = (
    ("diffusion", "without_diffusion"),
    ("multi-view", "without_multiview"),
    ("both", "baseline"),
)
for label, other in pairs:
    deltas = [scores["full_model"] - scores[other] for scores in per_seed.values()]
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    same_sign = len({value > 0 for value in deltas}) == 1
    detail = "  ".join(
        f"seed{seed}={value:+.3f}" for seed, value in zip(per_seed, deltas, strict=True)
    )
    print(f"  {label:11s} {detail}  mean={mean:+.3f} sd={sd:.3f} consistent={same_sign}")

print("\nXGBOOST ON THE IDENTICAL SPLIT")
xgb: dict[str, list[float]] = {"xgboost_raw": [], "xgboost_balanced": []}
for path in sorted(FINAL.glob("xgb_seed*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for arm, collected in xgb.items():
        collected.append(payload[arm]["macro_f1"] * 100)
for arm, values in xgb.items():
    print(f"  {arm:20s} {statistics.mean(values):.3f} +/- {statistics.stdev(values):.3f}")

print("\nHEADLINE COMPARISON (same split, same preprocessing)")
gap_raw = means["full_model"] - statistics.mean(xgb["xgboost_raw"])
gap_balanced = means["full_model"] - statistics.mean(xgb["xgboost_balanced"])
print(f"  RM-DMVT full        {means['full_model']:.3f}")
print(f"  vs xgboost_raw      {gap_raw:+.3f}")
print(f"  vs xgboost_balanced {gap_balanced:+.3f}")
