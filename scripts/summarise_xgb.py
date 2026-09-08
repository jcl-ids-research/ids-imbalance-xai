"""Summarise the XGBoost baseline measured on the frozen outer holdout."""

import json
import statistics
from pathlib import Path

rows: dict[str, list[float]] = {"xgboost_raw": [], "xgboost_balanced": []}

for path in sorted(Path("runs/outer_final").glob("xgb_seed*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    seed = path.stem.replace("xgb_seed", "")
    line = [f"seed{seed}"]
    for arm, collected in rows.items():
        value = payload[arm]["macro_f1"] * 100
        collected.append(value)
        line.append(f"{arm}={value:.3f}")
    print("  ".join(line))

print()
for arm, values in rows.items():
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f"{arm:20s} {mean:.3f} +/- {sd:.3f}")
