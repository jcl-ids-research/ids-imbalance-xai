"""Inspect what per-class information the existing result files contain."""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("/opt/ids_revision/results/shift_experiment")
sample = RESULTS / "unsw_keep005_seed42.json"

payload = json.loads(sample.read_text(encoding="utf-8"))
print("top-level keys:", list(payload))
print("variants:", list(payload["variants"]))
print("metrics per variant:", list(payload["variants"]["full_model"]))
print()
print("balance report:", json.dumps(payload.get("balance_report", {}), indent=2)[:400])
print()
print("Does any file carry per-class or confusion data?")
for path in sorted(RESULTS.glob("*.json")):
    text = path.read_text(encoding="utf-8")
    markers = [
        marker
        for marker in ("per_class", "confusion", "recall_0", "recall_1", "support", "classes")
        if marker in text
    ]
    if markers:
        print(f"  {path.name}: {markers}")
print("  scan complete")
