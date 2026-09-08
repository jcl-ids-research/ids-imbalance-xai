"""Print the structure of a baseline metrics file so the right key can be used.

The baseline reconciliation found no runs, which is far more likely to be a key
mismatch than missing data - the files exist and carry a balance report. This
shows the top-level keys and the shape of whatever holds the model scores.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SAMPLE = Path("/opt/ids_revision/deploy/results/baseline_fair/unsw/seed42/metrics.json")


def main() -> None:
    """Describe the file's layout and locate the model scores."""
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))

    print("TOP-LEVEL KEYS")
    for key, value in payload.items():
        kind = type(value).__name__
        preview = ""
        if isinstance(value, dict):
            preview = f"  keys: {list(value)[:8]}"
        print(f"  {key:22s} {kind:6s}{preview}")

    print("\nCANDIDATE SCORE BLOCKS")
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        for name, entry in value.items():
            if isinstance(entry, dict) and any(
                metric in entry for metric in ("f1_macro", "macro_f1")
            ):
                score = entry.get("f1_macro", entry.get("macro_f1"))
                print(f"  {key}.{name}: macro F1 = {score}")


if __name__ == "__main__":
    main()
