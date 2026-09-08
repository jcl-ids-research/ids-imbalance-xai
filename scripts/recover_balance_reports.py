"""Recover the recorded balance report for all four datasets.

The UNSW ratio was recomputed from its cache, but the other three caches are not
on disk. Their runs recorded a balance report at the time, which carries the
per-class counts before and after; this reads those reports so every ratio in
Table 1 rests on a measurement rather than on recollection.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOTS = (
    Path("/opt/ids_revision/results/ablation_cross_correct"),
    Path("/opt/ids_revision/results/ablation_repeat_correct"),
    Path("/opt/ids_revision/results/phase1_corrected"),
    Path("/opt/ids_revision/results/instrumented"),
)


def ratio(counts: dict) -> str:
    """Return the majority-to-minority ratio for a class-count mapping."""
    values = [int(v) for v in counts.values()]
    if not values or min(values) == 0:
        return "n/a"
    return f"{max(values) / min(values):.2f} : 1"


def main() -> None:
    """Print every balance report found, grouped by file."""
    seen = 0
    for root in ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            report = payload.get("balance_report")
            if not isinstance(report, dict):
                continue

            before = report.get("before")
            after = report.get("after")
            if not isinstance(before, dict) or not isinstance(after, dict):
                continue

            seen += 1
            print(f"\n{path.relative_to(path.parents[1])}")
            print(f"  before {before}  ratio {ratio(before)}")
            print(f"  after  {after}  ratio {ratio(after)}")
            if "imbalance_ratio_before" in report:
                print(
                    f"  recorded: {report['imbalance_ratio_before']:.4f}"
                    f" -> {report.get('imbalance_ratio_after', float('nan')):.4f}"
                )

    if seen == 0:
        print("no balance reports found in the searched directories")


if __name__ == "__main__":
    main()
