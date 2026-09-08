"""Read the recorded class balance from runs that saved a balance report.

The prepared caches for three of the four datasets are no longer on disk, but
several completed runs stored the class counts before and after rebalancing.
Those reports are the evidence behind the ratios printed in Table 1, so they are
read here rather than reconstructed.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RESULTS = Path("/opt/ids_revision/results")

WANTED = (
    "nslkdd_semantic/nslkdd_seed42.json",
    "nslkdd_semantic/nslkdd_seed123.json",
    "unsw_xattn/unsw_seed42.json",
    "shift_experiment/unsw_keep100_seed42.json",
)


def ratio(counts: dict) -> float:
    """Return the majority-to-minority ratio for a class-count mapping."""
    values = [int(value) for value in counts.values()]
    return max(values) / min(values)


def main() -> None:
    """Print before/after class counts and ratios for each available run."""
    for relative in WANTED:
        path = RESULTS / relative
        if not path.is_file():
            print(f"{relative}: not present")
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))
        report = payload.get("balance_report", {})
        before = report.get("before")
        after = report.get("after")

        if not before or not after:
            print(f"{relative}: no before/after counts")
            continue

        print(f"\n{relative}")
        print(f"  before {before}   ratio {ratio(before):.2f} : 1")
        print(f"  after  {after}   ratio {ratio(after):.2f} : 1")
        recorded_before = report.get("imbalance_ratio_before")
        recorded_after = report.get("imbalance_ratio_after")
        if recorded_before is not None:
            print(f"  recorded {recorded_before:.4f} -> {recorded_after:.4f}")

    print("\nSEARCHING FOR ANY CIC BALANCE REPORT")
    for path in sorted(RESULTS.rglob("*.json")):
        if "cic" not in path.name.lower() and "cic" not in str(path.parent).lower():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        report = payload.get("balance_report", {})
        if report.get("before") and report.get("after"):
            print(f"  {path}")
            print(f"    before {report['before']} ratio {ratio(report['before']):.2f}")
            print(f"    after  {report['after']} ratio {ratio(report['after']):.2f}")


if __name__ == "__main__":
    main()
