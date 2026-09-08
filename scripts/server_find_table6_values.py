"""Find the exact source of every number printed in Table 6.

The fidelity dump contains the current per-dataset metrics, but Table 6 is a
compact held-out-data summary (MMD, KL, JS, Wasserstein, correlation, coverage)
and does not map one-to-one to that record. This scans every JSON, CSV, Markdown
and text artifact under the authoritative results tree for the printed values,
then prints their path and context. A number is not accepted merely because it
looks plausible; it must lead back to a stored artifact.
"""

from __future__ import annotations

import re
from pathlib import Path

BASE = Path("/opt/ids_revision")
TARGETS = ("0.144", "0.005", "0.681", "0.147", "0.416", "1.019", "0.020", "P@5", "R@5")
SUFFIXES = {".json", ".csv", ".md", ".txt", ".log"}


def context(text: str, needle: str) -> str:
    """One compact line around a match."""
    index = text.find(needle)
    start = max(0, index - 90)
    end = min(len(text), index + len(needle) + 120)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def main() -> None:
    """Print every artifact containing each Table 6 value."""
    files = [p for p in BASE.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES]
    print(f"files scanned: {len(files)}")

    for target in TARGETS:
        print(f"\n=== {target} ===")
        hits = 0
        for path in files:
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if target not in text:
                continue
            hits += 1
            print(f"  {path.relative_to(BASE)}")
            print(f"    {context(text, target)}")
        if not hits:
            print("  NOT FOUND")


if __name__ == "__main__":
    main()
