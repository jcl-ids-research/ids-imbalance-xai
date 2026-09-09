"""Fail when the type-debt baseline gets worse.

The strict `basedpyright` configuration is an audit rather than a passing gate:
the project carries a known backlog, largely untyped third-party stubs and JSON
parsing at the archive boundary. Suppressing it would hide the debt, and fixing
all of it in one pass would touch every module at once.

This script pins the current count instead. New code cannot add diagnostics, and
the number can only be lowered, which turns a static backlog into a ratchet.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

BASELINE = Path(__file__).resolve().parents[1] / "docs" / "type-debt-baseline.json"


def measure() -> tuple[int, int]:
    """Return the current error and warning counts reported by basedpyright."""
    uv = shutil.which("uv")
    if uv is None:
        print("uv is not on PATH", file=sys.stderr)
        raise SystemExit(2)
    result = subprocess.run(  # noqa: S603 - fixed argument vector, no shell
        [uv, "run", "basedpyright", "--outputjson"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("basedpyright produced no JSON; raw output follows", file=sys.stderr)
        print(result.stdout[-2000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(2) from None
    summary = payload.get("summary", {})
    return int(summary.get("errorCount", 0)), int(summary.get("warningCount", 0))


def main() -> int:
    errors, warnings = measure()
    if "--update" in sys.argv:
        BASELINE.write_text(
            json.dumps({"errors": errors, "warnings": warnings}, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"baseline written: {errors} errors, {warnings} warnings")
        return 0

    if not BASELINE.is_file():
        print(f"no baseline at {BASELINE}; run with --update once", file=sys.stderr)
        return 2

    recorded = json.loads(BASELINE.read_text(encoding="utf-8"))
    allowed_errors = int(recorded["errors"])
    allowed_warnings = int(recorded["warnings"])

    print(f"errors   {errors} (baseline {allowed_errors})")
    print(f"warnings {warnings} (baseline {allowed_warnings})")

    if errors > allowed_errors or warnings > allowed_warnings:
        print("type debt increased; fix the new diagnostics or justify them", file=sys.stderr)
        return 1
    if errors < allowed_errors or warnings < allowed_warnings:
        print("type debt fell; lower the baseline with --update")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
