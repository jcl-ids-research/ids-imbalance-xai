"""Show how the authoritative result files are actually laid out.

The first attempt guessed the metrics were nested under a variant key and found
nothing. Rather than guess again, this prints the real structure: the top-level
keys, and the full contents of anything that is not bulk data, so the location
of the macro F1 figures can be read off instead of assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("/opt/ids_revision/deploy/results/phase1_corrected")
BULKY = {"balance_report", "sizes", "timings_sec"}


def walk(node: object, path: str = "", depth: int = 0) -> None:
    """Print every leaf that looks like a metric, with its full path."""
    if depth > 4:
        return
    if isinstance(node, dict):
        for key, value in node.items():
            walk(value, f"{path}.{key}" if path else key, depth + 1)
    elif (
        isinstance(node, (int, float))
        and not isinstance(node, bool)
        and any(term in path.lower() for term in ("f1", "acc", "prec", "recall"))
    ):
        print(f"    {path:56s} {node}")


def main() -> None:
    """Describe one file in full, then locate the metrics in each dataset."""
    sample = RESULTS / "nslkdd_seed42.json"
    payload = json.loads(sample.read_text())

    print(f"=== {sample.name} ===")
    print(f"top-level keys : {list(payload)}")
    print(f"artifact_dir   : {payload.get('artifact_dir')}")
    print(f"protocol       : {payload.get('protocol')}")
    print(f"official_split : {payload.get('official_split')}")

    print("\n--- everything except bulk sections ---")
    slim = {k: v for k, v in payload.items() if k not in BULKY}
    print(json.dumps(slim, indent=2)[:1500])

    print("\n--- metric-looking leaves in every dataset ---")
    for path in sorted(RESULTS.glob("*_seed42.json")):
        print(f"\n  {path.name}")
        walk(json.loads(path.read_text()))

    print("\n--- what sits in the artifact directory ---")
    artifact = payload.get("artifact_dir")
    if artifact and Path(artifact).is_dir():
        for item in sorted(Path(artifact).iterdir())[:15]:
            print(f"    {item.name}")
    else:
        print(f"    not present locally: {artifact}")


if __name__ == "__main__":
    main()
