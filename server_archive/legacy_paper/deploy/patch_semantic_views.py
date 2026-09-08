"""Add a semantic view partition for NSL-KDD to correct_cross_dataset.py.

The script currently splits every cross-dataset feature vector into three equal
positional blocks. UNSW-NB15 is the only benchmark that received a semantic
partition, and it is also the only one where the multi-view contribution was
measured against a positional control. This patch gives NSL-KDD the same
treatment so the two can be compared on equal terms.

The patch is additive:
  - positional_view_splits is left untouched and stays the default
  - NSL-KDD resolves its views by column name, falling back to positional if
    any name is missing after zero-variance columns are dropped
  - the emitted JSON records which partition was used and the exact grouping

Run on the server:
    /usr/bin/python3 patch_semantic_views.py --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

TARGET = Path("/opt/ids_revision/deploy/correct_cross_dataset.py")

ANCHOR_HELPERS = "def _encode_and_scale("

HELPERS = '''
# NSL-KDD inherits the KDD Cup 1999 schema, whose documentation groups the
# attributes into basic, content, time-based traffic and host-based traffic
# families. The two traffic families share a measurement basis and differ only
# in window, so they are encoded together; the content family is kept whole
# because it carries the attributes that define privilege escalation and
# brute-force behaviour.
NSL_VIEW_GROUPS: dict[str, list[str]] = {
    "view1_basic_connection": [
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    ],
    "view2_content_semantics": [
        "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login",
    ],
    "view3_traffic_statistics": [
        "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ],
}


def semantic_view_splits(feature_names: tuple[str, ...]):
    """Map surviving NSL-KDD columns onto three semantic views.

    Returns (splits, grouping) or (None, None) when the partition cannot be
    formed - an empty view or an unassigned column both disqualify it, and the
    caller then falls back to the positional split.
    """
    index = {name: i for i, name in enumerate(feature_names)}
    splits: list[list[int]] = []
    grouping: dict[str, list[str]] = {}

    for view, names in NSL_VIEW_GROUPS.items():
        present = [n for n in names if n in index]
        if not present:
            print(f"  [VIEWS] {view} empty after preprocessing", flush=True)
            return None, None
        splits.append([index[n] for n in present])
        grouping[view] = present

    assigned = sum(len(s) for s in splits)
    if assigned != len(feature_names):
        stray = [n for n in feature_names if not any(n in g for g in grouping.values())]
        print(f"  [VIEWS] {len(feature_names) - assigned} column(s) unassigned: {stray}",
              flush=True)
        return None, None

    return splits, grouping


'''

ANCHOR_CALL = """    view_splits = positional_view_splits(data.n_features)
    print(f"  [VIEWS] sizes={[len(s) for s in view_splits]}", flush=True)"""

CALL = """    view_grouping = None
    view_kind = "positional"
    view_splits = None
    if dataset == "nslkdd":
        view_splits, view_grouping = semantic_view_splits(data.feature_names)
        if view_splits is not None:
            view_kind = "semantic"
        else:
            print("  [VIEWS] semantic mapping unavailable, using positional", flush=True)
    if view_splits is None:
        view_splits = positional_view_splits(data.n_features)
    print(f"  [VIEWS] kind={view_kind} sizes={[len(s) for s in view_splits]}", flush=True)"""

ANCHOR_PROTO = (
    '        "protocol": "train-only-preprocessing + conditional-DDPM + '
    'paired-init + positional-views",'
)

PROTO = (
    '        "protocol": "train-only-preprocessing + conditional-DDPM + '
    'paired-init + " + view_kind + "-views",\n'
    '        "view_partition_type": view_kind,\n'
    '        "view_partition": view_grouping,\n'
    '        "feature_names": list(data.feature_names),'
)


def patch(text: str) -> str:
    if "semantic_view_splits" in text:
        raise SystemExit("already patched - aborting")

    for anchor in (ANCHOR_HELPERS, ANCHOR_CALL, ANCHOR_PROTO):
        if anchor not in text:
            raise SystemExit(f"anchor not found:\n{anchor[:90]}")

    text = text.replace(ANCHOR_HELPERS, HELPERS.lstrip("\n") + ANCHOR_HELPERS, 1)
    text = text.replace(ANCHOR_CALL, CALL, 1)
    text = text.replace(ANCHOR_PROTO, PROTO, 1)
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write the patched file")
    args = parser.parse_args()

    original = TARGET.read_text(encoding="utf-8")
    patched = patch(original)

    print(f"original lines: {len(original.splitlines())}")
    print(f"patched  lines: {len(patched.splitlines())}")

    if not args.apply:
        print("\ndry run - pass --apply to write")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_name(f"correct_cross_dataset.py.bak_{stamp}")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(patched, encoding="utf-8")
    print(f"\nbackup : {backup}")
    print(f"written: {TARGET}")

    check = TARGET.read_text(encoding="utf-8")
    for token in ("semantic_view_splits", "view_partition_type", "NSL_VIEW_GROUPS"):
        print(f"  verify {token}: {'OK' if token in check else 'MISSING'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
