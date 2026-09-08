"""Switch correct_cross_dataset.py from augment() to rebalance().

Finding from the probe: augment() is not defined in this file. It is imported
from correct_unsw_ablation, and rebalance() lives in that same module. So the
port is an import change plus a call-site change - no function body is copied,
and both arms keep sharing one definition.

Why it matters: augment() expands every class by AUGMENT_FRAC and preserves the
original class ratio (NSL-KDD stays at 1.149 before and after). The paper's
Table 3/4/5 numbers come from phase1_corrected, which uses rebalance(). Without
this change the semantic-view run is not comparable with anything in the paper.

Run on the server:
    /usr/bin/python3 patch_rebalance.py --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

TARGET = Path("/opt/ids_revision/deploy/correct_cross_dataset.py")

# 1. pull rebalance in alongside augment
ANCHOR_IMPORT = "    augment,"
IMPORT = "    augment,\n    rebalance,"

# 2. call rebalance instead, and keep its report
ANCHOR_CALL = "    X_aug, y_aug = augment(ddpm_models, X_sub, y_sub, qt, device, AUGMENT_FRAC)"
CALL = """    X_aug, y_aug, balance_report = rebalance(
        ddpm_models, X_sub, y_sub, qt, device, seed
    )"""

# 3. record the arm in the protocol string
ANCHOR_PROTO = (
    '"train-only-preprocessing + conditional-DDPM + paired-init + " '
    '+ view_kind + "-views"'
)
PROTO = (
    '"train-only-preprocessing + conditional-DDPM + mean-target-rebalance + '
    'paired-init + " + view_kind + "-views"'
)

# 4. the field is a balanced count now, not an augmented one
ANCHOR_SIZES = '"test": int(len(data.X_test)), "augmented_train": int(len(X_aug)),'
SIZES = '"test": int(len(data.X_test)), "balanced_train": int(len(X_aug)),'

# 5. emit the before/after distribution the editor asked for
ANCHOR_REPORT = '        "view_partition_type": view_kind,'
REPORT = '        "balance_report": balance_report,\n        "view_partition_type": view_kind,'

PAIRS = [
    ("import", ANCHOR_IMPORT, IMPORT),
    ("call site", ANCHOR_CALL, CALL),
    ("protocol", ANCHOR_PROTO, PROTO),
    ("sizes", ANCHOR_SIZES, SIZES),
    ("balance_report", ANCHOR_REPORT, REPORT),
]


def patch(text: str) -> str:
    if "rebalance," in text:
        raise SystemExit("already patched - aborting")

    for label, anchor, _ in PAIRS:
        if anchor not in text:
            raise SystemExit(f"anchor not found [{label}]:\n  {anchor[:100]}")

    for label, anchor, replacement in PAIRS:
        text = text.replace(anchor, replacement, 1)
        print(f"  patched: {label}")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    original = TARGET.read_text(encoding="utf-8")
    patched = patch(original)
    print(f"\noriginal lines: {len(original.splitlines())}")
    print(f"patched  lines: {len(patched.splitlines())}")

    if not args.apply:
        print("\ndry run - pass --apply to write")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_name(f"correct_cross_dataset.py.bak_reb_{stamp}")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(patched, encoding="utf-8")
    print(f"\nbackup : {backup}")
    print(f"written: {TARGET}")

    # re-read from disk; never trust the write
    check = TARGET.read_text(encoding="utf-8")
    for token in ("    rebalance,", "balance_report", "mean-target-rebalance", "balanced_train"):
        print(f"  verify {token.strip()!r}: {'OK' if token in check else 'MISSING'}")
    if "y_aug = augment(" in check:
        print("  WARNING: augment() call still present in run_seed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
