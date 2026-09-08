"""Data-only check before the NSL-KDD semantic-view run. No training.

Three questions to settle:
  1. how many zero-variance columns are dropped on the FULL split, not a sample
  2. what the semantic partition actually measures once those columns are gone
  3. which augmentation path the cross-dataset script takes, and how its class
     ratio compares with the rebalance() arm the paper reports
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/opt/ids_revision/deploy")

import numpy as np  # noqa: E402

import correct_cross_dataset as ccd  # noqa: E402


def rule(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def main() -> int:
    rule("1. FULL NSL-KDD LOAD (official split, no sampling)")
    data = ccd.load_nsl_kdd(smoke=False)
    kept = list(data.feature_names)
    declared = [c for c in ccd.NSL_COLS if c not in ("label", "difficulty")]
    dropped = [c for c in declared if c not in kept]

    print(f"  train rows      : {len(data.X_train):,}")
    print(f"  test rows       : {len(data.X_test):,}")
    print(f"  declared columns: {len(declared)}")
    print(f"  surviving       : {len(kept)}")
    print(f"  dropped         : {len(dropped)} -> {dropped}")
    print(f"  train attack rate: {data.y_train.mean():.4f}")
    print(f"  test  attack rate: {data.y_test.mean():.4f}")

    rule("2. SEMANTIC PARTITION ON THE SURVIVING COLUMNS")
    splits, grouping = ccd.semantic_view_splits(data.feature_names)
    if splits is None:
        print("  RESULT: semantic mapping REJECTED, run would fall back to positional")
        return 1

    print(f"  accepted, sizes = {[len(s) for s in splits]}")
    for view, names in grouping.items():
        print(f"\n  {view}  ({len(names)})")
        for i in range(0, len(names), 4):
            print("      " + ", ".join(names[i : i + 4]))

    missing_by_view = {
        view: [n for n in ccd.NSL_VIEW_GROUPS[view] if n not in grouping[view]]
        for view in grouping
    }
    print()
    for view, gone in missing_by_view.items():
        if gone:
            print(f"  {view}: lost {gone}")

    positional = ccd.positional_view_splits(data.n_features)
    print(f"\n  positional control sizes = {[len(s) for s in positional]}")

    # do the two partitions actually differ in membership?
    sem_sets = [set(s) for s in splits]
    pos_sets = [set(s) for s in positional]
    identical = sem_sets == pos_sets
    print(f"  semantic identical to positional? {identical}")

    rule("3. COLLAPSE-PRONE ATTRIBUTES, WHERE THEY LANDED")
    prone = ["root_shell", "num_failed_logins", "land", "su_attempted", "urgent"]
    for name in prone:
        where = next((v for v, names in grouping.items() if name in names), None)
        if where is None:
            print(f"  {name:20s} DROPPED as zero-variance")
            continue
        col = kept.index(name)
        values = data.X_train[:, col]
        uniq = len(np.unique(values))
        modal_share = float(np.bincount(
            np.unique(values, return_inverse=True)[1]
        ).max()) / len(values)
        print(f"  {name:20s} {where:26s} unique={uniq:4d} modal_share={modal_share:.4f}")

    rule("4. AUGMENTATION PATH IN THIS SCRIPT")
    has_rebalance = hasattr(ccd, "rebalance")
    print(f"  augment()   present: {hasattr(ccd, 'augment')}")
    print(f"  rebalance() present: {has_rebalance}")
    print(f"  AUGMENT_FRAC       : {getattr(ccd, 'AUGMENT_FRAC', 'n/a')}")
    print("  run_seed calls     : augment()  (proportional expansion, not parity)")

    counts = np.bincount(data.y_train)
    print(f"\n  raw train class counts : {counts.tolist()}")
    print(f"  raw imbalance ratio    : {counts.max() / counts.min():.3f}")
    frac = getattr(ccd, "AUGMENT_FRAC", 0.0)
    projected = [int(c * (1 + frac)) for c in counts]
    print(f"  after augment(frac={frac}) : {projected} "
          f"-> ratio {max(projected) / min(projected):.3f}")
    print("  NOTE: proportional expansion preserves the ratio; it does not balance.")

    rule("VERDICT")
    print(f"  usable features            : {len(kept)}")
    print(f"  semantic sizes             : {[len(s) for s in splits]}")
    print(f"  differs from positional    : {not identical}")
    print(f"  augmentation arm           : augment() proportional")
    print("  -> comparable with phase1/phase1_corrected? NO, those use rebalance()")
    return 0


if __name__ == "__main__":
    sys.exit(main())
