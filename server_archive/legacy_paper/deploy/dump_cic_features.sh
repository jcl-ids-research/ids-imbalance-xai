#!/bin/bash
# Dump the surviving CIC feature names, as the loader produces them.
# The semantic mapping must be built against these exact names, not against
# the raw CSV headers, because CIC_DROP and the zero-variance filter both
# remove columns.
cd /opt/ids_revision/deploy

/usr/bin/python3 - <<'PY'
import sys
sys.path.insert(0, "/opt/ids_revision/deploy")
import correct_cross_dataset as ccd

for name, path in (
    ("cicids2017", "/opt/CIC-IDS-2017/MachineLearningCVE"),
    ("cicddos2019", "/opt/CIC-DDoS2019/all"),
):
    print("=" * 70)
    print(name)
    print("=" * 70)
    data = ccd.load_cic(path, seed=42, smoke=True)
    names = list(data.feature_names)
    print(f"surviving features: {len(names)}")
    for i, n in enumerate(names):
        print(f"  {i:3d}  {n}")
    print()
PY
