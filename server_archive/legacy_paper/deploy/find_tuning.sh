#!/bin/bash
# Locate the baseline scripts and read what tuning each side actually got.
cd /opt/ids_revision/deploy || exit 1

echo "=== where are the baseline scripts ==="
ls -1 baseline*.py scripts/baseline*.py 2>/dev/null
echo "--- searching wider ---"
find /opt/ids_revision -name "baseline*.py" -not -path "*/node_modules/*" 2>/dev/null | head

echo
echo "=== XGBoost construction, wherever it lives ==="
for f in $(find /opt/ids_revision -name "baseline*.py" 2>/dev/null); do
  if grep -q "XGB" "$f" 2>/dev/null; then
    echo "--- $f ---"
    grep -n "XGB" -A 12 "$f" | head -30
  fi
done

echo
echo "=== any sweep at all, anywhere ==="
grep -rn "GridSearch\|RandomizedSearch\|param_grid\|lr_sweep" /opt/ids_revision --include=*.py 2>/dev/null | head -12

echo
echo "=== recorded protocol from a results file ==="
/usr/bin/python3 - <<'PY'
import json, pathlib
for name in ("baseline_fair", "baseline_deep"):
    for p in sorted(pathlib.Path("/opt/ids_revision/results", name).rglob("metrics.json"))[:1]:
        d = json.loads(p.read_text())
        print(f"\n{name}:")
        for k in ("protocol", "lr_sweep", "epochs", "patience", "batch"):
            if k in d:
                print(f"  {k}: {d[k]}")
PY
