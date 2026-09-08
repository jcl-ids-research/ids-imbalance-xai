#!/bin/bash
# Find where augment() comes from in correct_cross_dataset.py.
cd /opt/ids_revision/deploy

echo "=== IMPORT LINES ==="
grep -n '^import \|^from \|^    from \|^    import ' correct_cross_dataset.py | head -40

echo
echo "=== ANY MENTION OF augment ==="
grep -n 'augment' correct_cross_dataset.py

echo
echo "=== ANY MENTION OF rebalance ==="
grep -n 'rebalance' correct_cross_dataset.py

echo
echo "=== TOP-LEVEL DEFS IN THIS FILE ==="
grep -n '^def \|^class ' correct_cross_dataset.py

echo
echo "=== FILE SIZE ==="
wc -l correct_cross_dataset.py

echo
echo "=== WHICH MODULES DEFINE augment ==="
grep -ln 'def augment' *.py

echo
echo "=== WHICH MODULES DEFINE rebalance ==="
grep -ln 'def rebalance' *.py
