#!/bin/bash
# Server status probe. Kept as a file so PowerShell quoting cannot mangle it.
echo "=== TIME ==="
date
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo
echo "=== RUNNING PYTHON ==="
n=$(ps aux | grep -c '[p]ython3')
echo "python3 processes: $n"
ps aux | grep '[p]ython3' | awk '{print $2, $10, $11, $12, $13, $14, $15}'
echo
echo "=== correct_cross_dataset.py FUNCTION DEFS ==="
grep -n 'def augment' correct_cross_dataset.py
grep -n 'def rebalance' correct_cross_dataset.py
echo
echo "=== AUGMENT CALL SITE ==="
grep -n 'augment(ddpm' correct_cross_dataset.py
echo
echo "=== VIEW PATCH STATE ==="
grep -c 'NSL_VIEW_GROUPS' correct_cross_dataset.py
grep -n 'view_kind' correct_cross_dataset.py | head -3
echo
echo "=== RESULT DIRS, NEWEST FIRST ==="
ls -lt /opt/ids_revision/results/ | head -12
echo
echo "=== SMOKE OUTPUT ==="
ls -l /tmp/smoke_sem/ 2>/dev/null || echo "no smoke dir"
