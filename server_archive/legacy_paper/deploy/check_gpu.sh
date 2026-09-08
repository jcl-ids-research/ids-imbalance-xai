#!/bin/bash
# Who holds the 6.7 GB of GPU memory, and is anything actually running?
cd /opt/ids_revision/deploy || exit 1

echo "=== TIME ==="
date '+%F %T'

echo
echo "=== GPU COMPUTE APPS ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo
echo "=== WHAT IS PID 35652 ==="
ps -p 35652 -o pid,etime,user,cmd 2>/dev/null || echo "no such process - memory is stale"

echo
echo "=== ALL PYTHON PROCESSES ==="
ps aux | grep '[p]ython3' | sed 's/  */ /g' | cut -d' ' -f2,10,11,12,13,14

echo
echo "=== GPU UTILISATION ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
