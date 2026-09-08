#!/bin/bash
cd /opt/ids_revision
python3 -u deploy/get_gnn_metrics.py 2>&1 | tee /opt/ids_revision/results/gnn_metrics.log
