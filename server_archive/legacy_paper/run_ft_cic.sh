#!/bin/bash
cd /opt/ids_revision
python3 -u deploy/feattrans_cic_quick.py 2>&1 | tee results/ft_cic_run.log
