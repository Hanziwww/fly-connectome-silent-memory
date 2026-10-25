#!/bin/sh
# Wait for Phase 3 regime A to finish on the GPU host, then write the campaign job list and run it.
#   nohup sh studies/brain-supplement-v2/code/after_regime_a.sh > runtime/campaign/launcher.log 2>&1 &
while [ ! -e studies/brain-pathology-v1/runs/phase3-regime-a/done.json ]; do sleep 30; done
.venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/campaign_jobs.py
sh studies/brain-supplement-v2/code/campaign.sh runtime/campaign/jobs.txt
