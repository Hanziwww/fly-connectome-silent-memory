#!/bin/sh
# Seventh GPU batch (plan v2, Phase 4): the dose series (C01 fine grid, random removal; about 2-4 h). Runs after the
# sixth batch (X3): waits until runtime/batch6/batch6.log reports it ended, then takes the GPU lock. Resumable.
# STOP: studies/brain-pathology-v1/STOP.
#   nohup sh studies/brain-pathology-v1/code/batch7.sh > runtime/batch7/runner.log 2>&1 < /dev/null &
log=runtime/batch7/batch7.log
mkdir -p runtime/batch7 runtime/campaign
echo "$(date -u +%FT%TZ) batch7 runner $$ waiting for batch6" >> "$log"
until [ -e runtime/batch6/batch6.log ] && grep -qE "batch6 done|STOP" runtime/batch6/batch6.log; do sleep 60; done
echo "$(date -u +%FT%TZ) batch6 ended: $(tail -n 1 runtime/batch6/batch6.log)" >> "$log"
exec 9> runtime/campaign/campaign.lock
flock 9
echo "$(date -u +%FT%TZ) batch7 runner $$ holds the lock" >> "$log"
if [ -e studies/brain-pathology-v1/STOP ]; then echo "$(date -u +%FT%TZ) STOP before phase4_dose_series" >> "$log"; exit 75; fi
echo "$(date -u +%FT%TZ) start phase4_dose_series" >> "$log"
.venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/phase4_dose_series.py > runtime/batch7/phase4_dose_series.log 2>&1 9>&-
code=$?
echo "$(date -u +%FT%TZ) finished phase4_dose_series exit $code" >> "$log"
[ "$code" -ne 0 ] && exit "$code"
echo "$(date -u +%FT%TZ) batch7 done" >> "$log"
