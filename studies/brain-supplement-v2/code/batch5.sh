#!/bin/sh
# Fifth GPU batch (plan v2, Phase 1C): X6 and X7, causal tests of the cue ensemble (about 1-2 h). Runs after the third
# batch: waits until runtime/batch3/batch3.log reports it ended, then takes the GPU lock. The X8 pilot (batch4.sh) waits
# for this batch. Resumable per scene. STOP: studies/brain-supplement-v2/STOP.
#   nohup sh studies/brain-supplement-v2/code/batch5.sh > runtime/batch5/runner.log 2>&1 < /dev/null &
log=runtime/batch5/batch5.log
mkdir -p runtime/batch5 runtime/campaign
echo "$(date -u +%FT%TZ) batch5 runner $$ waiting for batch3" >> "$log"
until [ -e runtime/batch3/batch3.log ] && grep -qE "batch3 done|exit [1-9]|STOP" runtime/batch3/batch3.log; do sleep 60; done
echo "$(date -u +%FT%TZ) batch3 ended: $(tail -n 1 runtime/batch3/batch3.log)" >> "$log"
exec 9> runtime/campaign/campaign.lock
flock 9
echo "$(date -u +%FT%TZ) batch5 runner $$ holds the lock" >> "$log"
if [ -e studies/brain-supplement-v2/STOP ]; then echo "$(date -u +%FT%TZ) STOP before x6_x7_ensemble" >> "$log"; exit 75; fi
echo "$(date -u +%FT%TZ) start x6_x7_ensemble" >> "$log"
.venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/x6_x7_ensemble.py > runtime/batch5/x6_x7_ensemble.log 2>&1 9>&-
code=$?
echo "$(date -u +%FT%TZ) finished x6_x7_ensemble exit $code" >> "$log"
[ "$code" -ne 0 ] && exit "$code"
echo "$(date -u +%FT%TZ) batch5 done" >> "$log"
