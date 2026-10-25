#!/bin/sh
# Third GPU batch (plan v2, Phase 1C): X5 swaps at the trained timings, then X4 scrambled-shift null. Runs after the
# second batch: it waits until runtime/batch2/batch2.log reports the batch finished, failed, refused or stopped (flock
# does not queue waiters in order), then takes the GPU lock (runtime/campaign/campaign.lock) and runs the jobs one after
# another, stopping at the first failure. Each job is resumable. STOP: studies/brain-supplement-v2/STOP.
#   nohup sh studies/brain-supplement-v2/code/batch3.sh > runtime/batch3/runner.log 2>&1 < /dev/null &
log=runtime/batch3/batch3.log
mkdir -p runtime/batch3 runtime/campaign
echo "$(date -u +%FT%TZ) batch3 runner $$ waiting for batch2" >> "$log"
until [ -e runtime/batch2/batch2.log ] && grep -qE "batch2 done|exit [1-9]|refused|STOP" runtime/batch2/batch2.log; do sleep 60; done
echo "$(date -u +%FT%TZ) batch2 ended: $(tail -n 1 runtime/batch2/batch2.log)" >> "$log"
exec 9> runtime/campaign/campaign.lock
flock 9
echo "$(date -u +%FT%TZ) batch3 runner $$ holds the lock" >> "$log"
for job in x5_swaps x4_scrambles; do
    if [ -e studies/brain-supplement-v2/STOP ]; then echo "$(date -u +%FT%TZ) STOP before $job" >> "$log"; exit 75; fi
    echo "$(date -u +%FT%TZ) start $job" >> "$log"
    .venv-linux/bin/python -B -X utf8 "studies/brain-supplement-v2/code/$job.py" > "runtime/batch3/$job.log" 2>&1 9>&-
    code=$?
    echo "$(date -u +%FT%TZ) finished $job exit $code" >> "$log"
    if [ "$code" -ne 0 ]; then exit "$code"; fi
done
echo "$(date -u +%FT%TZ) batch3 done" >> "$log"
