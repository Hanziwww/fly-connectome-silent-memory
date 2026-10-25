#!/bin/sh
# Second GPU batch (plan v2, Phase 3): G3a dependency map, late rescue, regime C, one after another, after the training
# campaign. It waits on the campaign runner's lock (blocking flock on runtime/campaign/campaign.lock), so it never shares
# the GPU with the campaign (whose frozen trainer has wall-clock phase limits), refuses to start unless the campaign
# log says "campaign done", and stops at the first failing job. Each job is resumable (finished outputs are skipped).
# STOP: create studies/brain-pathology-v1/STOP to stop before the next job.
#   nohup sh studies/brain-pathology-v1/code/batch2.sh > runtime/batch2/runner.log 2>&1 < /dev/null &
log=runtime/batch2/batch2.log
mkdir -p runtime/batch2 runtime/campaign
exec 9> runtime/campaign/campaign.lock
echo "$(date -u +%FT%TZ) batch2 runner $$ waiting for the GPU lock" >> "$log"
flock 9
echo "$(date -u +%FT%TZ) batch2 runner $$ holds the lock" >> "$log"
if ! grep -q "campaign done" runtime/campaign/campaign.log; then
    echo "$(date -u +%FT%TZ) refused: the campaign log has no 'campaign done'" >> "$log"
    exit 1
fi
for job in phase3_dependency phase3_rescue phase3_regime_c; do
    if [ -e studies/brain-pathology-v1/STOP ]; then echo "$(date -u +%FT%TZ) STOP before $job" >> "$log"; exit 75; fi
    echo "$(date -u +%FT%TZ) start $job" >> "$log"
    .venv-linux/bin/python -B -X utf8 "studies/brain-pathology-v1/code/$job.py" > "runtime/batch2/$job.log" 2>&1 9>&-
    code=$?
    echo "$(date -u +%FT%TZ) finished $job exit $code" >> "$log"
    if [ "$code" -ne 0 ]; then exit "$code"; fi
done
echo "$(date -u +%FT%TZ) batch2 done" >> "$log"
