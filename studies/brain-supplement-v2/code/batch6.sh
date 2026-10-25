#!/bin/sh
# Sixth GPU batch (plan v2, Phase 1C): X3, M0 readout-only training on jittered timings, replicates 0-9 (about 1-2 h).
# Runs after the fifth batch: waits until runtime/batch5/batch5.log reports it ended, then takes the GPU lock. Each job
# trains, then verifies in a separate process; resumable. The X8 pilot (batch4.sh) waits for this batch.
# STOP: studies/brain-supplement-v2/STOP before the next job.
#   nohup sh studies/brain-supplement-v2/code/batch6.sh > runtime/batch6/runner.log 2>&1 < /dev/null &
log=runtime/batch6/batch6.log
mkdir -p runtime/batch6 runtime/campaign
echo "$(date -u +%FT%TZ) batch6 runner $$ waiting for batch5" >> "$log"
until [ -e runtime/batch5/batch5.log ] && grep -qE "batch5 done|exit [1-9]|STOP" runtime/batch5/batch5.log; do sleep 60; done
echo "$(date -u +%FT%TZ) batch5 ended: $(tail -n 1 runtime/batch5/batch5.log)" >> "$log"
exec 9> runtime/campaign/campaign.lock
flock 9
echo "$(date -u +%FT%TZ) batch6 runner $$ holds the lock" >> "$log"
for r in 0 1 2 3 4 5 6 7 8 9; do
    id="x3-M0-head_only-r$r"
    folder="studies/brain-supplement-v2/runs/$id"
    if [ -e studies/brain-supplement-v2/STOP ]; then echo "$(date -u +%FT%TZ) STOP before $id" >> "$log"; exit 75; fi
    if [ -e "$folder/result.json" ] && [ -e "$folder/verification.json" ]; then continue; fi
    mkdir -p "$folder"
    echo "$(date -u +%FT%TZ) start $id" >> "$log"
    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/x3_jittered_training.py --replicate "$r" --out "$folder" > "$folder/train.log" 2>&1 9>&-
    code=$?
    echo "$(date -u +%FT%TZ) trained $id exit $code" >> "$log"
    if [ "$code" -eq 0 ]; then
        .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/x3_jittered_training.py --replicate "$r" --out "$folder" --verify > "$folder/verify.log" 2>&1 9>&-
        echo "$(date -u +%FT%TZ) verified $id exit $?" >> "$log"
    fi
done
echo "$(date -u +%FT%TZ) batch6 done" >> "$log"
