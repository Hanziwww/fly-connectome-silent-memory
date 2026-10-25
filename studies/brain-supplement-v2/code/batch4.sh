#!/bin/sh
# Fourth GPU batch (plan v2, Phase 1C): the X8 pilot, Sokoban training from a neutral brain, readout-only vs joint, maps
# D1/D3/D5/D7-T0, replicates 0-2 (readout-only first; joint D7, the longest, last). Runs after the eleventh batch: waits
# until runtime/batch11/batch11.log reports it ended, then takes the GPU lock. Each job trains, then verifies in a
# separate process; a job whose result.json and verification.json exist is skipped (resumable). Unlike the analysis
# batches it does not stop at a failed job (a run that fails to learn is an outcome, not an error).
# STOP: studies/brain-supplement-v2/STOP before the next job.
#   nohup sh studies/brain-supplement-v2/code/batch4.sh > runtime/batch4/runner.log 2>&1 < /dev/null &
log=runtime/batch4/batch4.log
mkdir -p runtime/batch4 runtime/campaign
echo "$(date -u +%FT%TZ) batch4 runner $$ waiting for batch11" >> "$log"
until [ -e runtime/batch11/batch11.log ] && grep -qE "batch11 done|exit [1-9]|STOP" runtime/batch11/batch11.log; do sleep 60; done
echo "$(date -u +%FT%TZ) batch11 ended: $(tail -n 1 runtime/batch11/batch11.log)" >> "$log"
exec 9> runtime/campaign/campaign.lock
flock 9
echo "$(date -u +%FT%TZ) batch4 runner $$ holds the lock" >> "$log"
for arm in head_only joint; do
    for map in D1-T0 D3-T0 D5-T0 D7-T0; do
        for r in 0 1 2; do
            id="x8-$map-$arm-r$r"
            folder="studies/brain-supplement-v2/runs/$id"
            if [ -e studies/brain-supplement-v2/STOP ]; then echo "$(date -u +%FT%TZ) STOP before $id" >> "$log"; exit 75; fi
            if [ -e "$folder/result.json" ] && [ -e "$folder/verification.json" ]; then continue; fi
            mkdir -p "$folder"
            echo "$(date -u +%FT%TZ) start $id" >> "$log"
            .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/x8_neutral_training.py --map "$map" --arm "$arm" --replicate "$r" --out "$folder" > "$folder/train.log" 2>&1 9>&-
            code=$?
            echo "$(date -u +%FT%TZ) trained $id exit $code" >> "$log"
            if [ "$code" -eq 0 ]; then
                .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/x8_neutral_training.py --map "$map" --arm "$arm" --replicate "$r" --out "$folder" --verify > "$folder/verify.log" 2>&1 9>&-
                echo "$(date -u +%FT%TZ) verified $id exit $?" >> "$log"
            fi
        done
    done
done
echo "$(date -u +%FT%TZ) batch4 done" >> "$log"
