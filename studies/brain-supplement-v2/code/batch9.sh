#!/bin/sh
# Ninth GPU batch (plan v2, Phase 1C, X10): replicates 1-5 of the 24 Fig. 2 maps (joint training from P05 with the X1
# readout jitter; about 31 h), replicate by replicate so complete sets accumulate. Runs after batch 10: waits until
# runtime/batch10/batch10.log reports it ended, then takes the GPU lock. Training jobs verify in a separate process; the
# batch continues past unsolved runs. STOP: studies/brain-supplement-v2/STOP.
#   nohup sh studies/brain-supplement-v2/code/batch9.sh > runtime/batch9/runner.log 2>&1 < /dev/null &
log=runtime/batch9/batch9.log
mkdir -p runtime/batch9 runtime/campaign
echo "$(date -u +%FT%TZ) batch9 runner $$ waiting for batch10" >> "$log"
until [ -e runtime/batch10/batch10.log ] && grep -qE "batch10 done|STOP" runtime/batch10/batch10.log; do sleep 60; done
echo "$(date -u +%FT%TZ) batch10 ended: $(tail -n 1 runtime/batch10/batch10.log)" >> "$log"
exec 9> runtime/campaign/campaign.lock
flock 9
echo "$(date -u +%FT%TZ) batch9 runner $$ holds the lock" >> "$log"
for r in 1 2 3 4 5; do
    for tier in 0 1 2 3 4 5 6 7; do
        for k in 0 1 2; do
            map="D$tier-T$k"
            id="difficulty-$map-joint-r$r"
            folder="studies/brain-supplement-v2/runs/$id"
            if [ -e studies/brain-supplement-v2/STOP ]; then echo "$(date -u +%FT%TZ) STOP before $id" >> "$log"; exit 75; fi
            if [ -e "$folder/result.json" ] && [ -e "$folder/verification.json" ]; then continue; fi
            mkdir -p "$folder"
            echo "$(date -u +%FT%TZ) start $id" >> "$log"
            .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/sokoban_replicate.py --map "$map" --condition joint --replicate "$r" --module difficulty --out "$folder" > "$folder/train.log" 2>&1 9>&-
            code=$?
            echo "$(date -u +%FT%TZ) trained $id exit $code" >> "$log"
            if [ "$code" -eq 0 ]; then
                .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/sokoban_replicate.py --map "$map" --condition joint --replicate "$r" --module difficulty --out "$folder" --verify > "$folder/verify.log" 2>&1 9>&-
                echo "$(date -u +%FT%TZ) verified $id exit $?" >> "$log"
            fi
        done
    done
done
echo "$(date -u +%FT%TZ) batch9 done" >> "$log"
