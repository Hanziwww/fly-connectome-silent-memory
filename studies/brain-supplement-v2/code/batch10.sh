#!/bin/sh
# Tenth GPU batch (plan v2, Phase 1C, X2): readout-only training on six new scenes (M1, M2, S1 mirror, S1 minus 45,
# S2 mirror, S2 minus 48), replicates 0-6 each (about 5 h). Runs after batch 8: waits until runtime/batch8/batch8.log
# reports it ended, then takes the GPU lock. X10 (batch9.sh) waits for this batch. Training jobs verify in a separate
# process; the batch continues past unsolved runs. STOP: studies/brain-supplement-v2/STOP.
#   nohup sh studies/brain-supplement-v2/code/batch10.sh > runtime/batch10/runner.log 2>&1 < /dev/null &
log=runtime/batch10/batch10.log
mkdir -p runtime/batch10 runtime/campaign
echo "$(date -u +%FT%TZ) batch10 runner $$ waiting for batch8" >> "$log"
until [ -e runtime/batch8/batch8.log ] && grep -qE "batch8 done|STOP" runtime/batch8/batch8.log; do sleep 60; done
echo "$(date -u +%FT%TZ) batch8 ended: $(tail -n 1 runtime/batch8/batch8.log)" >> "$log"
exec 9> runtime/campaign/campaign.lock
flock 9
echo "$(date -u +%FT%TZ) batch10 runner $$ holds the lock" >> "$log"
for r in 0 1 2 3 4 5 6; do
    for scene in M1 M2 S1-mirror S1-minus45 S2-mirror S2-minus48; do
        id="x2-$scene-r$r"
        folder="studies/brain-supplement-v2/runs/$id"
        if [ -e studies/brain-supplement-v2/STOP ]; then echo "$(date -u +%FT%TZ) STOP before $id" >> "$log"; exit 75; fi
        if [ -e "$folder/result.json" ] && [ -e "$folder/verification.json" ]; then continue; fi
        mkdir -p "$folder"
        echo "$(date -u +%FT%TZ) start $id" >> "$log"
        .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/x2_scene_training.py --scene "$scene" --replicate "$r" --out "$folder" > "$folder/train.log" 2>&1 9>&-
        code=$?
        echo "$(date -u +%FT%TZ) trained $id exit $code" >> "$log"
        if [ "$code" -eq 0 ]; then
            .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/x2_scene_training.py --scene "$scene" --replicate "$r" --out "$folder" --verify > "$folder/verify.log" 2>&1 9>&-
            echo "$(date -u +%FT%TZ) verified $id exit $?" >> "$log"
        fi
    done
done
echo "$(date -u +%FT%TZ) batch10 done" >> "$log"
