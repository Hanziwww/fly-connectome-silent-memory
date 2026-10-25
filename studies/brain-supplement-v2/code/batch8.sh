#!/bin/sh
# Eighth GPU batch (plan v2, Phase 1C, X9): replays for gaps G06 and G10 (about 2-3 h), then the never-run D7 training
# controls of gap G01 (frozen protocol, about 10 h; cheap arms first, brain_only last). Runs after the X8 pilot: waits
# until runtime/batch4/batch4.log reports it ended, then takes the GPU lock. Training jobs verify in a separate
# process; a failed or unsolved run is an outcome, so the batch continues. STOP: studies/brain-supplement-v2/STOP.
#   nohup sh studies/brain-supplement-v2/code/batch8.sh > runtime/batch8/runner.log 2>&1 < /dev/null &
log=runtime/batch8/batch8.log
mkdir -p runtime/batch8 runtime/campaign
echo "$(date -u +%FT%TZ) batch8 runner $$ waiting for batch4" >> "$log"
until [ -e runtime/batch4/batch4.log ] && grep -qE "batch4 done|STOP" runtime/batch4/batch4.log; do sleep 60; done
echo "$(date -u +%FT%TZ) batch4 ended: $(tail -n 1 runtime/batch4/batch4.log)" >> "$log"
exec 9> runtime/campaign/campaign.lock
flock 9
echo "$(date -u +%FT%TZ) batch8 runner $$ holds the lock" >> "$log"
if [ ! -e studies/brain-supplement-v2/STOP ]; then
    echo "$(date -u +%FT%TZ) start x9_replays" >> "$log"
    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/x9_replays.py > runtime/batch8/x9_replays.log 2>&1 9>&-
    echo "$(date -u +%FT%TZ) finished x9_replays exit $?" >> "$log"
fi
for arm in frozen board_mlp entry_ce alternating neutral rewired weight_shuffle no_recovery brain_only; do
    id="controls-D7-T0-$arm-r0"
    folder="studies/brain-supplement-v2/runs/$id"
    if [ -e studies/brain-supplement-v2/STOP ]; then echo "$(date -u +%FT%TZ) STOP before $id" >> "$log"; exit 75; fi
    if [ -e "$folder/result.json" ] && [ -e "$folder/verification.json" ]; then continue; fi
    mkdir -p "$folder"
    echo "$(date -u +%FT%TZ) start $id" >> "$log"
    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/sokoban_replicate.py --map D7-T0 --condition "$arm" --replicate 0 --module controls --out "$folder" > "$folder/train.log" 2>&1 9>&-
    code=$?
    echo "$(date -u +%FT%TZ) trained $id exit $code" >> "$log"
    if [ "$code" -eq 0 ]; then
        .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/sokoban_replicate.py --map D7-T0 --condition "$arm" --replicate 0 --module controls --out "$folder" --verify > "$folder/verify.log" 2>&1 9>&-
        echo "$(date -u +%FT%TZ) verified $id exit $?" >> "$log"
    fi
done
echo "$(date -u +%FT%TZ) batch8 done" >> "$log"
