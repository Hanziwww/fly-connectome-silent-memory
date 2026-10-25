#!/bin/sh
# Run the training campaign on the GPU host, one job at a time (concurrent workers give no throughput on this GPU;
# Phase 0b). Each job: train (replicate_training.py) then verify in a separate process (--verify). A job whose
# result.json and verification.json exist is skipped, so the script resumes after an interruption.
# Single instance: the runner holds runtime/campaign/campaign.lock (flock) for its whole life and refuses to start if
# another runner holds it (two runners once trained the same job into the same folder; 2026-10-19).
# STOP: create studies/brain-supplement-v2/STOP to stop before the next job (the frozen trainer also stops at a safe
# point when <run folder>/STOP exists).
#   nohup sh studies/brain-supplement-v2/code/campaign.sh [runtime/campaign/jobs.txt] > runtime/campaign/runner.log 2>&1 < /dev/null &
jobs=${1:-runtime/campaign/jobs.txt}
log=runtime/campaign/campaign.log
mkdir -p runtime/campaign
exec 9> runtime/campaign/campaign.lock
if ! flock -n 9; then
    echo "$(date -u +%FT%TZ) refused: another campaign runner holds runtime/campaign/campaign.lock" | tee -a "$log"
    exit 1
fi
echo "$(date -u +%FT%TZ) runner $$ started" >> "$log"
while IFS="$(printf '\t')" read -r id folder model rep lesion; do
    [ -z "$id" ] && continue
    if [ -e studies/brain-supplement-v2/STOP ]; then echo "$(date -u +%FT%TZ) STOP before $id" >> "$log"; exit 75; fi
    if [ -e "$folder/result.json" ] && [ -e "$folder/verification.json" ]; then continue; fi
    mkdir -p "$folder"
    les=""
    [ "$lesion" != "-" ] && les="--lesion $lesion"
    echo "$(date -u +%FT%TZ) start $id" >> "$log"
    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/replicate_training.py --model "$model" --replicate "$rep" --out "$folder" $les > "$folder/train.log" 2>&1 9>&-
    code=$?
    echo "$(date -u +%FT%TZ) trained $id exit $code" >> "$log"
    if [ "$code" -eq 0 ]; then
        .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/replicate_training.py --model "$model" --replicate "$rep" --out "$folder" $les --verify > "$folder/verify.log" 2>&1 9>&-
        echo "$(date -u +%FT%TZ) verified $id exit $?" >> "$log"
    fi
done < "$jobs"
echo "$(date -u +%FT%TZ) campaign done" >> "$log"
