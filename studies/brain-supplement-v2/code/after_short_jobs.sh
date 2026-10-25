#!/bin/sh
# Start the training campaign once the short GPU jobs (olfactory curves, linear response, mean field) have finished or
# failed, so the campaign's wall-clock stopping rules never share the GPU.
#   nohup sh studies/brain-supplement-v2/code/after_short_jobs.sh > runtime/campaign/launcher.log 2>&1 &
done_or_failed() {
    [ -e studies/brain-pathology-v1/runs/g4-meanfield/result.json ] && return 0
    for f in runtime/phase3_olfactory.log runtime/phase4_linear_response.log runtime/phase4_meanfield.log; do
        [ -e "$f" ] && grep -q Traceback "$f" && return 0
    done
    return 1
}
until done_or_failed; do sleep 30; done
echo "$(date -u +%FT%TZ) short jobs done; starting the campaign"
.venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/campaign_jobs.py && sh studies/brain-supplement-v2/code/campaign.sh runtime/campaign/jobs.txt
