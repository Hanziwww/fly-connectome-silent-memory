#!/bin/sh
# T-SC6 worker for one GPU on rtx5090x4 (tasks/sc-variant-design.json; amendment 0012). Claims each run once (mkdir),
# S1 runs first (longest), then M0: train with the brain's adaptation lesion (X1 protocol), verify, evaluate.
#   CUDA_VISIBLE_DEVICES=<g> setsid -f sh studies/brain-pathology-v1/code/sc_variant_queue.sh <g> > runtime/sc-variant/worker-<g>.log 2>&1 < /dev/null
cd /root/autodl-tmp/flybrain || exit 1
PY=".venv-linux/bin/python -B -X utf8"
CLAIMS=studies/brain-pathology-v1/runs/sc-variant-claims
mkdir -p "$CLAIMS" runtime/sc-variant
$PY studies/brain-pathology-v1/code/sc_variant.py | sort -k2,2r -k1,1 -k4,4n | while read -r brain family model rep out; do
    name=$(basename "$out")
    mkdir "$CLAIMS/$name" 2>/dev/null || continue
    lesion="studies/brain-pathology-v1/tasks/sc-variant-lesion.json"
    [ "$brain" = null ] && lesion="studies/brain-pathology-v1/tasks/sc-variant-null-lesion.json"
    echo "$(date '+%F %T') GPU $1 start $name"
    if $PY studies/brain-supplement-v2/code/replicate_training.py --model "$model" --replicate "$rep" --out "$out" --lesion "$lesion" \
        && $PY studies/brain-supplement-v2/code/replicate_training.py --model "$model" --replicate "$rep" --out "$out" --lesion "$lesion" --verify \
        && $PY studies/brain-pathology-v1/code/sc_variant_evaluate.py --run "$out" --lesion "$lesion"; then
        echo "$(date '+%F %T') GPU $1 done $name"
    else
        touch "$CLAIMS/$name/failed"
        echo "$(date '+%F %T') GPU $1 FAILED $name"
    fi
done
echo "$(date '+%F %T') GPU $1 no runs left"
