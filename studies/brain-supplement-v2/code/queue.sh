#!/bin/sh
# One GPU's worker on a shared job queue (plan v2, once a second host joined). Start one per GPU:
#   nohup sh studies/brain-supplement-v2/code/queue.sh <gpu> > runtime/queue/worker-<gpu>.out 2>&1 < /dev/null &
# The queue is runtime/queue/jobs.tsv (queue_jobs.py): id, done marker, minutes, required markers, command. Under the
# lock runtime/queue/claim.lock a worker takes the first job that is neither done (marker present) nor claimed by a
# live worker and whose required markers all exist, records the claim in runtime/queue/claims/<id>, runs the command
# from the project root with CUDA_VISIBLE_DEVICES=<gpu>, and logs the exit code in runtime/queue/queue.log. A failed
# job keeps its claim (a failure is logged, not retried); a claim of a dead worker is released. Each GPU also holds its
# own lock (runtime/queue/gpu-<gpu>.lock) for its whole life, so no two workers share a GPU. The worker exits when no
# job is left, or waits (60 s) while the only jobs left wait for requirements. STOP: runtime/queue/STOP.
gpu=$1
[ -z "$gpu" ] && { echo "usage: queue.sh <gpu>"; exit 2; }
q=runtime/queue
mkdir -p "$q/claims" "$q/logs"
exec 8> "$q/gpu-$gpu.lock"
if ! flock -n 8; then echo "$(date -u +%FT%TZ) gpu $gpu: another worker holds this GPU" >> "$q/queue.log"; exit 1; fi
echo "$(date -u +%FT%TZ) gpu $gpu: worker $$ started" >> "$q/queue.log"
while true; do
    if [ -e "$q/STOP" ]; then echo "$(date -u +%FT%TZ) gpu $gpu: STOP" >> "$q/queue.log"; exit 75; fi
    job=$(
        exec 9> "$q/claim.lock"
        flock 9
        waiting=0
        while IFS="$(printf '\t')" read -r id marker minutes requires cmd; do
            [ -z "$id" ] && continue
            [ -e "$marker" ] && continue
            if [ -e "$q/claims/$id" ]; then
                owner=$(cut -d' ' -f1 "$q/claims/$id")
                if kill -0 "$owner" 2>/dev/null || [ -e "$q/claims/$id.failed" ]; then continue; fi
            fi
            ready=1
            if [ "$requires" != "-" ]; then
                for m in $(echo "$requires" | tr ',' ' '); do [ -e "$m" ] || ready=0; done
            fi
            if [ "$ready" -eq 0 ]; then waiting=1; continue; fi
            echo "$$ gpu$gpu $(date -u +%FT%TZ)" > "$q/claims/$id"
            echo "$id"
            exit 0
        done < "$q/jobs.tsv"
        [ "$waiting" -eq 1 ] && echo "__wait__"
        exit 0
    )
    if [ -z "$job" ]; then echo "$(date -u +%FT%TZ) gpu $gpu: queue empty, worker exits" >> "$q/queue.log"; exit 0; fi
    if [ "$job" = "__wait__" ]; then sleep 60; continue; fi
    cmd=$(awk -F '\t' -v id="$job" '$1 == id {print $5}' "$q/jobs.tsv")
    echo "$(date -u +%FT%TZ) gpu $gpu: start $job" >> "$q/queue.log"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 sh -c "$cmd" 8>&- 9>&-
    code=$?
    echo "$(date -u +%FT%TZ) gpu $gpu: finished $job exit $code" >> "$q/queue.log"
    [ "$code" -ne 0 ] && touch "$q/claims/$job.failed"
done
