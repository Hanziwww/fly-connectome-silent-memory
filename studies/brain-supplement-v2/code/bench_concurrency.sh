#!/bin/sh
# Phase 0b(e): run 1, 2, 4 and 6 concurrent assessment workers and record the wall time of each wave.
# Usage (on the GPU host, from the project root): sh studies/brain-supplement-v2/code/bench_concurrency.sh
set -e
out=studies/brain-supplement-v2/runs/g0-bench-concurrency
mkdir -p "$out"
for n in 1 2 4 6; do
    start=$(date +%s.%N)
    i=1
    while [ "$i" -le "$n" ]; do
        .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/bench_concurrency.py --tag "n$n-$i" --k 4 > "$out/n$n-$i.log" 2>&1 &
        i=$((i + 1))
    done
    wait
    end=$(date +%s.%N)
    echo "n=$n wall=$(awk "BEGIN{print $end - $start}")" >> "$out/waves.txt"
done
echo done >> "$out/waves.txt"
