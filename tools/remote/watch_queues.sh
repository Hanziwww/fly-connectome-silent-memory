#!/bin/bash
# Watch the GPU job queues on both hosts (runtime/queue/queue.log; the first host's campaign log too) from the local
# machine, read-only over SSH (key login, BatchMode). Exits with one line when something needs attention:
#   exit 1  a job failed (a new "finished|trained|verified ... exit [1-9]" line since the watcher started);
#   exit 0  a host's queue drained (no worker alive where some were);
#   exit 2  a host was unreachable at start or for 6 polls in a row.
# Run it in the background from a Claude Code session (Bash run_in_background) so its exit notifies the session:
#   bash tools/remote/watch_queues.sh [POLL_SECONDS=900]
# Its baseline is the state at start, so earlier failures do not trigger it; re-arm it after handling each exit.
poll=${1:-900}
hosts="rtx5090 rtx5090x6"
state() {   # host -> "failures workers finished"
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$1" 'cd /root/autodl-tmp/flybrain/runtime 2>/dev/null || exit 3
        f=$(cat queue/queue.log campaign/campaign.log 2>/dev/null | grep -cE "(finished|trained|verified) .* exit [1-9]")
        w=$(ps -eo args | grep -cE "^sh studies/brain-supplement-v2/code/queue.sh|^sh runtime/queue/launch-first.sh")
        d=$(grep -cE "finished .* exit 0" queue/queue.log 2>/dev/null)
        echo "$f $w $d"' 2>/dev/null
}
declare -A base fails
for h in $hosts; do
    base[$h]=$(state $h); fails[$h]=0
    [ -z "${base[$h]}" ] && { echo "$h unreachable at start"; exit 2; }
done
while true; do
    sleep "$poll"
    for h in $hosts; do
        now=$(state $h)
        if [ -z "$now" ]; then
            fails[$h]=$((fails[$h] + 1))
            [ ${fails[$h]} -ge 6 ] && { echo "$h unreachable for 6 polls"; exit 2; }
            continue
        fi
        fails[$h]=0
        read -r f0 w0 d0 <<< "${base[$h]}"; read -r f w d <<< "$now"
        if [ "$f" != "$f0" ]; then echo "$h: failed jobs $f0 -> $f (finished ok: $d)"; exit 1; fi
        if [ "$w" = "0" ] && [ "$w0" != "0" ]; then echo "$h: queue drained (finished ok: $d)"; exit 0; fi
    done
done
