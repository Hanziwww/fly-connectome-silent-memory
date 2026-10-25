"""X5 (plan v2, Phase 1C; gap G07): opposite-cue swaps at all 9 trained timings, hidden vs visible goal, and S2
cross-orientation swaps. GPU host.

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/x5_swaps.py

The frozen supplement's rollout and manipulate (studies/brain-sokoban-v1/mechanisms-supplement-v1/code/supp_replay.py)
run unchanged; the donor of a cross-orientation swap is chosen by wrapping supp_replay.paired_indices.
- Models: the supplement's five memory models (the shared readout carries hidden- and visible-goal histories).
- Timings: the 9 trained timings (delays 0, 40, 160 ms x prefix 0, 1, 4); the frozen swaps covered 4 of them.
- Conditions per model and timing: the baseline; the opposite-cue swap (kind 'cue', dose 1: the whole continuous state
  replaced by the opposite cue's, read window); the same-cue control (must equal the baseline). S2 also: donor = the
  same cue in the other context, and donor = the opposite cue in the other context.
Outputs runs/x5-swaps/<model>-d<delay>-p<prefix>.json; resumable.
"""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
FROZEN_SUPP = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-supplement-v1'
OUT = SUPP / 'runs' / 'x5-swaps'
TRAINED = [(d, p) for d in (0, 40, 160) for p in (0, 1, 4)]


def cross_pairs(same_cue):
    def paired(cases):
        return [next(j for j, d in enumerate(cases) if d['cue'] == (c['cue'] if same_cue else 1 - c['cue'])
                     and d['context'] != c['context'] and d.get('visible', False) == c.get('visible', False)) for c in cases]
    return paired


def main():
    sys.path.insert(0, str(FROZEN_SUPP / 'code'))
    sys.path.insert(0, str(SUPP / 'code'))
    import torch
    import supp_base
    import supp_replay
    from x4_scrambles import compact
    frozen_pairs = supp_replay.paired_indices
    OUT.mkdir(parents=True, exist_ok=True)
    swap = dict(window='read', part='continuous', fraction=1.0, kind='cue')
    for mid in supp_base.MODEL_IDS:
        model = supp_base.get_model(mid)
        task = supp_base.task_for(model)
        policy = None
        for delay, prefix in TRAINED:
            target = OUT / f'{mid}-d{delay}-p{prefix}.json'
            if target.exists():
                continue
            if policy is None:
                policy = supp_replay.FrozenFollowup(model, OUT / 'meta' / mid, batch=len(task['cases']))
                policy.load(supp_base.checkpoint(model))
            t0 = time.time()
            base, _ = supp_replay.rollout(policy, task, delay, prefix)
            conds = [dict(kind='baseline', rows=compact(base), audit=[])]
            rows, audit = supp_replay.rollout(policy, task, delay, prefix, spec=swap)
            conds.append(dict(kind='opposite_cue', rows=compact(rows), audit=audit))
            same, _ = supp_replay.rollout(policy, task, delay, prefix, spec=dict(swap, kind='same_cue'))
            supp_replay.assert_equal(same, base)
            if len({c['context'] for c in task['cases']}) > 1:
                for name, same_cue in (('cross_same_cue', True), ('cross_opposite_cue', False)):
                    supp_replay.paired_indices = cross_pairs(same_cue)
                    try:
                        rows, audit = supp_replay.rollout(policy, task, delay, prefix, spec=swap)
                    finally:
                        supp_replay.paired_indices = frozen_pairs
                    conds.append(dict(kind=name, rows=compact(rows), audit=audit))
            target.write_text(json.dumps(dict(model=mid, delay=delay, prefix=prefix, same_cue_equals_baseline=True, conditions=conds,
                                              seconds=time.time() - t0)), encoding='utf-8', newline='\n')
            print(f'{mid} d{delay} p{prefix}: ' + ', '.join(f"{c['kind']} {sum(r['status'] == 'solved' for r in c['rows'])}/{len(c['rows'])}" for c in conds)
                  + f', {time.time() - t0:.0f} s', flush=True)
        if policy is not None:
            del policy
            gc.collect()                   # operators.apply leaves a reference cycle (enc.bind); free the GPU memory now
            torch.cuda.empty_cache()
    (OUT / 'done.json').write_text(json.dumps(dict(finished_unix=time.time())), encoding='utf-8')


if __name__ == '__main__':
    main()
