"""X4 (plan v2, Phase 1C; gap G08): the cue-direction state shift against 20 norm-matched scrambled shifts at 5 doses.
GPU host.

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/x4_scrambles.py

The frozen supplement's selective intervention (studies/brain-sokoban-v1/mechanisms-supplement-v1/code/supp_replay.py:
rollout, manipulate, matched_control) runs unchanged; only the scramble seed varies, by wrapping matched_control
(seed + 1000 k). The paper had one scrambled draw per condition.
- Models and timings: the supplement's five memory models at its 6 timings (delays 40, 160, 640 ms x prefix 0, 4).
- Intervention: read window (after the last pre-decision input), combined continuous state (v, g, g_slow, g_slow_cls,
  adapt), dose a in 0.125, 0.25, 0.5, 0.75, 1. The cue shift moves each history's state a of the way to its
  opposite-cue partner's; a scrambled shift has the same per-case, per-field norm in a random direction orthogonal to it.
- Per model and timing: the baseline; the cue shift at the 5 doses; scrambled draws k = 1..20 at the 5 doses; the frozen
  draw k = 0 at the frozen doses 0.25, 0.5 and 1 (it must reproduce the frozen outcomes; checked by analyze_x4_x5.py
  against mechanisms-supplement-v1/analysis/condition-outcomes.csv); the same-cue control, which must equal the baseline.
Outputs runs/x4-scrambles/<model>-d<delay>-p<prefix>.json: per history and condition the first-decision action, logits
and legal mask, the status and steps, and the norm audit; resumable.
"""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
FROZEN_SUPP = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-supplement-v1'
OUT = SUPP / 'runs' / 'x4-scrambles'
DOSES = (0.125, 0.25, 0.5, 0.75, 1.0)
FROZEN_DOSES = (0.25, 0.5, 1.0)
DRAWS = 20
SEED_STEP = 1000


def compact(rows):
    out = []
    for r in rows:
        d0 = r['decisions'][0] if r['decisions'] else None
        out.append(dict(case=r['case'], cue=r['cue'], context=r['context'], visible=r['visible'], status=r['status'], steps=r['steps'],
                        first_push_correct=r['first_push_correct'], first_action=d0['action'] if d0 else None,
                        first_logits=d0['logits'] if d0 else None, first_legal=d0['legal'] if d0 else None))
    return out


def main():
    sys.path.insert(0, str(FROZEN_SUPP / 'code'))
    import torch
    import supp_base
    import supp_replay
    original = supp_replay.matched_control
    offset = [0]

    def matched_control(value, delta, nonnegative=False, seed=10):
        return original(value, delta, nonnegative, seed + offset[0])

    supp_replay.matched_control = matched_control           # looked up at call time by supp_replay.manipulate
    OUT.mkdir(parents=True, exist_ok=True)
    plan = ([('cue', a, None) for a in DOSES] + [('scrambled', a, k) for k in range(1, DRAWS + 1) for a in DOSES]
            + [('scrambled', a, 0) for a in FROZEN_DOSES])
    for mid in supp_base.MODEL_IDS:
        model = supp_base.get_model(mid)
        task = supp_base.task_for(model)
        policy = None
        for delay, prefix in supp_base.TIMINGS:
            target = OUT / f'{mid}-d{delay}-p{prefix}.json'
            if target.exists():
                continue
            if policy is None:
                policy = supp_replay.FrozenFollowup(model, OUT / 'meta' / mid, batch=len(task['cases']))
                policy.load(supp_base.checkpoint(model))
            t0 = time.time()
            base, _ = supp_replay.rollout(policy, task, delay, prefix)
            conds = [dict(kind='baseline', dose=0.0, draw=None, rows=compact(base), audit=[])]
            for kind, a, k in plan:
                offset[0] = SEED_STEP * (k or 0)
                rows, audit = supp_replay.rollout(policy, task, delay, prefix, spec=dict(window='read', part='continuous', fraction=a, kind=kind))
                conds.append(dict(kind=kind, dose=a, draw=k, rows=compact(rows), audit=audit))
            offset[0] = 0
            same, _ = supp_replay.rollout(policy, task, delay, prefix, spec=dict(window='read', part='continuous', fraction=1.0, kind='same_cue'))
            supp_replay.assert_equal(same, base)
            invalid = sum(not x['norm_matched'] for c in conds if c['kind'] == 'scrambled' for x in c['audit'])
            target.write_text(json.dumps(dict(model=mid, delay=delay, prefix=prefix, doses=DOSES, draws=DRAWS, seed_step=SEED_STEP,
                                              same_cue_equals_baseline=True, invalid_norm_matches=invalid, conditions=conds,
                                              seconds=time.time() - t0)), encoding='utf-8', newline='\n')
            print(f'{mid} d{delay} p{prefix}: {len(conds)} conditions, {time.time() - t0:.0f} s, invalid norm matches {invalid}', flush=True)
        if policy is not None:
            del policy
            gc.collect()                   # operators.apply leaves a reference cycle (enc.bind); free the GPU memory now
            torch.cuda.empty_cache()
    (OUT / 'done.json').write_text(json.dumps(dict(finished_unix=time.time())), encoding='utf-8')


if __name__ == '__main__':
    main()
