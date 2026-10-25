"""Phase 3, regime A (lesion after training, evaluation only) for the G2 shortlist. GPU host.

    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/phase3_regime_a.py

Conditions (g2-shortlist.json; doses from the Phase 2 calibration):
- C01 Abeta42 pan-neuronal Kv4 loss: threshold offset -3.22 mV x {0.5, 1, 1.5, 2} on every cell;
- C04 Abeta42 late KC loss: 17% of KCs removed x {1, 2}, plus 17% of size-matched random non-KC cells (control);
- C13 Fragile X: GABAergic ALLN release x 0.5 (dose 1) and x 0 (dose 2), plus the same release change on a random
  set of cells of equal size (control);
- C16 tau KC hypoexcitability: KC threshold +3.55 mV (dose 1) and +7.1 mV (dose 2), plus the same offset on random
  non-KC cells of equal size (control);
- healthy (no operator).
For each condition: the five memory models (M0 readout-only, M0 joint, shared readout, S1, S2) replayed at all 35
timings with the frozen free-after protocol (capped_memory_rollout: mechanism_model.memory_rollout without capture,
intervention or gate, and with the decision limit as a parameter, 256 here instead of 512); the giant-fibre response to
a looming ramp; mean activity under the M0 task input. Outputs: runs/phase3-regime-a/<condition>.json.

Decision limit: in the healthy brain every solved history needs at most 161 decisions (S2; 36 for M0) and every
unsolved one ends in a deadlock within 65, so a limit of 256 changes no healthy outcome. Lesioned policies that wander
without solving or deadlocking stop at 256 with status action_limit instead of 512; they count as unsolved either way.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
OUT = STUDY / 'runs' / 'phase3-regime-a'
MODELS = ['remote-dynamic-M0-head_only', 'remote-dynamic-M0-joint', 'supp-M0-visibility', 'supp-S1-ring-brain', 'supp-S2-context-brain']
TIMINGS = [(d, p) for d in (0, 20, 40, 80, 160, 320, 640) for p in (0, 1, 2, 4, 8)]
MAX_STEPS = 256


def capped_memory_rollout(policy, task, delay, prefix, max_steps=MAX_STEPS, variant=0):
    """mechanism_model.memory_rollout (frozen) without capture, intervention or gate, with the decision limit as a
    parameter; otherwise line for line the same inputs, decisions, statuses and records."""
    import mechanism_model as mm
    from game import legal, restore, solved, state_from, step
    lib = mm.tasklib()
    cases = task['cases']
    policy.reset(len(cases))
    xs = [lib.frames(c, delay, prefix, False, variant)[:-1] for c in cases]
    for i, frame in enumerate(xs[0]):
        mm.check_stop()
        policy.advance(np.stack([x[i]['board'] for x in xs]), frame['ms'])
    maps = [restore(c['level']) for c in cases]
    states = [state_from(c['level']['start']) for c in cases]
    safe = [set(c['solvable_states']) for c in cases]
    records = [dict(case=c['id'], cue=c['cue'], context=c['context'], visible=c.get('visible', False), delay_ms=delay,
                    prefix_windows=prefix, status='running', first_push_correct=None, decisions=[]) for c in cases]
    for index in range(max_steps):
        mm.check_stop()
        q, f = policy.advance(np.stack([lib.board(c, st_) for c, st_ in zip(cases, states)]), 40)
        if not np.isfinite(q).all():
            raise FloatingPointError('Nonfinite logits')
        for row, (r, c, m, st_) in enumerate(zip(records, cases, maps, states)):
            if r['status'] != 'running':
                continue
            mask = legal(m, st_)
            if not mask.any():
                r['status'] = 'blocked'
                continue
            a = int(np.where(mask, q[row], -np.inf).argmax())
            nxt = step(m, st_, a)
            r['decisions'].append(dict(state=[st_[0], list(st_[1])], action=a, legal=mask.tolist(), logits=q[row].tolist(),
                                       features=f[row].tolist()))
            if nxt[1] != st_[1] and r['first_push_correct'] is None:
                r['first_push_correct'] = lib.state_key(nxt) in safe[row]
                r['first_push_action'] = a
            if solved(m, nxt):
                r['status'] = 'solved'
            elif nxt[1] != st_[1] and lib.state_key(nxt) not in safe[row]:
                r['status'] = 'irreversible_deadlock'
            states[row] = nxt
        if all(r['status'] != 'running' for r in records):
            break
    for r in records:
        if r['status'] == 'running':
            r['status'] = 'action_limit'
        r['steps'] = len(r['decisions'])
        r['history_split'] = 'train' if delay in (0, 40, 160) and prefix in (0, 1, 4) else 'heldout'
    return records
C01_MV, C16_MV = -3.22, 3.55          # Phase 2 calibration (C16: mean of the two brains' fitted offsets)


def conditions(sets, n, rng):
    """[(name, case, dose, specs-builder)] with cell sets drawn once (seeded) so every model sees the same lesion."""
    import operators as op
    kc, gaba = sets['kc'], sets['alln_gaba']
    nonkc = np.setdiff1d(np.arange(n), kc)
    nongaba = np.setdiff1d(np.arange(n), gaba)
    kc17 = rng.choice(kc, int(round(0.17 * len(kc))), replace=False)
    kc34 = rng.choice(kc, int(round(0.34 * len(kc))), replace=False)
    rand_kcsize = rng.choice(nonkc, len(kc17), replace=False)
    rand_gaba = rng.choice(nongaba, len(gaba), replace=False)
    rand_kc_all = rng.choice(nonkc, len(kc), replace=False)
    every = np.arange(n)
    out = [('healthy', None, 0.0, [])]
    for dose in (0.5, 1.0, 1.5, 2.0):
        out.append((f'C01-dose{dose:g}', 'C01', dose, [op.spec('threshold_offset', every, C01_MV * dose, 'pan-neuronal Kv4 loss')]))
    out += [('C04-dose1', 'C04', 1.0, [op.spec('remove', kc17, label='17% of KCs')]),
            ('C04-dose2', 'C04', 2.0, [op.spec('remove', kc34, label='34% of KCs')]),
            ('C04-random', 'C04', 1.0, [op.spec('remove', rand_kcsize, label='random non-KC cells, size of 17% of KCs')]),
            ('C13-dose1', 'C13', 1.0, [op.spec('release_scale', gaba, 0.5, 'GABAergic ALLN release x0.5')]),
            ('C13-dose2', 'C13', 2.0, [op.spec('release_scale', gaba, 0.0, 'GABAergic ALLN release x0')]),
            ('C13-random', 'C13', 1.0, [op.spec('release_scale', rand_gaba, 0.5, 'random cells release x0.5')]),
            ('C16-dose1', 'C16', 1.0, [op.spec('threshold_offset', kc, C16_MV, 'KC threshold +3.55 mV')]),
            ('C16-dose2', 'C16', 2.0, [op.spec('threshold_offset', kc, 2 * C16_MV, 'KC threshold +7.1 mV')]),
            ('C16-random', 'C16', 1.0, [op.spec('threshold_offset', rand_kc_all, C16_MV, 'random non-KC cells +3.55 mV')])]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    import mechanism_base as mb
    import torch
    from mechanism_model import Frozen, memory_rollout
    import operators as op
    from credibility import Rig, cell_sets, SETTLE_MS
    rig = Rig(ROOT / 'runs' / 'linear-joint' / '20261002-p05-01' / 'model.pt', OUT / 'meta-rig')
    sets, n = rig.sets, rig.n
    conds = conditions(sets, n, np.random.default_rng(10))
    (OUT / 'conditions.json').write_text(json.dumps([dict(name=c[0], case=c[1], dose=c[2], operators=[dict(kind=s['kind'], cells=s['cells'].tolist(),
                                                                                                        value=s['value'], label=s['label']) for s in c[3]])
                                                      for c in conds]), encoding='utf-8')
    tasks = json.loads((MECH / 'inputs' / 'memory-tasks.json').read_text(encoding='utf-8'))
    seq = rig.memory_input()
    ramp = np.linspace(0, 20, 400)
    for name, case, dose, specs in conds:
        target = OUT / f'{name}.json'
        if target.exists():
            continue
        t0 = time.time()
        res = dict(condition=name, case=case, dose=dose, max_decisions=MAX_STEPS, operators=[dict(kind=s['kind'], cells=int(len(s['cells'])), value=s['value'], label=s['label']) for s in specs])
        rig.lesion(specs)
        counts, total, _ = rig.run_sequence(seq)
        res['activity'] = dict(task_mean_hz=float((counts / (total / 1000)).mean()), fraction_active=float((counts > 0).mean()))
        e, b = rig.enc, rig.enc.brain
        with torch.no_grad():
            e.reset()
            e.bind()
            b.set_drive(e.input_index, rig.blank)
            for _ in range(SETTLE_MS):
                b.step()
            gf = 0.0
            first = None
            for t in range(400):
                b.set_drive(sets['loom'], torch.full((1, len(sets['loom'])), float(ramp[t]), device='cuda'))
                b.step()
                k = float(b.spikes[0, sets['gf']].sum())
                gf += k
                if k and first is None:
                    first = t + 1
        res['loom'] = dict(gf_spikes=gf, gf_first_spike_ms=first)
        rig.lesion([])
        res['memory'] = {}
        for mid in MODELS:
            model = mb.model_by_id(mid)
            task = next(x for x in tasks if x['id'] == model['task'])
            policy = Frozen(model, OUT / 'meta-policies' / name / mid, batch=len(task['cases']))
            policy.load(MECH / model['checkpoints']['after']['path'])
            audit = op.apply(policy.encoder, specs)
            rows = []
            for d, p in TIMINGS:
                for r in capped_memory_rollout(policy, task, d, p):
                    first_logits = r['decisions'][0]['logits'] if r['decisions'] else None
                    rows.append(dict(case=r['case'], cue=r['cue'], context=r['context'], visible=r['visible'], delay_ms=d, prefix_windows=p,
                                     status=r['status'], steps=r['steps'], first_push_correct=r['first_push_correct'],
                                     first_action=r['decisions'][0]['action'] if r['decisions'] else None, first_logits=first_logits,
                                     history_split=r['history_split']))
            res['memory'][mid] = dict(audit=audit, rows=rows)
            del policy
            torch.cuda.empty_cache()
        res['seconds'] = time.time() - t0
        target.write_text(json.dumps(res), encoding='utf-8', newline='\n')
        solved = {m: sum(r['status'] == 'solved' for r in v['rows']) for m, v in res['memory'].items()}
        print(f'{name}: {res["seconds"]:.0f} s, GF {gf:.0f}, task {res["activity"]["task_mean_hz"]:.2f} Hz, solved {solved}', flush=True)
    (OUT / 'done.json').write_text(json.dumps(dict(finished_unix=time.time())), encoding='utf-8')


if __name__ == '__main__':
    main()
