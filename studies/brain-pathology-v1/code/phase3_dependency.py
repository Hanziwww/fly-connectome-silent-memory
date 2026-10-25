"""G3a, circuit dependency map (GPU host): every condition of tasks/phase3-dependency-conditions.json (built locally by
phase3_dependency_prepare.py, committed before this run) applied to the healthy brain and measured as in regime A, but
at the 9 trained timings only.

    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/phase3_dependency.py

Per condition (cell removal, operators.py 'remove'):
- memory: the five trained memory models replayed at the 9 trained timings (regime A's capped rollout, 256 decisions);
- movement: the giant-fibre response to the looming ramp of regime A;
- activity: mean rate under the M0 task input.
The target, monoamine and descending-neuron sets are recomputed from the brain built here and must equal the prepared
ones. Outputs runs/phase3-dependency/<condition>.json; a finished condition is skipped (resumable).
"""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
OUT = STUDY / 'runs' / 'phase3-dependency'
CONDITIONS = STUDY / 'tasks' / 'phase3-dependency-conditions.json'
TRAINED = [(d, p) for d in (0, 40, 160) for p in (0, 1, 4)]


def main():
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    import mechanism_base as mb
    import torch
    from mechanism_model import Frozen
    import operators as op
    from credibility import Rig, SETTLE_MS
    from phase3_regime_a import MODELS, capped_memory_rollout
    OUT.mkdir(parents=True, exist_ok=True)
    raw = CONDITIONS.read_bytes()
    spec = json.loads(raw)
    rig = Rig(ROOT / 'runs' / 'linear-joint' / '20261002-p05-01' / 'model.pt', OUT / 'meta-rig')
    sets = dict(rig.sets, descending=np.flatnonzero((rig.enc.c.neurons['superclass'].fillna('') == 'descending_neuron').to_numpy()))
    by_name = {c['name']: c for c in spec['conditions']}
    for pop in ('kc', 'alln_gaba', 'monoamine', 'descending'):
        if sorted(by_name[f'{pop}-1']['cells']) != sorted(sets[pop].tolist()):
            raise RuntimeError(f'{pop}: the prepared cell set differs from the brain built here')
    tasks = json.loads((MECH / 'inputs' / 'memory-tasks.json').read_text(encoding='utf-8'))
    seq = rig.memory_input()
    ramp = np.linspace(0, 20, 400)
    sha = hashlib.sha256(raw).hexdigest()
    for c in spec['conditions']:
        target = OUT / f"{c['name']}.json"
        if target.exists():
            continue
        t0 = time.time()
        specs = [op.spec('remove', np.asarray(c['cells'], np.int64), label=c['name'])] if c['cells'] else []
        res = dict(condition=c['name'], population=c['population'], level=c['level'], kind=c['kind'], cells=len(c['cells']),
                   borrowed=c['borrowed'], conditions_sha256=sha, timings=TRAINED)
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
            gf, first = 0.0, None
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
            policy = Frozen(model, OUT / 'meta-policies' / c['name'] / mid, batch=len(task['cases']))
            policy.load(MECH / model['checkpoints']['after']['path'])
            audit = op.apply(policy.encoder, specs)
            rows = []
            for d, p in TRAINED:
                for r in capped_memory_rollout(policy, task, d, p):
                    rows.append(dict(case=r['case'], cue=r['cue'], context=r['context'], visible=r['visible'], delay_ms=d, prefix_windows=p,
                                     status=r['status'], steps=r['steps'], first_push_correct=r['first_push_correct'],
                                     first_action=r['decisions'][0]['action'] if r['decisions'] else None,
                                     first_logits=r['decisions'][0]['logits'] if r['decisions'] else None))
            res['memory'][mid] = dict(audit=audit, rows=rows)
            del policy
            gc.collect()                   # operators.apply leaves a reference cycle (enc.bind); free the GPU memory now
            torch.cuda.empty_cache()
        res['seconds'] = time.time() - t0
        target.write_text(json.dumps(res), encoding='utf-8', newline='\n')
        solved = {m: sum(r['status'] == 'solved' for r in v['rows']) for m, v in res['memory'].items()}
        print(f"{c['name']}: {res['seconds']:.0f} s, GF {gf:.0f}, task {res['activity']['task_mean_hz']:.2f} Hz, solved {solved}", flush=True)
    (OUT / 'done.json').write_text(json.dumps(dict(finished_unix=time.time(), conditions_sha256=sha)), encoding='utf-8')


if __name__ == '__main__':
    main()
