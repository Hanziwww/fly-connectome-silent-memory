"""Phase 4 dose series (GPU host; conditions built locally with --prepare): where does memory fail, compared with the
network's stability limit and with the structural (percolation) threshold?

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\phase4_dose_series.py --prepare
    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/phase4_dose_series.py [--set low]

Conditions (tasks/phase4-dose-conditions.json, written by --prepare before any run):
- C01 fine grid: threshold offset -3.22 mV x {0.05, 0.1, 0.2, 0.3, 0.4} on every cell (regime A has 0, 0.5, 1, 1.5, 2);
  the mean-field result (PLAN.md section 19) puts the runaway onset between -6.0 and -6.5 mV (dose 1.9-2.0);
- random cell removal: shares {0.002, 0.005, 0.01, 0.02, 0.05} of the cells that are neither input nor readout cells,
  3 seeded draws each (phase4_graph.py found no structural threshold up to 90%).
Measures as G3a (phase3_dependency.py): the five memory models at the 9 trained timings (regime A's capped rollout), the
giant-fibre looming response, mean task activity. Outputs runs/phase4-dose/<condition>.json; resumable.
The dose-response fits (4-parameter logistic, ED50 with a hierarchical bootstrap over models and histories; linear and
threshold models compared by AIC; studykit.stats_ext) are estimates, not tests (analyze_phase4_dose.py).

Low extension (--set low; post hoc, amendment 0001, declared before any of its runs): the main series put the C01 ED50
below its lowest dose, so tasks/phase4-dose-low-conditions.json adds C01 at 0 (the healthy brain through this code
path, which must reproduce regime A's healthy outcome), 0.003125, 0.00625, 0.0125 and 0.025, and random removal of
0.0002, 0.0005 and 0.001 of the pool (3 new draws each, seed 20261020); outputs runs/phase4-dose-low. Both sets now
also keep each history's first_push_correct (energy per bit).
Micro set (--set micro; T-DY1, F14.DY.sensitivity, amendment 0002, declared before any of its runs): C01 static shifts of
-1e-4, -3e-4, -1e-3 and -3e-3 mV (doses d / 3.22); outputs runs/phase4-dose-micro.
"""
import argparse
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
OUT = STUDY / 'runs' / 'phase4-dose'
CONDITIONS = STUDY / 'tasks' / 'phase4-dose-conditions.json'
C01_MV = -3.22
FINE = (0.05, 0.1, 0.2, 0.3, 0.4)
SHARES = (0.002, 0.005, 0.01, 0.02, 0.05)
DRAWS = 3
SEED = 20261019
SETS = {'main': dict(conditions=CONDITIONS, out=OUT, fine=FINE, shares=SHARES, seed=SEED),
        'low': dict(conditions=STUDY / 'tasks' / 'phase4-dose-low-conditions.json', out=STUDY / 'runs' / 'phase4-dose-low',
                    fine=(0.0, 0.003125, 0.00625, 0.0125, 0.025), shares=(0.0002, 0.0005, 0.001), seed=20261020),
        # T-DY1 (F14.DY.sensitivity, amendment 0002): static shifts of -1e-4, -3e-4, -1e-3 and -3e-3 mV, below the low extension
        'micro': dict(conditions=STUDY / 'tasks' / 'phase4-dose-micro-conditions.json', out=STUDY / 'runs' / 'phase4-dose-micro',
                      fine=tuple(d / 3.22 for d in (1e-4, 3e-4, 1e-3, 3e-3)), shares=(), seed=20261021)}


def prepare(cfg):
    import pandas as pd
    ann = pd.read_parquet(MECH / 'neurons' / 'annotations.parquet', columns=['direct_input', 'observed_by_policy'])
    n = len(ann)
    pool = np.flatnonzero(~ann['direct_input'].to_numpy(bool) & ~ann['observed_by_policy'].to_numpy(bool))
    rng = np.random.default_rng(cfg['seed'])
    conds = [dict(name=f'C01-dose{d:g}', kind='threshold_offset', dose=d, value=C01_MV * d, cells='all') for d in cfg['fine']]
    for q in cfg['shares']:
        m = int(round(q * len(pool)))
        for k in range(DRAWS):
            conds.append(dict(name=f'random-{q:g}-draw{k}', kind='remove', dose=q, draw=k, cells=np.sort(rng.choice(pool, m, replace=False)).tolist()))
    text = json.dumps(dict(meta=dict(seed=cfg['seed'], cells=n, pool=int(len(pool)), c01_mv=C01_MV), conditions=conds), separators=(',', ':'))
    if cfg['conditions'].exists():
        raise SystemExit(f"{cfg['conditions'].name} exists; task files are written once, before their runs")
    cfg['conditions'].write_text(text, encoding='utf-8', newline='\n')
    print(f'{len(conds)} conditions, sha256 {hashlib.sha256(text.encode()).hexdigest()}')


def run(cfg):
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    import mechanism_base as mb
    import torch
    from mechanism_model import Frozen
    import operators as op
    from credibility import Rig, SETTLE_MS
    from phase3_regime_a import MODELS, capped_memory_rollout
    from phase3_dependency import TRAINED
    OUT = cfg['out']
    raw = cfg['conditions'].read_bytes()
    spec = json.loads(raw)
    OUT.mkdir(parents=True, exist_ok=True)
    rig = Rig(ROOT / 'runs' / 'linear-joint' / '20261002-p05-01' / 'model.pt', OUT / 'meta-rig')
    if rig.n != spec['meta']['cells']:
        raise RuntimeError('cell count differs')
    tasks = json.loads((MECH / 'inputs' / 'memory-tasks.json').read_text(encoding='utf-8'))
    seq = rig.memory_input()
    ramp = np.linspace(0, 20, 400)
    sha = hashlib.sha256(raw).hexdigest()
    for c in spec['conditions']:
        target = OUT / f"{c['name']}.json"
        if target.exists():
            continue
        t0 = time.time()
        cells = np.arange(rig.n) if c['cells'] == 'all' else np.asarray(c['cells'], np.int64)
        specs = [op.spec(c['kind'], cells, c.get('value'), c['name'])]
        res = dict(condition=c['name'], kind=c['kind'], dose=c['dose'], cells=int(len(cells)), conditions_sha256=sha, timings=TRAINED)
        rig.lesion(specs)
        counts, total, _ = rig.run_sequence(seq)
        res['activity'] = dict(task_mean_hz=float((counts / (total / 1000)).mean()), fraction_active=float((counts > 0).mean()),
                               fraction_above_100hz=float((counts / (total / 1000) > 100).mean()))
        e, b = rig.enc, rig.enc.brain
        with torch.no_grad():
            e.reset()
            e.bind()
            b.set_drive(e.input_index, rig.blank)
            for _ in range(SETTLE_MS):
                b.step()
            gf, first = 0.0, None
            for t in range(400):
                b.set_drive(rig.sets['loom'], torch.full((1, len(rig.sets['loom'])), float(ramp[t]), device='cuda'))
                b.step()
                k = float(b.spikes[0, rig.sets['gf']].sum())
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--set', default='main', choices=list(SETS))
    a = ap.parse_args()
    prepare(SETS[a.set]) if a.prepare else run(SETS[a.set])
