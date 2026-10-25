"""T-SC4: Kv4-weighted C01 (evaluation only; F14.SC.kv4, .equivalence, .descriptive; tasks/sc-panels-v1.json, amendment 0005).

    local, once:   .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\sc_kv4.py --prepare
    host, per GPU: CUDA_VISIBLE_DEVICES=<g> .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/sc_kv4.py --worker <g>
                   (regime B conditions in a separate process: ... sc_kv4.py --worker-b <g>)

--prepare writes tasks/sc-kv4-conditions.json once, before any run: the per-cell weights w of the class and global
imputations (sc_index.kv4_weights over mapping v1; mean 1 over the 167,106 cells), grouped by value, and the conditions:
- check-uniform-dose0.5: w = 1 through the weighted code path; must reproduce runs/phase3-regime-a/C01-dose0.5.json;
- kv4-{class,global}-dose{0.5,1,1.5,2}: the regime A protocol of phase3_regime_a.py (five memory models at the 35
  timings, 256 decisions; task activity and the looming response on the P05 rig);
- kv4-class-dose{0.003125 ... 0.4}: the dose-series protocol of phase4_dose_series.py (9 trained timings) at the doses of
  runs/phase4-dose and runs/phase4-dose-low;
- loo-<superclass>: dose 1, class weights with the cells of one of the nine largest superclasses at 1, rescaled to mean
  1; dose-series protocol;
- b-<run>-{uniform,kv4}: the regime B runs C01 dose 0.5, M0 readout-only and joint, r1-r3 (inputs/regime-b/<run>:
  config.json, model.pt) under the frozen trainer's full assessment as in phase3_rescue.py, with the uniform lesion they
  learned with (must reproduce the run's verification; compared locally) and with the class-weighted lesion of the same
  mean (-1.61 mV x w).
Workers claim conditions longest first (runs/sc-kv4/claims/<name>, created exclusively); outputs runs/sc-kv4/<name>.json.
The third operator of the declaration (the AD-FCA-measured Shal change) is not built: T-SC4b found pan-neuronal Shal
mRNA higher, not lower, under Abeta42 (amendment 0006).
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
CONDITIONS = STUDY / 'tasks' / 'sc-kv4-conditions.json'
OUT = STUDY / 'runs' / 'sc-kv4'
REGIME_B = STUDY / 'inputs' / 'regime-b'
C01_MV = -3.22
DOSES = (0.5, 1.0, 1.5, 2.0)
SERIES = (0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4)
B_RUNS = [f'b-C01-dose0.5-M0-{learner}-r{r}' for learner in ('head_only', 'joint') for r in (1, 2, 3)]
# expected cost order (longest first): regime A at high doses, regime B assessments, then the trained-timing sets
COST = {'regime-a': 3, 'regime-b': 2, 'trained': 1}


def groups(w):
    values, inverse = np.unique(w, return_inverse=True)
    return [[float(v), np.flatnonzero(inverse == i).tolist()] for i, v in enumerate(values)]


def prepare():
    sys.path.insert(0, str(STUDY / 'code'))
    import pandas as pd
    from sc_common import INDEX_GENES, KV4_GENE
    from sc_index import cell_map, kv4_weights, unit_values
    if CONDITIONS.exists():
        raise SystemExit(f'{CONDITIONS.name} exists; task files are written once, before their runs')
    cells = cell_map()
    neurons = pd.read_parquet(ROOT / 'flybrain' / 'cache' / 'neurons.parquet', columns=['bodyId'])
    if not (neurons['bodyId'].to_numpy() == cells['bodyId'].to_numpy()).all():
        raise SystemExit('cell order differs from the brain')
    table = unit_values(cells, INDEX_GENES + (KV4_GENE,))
    w = {imp: kv4_weights(cells, table, imp) for imp in ('class', 'global')}
    sc = cells['superclass'].fillna('').astype(str).to_numpy()
    largest = pd.Series(sc[sc != '']).value_counts().index[:9].tolist()
    conds = [dict(name='check-uniform-dose0.5', protocol='regime-a', weights='uniform', dose=0.5)]
    conds += [dict(name=f'kv4-{imp}-dose{d:g}', protocol='regime-a', weights=imp, dose=d) for imp in ('class', 'global')
              for d in DOSES]
    conds += [dict(name=f'kv4-class-dose{d:g}', protocol='trained', weights='class', dose=d) for d in SERIES]
    conds += [dict(name=f'loo-{s}', protocol='trained', weights='class', dose=1.0, uniform_superclass=s) for s in largest]
    conds += [dict(name=f'{run}-{kind}', protocol='regime-b', weights='uniform' if kind == 'uniform' else 'class',
                   dose=0.5, run=run) for run in B_RUNS for kind in ('uniform', 'kv4')]
    stats = {imp: dict(min=float(v.min()), p01=float(np.quantile(v, 0.01)), median=float(np.median(v)),
                       p99=float(np.quantile(v, 0.99)), max=float(v.max()), mean=float(v.mean())) for imp, v in w.items()}
    spec = dict(meta=dict(cells=len(cells), c01_mv=C01_MV, mapping='tasks/sc-mapping-v1.json',
                          definitions='tasks/sc-panels-v1.json (definitions.kv4_weight)', weight_stats=stats,
                          mapped_cells=int(cells['tier'].ne('unmapped').sum())),
                weights={imp: groups(v) for imp, v in w.items()},
                superclass_cells={s: np.flatnonzero(sc == s).tolist() for s in largest},
                conditions=conds)
    text = json.dumps(spec, separators=(',', ':'))
    CONDITIONS.write_text(text, encoding='utf-8', newline='\n')
    print(f'{len(conds)} conditions; weights {json.dumps(stats)}; sha256 {hashlib.sha256(text.encode()).hexdigest()}')


def weight_vector(spec, c, n):
    if c['weights'] == 'uniform':
        return np.ones(n)
    w = np.empty(n)
    for value, cells in spec['weights'][c['weights']]:
        w[cells] = value
    if c.get('uniform_superclass'):
        w[spec['superclass_cells'][c['uniform_superclass']]] = 1.0
        w = w / w.mean()
    return w


def offset_specs(op, w, dose, label):
    """threshold_offset specs, one per distinct weight; each cell gets exactly -3.22 mV x dose x w."""
    values, inverse = np.unique(w, return_inverse=True)
    return [op.spec('threshold_offset', np.flatnonzero(inverse == i), C01_MV * dose * float(v), label) for i, v in enumerate(values)]


def rig_checks(rig, specs, torch, settle_ms, seq):
    """Task activity and the looming response of the P05 rig under the specs (as phase3_regime_a.py)."""
    rig.lesion(specs)
    counts, total, _ = rig.run_sequence(seq)
    activity = dict(task_mean_hz=float((counts / (total / 1000)).mean()), fraction_active=float((counts > 0).mean()),
                    fraction_above_100hz=float((counts / (total / 1000) > 100).mean()))
    e, b = rig.enc, rig.enc.brain
    ramp = np.linspace(0, 20, 400)
    with torch.no_grad():
        e.reset()
        e.bind()
        b.set_drive(e.input_index, rig.blank)
        for _ in range(settle_ms):
            b.step()
        gf, first = 0.0, None
        for t in range(400):
            b.set_drive(rig.sets['loom'], torch.full((1, len(rig.sets['loom'])), float(ramp[t]), device='cuda'))
            b.step()
            k = float(b.spikes[0, rig.sets['gf']].sum())
            gf += k
            if k and first is None:
                first = t + 1
    rig.lesion([])
    return activity, dict(gf_spikes=gf, gf_first_spike_ms=first)


def memory_rows(mb, Frozen, op, ra, tasks, specs, timings, meta, torch, split):
    out = {}
    for mid in ra.MODELS:
        model = mb.model_by_id(mid)
        task = next(x for x in tasks if x['id'] == model['task'])
        policy = Frozen(model, meta / mid, batch=len(task['cases']))
        policy.load(MECH / model['checkpoints']['after']['path'])
        audit = op.apply(policy.encoder, specs)
        rows = []
        for d, p in timings:
            for r in ra.capped_memory_rollout(policy, task, d, p):
                row = dict(case=r['case'], cue=r['cue'], context=r['context'], visible=r['visible'], delay_ms=d, prefix_windows=p,
                           status=r['status'], steps=r['steps'], first_push_correct=r['first_push_correct'],
                           first_action=r['decisions'][0]['action'] if r['decisions'] else None,
                           first_logits=r['decisions'][0]['logits'] if r['decisions'] else None)
                if split:
                    row['history_split'] = r['history_split']
                rows.append(row)
        out[mid] = dict(audit=audit, rows=rows)
        del policy
        gc.collect()                   # operators.apply leaves a reference cycle (enc.bind); free the GPU memory now
        torch.cuda.empty_cache()
    return out


def regime_b(c, specs, op, meta, torch, n):
    """The frozen trainer's full assessment of a regime B run under the given lesion specs (phase3_rescue.evaluate)."""
    import memory as legacy
    import phase3_rescue as rescue
    run = REGIME_B / c['run']
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    uniform = [op.spec('threshold_offset', np.arange(n), C01_MV * 0.5, 'pan-neuronal Kv4 loss, dose 0.5')]
    if [dict(kind=s['kind'], cells=int(len(s['cells'])), value=s['value'], label=s['label']) for s in uniform] != config['lesion']:
        raise RuntimeError(f'{run.name}: the uniform lesion differs from the run config')
    meta.mkdir(parents=True, exist_ok=True)
    policy = legacy.MemoryPolicy(config, meta)
    saved = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
    if saved['config'] != config:
        raise RuntimeError(f'{run.name}: model.pt was written with another config')
    policy.load(saved['pair'])
    audit = op.apply(policy.encoder, uniform if c['weights'] == 'uniform' else specs)
    task = next(t for t in legacy.prepare_memory() if t['id'] == config['task'])
    result = rescue.slim(legacy.assess(policy, task, True))
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return dict(config_id=config['id'], audit=audit, results=result)


def claim(name):
    (OUT / 'claims').mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(OUT / 'claims' / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def worker(gpu):
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    import mechanism_base as mb
    import torch
    from mechanism_model import Frozen
    import operators as op
    import phase3_regime_a as ra
    from credibility import Rig, SETTLE_MS
    from phase3_dependency import TRAINED
    raw = CONDITIONS.read_bytes()
    spec = json.loads(raw)
    sha = hashlib.sha256(raw).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = json.loads((MECH / 'inputs' / 'memory-tasks.json').read_text(encoding='utf-8'))
    rig = seq = None
    order = sorted(spec['conditions'], key=lambda c: (-COST[c['protocol']], -c['dose']))
    for c in order:
        target = OUT / f"{c['name']}.json"
        if c['protocol'] == 'regime-b' or target.exists() or not claim(c['name']):
            continue                   # regime B runs in its own process (--worker-b): its frozen trainer imports clash
        t0 = time.time()
        n = spec['meta']['cells']
        w = weight_vector(spec, c, n)
        specs = offset_specs(op, w, c['dose'], c['name'])
        res = dict(condition=c, conditions_sha256=sha, gpu=gpu, weights=dict(mean=float(w.mean()), min=float(w.min()),
                                                                               max=float(w.max()), distinct=len(specs)))
        if c['protocol'] == 'regime-b':
            res.update(regime_b(c, specs, op, OUT / 'meta' / c['name'], torch, n))
        else:
            if rig is None:
                rig = Rig(ROOT / 'runs' / 'linear-joint' / '20261002-p05-01' / 'model.pt', OUT / 'meta-rig' / f'gpu{gpu}')
                if rig.n != n:
                    raise RuntimeError('cell count differs')
                seq = rig.memory_input()
            res['activity'], res['loom'] = rig_checks(rig, specs, torch, SETTLE_MS, seq)
            timings = ra.TIMINGS if c['protocol'] == 'regime-a' else TRAINED
            res['max_decisions'] = ra.MAX_STEPS
            res['memory'] = memory_rows(mb, Frozen, op, ra, tasks, specs, timings, OUT / 'meta-policies' / c['name'], torch,
                                        c['protocol'] == 'regime-a')
        res['seconds'] = time.time() - t0
        tmp = target.with_suffix('.partial')
        tmp.write_text(json.dumps(res, default=str), encoding='utf-8', newline='\n')
        os.replace(tmp, target)
        print(f"{c['name']}: {res['seconds']:.0f} s", flush=True)
    print(f'worker {gpu}: no conditions left', flush=True)


def worker_b(gpu):
    """The regime B conditions alone, with phase3_rescue.py's imports only. The frozen trainer (memory.py) imports
    sokoban's evaluation module, which clashes in one process with the study code's evaluation module that the regime A
    rollouts load first (2026-10-22: every worker stopped at its first regime B condition)."""
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    import mechanism_base  # noqa: F401  (path setup: flybrain, sokoban, frozen study code)
    import torch
    import operators as op
    raw = CONDITIONS.read_bytes()
    spec = json.loads(raw)
    sha = hashlib.sha256(raw).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    for c in sorted((c for c in spec['conditions'] if c['protocol'] == 'regime-b'), key=lambda c: c['name']):
        target = OUT / f"{c['name']}.json"
        if target.exists() or not claim(c['name']):
            continue
        t0 = time.time()
        n = spec['meta']['cells']
        w = weight_vector(spec, c, n)
        specs = offset_specs(op, w, c['dose'], c['name'])
        res = dict(condition=c, conditions_sha256=sha, gpu=gpu, weights=dict(mean=float(w.mean()), min=float(w.min()),
                                                                               max=float(w.max()), distinct=len(specs)))
        res.update(regime_b(c, specs, op, OUT / 'meta' / c['name'], torch, n))
        res['seconds'] = time.time() - t0
        tmp = target.with_suffix('.partial')
        tmp.write_text(json.dumps(res, default=str), encoding='utf-8', newline='\n')
        os.replace(tmp, target)
        print(f"{c['name']}: {res['seconds']:.0f} s", flush=True)
    print(f'regime B worker {gpu}: no conditions left', flush=True)


def lock():
    """Task lock and amendment 0006 for the conditions file, before any T-SC4 run."""
    sys.path.insert(0, str(ROOT))
    from studykit.registry import Study
    study = Study.named('brain-pathology-v1')
    if any(a['reason'].startswith('T-SC4') for a in study.amendments()):
        raise SystemExit('the T-SC4 amendment exists')
    spec = json.loads(CONDITIONS.read_bytes())
    study.lock_tasks(status='draft', note='inputs written before their runs (plan v2 batches); re-locked for amendment '
                                          '0006 (T-SC4 Kv4-weighted C01 conditions)')
    path = study.amend('T-SC4 Kv4-weighted C01 conditions', dict(
        added_task_file=dict(file=f'tasks/{CONDITIONS.name}', sha256=hashlib.sha256(CONDITIONS.read_bytes()).hexdigest()),
        why='T-SC4 (declared in tasks/sc-panels-v1.json, amendment 0005): the per-cell Kv4 weights and the 39 conditions are '
            'written before any run.',
        conditions=[c['name'] for c in spec['conditions']], weight_stats=spec['meta']['weight_stats'],
        deviations_from_plan=dict(
            adfca_operator='the third operator (a cell-type Shal change measured in T-SC4b) is not built: in the AD Fly '
                           'Cell Atlas, pan-neuronal Shal mRNA is higher under Abeta42 (log2 fold change +0.13, '
                           'F14.SC.adfca.pan, Holm-significant) and rises in 45 of 58 neuron types, so the data give no '
                           'loss pattern; the C01 calibration rests on Kv4 current (protein) loss',
            check='the uniform identity check runs once (dose 0.5) through the weighted code path with w = 1',
            regime_b='evaluated under the uniform lesion (control) and the class-weighted lesion; restored is in '
                     'runs/rescue (F12.rescue)')))
    print(path.name)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--lock', action='store_true')
    ap.add_argument('--worker', type=int)
    ap.add_argument('--worker-b', type=int)
    a = ap.parse_args()
    if a.prepare:
        prepare()
    elif a.lock:
        lock()
    elif a.worker_b is not None:
        worker_b(a.worker_b)
    else:
        worker(a.worker)
