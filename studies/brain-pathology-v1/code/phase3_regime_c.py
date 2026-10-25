"""Regime C, lesion then retraining (GPU host): reserve of the learned memory under the C01 lesion.

    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/phase3_regime_c.py

Start: the healthy M0 readout-only X1 replicates 1-3 of the training campaign (studies/brain-supplement-v2/runs/
x1-M0-head_only-r<r>, their champion pair). The C01 lesion (threshold -3.22 mV x dose on every cell, the regime B lesion
files) is applied, then 100 updates of retraining with the frozen training step and optimiser (Adam, readout lr 3e-4,
brain lr 1e-4 when joint), in the replicate's own timing order (replicate_training.timing_iteration):
- readout-only: only the readout trains (the brain keeps the P05 parameters);
- joint: the brain's type-level parameters train as well.
Assessment: the frozen trainer's trained-timing assessment (18 histories, 512 decisions) after 0, 50 and 100 updates.
Recovery fraction = (solved after 100 updates - solved after 0) / (18 - solved after 0); undefined when the lesion
leaves all 18 solved. Jobs: 4 doses x 2 modes x 3 replicates; outputs runs/regime-c/<job>.json (resumable).
"""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
FROZEN = ROOT / 'studies' / 'brain-sokoban-v1'
MECH = FROZEN / 'mechanisms-v1'
OUT = STUDY / 'runs' / 'regime-c'
DOSES = (0.5, 1.0, 1.5, 2.0)
MODES = ('head_only', 'joint')
UPDATES = 100
ASSESS_AT = (0, 50, 100)


def lesion_specs(dose):
    import operators
    raw = json.loads((STUDY / 'runs' / 'regime-b-lesions' / f'C01-dose{dose:g}.json').read_text(encoding='utf-8'))
    return [operators.spec(s['kind'], s['cells'], s.get('value'), s.get('label')) for s in raw]


def summary(obs):
    return dict(solved=obs['solved'], total=obs['total'], probe_correct=obs['probe_correct'], probe_total=obs['probe_total'],
                min_choice_margin=obs['min_choice_margin'],
                rollouts=[dict(cue=r['cue'], delay_ms=r['delay_ms'], prefix_windows=r['prefix_windows'], status=r['status'], steps=r['steps'])
                          for r in obs['rollouts']])


def run_job(rep, dose, mode, target):
    import torch
    import memory as legacy
    import operators
    from replicate_training import timing_iteration
    start = SUPP / 'runs' / f'x1-M0-head_only-r{rep}'
    if not (start / 'verification.json').exists():
        raise RuntimeError(f'{start.name}: start replicate missing or not verified')
    base = json.loads((start / 'config.json').read_text(encoding='utf-8'))
    saved = torch.load(start / 'model.pt', map_location='cpu', weights_only=True)
    if saved['config'] != base:
        raise RuntimeError(f'{start.name}: model.pt was written with another config')
    config = dict(base, condition=mode, id=f'regime-c-C01-dose{dose:g}-{mode}-r{rep}')
    meta = OUT / 'meta' / target.stem
    meta.mkdir(parents=True, exist_ok=True)
    policy = legacy.MemoryPolicy(config, meta)
    policy.load(saved['pair'])
    audit = operators.apply(policy.encoder, lesion_specs(dose))
    task = next(t for t in legacy.prepare_memory() if t['id'] == config['task'])
    groups = [dict(params=policy.head.parameters(), lr=.0003)]
    if policy.train_core:
        groups.append(dict(params=policy.encoder.core.parameters(), lr=.0001))
    optimizer = torch.optim.Adam(groups)
    assessments, steps = {}, []
    t0 = time.time()
    for u in range(UPDATES + 1):
        if u in ASSESS_AT:
            assessments[str(u)] = summary(legacy.assess(policy, task))
        if u == UPDATES:
            break
        g = legacy.training_step(policy, task, optimizer, timing_iteration(u, rep))
        steps.append(dict(update=u + 1, loss=g['loss'], gradient_norm=g['gradient_norm'], delay_ms=g['delay_ms'], prefix_windows=g['prefix_windows']))
    s0, s1 = assessments['0']['solved'], assessments[str(UPDATES)]['solved']
    total = assessments['0']['total']
    rec = dict(job=target.stem, start=start.name, start_model_sha256=json.loads((start / 'result.json').read_text(encoding='utf-8'))['model_sha256'],
               replicate=rep, dose=dose, mode=mode, updates=UPDATES, audit=audit, trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad),
               assessments=assessments, steps=steps, seconds=time.time() - t0,
               recovery_fraction=(s1 - s0) / (total - s0) if s0 < total else None)
    del policy
    gc.collect()                   # operators.apply leaves a reference cycle (enc.bind); free the GPU memory now
    torch.cuda.empty_cache()
    return rec


def main():
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    sys.path.insert(0, str(SUPP / 'code'))
    import mechanism_base  # noqa: F401  (path setup: flybrain, sokoban, frozen study code)
    OUT.mkdir(parents=True, exist_ok=True)
    for rep in (1, 2, 3):
        for dose in DOSES:
            for mode in MODES:
                target = OUT / f'C01-dose{dose:g}-{mode}-r{rep}.json'
                if target.exists():
                    continue
                rec = run_job(rep, dose, mode, target)
                target.write_text(json.dumps(rec, default=str), encoding='utf-8', newline='\n')
                a = rec['assessments']
                print(f"{target.stem}: solved {a['0']['solved']} -> {a['50']['solved']} -> {a['100']['solved']} / {a['0']['total']}, "
                      f"recovery {rec['recovery_fraction']}, {rec['seconds']:.0f} s", flush=True)
    (OUT / 'done.json').write_text(json.dumps(dict(finished_unix=time.time())), encoding='utf-8')


if __name__ == '__main__':
    main()
