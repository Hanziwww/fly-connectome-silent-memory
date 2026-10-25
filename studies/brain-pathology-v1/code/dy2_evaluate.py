"""T-DY2 evaluation of one trained M0 readout-only model (GPU host; one queue job per model; F14.DY.robust, declared with
tasks/dy2-design.json before any T-DY2 run).

    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/dy2_evaluate.py --run RUN_FOLDER [--lesion SPEC.json]

The model is loaded as its own verify step loads it (MemoryPolicy from the run's config.json, the champion pair from
model.pt); every assessment is the frozen trainer's (memory.assess: 512 decisions; 'trained' = the 9 trained timings,
'full' = all 35), under the perturbations of dy2_perturbation.py, which leave the brain frozen when off.
Healthy-trained models (the T-DY2 runs and the X1 replicates):
- as-trained, full; it must reproduce the run's verification.json outcomes exactly;
- C01 direction: uniform threshold offsets SHIFTS_MV on every cell (the C01 operator's arithmetic), trained timings;
- random removal: the 15 cell sets of the dose series (tasks/phase4-dose-low-conditions.json shares 0.0002, 0.0005 and
  0.001, tasks/phase4-dose-conditions.json shares 0.002 and 0.005; 3 draws each), trained timings;
- perturbation panel: the two held-out draws of every T-DY2 condition (dy2_perturbation.heldout_draws) at the trained
  timings, and of the model's own training condition at all 35 (memory use and the lock under its own perturbation);
- mechanism: at the prefix-0 trained timings (delays 0, 40, 160), adaptation cleared, and the state swapped between the
  opposite cues, at decision onset (the frozen interventions 'adaptation' and 'swap' of memory.batched_assessment).
C01-trained models (--lesion, the lesion the run trained with, checked against its config): as-trained (full, must
reproduce the verification), the mechanism, the two held-out draws of th0.5 on top of the lesion, then restored (the
lesion removed; full), as in phase3_rescue.py.
Output: runs/dy2-eval/<run>.json (rollout and probe outcomes per condition, without features).
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
OUT = STUDY / 'runs' / 'dy2-eval'
SHIFTS_MV = (-3e-5, -1e-4, -3e-4, -1e-3, -3e-3, -1e-2, -3e-2, -0.1, -0.322, -1.61, -3.22, 1e-3, 0.1)
REMOVAL = ((STUDY / 'tasks' / 'phase4-dose-low-conditions.json', (0.0002, 0.0005, 0.001)),
           (STUDY / 'tasks' / 'phase4-dose-conditions.json', (0.002, 0.005)))
MECHANISM = dict(delays=(0, 40, 160), prefixes=(0,), interventions=('adaptation', 'swap'))


def outcome(rows):
    return [(r['cue'], r['delay_ms'], r['prefix_windows'], r['status']) for r in rows]


def slim(res):
    rolls = [dict(cue=r['cue'], delay_ms=r['delay_ms'], prefix_windows=r['prefix_windows'], status=r['status'], steps=r['steps'],
                  first_action=r['decisions'][0]['action'] if r['decisions'] else None,
                  first_logits=r['decisions'][0]['logits'] if r['decisions'] else None) for r in res['rollouts']]
    probes = [dict(delay_ms=p['delay_ms'], prefix_windows=p['prefix_windows'], correct=p['correct'],
                   records=[dict(cue=x['cue'], choice=x['choice'], correct=x['correct'], logits=x['logits']) for x in p['records']])
              for p in res['probes']]
    out = dict(rollouts=rolls, probes=probes, solved=sum(r['status'] == 'solved' for r in rolls), total=len(rolls))
    if 'min_choice_margin' in res:
        out['min_choice_margin'] = res['min_choice_margin']
    return out


def removal_sets():
    sets = []
    for path, shares in REMOVAL:
        raw = path.read_bytes()
        for c in json.loads(raw)['conditions']:
            if c['kind'] == 'remove' and c['dose'] in shares:
                sets.append(dict(name=c['name'], share=c['dose'], draw=c['draw'], cells=c['cells'],
                                 source=path.name, source_sha256=hashlib.sha256(raw).hexdigest()))
    if len(sets) != 15:
        raise RuntimeError(f'expected 15 removal sets, found {len(sets)}')
    return sets


def lesion_specs(path, config, operators):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    specs = [operators.spec(s['kind'], s['cells'], s.get('value'), s.get('label')) for s in raw]
    summary = [dict(kind=s['kind'], cells=int(len(s['cells'])), value=s['value'], label=s['label']) for s in specs]
    if summary != config['lesion']:
        raise RuntimeError('the lesion file disagrees with the run config')
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--lesion', default=None)
    args = ap.parse_args()
    run = (ROOT / args.run).resolve()
    target = OUT / f'{run.name}.json'
    if target.exists():
        return 0
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    import mechanism_base  # noqa: F401  (path setup: flybrain, sokoban, frozen study code)
    import torch
    import memory as legacy
    import operators
    import dy2_perturbation as dy2
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    ver = json.loads((run / 'verification.json').read_text(encoding='utf-8'))
    if ver.get('status') != 'verified':
        raise RuntimeError(f'{run.name}: verification status {ver.get("status")}')
    if config.get('variant') or config['condition'] != 'head_only':
        raise RuntimeError(f'{run.name}: not an M0 readout-only run on the default structure')
    if bool(config['lesion']) != bool(args.lesion):
        raise RuntimeError(f'{run.name}: --lesion must be given exactly for runs trained under a lesion')
    policy = legacy.MemoryPolicy(config, OUT / 'meta' / run.name)
    saved = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
    if saved['config'] != config:
        raise RuntimeError(f'{run.name}: model.pt was written with another config')
    policy.load(saved['pair'])
    specs = lesion_specs(args.lesion, config, operators) if args.lesion else []
    audit = operators.apply(policy.encoder, specs)
    pert = dy2.Perturbation(policy.encoder)             # last: wraps the lesion's bind
    b = policy.encoder.brain
    active = b.active.clone()
    task = next(t for t in legacy.prepare_memory() if t['id'] == config['task'])
    rec = dict(run=run.name, config_id=config['id'], training_perturbation=(config.get('training_perturbation') or {}).get('name'),
               lesion_audit=audit, model_sha256=ver.get('model_sha256'), conditions={}, seconds={})
    t_all = time.time()

    def put(name, fn):
        t0 = time.time()
        rec['conditions'][name] = slim(fn())
        rec['seconds'][name] = time.time() - t0

    def under(draw, full):
        def fn():
            pert.set_draw(draw)
            try:
                return legacy.assess(policy, task, full)
            finally:
                pert.set_off()
        return fn

    def mechanism(kind):
        def fn():
            probes, rolls = legacy.batched_assessment(policy, task, list(MECHANISM['delays']), list(MECHANISM['prefixes']), intervention=kind)
            return dict(probes=probes, rollouts=rolls)
        return fn

    pert.set_off()
    put('as-trained', lambda: legacy.assess(policy, task, True))
    rec['reproduces_verification'] = outcome(rec['conditions']['as-trained']['rollouts']) == outcome(ver['results']['rollouts'])
    if not rec['reproduces_verification']:
        target.with_suffix('.failed.json').write_text(json.dumps(rec, default=str), encoding='utf-8', newline='\n')
        raise RuntimeError(f'{run.name}: the as-trained assessment differs from the run verification')
    for kind in MECHANISM['interventions']:
        put(f'mechanism-{kind}', mechanism(kind))
    if specs:
        for k, draw in enumerate(dy2.heldout_draws('th0.5')):
            put(f'th0.5-s{k}', under(draw, True))
        operators.apply(policy.encoder, [])                  # restoration: the lesion removed (and the patches with it)
        put('restored', lambda: legacy.assess(policy, task, True))
    else:
        for s in SHIFTS_MV:
            put(f'shift{s:+g}', under(dy2.draw_for({}, s, f'dy2-eval|shift{s:+g}'), False))
        rec['removal_sets'] = []
        for r in removal_sets():
            idx = torch.as_tensor(r['cells'], device=b.device, dtype=torch.long)

            def removed():
                b.active[idx] = 0.0
                try:
                    return legacy.assess(policy, task, False)
                finally:
                    b.active.copy_(active)
            put(f"remove-{r['name']}", removed)
            rec['removal_sets'].append({k: v for k, v in r.items() if k != 'cells'} | dict(cells=len(r['cells'])))
        own = rec['training_perturbation']
        for name in dy2.CONDITIONS:
            for k, draw in enumerate(dy2.heldout_draws(name)):
                put(f'{name}-s{k}', under(draw, name == own))
    rec['total_seconds'] = time.time() - t_all
    target.write_text(json.dumps(rec, default=str), encoding='utf-8', newline='\n')
    print(f"{run.name}: {len(rec['conditions'])} conditions, {rec['total_seconds']:.0f} s, as-trained solved "
          f"{rec['conditions']['as-trained']['solved']}/{rec['conditions']['as-trained']['total']}", flush=True)
    del policy, pert
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    sys.exit(main())
