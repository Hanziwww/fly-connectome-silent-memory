"""Late rescue (GPU host): Kv4 restoration modelled as removing the C01 threshold offset from the M0 models that learned
under the lesion (regime B, the training campaign), then the frozen trainer's full assessment (35 timings, 512
decisions) as in the runs' own verification.

    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/phase3_rescue.py

In Abeta42 flies, transgenic Kv4 restoration attenuated learning and locomotor deficits (Ping 2015,
doi:10.1371/journal.pgen.1005025) and partially rescued short-term memory (Feng 2017,
doi:10.1016/j.neuroscience.2017.06.008). Restoration from the start is the healthy model by construction (the operator is
the identity at zero dose), so the model's question is the late one: does a policy that learned with hyperexcitable
neurons keep its memory when excitability is restored?
Each regime B run folder (studies/brain-pathology-v1/runs/b-C01-dose<d>-M0-<learner>-r<r>) is loaded as its own verify
step loads it (MemoryPolicy from the run's config.json, the champion pair from model.pt), without the lesion. Control:
for one run per learner (dose 0.5, replicate 1) the lesion is re-applied and the assessment must reproduce the run's
verification.json outcomes exactly. Outputs runs/rescue/<run>.json (and <run>-lesion-control.json); resumable.
"""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
FROZEN = ROOT / 'studies' / 'brain-sokoban-v1'
MECH = FROZEN / 'mechanisms-v1'
OUT = STUDY / 'runs' / 'rescue'
DOSES = (0.5, 1.0, 1.5, 2.0)
LEARNERS = ('M0-head_only', 'M0-joint')


def outcome(rows):
    return [(r['cue'], r['delay_ms'], r['prefix_windows'], r['status']) for r in rows]


def lesion_specs(run, config):
    """The run's lesion from the campaign's lesion file (campaign_jobs.py), checked against the summary in config.json."""
    import operators
    dose = run.name.split('-dose')[1].split('-')[0]
    raw = json.loads((STUDY / 'runs' / 'regime-b-lesions' / f'C01-dose{dose}.json').read_text(encoding='utf-8'))
    specs = [operators.spec(s['kind'], s['cells'], s.get('value'), s.get('label')) for s in raw]
    summary = [dict(kind=s['kind'], cells=int(len(s['cells'])), value=s['value'], label=s['label']) for s in specs]
    if summary != config['lesion']:
        raise RuntimeError(f'{run.name}: lesion file disagrees with the run config')
    return specs


def slim(result):
    """The assessment without per-decision features (the verification files keep them; outcomes are what is compared)."""
    rolls = [dict({k: v for k, v in r.items() if k != 'decisions'}, first_logits=r['decisions'][0]['logits'] if r['decisions'] else None)
             for r in result['rollouts']]
    probes = [dict(p, records=[{k: v for k, v in x.items() if k != 'features'} for x in p['records']]) for p in result['probes']]
    return dict(result, rollouts=rolls, probes=probes)


def evaluate(run, lesion, meta):
    import torch
    import memory as legacy
    import operators
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    meta.mkdir(parents=True, exist_ok=True)
    policy = legacy.MemoryPolicy(config, meta)
    saved = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
    if saved['config'] != config:
        raise RuntimeError(f'{run.name}: model.pt was written with another config')
    policy.load(saved['pair'])
    specs = lesion_specs(run, config) if lesion else []
    audit = operators.apply(policy.encoder, specs)
    task = next(t for t in legacy.prepare_memory() if t['id'] == config['task'])
    result = slim(legacy.assess(policy, task, True))
    del policy
    gc.collect()                   # operators.apply leaves a reference cycle (enc.bind); free the GPU memory now
    torch.cuda.empty_cache()
    return config, audit, result


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--learners', nargs='*', default=list(LEARNERS), choices=LEARNERS,
                    help='learners whose regime B runs are on this host (the two hosts each rescue their own)')
    learners = ap.parse_args().learners
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    import mechanism_base  # noqa: F401  (path setup: flybrain, sokoban, frozen study code)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(STUDY / 'runs' / f'b-C01-dose{d:g}-{learner}-r{r}', False) for learner in learners for d in DOSES for r in (1, 2, 3)]
    jobs += [(STUDY / 'runs' / f'b-C01-dose0.5-{learner}-r1', True) for learner in learners]
    for run, lesion in jobs:
        target = OUT / f'{run.name}{"-lesion-control" if lesion else ""}.json'
        if target.exists():
            continue
        if not (run / 'verification.json').exists():
            raise RuntimeError(f'{run.name}: regime B run missing or not verified')
        t0 = time.time()
        config, audit, result = evaluate(run, lesion, OUT / 'meta' / target.stem)
        ver = json.loads((run / 'verification.json').read_text(encoding='utf-8'))
        rec = dict(run=run.name, lesion_applied=lesion, config_id=config['id'], audit=audit, results=result, seconds=time.time() - t0,
                   as_trained=dict(solved=ver['results']['solved'], total=ver['results']['total']))
        if lesion:
            rec['reproduces_verification'] = outcome(result['rollouts']) == outcome(ver['results']['rollouts'])
            if not rec['reproduces_verification']:
                target.write_text(json.dumps(rec, default=str), encoding='utf-8', newline='\n')
                raise RuntimeError(f'{run.name}: re-lesioned assessment differs from the run verification')
        target.write_text(json.dumps(rec, default=str), encoding='utf-8', newline='\n')
        print(f'{target.stem}: solved {result["solved"]}/{result["total"]} (as trained {ver["results"]["solved"]}), {rec["seconds"]:.0f} s', flush=True)
    (OUT / ('done.json' if set(learners) == set(LEARNERS) else f"done-{'-'.join(sorted(learners))}.json")).write_text(
        json.dumps(dict(finished_unix=time.time(), learners=learners)), encoding='utf-8')


if __name__ == '__main__':
    main()
