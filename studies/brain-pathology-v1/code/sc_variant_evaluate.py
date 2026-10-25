"""T-SC6 evaluation of one model trained on a variant or null brain (GPU host; tasks/sc-variant-design.json).

    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/sc_variant_evaluate.py --run RUN_FOLDER --lesion LESION.json

The model is loaded as its verify step loads it (M0: memory.MemoryPolicy; S1: the frozen supplement's Policy and task
library), the brain's adaptation lesion applied (checked against the run config), then the frozen trainer's assessments:
- as-trained, all 35 timings; it must reproduce the run's verification.json outcomes exactly;
- mechanism: adaptation cleared, and the state swapped between the opposite cues, at decision onset at the prefix-0
  trained timings (the frozen interventions 'adaptation' and 'swap'; collection switched on for the evaluation);
- M0 only: the C01 direction of dy2_evaluate.SHIFTS_MV (uniform threshold offsets through dy2_perturbation) on top of
  the variant, trained timings.
Output: runs/sc-variant-eval/<run>.json (rollout and probe outcomes per condition, without features).
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
FROZEN = ROOT / 'studies' / 'brain-sokoban-v1'
MECH = FROZEN / 'mechanisms-v1'
OUT = STUDY / 'runs' / 'sc-variant-eval'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--lesion', required=True)
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
    from dy2_evaluate import MECHANISM, SHIFTS_MV, lesion_specs, outcome

    def slim(res):
        """Rollout outcomes (case, cue, context, timing, status, steps, first action and logits) and probe correctness."""
        rolls = [dict(case=r.get('case'), cue=r['cue'], context=r.get('context'), visible=r.get('visible'),
                      delay_ms=r['delay_ms'], prefix_windows=r['prefix_windows'], status=r['status'], steps=r.get('steps'),
                      first_action=r['decisions'][0]['action'] if r.get('decisions') else None,
                      first_logits=r['decisions'][0]['logits'] if r.get('decisions') else None) for r in res['rollouts']]
        probes = [dict(delay_ms=p.get('delay_ms'), prefix_windows=p.get('prefix_windows'), correct=p.get('correct'))
                  for p in res.get('probes', [])]
        return dict(rollouts=rolls, probes=probes, solved=sum(r['status'] == 'solved' for r in rolls), total=len(rolls))
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    ver = json.loads((run / 'verification.json').read_text(encoding='utf-8'))
    if ver.get('status') != 'verified':
        raise RuntimeError(f'{run.name}: verification status {ver.get("status")}')
    supplement = 'S1' in config['task'] or 'S2' in config['task'] or config['id'].startswith('supp-')
    if supplement:
        sys.path.insert(0, str(FROZEN / 'remote-supplement-20261003'))
        import supp_experiments as trainer
        policy = trainer.Policy(config, OUT / 'meta' / run.name)
        task = next(t for t in trainer.tasklib.prepare() if t['id'] == config['task'])
        assess, batched, delays = trainer.assess, trainer.batched_assessment, list(trainer.PROTOCOL['trained_delay_ms'])
    else:
        policy = legacy.MemoryPolicy(config, OUT / 'meta' / run.name)
        task = next(t for t in legacy.prepare_memory() if t['id'] == config['task'])
        assess, batched, delays = legacy.assess, legacy.batched_assessment, list(MECHANISM['delays'])
    saved = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
    if saved['config'] != config:
        raise RuntimeError(f'{run.name}: model.pt was written with another config')
    policy.load(saved['pair'])
    specs = lesion_specs(args.lesion, config, operators)
    audit = operators.apply(policy.encoder, specs)
    rec = dict(run=run.name, config_id=config['id'], family='S1' if supplement else 'M0', lesion=args.lesion, lesion_audit=audit,
               model_sha256=ver.get('model_sha256'), conditions={}, seconds={})
    t_all = time.time()

    def put(name, fn):
        t0 = time.time()
        rec['conditions'][name] = slim(fn())
        rec['seconds'][name] = time.time() - t0

    put('as-trained', lambda: assess(policy, task, True))
    rec['reproduces_verification'] = outcome(rec['conditions']['as-trained']['rollouts']) == outcome(ver['results']['rollouts'])
    if not rec['reproduces_verification']:
        target.with_suffix('.failed.json').write_text(json.dumps(rec, default=str), encoding='utf-8', newline='\n')
        raise RuntimeError(f'{run.name}: the as-trained assessment differs from the run verification')
    policy.config['collect_interventions'] = True                     # evaluation only; training is over
    for kind in MECHANISM['interventions']:
        put(f'mechanism-{kind}', lambda kind=kind: dict(zip(('probes', 'rollouts'),
                                                            batched(policy, task, delays, list(MECHANISM['prefixes']), intervention=kind))))
    if not supplement:
        import dy2_perturbation as dy2
        pert = dy2.Perturbation(policy.encoder)                       # wraps the bind; the adaptation lesion is not bind-time
        for s in SHIFTS_MV:
            def shifted(s=s):
                pert.set_draw(dy2.draw_for({}, s, f'sc-variant|shift{s:+g}'))
                try:
                    return assess(policy, task, False)
                finally:
                    pert.set_off()
            put(f'shift{s:+g}', shifted)
    rec['total_seconds'] = time.time() - t_all
    target.write_text(json.dumps(rec, default=str), encoding='utf-8', newline='\n')
    print(f"{run.name}: {len(rec['conditions'])} conditions, {rec['total_seconds']:.0f} s, as-trained solved "
          f"{rec['conditions']['as-trained']['solved']}/{rec['conditions']['as-trained']['total']}", flush=True)
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    sys.exit(main())
