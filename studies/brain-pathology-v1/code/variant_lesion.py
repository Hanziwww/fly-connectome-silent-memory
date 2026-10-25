"""O14, task level (GPU host): the C01 lesion at dose 1 on the M0 readout-only models trained under each structural
variant (runs/o14-<variant>-r<r>, written by studies/brain-supplement-v2/code/replicate_training.py --variant).

    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/variant_lesion.py

Per run: the policy is rebuilt with its variant (variants.apply) and its trained pair, then assessed with the frozen
assessment (memory.assess, all 35 timings) twice: healthy (must reproduce the run's verification.json outcomes) and under
C01 at dose 1 (the regime B lesion file C01-dose1.json: threshold -3.22 mV on every cell, as regime A and C).
Outputs runs/o14-lesion/<run>.json; a finished run is skipped (resumable).
"""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
OUT = STUDY / 'runs' / 'o14-lesion'
DOSE = 1.0


def outcomes(rollouts):
    return {(r['cue'], r['delay_ms'], r['prefix_windows'], r.get('case')): (r['status'], r['steps']) for r in rollouts}


def main():
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    import mechanism_base  # noqa: F401  (path setup: flybrain, sokoban, frozen study code)
    import torch
    import memory as legacy
    import operators
    import variants
    from phase3_regime_c import lesion_specs
    from phase3_rescue import slim
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(p for p in (STUDY / 'runs').glob('o14-*-r*') if (p / 'verification.json').exists())
    for run in runs:
        target = OUT / f'{run.name}.json'
        if target.exists():
            continue
        t0 = time.time()
        config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        saved = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
        if saved['config'] != config:
            raise RuntimeError(f'{run.name}: model.pt was written with another config')
        task = next(t for t in legacy.prepare_memory() if t['id'] == config['task'])
        rec = dict(run=run.name, variant=config['variant'], replicate=config['replicate'], dose=DOSE)
        for label, specs in (('healthy', []), ('C01', lesion_specs(DOSE))):
            policy = legacy.MemoryPolicy(config, OUT / 'meta' / run.name)
            rec.setdefault('variant_audit', variants.apply(policy.encoder, config['variant']))
            policy.load(saved['pair'])
            audit = operators.apply(policy.encoder, specs)
            rec[label] = dict(audit=audit, result=slim(legacy.assess(policy, task, True)))
            del policy
            gc.collect()                   # operators.apply leaves a reference cycle (enc.bind); free the GPU memory now
            torch.cuda.empty_cache()
        ver = json.loads((run / 'verification.json').read_text(encoding='utf-8'))['results']['rollouts']
        rec['healthy_reproduces_verification'] = outcomes(ver) == outcomes(rec['healthy']['result']['rollouts'])
        rec['seconds'] = time.time() - t0
        target.write_text(json.dumps(rec, default=str), encoding='utf-8', newline='\n')
        solved = {k: sum(r['status'] == 'solved' for r in rec[k]['result']['rollouts']) for k in ('healthy', 'C01')}
        print(f"{run.name}: solved healthy {solved['healthy']}, C01 {solved['C01']}; reproduces verification "
              f"{rec['healthy_reproduces_verification']}; {rec['seconds']:.0f} s", flush=True)
    (OUT / 'done.json').write_text(json.dumps(dict(finished_unix=time.time(), runs=[p.name for p in runs])), encoding='utf-8')


if __name__ == '__main__':
    main()
