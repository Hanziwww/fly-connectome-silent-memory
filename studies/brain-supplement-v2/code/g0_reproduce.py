"""Gate G0 on the GPU host: the frozen study's recorded behaviour is reproduced by its own code path on this machine.

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/g0_reproduce.py [--models ID ...]

1. P05: the formal P05 checkpoint (runs/linear-joint/20261002-p05-01/model.pt) is rolled out on its level with the
   frozen study's static rollout, once with the "mixed" backend and once in float64.
2. Memory models: at the 9 trained timings (delay 0/40/160 ms x 0/1/4 prefix windows) every history of each memory model
   is replayed with memory_rollout, exactly as the free-after recordings were made (batch = number of cases).

Outputs go to studies/brain-supplement-v2/runs/g0-reproduce/ (p05.json, memory-<model>.json, environment.json and the
backend metadata the frozen code writes). g0_compare.py compares them with the local records after the pull; the records
themselves stay on the local machine. Nothing under studies/brain-sokoban-v1 is written.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-supplement-v2'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
P05 = ROOT / 'runs' / 'linear-joint' / '20261002-p05-01'
TRAINED = [(d, p) for d in (0, 40, 160) for p in (0, 1, 4)]
MODELS = ['remote-dynamic-M0-head_only', 'remote-dynamic-M0-joint', 'supp-M0-visibility', 'supp-S1-ring-brain',
          'supp-S2-context-brain']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', nargs='*', default=MODELS)
    ap.add_argument('--out', default=str(STUDY / 'runs' / 'g0-reproduce'))
    ap.add_argument('--skip-p05', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(MECH / 'code'))
    import mechanism_base as mb  # noqa: E402  (adds flybrain, sokoban and the study code to sys.path)
    import torch
    from mechanism_model import Frozen, memory_rollout, static_rollout

    def write(name, obj):
        (out / name).write_text(json.dumps(obj), encoding='utf-8', newline='\n')

    write('environment.json', dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                                   capability=list(torch.cuda.get_device_capability(0)), started_unix=time.time()))
    if not args.skip_p05:
        level = json.loads((P05 / 'level.json').read_text(encoding='utf-8'))
        result = {}
        for precision in ('mixed', 'float64'):
            model = dict(id='p05', precision=precision, config=dict(condition=dict(structure='real')))
            policy = Frozen(model, out / f'p05-{precision}', batch=1)
            policy.load(P05 / 'model.pt')
            t = time.perf_counter()
            r = static_rollout(policy, level)
            r['seconds'] = time.perf_counter() - t
            r['audit'] = policy.frozen_audit()
            result[precision] = r
            print(f'P05 {precision}: {r["status"]} in {r["steps"]} steps, {r["seconds"]:.1f} s', flush=True)
            del policy
            torch.cuda.empty_cache()
        write('p05.json', result)
    tasks = json.loads((MECH / 'inputs' / 'memory-tasks.json').read_text(encoding='utf-8'))
    for mid in args.models:
        model = mb.model_by_id(mid)
        task = next(x for x in tasks if x['id'] == model['task'])
        policy = Frozen(model, out / f'memory-{mid}', batch=len(task['cases']))
        policy.load(MECH / model['checkpoints']['after']['path'])
        rows, t = [], time.perf_counter()
        for d, p in TRAINED:
            rows.extend(memory_rollout(policy, task, d, p))
        seconds = time.perf_counter() - t
        write(f'memory-{mid}.json', dict(model=mid, task=model['task'], precision=model['precision'], seconds=seconds,
                                         audit=policy.frozen_audit(), rows=rows))
        solved = sum(r['status'] == 'solved' for r in rows)
        print(f'{mid}: {solved}/{len(rows)} trained histories solved, {seconds:.1f} s', flush=True)
        del policy
        torch.cuda.empty_cache()
    write('done.json', dict(finished_unix=time.time()))


if __name__ == '__main__':
    main()
