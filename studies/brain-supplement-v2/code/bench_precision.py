"""Phase 0b(c): speed and agreement of the frozen backend's precision paths on the GPU host.

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/bench_precision.py

For each precision ('mixed' = float64 forward [reference], 'float32_segmented' = fixed-order float32, 'float32' = cuSPARSE):
1. throughput (simulated ms per second) at batch 1, 2, 18, 64;
2. determinism: the M0 memory replay at the 9 trained timings run twice, bitwise;
3. agreement with 'mixed' (E3): actions, status and decision logits of every history (M0 readout-only and S2).
Writes studies/brain-supplement-v2/runs/g0-bench-precision/bench.json.
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'studies' / 'brain-supplement-v2' / 'runs' / 'g0-bench-precision'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
TRAINED = [(d, p) for d in (0, 40, 160) for p in (0, 1, 4)]
PRECISIONS = ('mixed', 'float32_segmented', 'float32')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(MECH / 'code'))
    import mechanism_base as mb
    import numpy as np
    import torch
    from backend import StudyBrain
    from mechanism_model import Frozen, memory_rollout
    sync = torch.cuda.synchronize
    rng = np.random.default_rng(0)
    res = dict(throughput={}, replay={})
    for precision in PRECISIONS:
        enc = StudyBrain(batch=1, precision=precision, metadata_dir=OUT / f'meta-{precision}')
        rows = []
        for batch in (1, 2, 18, 64):
            enc.reset_batch(batch)
            boards = torch.as_tensor(rng.integers(0, 2, size=(batch, 4, 10, 10)).astype(np.float32), device='cuda')
            with torch.no_grad():
                enc.advance(boards, 40)
                times = []
                for _ in range(3):
                    sync()
                    t = time.perf_counter()
                    for _ in range(5):
                        enc.advance(boards, 40)
                    sync()
                    times.append(time.perf_counter() - t)
            s = statistics.median(times)
            rows.append(dict(batch=batch, ms_per_second=200 / s, column_ms_per_second=200 * batch / s))
            print(f'{precision:18s} batch {batch:3d}: {200 / s:8.1f} sim-ms/s', flush=True)
        res['throughput'][precision] = rows
        del enc
        torch.cuda.empty_cache()
    tasks = json.loads((MECH / 'inputs' / 'memory-tasks.json').read_text(encoding='utf-8'))
    for mid in ('remote-dynamic-M0-head_only', 'supp-S2-context-brain'):
        base = mb.model_by_id(mid)
        task = next(x for x in tasks if x['id'] == base['task'])
        res['replay'][mid] = {}
        for precision in PRECISIONS:
            runs = []
            for rep in (0, 1):
                model = dict(base, precision=precision)
                policy = Frozen(model, OUT / f'policy-{mid}-{precision}-{rep}', batch=len(task['cases']))
                policy.load(MECH / base['checkpoints']['after']['path'])
                t = time.perf_counter()
                rows = []
                for d, p in TRAINED:
                    rows.extend(memory_rollout(policy, task, d, p))
                runs.append(dict(seconds=time.perf_counter() - t, rows=rows))
                del policy
                torch.cuda.empty_cache()
            same = all(a['decisions'] == b['decisions'] and a['status'] == b['status'] for a, b in zip(runs[0]['rows'], runs[1]['rows']))
            res['replay'][mid][precision] = dict(seconds=[r['seconds'] for r in runs], deterministic=same, rows=runs[0]['rows'])
            print(f'{mid} {precision}: deterministic={same}, {runs[0]["seconds"]:.1f} s', flush=True)
    (OUT / 'bench.json').write_text(json.dumps(res), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
