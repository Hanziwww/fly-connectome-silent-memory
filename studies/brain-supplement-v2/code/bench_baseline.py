"""Phase 0b baseline benchmark on the GPU host (current code, no optimisation): where the time goes.

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/bench_baseline.py

1. Inference throughput of the study brain ("mixed" backend): simulated ms per second at batch 1...128 (200 ms per
   measurement after a 40 ms warm-up; 3 repeats).
2. Memory training costs with the frozen study's own MemoryPolicy and training_step on the M0 task:
   - readout-only (head_only): 9 cache-filling updates, then 20 cached updates, and one closed-loop assessment;
   - joint (brain and readout): 6 updates and one assessment.
Writes studies/brain-supplement-v2/runs/g0-bench-baseline/bench.json. Nothing of the frozen study is written: the
policies write their metadata into the run folder.
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'studies' / 'brain-supplement-v2' / 'runs' / 'g0-bench-baseline'
FROZEN = ROOT / 'studies' / 'brain-sokoban-v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(FROZEN / 'mechanisms-v1' / 'code'))
    import mechanism_base  # noqa: F401  (path setup: flybrain, sokoban, study code)
    import numpy as np
    import torch
    from backend import StudyBrain
    import memory as mem
    from memory_tasks import prepare_memory
    sync = torch.cuda.synchronize
    result = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, inference=[], training={})
    rng = np.random.default_rng(0)
    enc = StudyBrain(batch=1, precision='mixed', metadata_dir=OUT / 'inference')
    for batch in (1, 2, 4, 8, 16, 32, 64, 128):
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
        sec = statistics.median(times)
        result['inference'].append(dict(batch=batch, simulated_ms=200, seconds=sec, ms_per_second=200 / sec,
                                        column_ms_per_second=200 * batch / sec,
                                        peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30))
        print(f'batch {batch:4d}: {200 / sec:8.1f} sim-ms/s, {200 * batch / sec:9.1f} column-ms/s', flush=True)
    del enc
    torch.cuda.empty_cache()
    task = next(t for t in prepare_memory() if t['id'] == 'M0')
    for kind, updates in (('head_only', 29), ('joint', 6)):
        config = dict(id=f'bench-{kind}', task='M0', condition=kind, precision='mixed')
        policy = mem.MemoryPolicy(config, OUT / f'train-{kind}')
        groups = [dict(params=policy.head.parameters(), lr=.0003)]
        if policy.train_core:
            groups.append(dict(params=policy.encoder.core.parameters(), lr=.0001))
        opt = torch.optim.Adam(groups)
        steps = []
        for i in range(updates):
            sync()
            t = time.perf_counter()
            mem.training_step(policy, task, opt, i)
            sync()
            steps.append(time.perf_counter() - t)
        sync()
        t = time.perf_counter()
        obs = mem.assess(policy, task)
        sync()
        a = time.perf_counter() - t
        result['training'][kind] = dict(update_seconds=steps, assessment_seconds=a, assessment_solved=obs['solved'],
                                        assessment_total=obs['total'])
        print(f'{kind}: first updates {steps[:3]}, median later {statistics.median(steps[-5:]):.3f} s; assessment {a:.1f} s',
              flush=True)
        del policy, opt
        torch.cuda.empty_cache()
    (OUT / 'bench.json').write_text(json.dumps(result, indent=1), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
