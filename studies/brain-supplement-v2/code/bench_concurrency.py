"""Phase 0b(e): throughput of concurrent worker processes on one GPU. Each process runs the frozen study's closed-loop
M0 assessment (18 histories, batch 18) K times and writes its timings; run N copies at once to measure scaling.

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/bench_concurrency.py --tag N-i [--k 4]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'studies' / 'brain-supplement-v2' / 'runs' / 'g0-bench-concurrency'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--k', type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1' / 'code'))
    import mechanism_base  # noqa: F401
    import torch
    import memory as mem
    from memory_tasks import prepare_memory
    task = next(t for t in prepare_memory() if t['id'] == 'M0')
    policy = mem.MemoryPolicy(dict(id='bench', task='M0', condition='head_only', precision='mixed'), OUT / f'policy-{args.tag}')
    mem.assess(policy, task)                       # warm-up
    times, results = [], []
    for _ in range(args.k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        obs = mem.assess(policy, task)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        results.append(obs['solved'])
    (OUT / f'{args.tag}.json').write_text(json.dumps(dict(tag=args.tag, seconds=times, solved=results,
                                                          peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)),
                                          encoding='utf-8')


if __name__ == '__main__':
    main()
