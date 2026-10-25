"""X3 (plan v2, Phase 1C): M0 readout-only memory training on jittered timings vs the fixed 9 trained timings. GPU host.

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/x3_jittered_training.py --replicate R --out DIR [--verify]

The frozen memory worker (studies/brain-sokoban-v1/code/memory.py) runs unchanged, as in replicate_training.py; only
the timing of each training update differs. Update u of replicate r draws its delay uniformly from 0, 20, ..., 320 ms
and its prefix uniformly from 0, 1, 2, 4, 6, 8 windows (the lengths the task defines) with a generator seeded by
(3000 + r, u). The frozen training step is called unchanged while the protocol it reads for the timing (memory.read)
returns that single delay and prefix; everything else, including the stopping rule (all 18 histories at the 9 standard
trained timings solved) and the assessments, reads the frozen protocol. The readout starts as in X1 replicate r
(jitter 0.1 SD, seed 2000 + r; none for r = 0), so X3 replicate r and X1 replicate r share their initial state.
The verification (separate process, --verify) evaluates all 35 timings as for X1.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / 'studies' / 'brain-sokoban-v1'
MECH = FROZEN / 'mechanisms-v1'
DELAYS = tuple(range(0, 321, 20))
PREFIXES = (0, 1, 2, 4, 6, 8)
MODEL = 'remote-dynamic-M0-head_only'


def timing(update, replicate):
    g = np.random.default_rng([3000 + replicate, update])
    return int(DELAYS[g.integers(len(DELAYS))]), int(PREFIXES[g.integers(len(PREFIXES))])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replicate', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((MECH / 'inputs' / 'models' / MODEL / 'source-config.json').read_text(encoding='utf-8'))
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(ROOT / 'studies' / 'brain-supplement-v2' / 'code'))
    import mechanism_base  # noqa: F401  (path setup)
    import memory as legacy
    from replicate_training import jitter_head
    applied, drawn = {}, []
    frozen_read, frozen_step = legacy.read, legacy.training_step

    class X3Policy(legacy.MemoryPolicy):
        def __init__(self, config, out_dir):
            super().__init__(config, out_dir)
            applied['jitter'] = jitter_head(self.head, args.replicate)

    def x3_step(policy, task, optimizer, iteration):
        delay, prefix = timing(iteration, args.replicate)
        protocol = frozen_read(FROZEN / 'protocol.json')

        def read(path):
            if Path(path).resolve() == (FROZEN / 'protocol.json').resolve():
                return dict(protocol, trained_delay_ms=[delay], trained_prefix_windows=[prefix])
            return frozen_read(path)

        legacy.read = read
        try:
            result = frozen_step(policy, task, optimizer, iteration)
        finally:
            legacy.read = frozen_read
        if (result['delay_ms'], result['prefix_windows']) != (delay, prefix):
            raise RuntimeError('the frozen step did not use the drawn timing')
        drawn.append((delay, prefix))
        return result

    legacy.MemoryPolicy = X3Policy
    legacy.training_step = x3_step
    config = dict(base, id=f'{MODEL}-x3-r{args.replicate}', replicate=args.replicate,
                  replicate_definition='X3: per-update timing drawn from delays 0-320 ms (step 20) x prefixes 0,1,2,4,6,8 (seed 3000+r, u); '
                                       'readout jitter 0.1 SD (seed 2000+r; none for r = 0)')
    config_path = out / 'config.json'
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=1), encoding='utf-8', newline='\n')
    code = legacy.worker(str(config_path), args.verify)
    (out / ('replicate-verify.json' if args.verify else 'replicate.json')).write_text(
        json.dumps(dict(model=MODEL, replicate=args.replicate, exit_code=code, applied=applied,
                        timings_drawn=len(drawn), distinct_timings=len(set(drawn))), indent=1, default=str), encoding='utf-8', newline='\n')
    return code


if __name__ == '__main__':
    sys.exit(main() or 0)
