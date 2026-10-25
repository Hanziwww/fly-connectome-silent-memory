"""X8 (plan v2, Phase 1C): Sokoban training from a neutral brain, readout-only vs joint. GPU host only.

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/x8_neutral_training.py --map D1-T0 --arm head_only
        --replicate R --out DIR [--verify]

The frozen Sokoban worker (studies/brain-sokoban-v1/code/execution.py run_worker -> training.fit) runs unchanged. The
configuration follows the frozen control arms (mechanisms-v1/inputs/models/control-D1-head_only/source-config.json):
batch 16, "mixed" precision, segments 1200/2400/4800 s, the P05 source for the provenance check. Arms:
- joint: the frozen 'neutral' condition (neutral brain and readout initialisation, brain and readout trained);
- head_only: the same with the brain fixed (train_core False), a condition the frozen study did not run.
A replicate r changes only the readout initialisation: every readout parameter + N(0, (0.1 x its SD)^2), seed 2000 + r
(replicate_training.jitter_head), applied by wrapping training.fit, i.e. after the frozen neutral readout initialisation
and after initial.pt is written (initial.pt holds the unjittered readout). r = 0 is the frozen protocol; for the joint
arm it repeats the frozen control-D<t>-neutral runs (D1, D3, D5; the D7 run was stopped before it finished).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / 'studies' / 'brain-sokoban-v1'
MECH = FROZEN / 'mechanisms-v1'
TEMPLATE = MECH / 'inputs' / 'models' / 'control-D1-head_only' / 'source-config.json'
P05 = 'runs/linear-joint/20261002-p05-01/model.pt'
MAPS = ('D1-T0', 'D3-T0', 'D5-T0', 'D7-T0')


def make_config(task, arm, replicate):
    sys.path.insert(0, str(FROZEN / 'code'))
    from registry import condition
    t = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    cond = condition('neutral')
    if arm == 'head_only':
        cond = dict(cond, name='neutral_head_only', train_core=False)
    elif arm != 'joint':
        raise ValueError(arm)
    return dict(id=f'x8-{task}-{arm}-r{replicate}', task=task, condition=cond, module='controls', parent=None, seed=10, source=P05,
                source_sha256=t['source_sha256'], precision=t['precision'], batch=t['batch'], segments_seconds=t['segments_seconds'],
                code_version=t['code_version'], task_lock_sha256=t['task_lock_sha256'], protocol_sha256=t['protocol_sha256'],
                replicate=replicate, replicate_definition='readout jitter 0.1 SD (seed 2000+r) after the frozen neutral initialisation; r = 0 is the frozen protocol')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map', required=True, choices=MAPS)
    ap.add_argument('--arm', required=True, choices=('head_only', 'joint'))
    ap.add_argument('--replicate', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = make_config(args.map, args.arm, args.replicate)
    config_path = out / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise RuntimeError('config.json differs from this job')
    else:
        config_path.write_text(json.dumps(config, indent=1), encoding='utf-8', newline='\n')
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(ROOT / 'studies' / 'brain-supplement-v2' / 'code'))
    import mechanism_base  # noqa: F401  (path setup: flybrain, sokoban, frozen study code)
    import execution
    import training
    from replicate_training import jitter_head
    applied = {}
    frozen_fit = training.fit

    def fit(encoder, head, level, config, out, check):
        applied['jitter'] = jitter_head(head, args.replicate)
        return frozen_fit(encoder, head, level, config, out, check)

    training.fit = fit                                     # run_worker imports training.fit at call time
    code = execution.run_worker(str(config_path), args.verify)
    (out / ('replicate-verify.json' if args.verify else 'replicate.json')).write_text(
        json.dumps(dict(map=args.map, arm=args.arm, replicate=args.replicate, exit_code=code, applied=applied), indent=1, default=str),
        encoding='utf-8', newline='\n')
    return code


if __name__ == '__main__':
    sys.exit(main() or 0)
