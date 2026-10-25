"""Replicated Sokoban training with the frozen worker (plan v2, Phase 1C: X9 G01 D7 controls, X10 Fig. 2 replicates).
GPU host only.

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/sokoban_replicate.py --map D7-T0 --condition brain_only
        --replicate R --out DIR [--module controls|difficulty] [--verify]

The frozen Sokoban worker (studies/brain-sokoban-v1/code/execution.py run_worker -> training.fit) runs unchanged with a
frozen condition (registry.condition: joint, frozen, brain_only, head_only, alternating, no_recovery, entry_ce,
demonstration, neutral, rewired, weight_shuffle, board_mlp) and the frozen control settings (batch 16, "mixed",
segments 1200/2400/4800 s; mechanisms-v1/inputs/models/control-D1-head_only/source-config.json). As in X8, a replicate
r > 0 differs only by the readout jitter of X1 (0.1 SD, seed 2000 + r) applied by wrapping training.fit; r = 0 is the
frozen protocol (for X9 G01 the never-run registered D7 controls; for X10 replicates 1-5 of the Fig. 2 maps).
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


def make_config(task, name, replicate, module):
    sys.path.insert(0, str(FROZEN / 'code'))
    from registry import condition
    t = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    return dict(id=f'{module}-{task}-{name}-r{replicate}', task=task, condition=condition(name), module=module, parent=None, seed=10,
                source=P05, source_sha256=t['source_sha256'], precision=t['precision'], batch=t['batch'],
                segments_seconds=t['segments_seconds'], code_version=t['code_version'], task_lock_sha256=t['task_lock_sha256'],
                protocol_sha256=t['protocol_sha256'], replicate=replicate,
                replicate_definition='readout jitter 0.1 SD (seed 2000+r) applied at the start of training.fit; r = 0 is the frozen protocol')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map', required=True)
    ap.add_argument('--condition', required=True)
    ap.add_argument('--replicate', type=int, required=True)
    ap.add_argument('--module', default='controls', choices=('controls', 'difficulty'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = make_config(args.map, args.condition, args.replicate, args.module)
    config_path = out / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise RuntimeError('config.json differs from this job')
    else:
        config_path.write_text(json.dumps(config, indent=1), encoding='utf-8', newline='\n')
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(ROOT / 'studies' / 'brain-supplement-v2' / 'code'))
    import mechanism_base  # noqa: F401  (path setup)
    import execution
    import training
    from replicate_training import jitter_head
    applied = {}
    frozen_fit = training.fit

    def fit(encoder, head, level, config, out, check):
        applied['jitter'] = jitter_head(head, args.replicate) if head is not None else dict(jitter=False)
        return frozen_fit(encoder, head, level, config, out, check)

    training.fit = fit
    code = execution.run_worker(str(config_path), args.verify)
    (out / ('replicate-verify.json' if args.verify else 'replicate.json')).write_text(
        json.dumps(dict(map=args.map, condition=args.condition, replicate=args.replicate, module=args.module, exit_code=code,
                        applied=applied), indent=1, default=str), encoding='utf-8', newline='\n')
    return code


if __name__ == '__main__':
    sys.exit(main() or 0)
