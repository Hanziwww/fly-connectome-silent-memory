"""X2 (plan v2, Phase 1C; gap G05): readout-only memory training on new scenes, as X1. GPU host only.

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/x2_scene_training.py --scene S1-mirror --replicate R
        --out DIR [--verify]

Scenes (code/x2_scenes.py): M1 and M2 (registered in the frozen study, never trained; legacy worker, like M0) and the
S1/S2 left-right mirrors and certified single-cell removals (tasks/x2-tasks.json; supplement worker, like S1 and S2).
The family's frozen readout-only model gives the configuration (M0: remote-dynamic-M0-head_only; S1: supp-S1-ring-brain;
S2: supp-S2-context-brain) with the task replaced. The frozen trainers run unchanged; as in replicate_training.py a
replicate changes only the order of the 9 trained timings and the readout jitter (r = 0: frozen order, no jitter). The
supplement trainer finds the new scenes because supp_tasks.prepare is wrapped to return the locked tasks plus these.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / 'studies' / 'brain-sokoban-v1'
MECH = FROZEN / 'mechanisms-v1'
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
X2_TASKS = SUPP / 'tasks' / 'x2-tasks.json'
SCENES = {'M1': 'remote-dynamic-M0-head_only', 'M2': 'remote-dynamic-M0-head_only',
          'S1-mirror': 'supp-S1-ring-brain', 'S1-minus45': 'supp-S1-ring-brain',
          'S2-mirror': 'supp-S2-context-brain', 'S2-minus48': 'supp-S2-context-brain'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene', required=True, choices=sorted(SCENES))
    ap.add_argument('--replicate', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = SCENES[args.scene]
    base = json.loads((MECH / 'inputs' / 'models' / model / 'source-config.json').read_text(encoding='utf-8'))
    supplement = model.startswith('supp-')
    sys.path.insert(0, str(MECH / 'code'))
    sys.path.insert(0, str(SUPP / 'code'))
    import mechanism_base  # noqa: F401  (path setup)
    import memory as legacy
    from replicate_training import jitter_head, record_verification_adapter, supplement_verification_adapter, timing_iteration
    extra, adapter = {}, None
    if supplement:
        sys.path.insert(0, str(FROZEN / 'remote-supplement-20261003'))
        import supp_experiments as trainer
        import supp_tasks
        raw = X2_TASKS.read_bytes()
        scenes = json.loads(raw)
        frozen_prepare = supp_tasks.prepare

        def prepare(rebuild=False):
            locked = frozen_prepare(rebuild)
            return locked + [t for t in scenes if t['id'] not in {x['id'] for x in locked}]

        supp_tasks.prepare = prepare                      # supp_experiments.train reads tasklib.prepare at call time
        if not any(t['id'] == args.scene for t in scenes):
            raise RuntimeError(f'{args.scene} is not in {X2_TASKS.name}')
        extra = dict(x2_tasks_sha256=hashlib.sha256(raw).hexdigest())
        policy_cls, step = trainer.Policy, trainer.training_step
        if args.verify:                                   # the frozen supplement's verification adapter (replicate_training.py)
            adapter = supplement_verification_adapter(trainer, legacy)
    else:
        trainer = legacy
        policy_cls, step = legacy.MemoryPolicy, legacy.training_step
    applied = {}

    class ReplicatePolicy(policy_cls):
        def __init__(self, config, out_dir):
            super().__init__(config, out_dir)
            if self.head is not None:
                applied['jitter'] = jitter_head(self.head, args.replicate)

    def replicate_step(policy, task, optimizer, iteration):
        frozen_iteration = timing_iteration(iteration, args.replicate)
        result = step(policy, task, optimizer, frozen_iteration)
        result['frozen_iteration'] = frozen_iteration
        return result

    if supplement:
        trainer.Policy = ReplicatePolicy
        trainer.training_step = replicate_step
    else:
        legacy.MemoryPolicy = ReplicatePolicy
        legacy.training_step = replicate_step
    config = dict(base, id=f'x2-{args.scene}-r{args.replicate}', task=args.scene, replicate=args.replicate,
                  replicate_definition='timing-order permutation (seed 1000+r) and head jitter 0.1 SD (seed 2000+r); r = 0 is the frozen protocol',
                  **extra)
    config_path = out / 'config.json'
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=1), encoding='utf-8', newline='\n')
    code = trainer.train(str(config_path), args.verify) if supplement else legacy.worker(str(config_path), args.verify)
    record_verification_adapter(out, adapter, legacy)
    (out / ('replicate-verify.json' if args.verify else 'replicate.json')).write_text(
        json.dumps(dict(scene=args.scene, model=model, replicate=args.replicate, exit_code=code, applied=applied,
                        verification_adapter=adapter), indent=1, default=str),
        encoding='utf-8', newline='\n')
    return code


if __name__ == '__main__':
    sys.exit(main() or 0)
