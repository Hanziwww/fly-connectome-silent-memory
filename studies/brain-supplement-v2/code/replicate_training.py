"""Replicate memory training (plan v2: X1 replicates, X8, and Phase 3 learning under lesion). GPU host only.

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/replicate_training.py --model ID --replicate R
        --out DIR [--lesion SPEC.json] [--variant NAME] [--perturb NAME] [--verify]

The frozen trainers run unchanged; this module only patches names they look up at run time (as the frozen supplement
itself does with the legacy worker):
- M0 family: studies/brain-sokoban-v1/code/memory.py (worker, MemoryPolicy, training_step);
- S1/S2 families: studies/brain-sokoban-v1/remote-supplement-20261003/supp_experiments.py (train -> legacy worker).

A replicate r changes exactly two things:
1. the order of the 9 trained timings: update u uses timing perm_b[u mod 9], where perm_b is a permutation of the 9
   (delay, prefix) combinations drawn for block b = u div 9 from seed (1000 + r, b). The frozen order is u -> (delay index
   u mod 3, prefix index (u div 3) mod 3); a replicate reaches its timing by calling the frozen step with the iteration
   index of that combination. The two cues are summed before the optimiser step, so their order is immaterial.
2. the readout initialisation: every head parameter + N(0, (0.1 x its SD)^2), seeded by (2000 + r).
Replicate 0 is the frozen protocol: identity order, no jitter. Its result must equal the original run (checked by
verify_replicate0 in g0-style comparisons).

Optional lesion (Phase 3, regime B): a JSON list of operator specs (studies/brain-pathology-v1/code/operators.py) applied
to the brain after the source checkpoint is loaded and before training; the operators stay on for training and for
every assessment.

Optional training perturbation (T-DY2, robust readout training; M0 trainer only): --perturb NAME, a condition of
studies/brain-pathology-v1/code/dy2_perturbation.py (input noise, membrane noise, excitability randomisation). Each
training episode draws its own perturbation (seeded by condition, replicate, update and episode), the readout-only
feature cache is off, the worker's assessment requires success unperturbed and under the condition's two fixed
acceptance draws, and the worker's clock runs at a third of real time (dy2_perturbation.CLOCK_FACTOR; the assessment is
run three times). The --verify step is the frozen, unperturbed verification. config.json records the condition.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / 'studies' / 'brain-sokoban-v1'
MECH = FROZEN / 'mechanisms-v1'
PATHO = ROOT / 'studies' / 'brain-pathology-v1' / 'code'
COMBOS = 9
JITTER_SD = 0.1


def timing_iteration(update, replicate):
    """Frozen-step iteration index whose (delay, prefix) is the replicate's timing for this update."""
    if replicate == 0:
        return update
    block, pos = divmod(update, COMBOS)
    perm = np.random.default_rng([1000 + replicate, block]).permutation(COMBOS)
    return int(perm[pos])


def jitter_head(head, replicate):
    import torch
    if replicate == 0:
        return dict(jitter=False)
    g = torch.Generator().manual_seed(2000 + replicate)
    stats = {}
    with torch.no_grad():
        for name, p in head.named_parameters():
            sd = float(p.detach().double().std()) if p.numel() > 1 else float(p.detach().abs().mean())
            noise = torch.randn(p.shape, generator=g, dtype=torch.float64) * (JITTER_SD * sd)
            p.add_(noise.to(p.device, p.dtype))
            stats[name] = dict(sd=sd, noise_sd=JITTER_SD * sd)
    return dict(jitter=True, parameters=stats)


def lesion_specs(path):
    if not path:
        return []
    sys.path.insert(0, str(PATHO))
    import operators
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    return [operators.spec(s['kind'], np.asarray(s['cells'], dtype=np.int64), s.get('value'), s.get('label')) for s in raw]


def supplement_verification_adapter(trainer, legacy):
    """The frozen supplement verified its S1/S2 runs through verify_compat.py, a verification-only adapter: the
    supplement's intervention rollouts carry an 'intervention' key that memory.worker adds again when archiving them.
    Patch trainer.batched_assessment the same way (verify_compat.legacy_rows; read by supp_experiments.train at call
    time) and return the record the frozen supplement adds to verification.json. Training, forward pass and acceptance
    are unchanged. Needs remote-supplement-20261003 on sys.path."""
    import verify_compat
    original = trainer.batched_assessment

    def compatible(policy, task, delays, prefixes, max_steps=512, intervention=None):
        probes, rows = original(policy, task, delays, prefixes, max_steps, intervention)
        return verify_compat.legacy_rows(probes, rows, intervention)

    trainer.batched_assessment = compatible
    return dict(sha256=legacy.digest(Path(verify_compat.__file__)),
                reason='Remove duplicate intervention metadata at the legacy archive boundary',
                changes_to_training=False, changes_to_forward_pass=False, changes_to_acceptance=False)


def record_verification_adapter(out, adapter, legacy):
    """Add the adapter record to <out>/verification.json, as the frozen supplement does."""
    verification = Path(out) / 'verification.json'
    if adapter and verification.exists():
        legacy.write(verification, dict(legacy.read(verification), verification_adapter=adapter))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, help='mechanisms-v1 model id (its source-config.json is the base config)')
    ap.add_argument('--replicate', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lesion', default=None, help='JSON list of operator specs')
    ap.add_argument('--variant', default=None, help='structural variant of the brain (studies/brain-pathology-v1/code/variants.py)')
    ap.add_argument('--perturb', default=None, help='T-DY2 training perturbation (studies/brain-pathology-v1/code/dy2_perturbation.py)')
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((MECH / 'inputs' / 'models' / args.model / 'source-config.json').read_text(encoding='utf-8'))
    sys.path.insert(0, str(MECH / 'code'))
    import mechanism_base  # noqa: F401  (path setup: flybrain, sokoban, frozen study code)
    supplement = 'S1' in base['task'] or 'S2' in base['task'] or base['id'].startswith('supp-')
    perturb = None
    if args.perturb:
        if supplement:
            raise SystemExit('--perturb: T-DY2 stage 1 covers the M0 trainer only')
        sys.path.insert(0, str(PATHO))
        import dy2_perturbation as dy2
        perturb = dy2.describe(args.perturb)
    specs = lesion_specs(args.lesion)
    record = dict(model=args.model, replicate=args.replicate, lesion=[dict(kind=s['kind'], cells=int(len(s['cells'])), value=s['value'],
                                                                           label=s['label']) for s in specs], frozen_trainer='supplement' if supplement else 'memory')
    import memory as legacy
    adapter = None
    if supplement:
        sys.path.insert(0, str(FROZEN / 'remote-supplement-20261003'))
        import supp_experiments as trainer
        policy_cls, step = trainer.Policy, trainer.training_step
        if args.verify:
            adapter = supplement_verification_adapter(trainer, legacy)
    else:
        trainer = legacy
        policy_cls, step = legacy.MemoryPolicy, legacy.training_step
    applied = {}
    holder = {}                                         # the training policy's T-DY2 episode draws (--perturb only)

    class ReplicatePolicy(policy_cls):
        def __init__(self, config, out_dir):
            super().__init__(config, out_dir)
            if args.variant and self.encoder is not None:     # before the brain first runs (the constructor only loads)
                sys.path.insert(0, str(PATHO))
                import variants
                applied['variant'] = variants.apply(self.encoder, args.variant)
            if self.head is not None:
                applied['jitter'] = jitter_head(self.head, args.replicate)
            if specs and self.encoder is not None:
                sys.path.insert(0, str(PATHO))
                import operators
                applied['lesion'] = operators.apply(self.encoder, specs)
            if perturb and not args.verify and self.encoder is not None:   # last: wraps the lesion's bind
                pert = dy2.Perturbation(self.encoder)
                holder.update(policy=self, training=dy2.Training(pert, args.perturb, args.replicate))
                self.frozen_features = dy2.NoCache()
                applied['perturbation'] = dict(name=args.perturb, membrane_factor=pert.a_m, membrane_step_factor=pert.step_factor)

        def reset(self, *a, **k):
            if holder.get('policy') is self:
                holder['training'].episode()             # a new episode's draws, only inside a training update
            return super().reset(*a, **k)

    def replicate_step(policy, task, optimizer, iteration):
        frozen_iteration = timing_iteration(iteration, args.replicate)
        training = holder.get('training')
        if training is not None:
            training.begin(iteration)
        try:
            result = step(policy, task, optimizer, frozen_iteration)
        finally:
            if training is not None:
                training.end()
        result['frozen_iteration'] = frozen_iteration       # the worker logs the update number itself
        return result

    if perturb and not args.verify:
        frozen_assess = legacy.assess

        def robust_assess(policy, task, full=False):
            training = holder.get('training')
            if training is None:
                return frozen_assess(policy, task, full)
            return dy2.merged_assess(frozen_assess, training.pert, perturb['accept_draws'])(policy, task, full)
        legacy.assess = robust_assess
        legacy.time = dy2.ScaledClock(perturb['clock_factor'])   # the worker's segment clock (memory.time.perf_counter)

    if supplement:
        trainer.Policy = ReplicatePolicy
        trainer.training_step = replicate_step
    else:
        legacy.MemoryPolicy = ReplicatePolicy
        legacy.training_step = replicate_step
    config = dict(base, id=f'{args.model}-r{args.replicate}' + ('-lesion' if specs else '') + (f'-{args.variant}' if args.variant else '')
                  + (f'-{args.perturb}' if args.perturb else ''),
                  replicate=args.replicate,
                  replicate_definition='timing-order permutation (seed 1000+r) and head jitter 0.1 SD (seed 2000+r); r = 0 is the frozen protocol',
                  lesion=record['lesion'])
    if args.variant:                                   # absent otherwise, so runs without a variant keep their config
        config['variant'] = record['variant'] = args.variant
    if perturb:                                        # likewise
        config['training_perturbation'] = record['training_perturbation'] = perturb
    config_path = out / 'config.json'
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=1), encoding='utf-8', newline='\n')
    written = json.loads(config_path.read_text(encoding='utf-8'))
    if written.get('variant') != args.variant:
        raise SystemExit(f'{out}: the run was configured with another structural variant than --variant {args.variant}')
    if written.get('training_perturbation') != (json.loads(json.dumps(perturb)) if perturb else None):
        raise SystemExit(f'{out}: the run was configured with another training perturbation than --perturb {args.perturb}')
    started = time.time()
    code = trainer.train(str(config_path), args.verify) if supplement else legacy.worker(str(config_path), args.verify)
    record['real_seconds'] = time.time() - started
    record_verification_adapter(out, adapter, legacy)
    if holder.get('training') is not None:
        applied['perturbation']['episodes'] = holder['training'].episodes
    record.update(exit_code=code, applied=applied, verification_adapter=adapter)
    (out / ('replicate-verify.json' if args.verify else 'replicate.json')).write_text(json.dumps(record, indent=1, default=str),
                                                                                      encoding='utf-8', newline='\n')
    return code


if __name__ == '__main__':
    sys.exit(main() or 0)
