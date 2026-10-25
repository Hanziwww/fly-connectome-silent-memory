"""Matched linear, MLP and E/I readouts on one frozen learned connectome."""
from workspace import ARCHIVE, RUNS
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from bootstrap import HERE, ROOT
from boundary_worker import initialize, stop_check
from common import configure, cpu_state, file_hash, save_checkpoint, save_json, tensor_hash
from continual import evaluate
from game import legal, restore
from readout_models import make_readout
from hybrid_runtime import FeatureBank, assessment, audit_states, policy_audit
from hybrid_tasks import dataset, descriptor
from hybrid_training import fit
from population_gpu import PopulationGPU, logits
from progressive_worker import same_rollouts
from run_boundary import supervise
from run_cycle import acquire_worker_lock

DEFAULT_SOURCE = ARCHIVE / 'frontier/20261001-directions01/trials/G'
DONE = ('budget', 'original_solved')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def prepare(args, out):
    source = Path(args.source).resolve()
    verified = read(source / 'verification.json')
    assert verified['status'] == 'verified'
    assert file_hash(source / 'model.pt') == verified['model_sha256']
    out.mkdir(parents=True, exist_ok=False)
    for src, dst in [('model.pt', 'source.pt'), ('level.json', 'level.json'), ('lessons.json', 'feedback.json')]:
        shutil.copyfile(source / src, out / dst)
    original = torch.load(out / 'source.pt', map_location='cpu', weights_only=True)
    cases = []
    for arm, kind in [('L', 'linear'), ('M', 'mlp'), ('E', 'ei')]:
        configure(10)
        head = make_readout(kind, original['head'])
        name = f'initial-head-{arm}.pt'
        save_checkpoint(out / name, cpu_state(head))
        cases.append(dict(arm=arm, mode='head_only', readout_kind=kind,
            architecture=head.architecture, initial_head_file=name,
            head_trainable_parameters=sum(p.numel() for p in head.parameters())))
    order = np.random.default_rng(10).permutation(['L', 'M', 'E']).tolist()
    config = dict(experiment='frozen_brain_readout_comparison', seed=10,
        source_directory=str(source), source_checkpoint_sha256=file_hash(out / 'source.pt'),
        training_seconds=args.seconds, head_learning_rate=.0003, head_steps_per_round=8,
        directions=4, starting_depth=6, cases=cases, order=order, allocator_limit_gib=11,
        brain_frozen=True, shortest_path_required=False, old_unrelated_tasks_not_gated=True,
        original_success_overrides_auxiliary_requirements=True,
        common_loss='softplus entry + 0.5 mean squared anchor violations + 0.5 max squared anchor violation',
        optimizer='Adam, shared lr=0.0003, clip gradient norm=5, reset optimizer on curriculum advancement',
        feedback='Same reverse suffix curriculum from depth 6; accumulate successful dependencies within each arm',
        inference='Brain -> 512 activity features -> readout -> physical legal mask -> action',
        head_initialization='All arms embed the source affine readout; all 39 solvable nonterminal actions must match',
        normalization='Identical frozen mean, scale and clamp(-10,10) in all arms',
        ei_constraint='32 excitatory and 32 inhibitory nonnegative hidden rates; fixed outgoing column signs through softplus magnitudes',
        ei_scope='Added hidden-to-action weights only; afferent weights unconstrained; no recurrent dynamics, spikes or local biological learning',
        ei_ratio='1:1 is an engineering initialization choice, not an inferred fruit-fly population ratio',
        ei_reference='https://doi.org/10.1371/journal.pcbi.1004792',
        comparison_scope='One seed, one fixed task, one inherited brain, one shared optimizer setting; no generalization or optimality claim',
        trainer='Unmodified hybrid_training.fit in head_only mode; its legacy checkpoint experiment and phase labels include MLP',
        no_automatic_retry=True, population_inference=False,
        source_files={str(source / name): file_hash(source / name)
                      for name in ('model.pt', 'level.json', 'lessons.json', 'verification.json')},
        copied_files={name: file_hash(out / name) for name in
                      ['source.pt', 'level.json', 'feedback.json'] + [c['initial_head_file'] for c in cases]})
    save_json(out / 'config.json', config)
    save_json(out / 'protocol.json', dict(status='prepared', **config,
        initialization_baseline_and_initial_cache_excluded_from_training_clock=True,
        independent_verification_excluded_from_training_clock=True,
        total_worker_and_supervisor_wall_time_reported=True,
        substitutions=['source_brain_same_final_readout', 'zero_plasticity_same_final_readout', 'final_brain_initial_readout'],
        substitution_drop_is_not_proof_of_irreplaceability=True))
    files = list(HERE.glob('*.py')) + list((HERE / 'lib').glob('*.py')) + [ROOT / 'workflow.py']
    save_json(out / 'source-files.json', {str(p.relative_to(ROOT)): file_hash(p) for p in files})
    for case in cases:
        trial = out / 'trials' / case['arm']
        trial.mkdir(parents=True)
        save_json(trial / 'config.json', dict(config, **case))
    print(json.dumps(dict(event='prepared', out=str(out), order=order, cases=cases)), flush=True)


def integrity(out):
    config = read(out / 'config.json')
    for path, expected in config['source_files'].items():
        assert file_hash(path) == expected, path
    for name, expected in config['copied_files'].items():
        assert file_hash(out / name) == expected, name
    for path, expected in read(out / 'source-files.json').items():
        assert file_hash(ROOT / path) == expected, path
    return config


def preflight(encoder, old_head, head, source, level, check):
    states, boards, _ = audit_states(level)
    source_features = encoder.encode(boards, check)
    old_q, new_q = logits(old_head, source_features), logits(head, source_features)
    error = float((old_q - new_q).abs().max())
    masks = torch.as_tensor(np.stack([legal(restore(level), s) for s in states]), device='cuda')
    old_actions = old_q.masked_fill(~masks, -1e9).softmax(1).argmax(1)
    new_actions = new_q.masked_fill(~masks, -1e9).softmax(1).argmax(1)
    assert error < 1e-5 and torch.equal(old_actions, new_actions)
    zero = {k: torch.zeros_like(v) for k, v in source['core'].items()}
    encoder.set_core(zero)
    zero_features = encoder.encode(boards, check)
    engine = PopulationGPU(encoder, head)
    engine.bind([source['core'], zero])
    try:
        together = engine.encode(np.concatenate([boards, boards]),
                                 np.repeat(np.arange(2), len(boards)), check)
    finally:
        engine.release()
    np.testing.assert_array_equal(together[:len(boards)], source_features)
    np.testing.assert_array_equal(together[len(boards):], zero_features)
    encoder.set_core(source['core'])
    return dict(initial_action_parity_states=len(states), initial_maximum_logit_difference=error,
                population_feature_parity_exact=True, distinct_brains_checked=2), FeatureBank(encoder, source['core'], boards, source_features)


def load_pair(path, encoder, head):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['architecture'] == head.architecture
    encoder.set_core(saved['core'])
    head.load_state_dict(saved['head'], strict=True)
    return saved


def substitutions(encoder, head, saved, source, initial_head, levels, check):
    results = {}
    try:
        for name, core, readout in [
            ('source_brain_same_final_readout', source['core'], saved['head']),
            ('zero_plasticity_same_final_readout', {k: torch.zeros_like(v) for k, v in saved['core'].items()}, saved['head']),
            ('final_brain_initial_readout', saved['core'], initial_head),
        ]:
            encoder.set_core(core)
            head.load_state_dict(readout)
            results[name] = evaluate(encoder, head, levels, name, check)
    finally:
        encoder.set_core(saved['core'])
        head.load_state_dict(saved['head'])
    return results


def worker(out, arm, verify=False):
    integrity(out)
    trial = out / 'trials' / arm
    config, level = read(trial / 'config.json'), read(out / 'level.json')
    configure(10)
    check = lambda: stop_check(out)
    started = time.monotonic()
    encoder = None
    result = dict(status='initializing', arm=arm, mode=config['mode'], readout_kind=config['readout_kind'], architecture=config['architecture'], head_trainable_parameters=config['head_trainable_parameters'], pid=os.getpid())
    try:
        encoder, old_head, source, resources = initialize(out)
        initial_head = torch.load(out / config['initial_head_file'], map_location='cpu', weights_only=True)
        head = make_readout(config['readout_kind']).cuda()
        head.load_state_dict(initial_head)
        result['resources'] = resources
        if verify:
            previous = read(trial / 'result.json')
            assert previous['pid'] != os.getpid()
            assert file_hash(trial / 'model.pt') == previous['model_sha256']
            saved = load_pair(trial / 'model.pt', encoder, head)
            _, data, levels = dataset(level, saved['feedback_snapshot'], saved['lesson_depth'])
            after, suite = assessment(encoder, head, data, levels, check)
            assert same_rollouts(previous['after_suite'], suite)
            for key in ('loss', 'margins', 'bridge_margin', 'critical_margin', 'retained', 'solved', 'joint_passed'):
                assert after[key] == previous['after'][key], key
            policy = policy_audit(encoder, head, level, check)
            assert policy == previous['after_policy']
            ablated = substitutions(encoder, head, saved, source, initial_head, levels, check)
            for name, tested in ablated.items():
                assert same_rollouts(previous['substitutions'][name], tested), name
            promoted = []
            for entry in previous['training']['promotions']:
                intermediate = load_pair(trial / entry['checkpoint'], encoder, head)
                _, d, ls = dataset(level, intermediate['feedback_snapshot'], intermediate['lesson_depth'])
                obs, tested = assessment(encoder, head, d, ls, check)
                assert obs['joint_passed'] and same_rollouts(entry['after_suite'], tested)
                promoted.append(dict(depth=entry['depth'], sha256=file_hash(trial / entry['checkpoint']), verified=True))
            load_pair(trial / 'model.pt', encoder, head)
            assert encoder.core.bound_audit()['in_bounds']
            assert encoder.graph_digest() == encoder.base_digest
            assert after['retained'] or after['solved']
            assert tensor_hash(saved['core']) == tensor_hash(source['core'])
            sign_check = head.sign_audit() if hasattr(head, 'sign_audit') else None
            assert sign_check == previous['checks']['ei_sign_audit']
            integrity(out)
            save_json(trial / 'verification.json', dict(status='verified', arm=arm,
                pid=os.getpid(), training_pid=previous['pid'], model_sha256=previous['model_sha256'],
                original_solved=after['solved'], original_steps=after['rollout_steps'],
                actions_logits_scores_exact=True, policy_audit_exact=True, substitutions_exact=True,
                graph_unchanged=True, bounds_valid=True, no_inference_planner=True, brain_frozen=True, ei_sign_audit=sign_check,
                head_activity_only=True, promotions=promoted, wall_seconds=time.monotonic() - started))
            print(json.dumps(dict(event='readout-verified', arm=arm, solved=after['solved'],
                                  steps=after['rollout_steps'])), flush=True)
            return 0
        if (trial / 'result.json').exists() or (trial / 'history.jsonl').exists():
            raise RuntimeError('Preserve existing attempts; never retrain them in place')
        result['preflight'], bank = preflight(encoder, old_head, head, source, level, check)
        snapshot = read(out / 'feedback.json')
        _, data, levels = dataset(level, snapshot, 6)
        result['before'], result['before_suite'] = assessment(bank, head, data, levels, check)
        result['before_policy'] = policy_audit(bank, head, level, check)
        assert result['before']['retained']
        result['initialization_seconds'] = time.monotonic() - started
        print(json.dumps(dict(event='readout-baseline', arm=arm, mode=config['mode'],
            initial=result['before']['rollout_status'], bridge=result['before']['bridge_margin'])), flush=True)
        result['training'] = fit(encoder, head, level, snapshot, config, trial, out, bank)
        saved = load_pair(trial / 'model.pt', encoder, head)
        _, data, levels = dataset(level, saved['feedback_snapshot'], saved['lesson_depth'])
        result['after'], result['after_suite'] = assessment(encoder, head, data, levels, check)
        result['after_policy'] = policy_audit(encoder, head, level, check)
        result['substitutions'] = substitutions(encoder, head, saved, source, initial_head, levels, check)
        assert result['after']['retained'] or result['after']['solved']
        if result['training']['stop'] == 'original_solved':
            assert result['after']['solved']
        core_changes = {k: int((v != source['core'][k]).sum()) for k, v in saved['core'].items()}
        if config['mode'] == 'head_only':
            assert sum(core_changes.values()) == 0
        assert all(torch.equal(saved['head'][k], initial_head[k]) for k in ('mean', 'scale'))
        result['checks'] = dict(graph_unchanged=encoder.graph_digest() == encoder.base_digest,
            bounds=encoder.core.bound_audit(), source_and_code_unchanged=bool(integrity(out)),
            head_changed=tensor_hash(saved['head']) != tensor_hash(initial_head),
            core_changes_from_source=core_changes, core_changed=bool(sum(core_changes.values())),
            final_head_sha256=tensor_hash(saved['head']), final_core_sha256=tensor_hash(saved['core']),
            normalization_frozen=True, head_activity_only=True, inference_planner=False,
            ei_sign_audit=head.sign_audit() if hasattr(head, 'sign_audit') else None)
        if result['checks']['ei_sign_audit'] is not None:
            assert all(result['checks']['ei_sign_audit'][k] for k in ('fixed_signs_valid', 'all_outgoing_signs_valid'))
        assert result['checks']['graph_unchanged'] and result['checks']['bounds']['in_bounds']
        save_json(trial / 'data.json', descriptor(data))
        save_json(trial / 'lessons.json', saved['feedback_snapshot'])
        shutil.copyfile(out / 'level.json', trial / 'level.json')
        result['model_sha256'] = file_hash(trial / 'model.pt')
        result['status'] = result['training']['stop']
    except BaseException as error:
        if verify:
            raise
        result.update(status='stopped' if isinstance(error, KeyboardInterrupt) else 'error',
                      error=repr(error), traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    finally:
        if encoder is not None:
            encoder.release_graph()
        if not verify:
            result['wall_seconds'] = time.monotonic() - started
            if torch.cuda.is_initialized():
                result.setdefault('resources', {}).update(
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
            save_json(trial / 'result.json', result)
            save_json(trial / 'progress.json', dict(status=result['status']))
    return 0 if result['status'] in DONE else 1


def summarize(out):
    config, rows = read(out / 'config.json'), []
    for case in config['cases']:
        trial = out / 'trials' / case['arm']
        if not (trial / 'result.json').exists():
            continue
        r = read(trial / 'result.json')
        a, t = r.get('after', {}), r.get('training', {})
        verified = (trial / 'verification.json').exists() and read(trial / 'verification.json')['status'] == 'verified'
        rows.append(dict(**case, status=r['status'], verified=verified, original_solved=a.get('solved'),
            original_steps=a.get('rollout_steps'), training_seconds=t.get('training_seconds'),
            head_steps=t.get('head_steps'), core_rounds=t.get('core_rounds'),
            working_core_updates=t.get('working_core_updates'),
            saved_core_changed=r.get('checks', {}).get('core_changed'),
            saved_head_changed=r.get('checks', {}).get('head_changed'),
            peak_gib=r.get('resources', {}).get('peak_allocated_gib'),
            goal_reaching_states=r.get('after_policy', {}).get('goal_reaching_nonterminal_states')))
    status = 'complete' if len(rows) == len(config['cases']) and all(r['verified'] for r in rows) else 'running'
    summary = dict(status=status, seed=10, rows=rows, order=config['order'])
    save_json(out / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ARCHIVE / 'readouts/20261001-compare01'))
    parser.add_argument('--source', default=str(DEFAULT_SOURCE))
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--arm', choices=['L', 'M', 'E'])
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if args.worker or args.verify:
        return worker(out, args.arm, args.verify)
    if not 5 <= args.seconds <= 300:
        parser.error('Training budget must be 5..300 seconds per arm')
    if not args.resume:
        prepare(args, out)
    config = integrity(out)
    if args.seconds != config['training_seconds']:
        parser.error('Resume budget must match the saved protocol')
    if args.prepare_only:
        return 0
    lock = acquire_worker_lock()
    started = time.monotonic()
    identity = dict(supervisor_pid=os.getpid(), status='starting', order=config['order'])
    try:
        for arm in config['order']:
            trial = out / 'trials' / arm
            command = [sys.executable, '-B', '-X', 'utf8', '-u', str(HERE / 'run_readouts.py'),
                       '--out', str(out), '--arm', arm]
            identity.update(arm=arm, phase='training')
            if (trial / 'result.json').exists():
                assert read(trial / 'result.json')['status'] in DONE, 'Failed attempts cannot be retried'
            else:
                code, state = supervise(command + ['--worker'], trial / 'worker.log', out, args.seconds + 180, identity)
                summarize(out)
                if code or state != 'running':
                    identity['status'] = 'stopped' if state == 'stopped' else 'error'
                    return 1
            if not (trial / 'verification.json').exists():
                identity['phase'] = 'independent-verification'
                code, state = supervise(command + ['--verify'], trial / 'verification.log', out, 240, identity)
                if code or state != 'running':
                    identity['status'] = 'error'
                    return 1
            print(json.dumps(dict(event='readout-arm-complete', arm=arm, summary=summarize(out))), flush=True)
        identity['status'] = 'complete'
        return 0
    finally:
        identity['wall_seconds'] = time.monotonic() - started
        identity['summary'] = summarize(out)
        save_json(out / 'supervisor.json', identity)
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
