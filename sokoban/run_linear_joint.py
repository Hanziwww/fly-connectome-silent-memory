"""Bounded P03 experiment: inherited full brain and linear readout learn together."""
from workspace import ARCHIVE, RUNS
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import torch

from bootstrap import HERE, ROOT
from boundary_worker import CountedBrain, stop_check
from common import configure, cpu_state, file_hash, save_json, tensor_hash
from continual import evaluate
from linear_joint_tasks import SuffixCurriculum, descriptor
from linear_joint_training import assessment, fit
from progressive_tasks import forward_certificate, reverse_distances
from progressive_worker import same_rollouts
from readout_models import ActivityLinear
from run_boundary import supervise
from run_cycle import acquire_worker_lock
from run_readouts import read
from workspace import recorded_path, source_manifest, verify_code_manifest

SOURCE = ARCHIVE / 'readouts/20261001-compare01/trials/L'
LEVELS = ARCHIVE / 'progressive/20261001-single-seed01/levels.json'
DONE = ('original_solved', 'budget', 'already_solved')


def prepare(args, out):
    verification = read(SOURCE / 'verification.json')
    assert verification['status'] == 'verified' and verification['original_solved']
    assert file_hash(SOURCE / 'model.pt') == verification['model_sha256']
    level = next(l for l in read(LEVELS) if l['id'] == 'P03')
    distances, certificate = reverse_distances(level)
    assert distances[(level['start'][0], tuple(level['start'][1]))] == 22
    certificate['independent_forward_bfs'] = forward_certificate(level)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SOURCE / 'model.pt', out / 'source.pt')
    shutil.copyfile(SOURCE / 'level.json', out / 'previous-level.json')
    save_json(out / 'level.json', level)
    save_json(out / 'task-certificate.json', certificate)
    config = dict(experiment='joint_linear_new_task', source=str(SOURCE),
        source_checkpoint_sha256=file_hash(out / 'source.pt'), seed=10, task_id='P03',
        training_seconds=args.seconds, core_learning_rate=.0001, head_learning_rate=.0003,
        steps_per_assessment=4, brain_parameters=35310, head_parameters=2052,
        joint_optimization='Same loss and parameter point; one Adam step updates both parameter groups',
        core_gradient='Existing surrogate spike derivatives with hard LIF forward; all full-connectome edges retained',
        activation_memory='Two-pass chain rule; one 16-board differentiable simulation at a time',
        normalization='Inherited frozen activity mean/scale and clipping',
        curriculum='Generic reverse suffixes 1..22, one entry plus recovery loss on demonstrated successful dependencies',
        loss='Shared hybrid softplus entry plus .5 mean and .5 max squared anchor violation',
        acceptance='Any autonomous original-start solution; otherwise preserve learned suffixes and improve current loss',
        shortest_path_required=False, old_task_retention_gate=False,
        training_teacher_only=True, inference_planner=False, head_board_input=False,
        inference='Board -> full brain -> 512 activities -> affine readout -> physical legal mask -> action',
        initialization_and_verification_excluded_from_training_clock=True,
        allocator_limit_gib=11, minimum_free_gib=13, serial_worker=True,
        source_files={str(p):file_hash(p) for p in (SOURCE/'model.pt', SOURCE/'verification.json', SOURCE/'result.json', SOURCE/'level.json', LEVELS)},
        copied_files={name:file_hash(out/name) for name in ('source.pt','previous-level.json','level.json','task-certificate.json')},
        no_automatic_retry=True)
    save_json(out / 'config.json', config)
    save_json(out / 'protocol.json', dict(status='prepared', **config,
        substitutions=['source_brain_final_readout','final_brain_source_readout','zero_plasticity_final_readout'],
        interpretation='Joint learning attempts and actual saved core changes are reported separately from causal necessity'))
    save_json(out/'source-files.json', source_manifest())
    print(json.dumps(dict(event='prepared', out=str(out), task='P03', boxes=2, size=8, floor_cells=31, reference_moves=22)), flush=True)


def integrity(out):
    config = read(out/'config.json')
    for name, expected in config['source_files'].items():
        assert file_hash(recorded_path(name)) == expected, name
    for name, expected in config['copied_files'].items():
        assert file_hash(out/name) == expected, name
    verify_code_manifest(out)
    return config


def load_pair(path, encoder, head):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['architecture'] == head.architecture
    encoder.set_core(saved['core'])
    head.load_state_dict(saved['head'], strict=True)
    return saved


def substitutions(encoder, head, saved, source, levels, check):
    result = {}
    try:
        for name, core, readout in [
            ('source_brain_final_readout', source['core'], saved['head']),
            ('final_brain_source_readout', saved['core'], source['head']),
            ('zero_plasticity_final_readout', {k:torch.zeros_like(v) for k,v in source['core'].items()}, saved['head'])]:
            encoder.set_core(core)
            head.load_state_dict(readout)
            result[name] = evaluate(encoder, head, levels, name, check)
    finally:
        encoder.set_core(saved['core'])
        head.load_state_dict(saved['head'])
    return result


def worker(out, verify=False):
    config = integrity(out)
    configure(config['seed'])
    started = time.monotonic()
    check = lambda: stop_check(out)
    encoder = None
    result = dict(status='initializing', pid=os.getpid(), mode='simultaneous_joint_adam')
    try:
        free, total = torch.cuda.mem_get_info()
        if free < 13 * 2**30:
            raise RuntimeError('Need at least 13 GiB free GPU memory')
        torch.cuda.set_per_process_memory_fraction(11 * 2**30 / total)
        encoder = CountedBrain(ARCHIVE/'campaign/run', batch=16, persist_mapping=False)
        head = ActivityLinear().cuda()
        source = load_pair(out/'source.pt', encoder, head)
        level, old_level = read(out/'level.json'), read(out/'previous-level.json')
        result['resources'] = dict(free_before_gib=free / 2**30, allocator_limit_gib=11, batch=16)
        if verify:
            previous = read(out/'result.json')
            assert previous['pid'] != os.getpid()
            assert file_hash(out/'model.pt') == previous['model_sha256']
            saved = load_pair(out/'model.pt', encoder, head)
            lessons = SuffixCurriculum(level, saved['feedback_snapshot'])
            after, suite = assessment(encoder, head, lessons, saved['lesson_depth'], check)
            assert after == previous['after']
            assert same_rollouts(suite, previous['after_suite'])
            old = evaluate(encoder, head, [old_level], 'previous-task-diagnostic', check)
            assert same_rollouts(old, previous['old_task_after'])
            ablations = substitutions(encoder, head, saved, source, [level], check)
            for name, observation in ablations.items():
                assert same_rollouts(observation, previous['substitutions'][name]), name
            promoted = []
            for entry in previous.get('training', {}).get('promotions', []):
                checkpoint = load_pair(out/entry['checkpoint'], encoder, head)
                ls = SuffixCurriculum(level, checkpoint['feedback_snapshot'])
                obs, run = assessment(encoder, head, ls, checkpoint['lesson_depth'], check)
                assert obs['stage_passed'] and same_rollouts(run, entry['after_suite'])
                promoted.append(dict(depth=entry['depth'], sha256=file_hash(out/entry['checkpoint']), verified=True))
            load_pair(out/'model.pt', encoder, head)
            assert encoder.graph_digest() == encoder.base_digest and encoder.core.bound_audit()['in_bounds']
            assert all(torch.equal(saved['head'][k], source['head'][k]) for k in ('mean','scale'))
            integrity(out)
            save_json(out/'verification.json', dict(status='verified', pid=os.getpid(), training_pid=previous['pid'],
                model_sha256=previous['model_sha256'], original_solved=after['solved'], original_steps=after['original_steps'],
                actions_logits_scores_exact=True, substitutions_exact=True, promotions=promoted,
                graph_unchanged=True, bounds_valid=True, normalization_frozen=True, head_activity_only=True,
                no_inference_teacher=True, wall_seconds=time.monotonic()-started))
            print(json.dumps(dict(event='joint-linear-verified', solved=after['solved'], steps=after['original_steps'])), flush=True)
            return 0
        if (out/'result.json').exists() or (out/'history.jsonl').exists():
            raise RuntimeError('Existing attempts cannot be overwritten or retried')
        result['before_suite'] = evaluate(encoder, head, [level, old_level], 'inherited-linear', check)
        old_expected = read(SOURCE/'result.json')['after_suite']['records'][0]
        old_actual = result['before_suite']['records'][1]
        assert old_actual['actions'] == old_expected['actions']
        assert [d['brain_logits'] for d in old_actual['decisions']] == [d['brain_logits'] for d in old_expected['decisions']]
        result['initialization_seconds'] = time.monotonic()-started
        print(json.dumps(dict(event='joint-linear-baseline', tasks=[dict(id=r['level_id'], status=r['status'], steps=r['steps']) for r in result['before_suite']['records']])), flush=True)
        if result['before_suite']['records'][0]['status'] == 'solved':
            raise RuntimeError(f"{level['id']} already solved before training; record baseline and reassess task selection")
        result['training'] = fit(encoder, head, level, config, out)
        saved = load_pair(out/'model.pt', encoder, head)
        lessons = SuffixCurriculum(level, saved['feedback_snapshot'])
        result['after'], result['after_suite'] = assessment(encoder, head, lessons, saved['lesson_depth'], check)
        result['old_task_after'] = evaluate(encoder, head, [old_level], 'previous-task-diagnostic', check)
        result['substitutions'] = substitutions(encoder, head, saved, source, [level], check)
        if result['training']['stop'] == 'original_solved':
            assert result['after']['solved']
        assert result['after']['retained'] or result['after']['solved']
        result['checks'] = dict(core_changes={k:int((v!=source['core'][k]).sum()) for k,v in saved['core'].items()},
            head_changes={k:int((v!=source['head'][k]).sum()) for k,v in saved['head'].items() if k not in ('mean','scale')},
            graph_unchanged=encoder.graph_digest()==encoder.base_digest, bounds=encoder.core.bound_audit(),
            normalization_frozen=all(torch.equal(saved['head'][k],source['head'][k]) for k in ('mean','scale')),
            final_core_sha256=tensor_hash(saved['core']), final_head_sha256=tensor_hash(saved['head']),
            source_and_code_unchanged=bool(integrity(out)))
        assert result['checks']['graph_unchanged'] and result['checks']['bounds']['in_bounds']
        assert result['checks']['normalization_frozen']
        save_json(out/'data.json', descriptor(lessons.data(saved['lesson_depth'])))
        save_json(out/'lessons.json', saved['feedback_snapshot'])
        result['status'] = result['training']['stop']
        result['model_sha256'] = file_hash(out/'model.pt')
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
            result['wall_seconds'] = time.monotonic()-started
            if torch.cuda.is_initialized():
                result.setdefault('resources', {}).update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            save_json(out/'result.json', result)
            save_json(out/'progress.json', dict(status=result['status']))
    return 0 if result['status'] in DONE else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RUNS/'linear-joint/20261002-p03-01')
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.worker or args.verify:
        return worker(out, args.verify)
    if not 5 <= args.seconds <= 600:
        parser.error('Use a 5..600 second bounded training attempt')
    if not args.resume:
        prepare(args, out)
    config = integrity(out)
    assert config['training_seconds'] == args.seconds
    if args.prepare_only:
        return 0
    lock, started = acquire_worker_lock(), time.monotonic()
    identity = dict(status='starting', supervisor_pid=os.getpid(), phase='training')
    command = [sys.executable,'-B','-X','utf8','-u',str(HERE/'run_linear_joint.py'),'--out',str(out)]
    try:
        if (out/'result.json').exists():
            assert read(out/'result.json')['status'] in DONE, 'Failed attempts are preserved'
        else:
            code, state = supervise(command+['--worker'], out/'worker.log', out, args.seconds+180, identity)
            if code or state != 'running':
                identity['status'] = 'stopped' if state == 'stopped' else 'error'
                return 1
        if not (out/'verification.json').exists():
            identity['phase'] = 'independent-verification'
            code, state = supervise(command+['--verify'], out/'verification.log', out, 360, identity)
            if code or state != 'running':
                identity['status'] = 'error'
                return 1
        identity['status'] = 'complete'
        print(json.dumps(dict(event='complete', verification=read(out/'verification.json'))), flush=True)
        return 0
    finally:
        identity['wall_seconds'] = time.monotonic()-started
        save_json(out/'supervisor.json', identity)
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
