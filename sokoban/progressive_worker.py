"""One isolated curriculum stage, or its separate-process checkpoint verification."""
from workspace import ARCHIVE, RUNS
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import torch
from bootstrap import HERE, ROOT
from boundary_tasks import load_task
from boundary_worker import initialize, stop_check
from common import configure, cpu_state, file_hash, save_checkpoint, save_json, tensor_hash
from continual import evaluate
from progressive_training import fit
from repair_training import measure, public


def integrity(out):
    config = json.loads((out / 'config.json').read_text())
    for path, digest in json.loads((out / 'protected-files.json').read_text()).items():
        assert file_hash(path) == digest, f'Protected file changed: {path}'
    for path, digest in json.loads((out / 'source-files.json').read_text()).items():
        assert file_hash(ROOT / path) == digest, f'Experiment source changed: {path}'
    for name, digest in config['task_files_sha256'].items():
        assert file_hash(out / name) == digest, f'Fixed task data changed: {name}'
    assert file_hash(out / 'source.pt') == config['source_checkpoint_sha256']
    return True


def assessments(encoder, head, out, task, tag):
    levels = json.loads((out / 'levels.json').read_text())
    position = next(i for i, level in enumerate(levels) if level['id'] == task)
    check = lambda: stop_check(out)
    return dict(current=evaluate(encoder, head, levels[position:position + 1], tag, check),
        previous=evaluate(encoder, head, levels[:position], tag + '-previous', check) if position else None,
        original=evaluate(encoder, head, json.loads((ARCHIVE / 'campaign/pilot-levels.json').read_text()),
                          tag + '-original', check))


def run_stage(out, task):
    config = json.loads((out / 'config.json').read_text())
    seed = config['seed']
    trial = out / 'stages' / task
    if (trial / 'progress.json').exists() or (trial / 'result.json').exists():
        raise RuntimeError('An attempted stage is preserved; no automatic retraining')
    started = time.monotonic()
    result = dict(status='initializing', task_id=task, seed=seed, pid=os.getpid())
    save_json(trial / 'progress.json', result)
    encoder = None
    try:
        integrity(out)
        stop_check(out)
        configure(seed)
        chain = json.loads((trial / 'source.json').read_text())
        assert file_hash(trial / 'source.pt') == chain['sha256']
        assert file_hash(ROOT / chain['path']) == chain['sha256']
        result['source'] = chain
        level, data = load_task(out, task)
        encoder, head, source, resources = initialize(trial)
        result['resources'] = resources
        assert tensor_hash(cpu_state(encoder.core)) == tensor_hash(source['core'])
        assert tensor_hash(cpu_state(head)) == tensor_hash(source['head'])
        result['baseline'] = public(measure(encoder, head, data, lambda: stop_check(out)))
        result['before'] = assessments(encoder, head, out, task, 'source')
        print(json.dumps(dict(event='baseline', task_id=task, correct=result['baseline']['correct'],
            total=len(data['boards']), autonomous_solved=result['before']['current']['solved'])), flush=True)
        if result['before']['current']['solved']:
            save_checkpoint(trial / 'model.pt', dict(core=cpu_state(encoder.core), head=cpu_state(head),
                task_id=task, seed=seed, source_checkpoint_sha256=chain['sha256'],
                checkpoint_training_seconds=0., output_frozen=True, correction_enabled=False,
                experiment='single_seed_continuous_curriculum'))
            result['training'] = dict(stop='already_solved', training_seconds=0., candidates=0,
                                      accepted_updates=0, checkpoint_training_seconds=0.)
        else:
            stage_config = dict(config, source_checkpoint_sha256=chain['sha256'])
            result['training'] = fit(encoder, head, level, data, stage_config, 'D', seed, out, trial)
        stop_check(out)
        saved = torch.load(trial / 'model.pt', map_location='cpu', weights_only=True)
        encoder.set_core(saved['core'])
        head.load_state_dict(saved['head'])
        result['teacher'] = public(measure(encoder, head, data, lambda: stop_check(out)))
        result['after'] = assessments(encoder, head, out, task, 'reloaded')
        assert result['after']['current']['complete']
        if result['training']['stop'] in ('already_solved', 'autonomous_success'):
            assert result['after']['current']['solved'], 'Checkpoint did not reproduce the claimed success'
        result['checks'] = dict(source_loaded_exactly=True,
            output_bitwise_frozen=tensor_hash(cpu_state(head)) == tensor_hash(source['head']),
            graph_unchanged=encoder.graph_digest() == encoder.base_digest,
            bounds=encoder.core.bound_audit(), protected_files_and_sources_unchanged=integrity(out))
        assert result['checks']['output_bitwise_frozen'] and result['checks']['graph_unchanged']
        assert result['checks']['bounds']['in_bounds']
        result['model_sha256'] = file_hash(trial / 'model.pt')
        result['status'] = ('already_solved' if result['before']['current']['solved'] else 'learned') \
            if result['after']['current']['solved'] else 'budget_exhausted'
    except KeyboardInterrupt as error:
        result.update(status='stopped', error=str(error))
    except Exception as error:
        result.update(status='error', error=repr(error), traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    finally:
        if encoder is not None:
            encoder.release_graph()
        if torch.cuda.is_initialized():
            result.setdefault('resources', {}).update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        result['wall_seconds'] = time.monotonic() - started
        save_json(trial / 'result.json', result)
        save_json(trial / 'progress.json', dict(status=result['status'], task_id=task, seed=seed))
        print(json.dumps(dict(event='stage_finished', task_id=task, status=result['status'],
                              seconds=result['wall_seconds'])), flush=True)
    return 0 if result['status'] in ('already_solved', 'learned', 'budget_exhausted') else 1


def same_rollouts(previous, current):
    if previous is None or current is None:
        return previous is current
    if not previous['complete'] or not current['complete'] or previous['total'] != current['total']:
        return False
    def signature(result):
        return [(r['level_id'], r['status'], r['actions'],
                 [d['brain_logits'] for d in r['decisions']]) for r in result['records']]
    return signature(previous) == signature(current)


def verify_stage(out, task):
    integrity(out)
    config = json.loads((out / 'config.json').read_text())
    configure(config['seed'])
    trial = out / 'stages' / task
    old = json.loads((trial / 'result.json').read_text())
    assert old['status'] in ('already_solved', 'learned', 'budget_exhausted')
    assert os.getpid() != old['pid'], 'Verification must be a new process'
    assert file_hash(trial / 'model.pt') == old['model_sha256']
    assert file_hash(trial / 'source.pt') == old['source']['sha256']
    assert file_hash(ROOT / old['source']['path']) == old['source']['sha256']
    encoder, head, source, resources = initialize(trial)
    try:
        saved = torch.load(trial / 'model.pt', map_location='cpu', weights_only=True)
        assert tensor_hash(saved['head']) == tensor_hash(source['head'])
        assert saved['source_checkpoint_sha256'] == old['source']['sha256']
        encoder.set_core(saved['core'])
        level, data = load_task(out, task)
        now = assessments(encoder, head, out, task, 'independent-reload')
        exact = {key: same_rollouts(old['after'][key], now[key]) for key in now}
        assert all(exact.values()), exact
        teacher = public(measure(encoder, head, data, lambda: stop_check(out)))
        assert teacher == old['teacher']
        assert encoder.graph_digest() == encoder.base_digest
        assert encoder.core.bound_audit()['in_bounds']
        integrity(out)
        verification = dict(status='verified', task_id=task, pid=os.getpid(), training_pid=old['pid'],
            model_sha256=old['model_sha256'], teacher_identical=True, actions_and_logits_identical=exact,
            solved=bool(now['current']['solved']), frozen_readout=True, graph_unchanged=True,
            bounds=encoder.core.bound_audit(), protected_files_and_sources_unchanged=True,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        save_json(trial / 'verification.json', verification)
        print(json.dumps(dict(event='independent_verified', **verification)), flush=True)
    finally:
        encoder.release_graph()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    raise SystemExit((verify_stage if args.verify else run_stage)(Path(args.out).resolve(), args.task))
