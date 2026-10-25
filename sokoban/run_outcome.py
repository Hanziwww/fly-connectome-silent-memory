"""Bounded P01 outcome-first experiment with a paired CUDA population evaluator."""
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
from boundary_worker import initialize, stop_check
from common import configure, cpu_state, file_hash, save_json, tensor_hash
from continual import evaluate
from outcome_training import fit
from progressive_worker import integrity as prior_integrity, same_rollouts
from run_boundary import supervise
from run_cycle import acquire_worker_lock

PREVIOUS = ARCHIVE / 'progressive/20261001-single-seed01'


def prepare(out, seconds):
    out.mkdir(parents=True, exist_ok=False)
    source = PREVIOUS / 'stages/P00/model.pt'
    prior_integrity(PREVIOUS)
    shutil.copyfile(source, out / 'source.pt')
    level = next(l for l in json.loads((PREVIOUS / 'levels.json').read_text()) if l['id'] == 'P01')
    save_json(out / 'level.json', level)
    config = dict(experiment='outcome_counterexamples', seed=10, task_id='P01',
        training_seconds=seconds, population_gpu=True, candidates_per_pair_group=8,
        source_checkpoint=str(source.relative_to(ROOT)), source_checkpoint_sha256=file_hash(source),
        success='Any autonomous solution within the operational 512-step bound; no shortest-path preference',
        failure_feedback='Only witnessed irreversible dead ends or closed deterministic policy cycles',
        training_labels='Any action preserving solvability; any viable exit from an actual cycle',
        shortest_path_actions_used=False, shortest_distances_used_for_scoring=False,
        old_task_retention_constraint=False, old_task_retention_gate=False,
        frozen_readout=True, graph_unchanged=True, correction_enabled=False,
        candidate_evaluation='All candidates execute their own policy; distinct cores batched through CUDA',
        search='D block directions/radii with counterexample gradients; greedy committed updates; no temporary branch search',
        auxiliary_order=['satisfied counterexample clauses', 'worst clause margin', 'focused clause loss'],
        protocol_note='Changed objective, feedback collection and candidate evaluation; this is not an isolated one-factor ablation',
        task_sha256=file_hash(out / 'level.json'), no_automatic_retry=True)
    save_json(out / 'config.json', config)
    files = ['outcome_objective.py', 'outcome_training.py', 'population_gpu.py', 'run_outcome.py']
    save_json(out / 'source-files.json', {str((HERE / name).relative_to(ROOT)): file_hash(HERE / name) for name in files})


def integrity(out):
    config = json.loads((out / 'config.json').read_text())
    assert file_hash(out / 'source.pt') == config['source_checkpoint_sha256']
    assert file_hash(ROOT / config['source_checkpoint']) == config['source_checkpoint_sha256']
    assert file_hash(out / 'level.json') == config['task_sha256']
    for path, digest in json.loads((out / 'source-files.json').read_text()).items():
        assert file_hash(ROOT / path) == digest, path
    prior_integrity(PREVIOUS)
    return True


def worker(out, verify=False):
    integrity(out)
    config = json.loads((out / 'config.json').read_text())
    level = json.loads((out / 'level.json').read_text())
    configure(config['seed'])
    encoder, head, source, resources = initialize(out)
    check = lambda: stop_check(out)
    started = time.monotonic()
    result = dict(status='initializing', pid=os.getpid(), resources=resources)
    try:
        if verify:
            old = json.loads((out / 'result.json').read_text())
            assert old['pid'] != os.getpid()
            assert file_hash(out / 'model.pt') == old['model_sha256']
            saved = torch.load(out / 'model.pt', weights_only=True, map_location='cpu')
            assert tensor_hash(saved['head']) == tensor_hash(source['head'])
            encoder.set_core(saved['core'])
            now = evaluate(encoder, head, [level], 'independent', check)
            assert same_rollouts(old['after'], now)
            assert encoder.graph_digest() == encoder.base_digest
            assert integrity(out)
            save_json(out / 'verification.json', dict(status='verified', pid=os.getpid(), training_pid=old['pid'],
                solved=bool(now['solved']), actions_and_logits_identical=True, frozen_readout=True,
                graph_unchanged=True, model_sha256=old['model_sha256'],
                steps=now['records'][0]['steps'], rollout_status=now['records'][0]['status']))
            print(json.dumps(dict(event='independent_verified', solved=bool(now['solved']))), flush=True)
            return 0
        if (out / 'result.json').exists() or (out / 'history.jsonl').exists():
            raise RuntimeError('Attempt already exists; no automatic retry')
        result['before'] = evaluate(encoder, head, [level], 'baseline', check)
        print(json.dumps(dict(event='baseline', status=result['before']['records'][0]['status'])), flush=True)
        result['training'] = fit(encoder, head, level, config, out, result['before'])
        saved = torch.load(out / 'model.pt', weights_only=True, map_location='cpu')
        encoder.set_core(saved['core'])
        result['after'] = evaluate(encoder, head, [level], 'uncached-reloaded', check)
        assert result['after']['complete']
        if result['training']['stop'] == 'autonomous_success':
            assert result['after']['solved'], 'Population success did not reproduce in ordinary inference'
        result['checks'] = dict(output_bitwise_frozen=tensor_hash(cpu_state(head)) == tensor_hash(source['head']),
            graph_unchanged=encoder.graph_digest() == encoder.base_digest, bounds=encoder.core.bound_audit(),
            old_files_unchanged=integrity(out))
        assert result['checks']['output_bitwise_frozen'] and result['checks']['graph_unchanged']
        assert result['checks']['bounds']['in_bounds']
        result['model_sha256'] = file_hash(out / 'model.pt')
        result['status'] = 'learned' if result['after']['solved'] else result['training']['stop']
    except KeyboardInterrupt as error:
        result.update(status='stopped', error=str(error))
    except Exception as error:
        if verify:
            raise
        result.update(status='error', error=repr(error), traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    finally:
        encoder.release_graph()
        if not verify:
            result['wall_seconds'] = time.monotonic() - started
            result['resources'].update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                                       peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
            save_json(out / 'result.json', result)
            save_json(out / 'progress.json', dict(status=result['status']))
            print(json.dumps(dict(event='finished', status=result['status'])), flush=True)
    return 0 if result['status'] in ('learned', 'training_budget') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ARCHIVE / 'outcome/20261001-success01'))
    parser.add_argument('--seconds', type=float, default=600.)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if args.worker or args.verify:
        return worker(out, args.verify)
    if not 5 <= args.seconds <= 600:
        parser.error('Budget must be 5..600 seconds')
    prepare(out, args.seconds)
    lock = acquire_worker_lock()
    identity = dict(supervisor_pid=os.getpid(), status='starting')
    try:
        command = [sys.executable, '-B', '-X', 'utf8', '-u', str(HERE / 'run_outcome.py'), '--out', str(out)]
        code, state = supervise(command + ['--worker'], out / 'worker.log', out, args.seconds + 180, identity)
        if code or state != 'running':
            identity['status'] = 'error' if state == 'running' else state
            return 1
        identity['phase'] = 'independent-verification'
        code, state = supervise(command + ['--verify'], out / 'verification.log', out, 180, identity)
        identity['status'] = 'complete' if code == 0 and state == 'running' else 'error'
        return code
    finally:
        save_json(out / 'supervisor.json', identity)
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
