"""Continue seed-10 P01 with a bounded point probe and reverse lesson training."""
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
from common import configure, cpu_state, file_hash, save_json, tensor_hash
from continual import evaluate
from lesson_tasks import Lessons, score
from lesson_training import fit
from lesson_population import evaluate_many
from population_gpu import PopulationGPU, logits
from progressive_worker import same_rollouts
from run_outcome import integrity as source_integrity
from run_boundary import supervise
from run_cycle import acquire_worker_lock

SOURCE = ARCHIVE / 'outcome/20261001-success01'


def prepare(out, critical_seconds, curriculum_seconds):
    source_integrity(SOURCE)
    verified = json.loads((SOURCE / 'verification.json').read_text())
    source = SOURCE / 'model.pt'
    assert verified['status'] == 'verified' and verified['model_sha256'] == file_hash(source)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, out / 'source.pt')
    shutil.copyfile(SOURCE / 'level.json', out / 'level.json')
    config = dict(experiment='reverse_lessons', seed=10, task_id='P01',
        critical_seconds=critical_seconds, curriculum_seconds=curriculum_seconds,
        source_checkpoint=str(source.relative_to(ROOT)), source_checkpoint_sha256=file_hash(source),
        source_training_seconds=600.141, source_status='verified_unsolved',
        continuous_parameter_inheritance=True, branches=4, gpu_candidates=8, simulation_batch=16,
        success='Any autonomous solution from the original start within 512 moves',
        shortest_path_gate=False, old_task_retention_gate=False,
        teacher='Existing valid solution witness; randomized acyclic completion on other solvable states',
        teacher_usage='Positive training examples and restart positions only; never inference actions',
        score='The same smooth margin penalty for direction estimates and candidate acceptance',
        explorations='Three persistent branches can temporarily regress; a separate best state is protected',
        frozen_readout=True, graph_unchanged=True, correction_enabled=False,
        protocol_note='Sequential continuation and a combined method change, not a matched-budget isolated ablation',
        no_automatic_retry=True, task_sha256=file_hash(out / 'level.json'))
    save_json(out / 'config.json', config)
    files = ['lesson_tasks.py', 'lesson_population.py', 'lesson_training.py', 'run_lessons.py']
    save_json(out / 'source-files.json', {str((HERE / p).relative_to(ROOT)): file_hash(HERE / p) for p in files})


def integrity(out):
    config = json.loads((out / 'config.json').read_text())
    assert file_hash(out / 'source.pt') == config['source_checkpoint_sha256']
    assert file_hash(ROOT / config['source_checkpoint']) == config['source_checkpoint_sha256']
    assert file_hash(out / 'level.json') == config['task_sha256']
    for path, digest in json.loads((out / 'source-files.json').read_text()).items():
        assert file_hash(ROOT / path) == digest, path
    source_integrity(SOURCE)
    return True


def point_score(encoder, head, level, check):
    data = Lessons(level).data(-1)
    features = encoder.encode(data['boards'], check)
    return score(logits(head, features), data)


def population_parity(encoder, head, level, check):
    """Compare distinct cores and both starts with uncached ordinary inference."""
    source = cpu_state(encoder.core)
    other = {k: v.clone() for k, v in source.items()}
    other['pre_log_gain'][:16] = (other['pre_log_gain'][:16] + .0003).clamp(
        float(np.log(.8)), float(np.log(1.25)))
    lessons = Lessons(level)
    levels, data = [level, lessons.level_for(2)], lessons.data(2)
    expected = []
    for state in (source, other):
        encoder.set_core(state)
        features = encoder.encode(data['boards'], check)
        metrics = score(logits(head, features), data)
        actual = evaluate(encoder, head, levels, 'serial-parity', check)
        expected.append((metrics, actual))
    engine = PopulationGPU(encoder, head)
    observed = evaluate_many(engine, [source, other], levels, data, check)
    for obs, (metrics, actual) in zip(observed, expected):
        assert obs['complete']
        assert all(obs[k] == v for k, v in metrics.items())
        assert same_rollouts(dict(complete=True, total=2, records=obs['records']), actual)
    encoder.set_core(source)
    return dict(passed=True, distinct_cores=2, starts_per_core=2,
                scores_actions_logits_exact=True, ordinary_inference_uncached=True)


def verify_stages(out, encoder, head, source, level, check):
    lessons, verified = Lessons(level), []
    for row in json.loads((out / 'stages.json').read_text()):
        if row['status'] not in ('passed', 'original_solved'):
            continue
        saved = torch.load(out / (row['stage'] + '.pt'), map_location='cpu', weights_only=True)
        assert tensor_hash(saved['head']) == tensor_hash(source['head'])
        encoder.set_core(saved['core'])
        if row['stage'] == 'critical':
            now = point_score(encoder, head, level, check)
            assert now['critical_margin'] == row['final']['critical_margin']
            assert now['critical_margin'] >= .1 or row['status'] == 'original_solved'
            verified.append(dict(stage=row['stage'], critical_margin=now['critical_margin'], exact=True))
        else:
            stage = lessons.depths.index(int(row['stage'].split('-')[1]))
            lesson_level = lessons.level_for(stage)
            now = evaluate(encoder, head, [lesson_level], 'lesson-independent', check)
            old = dict(complete=True, total=1, records=[row['records'][1]])
            # Original success can supersede an unfinished auxiliary rollout.
            if old['records'][0]['status'] != 'superseded_by_solution':
                assert same_rollouts(old, now)
            assert now['solved'] or row['status'] == 'original_solved'
            verified.append(dict(stage=row['stage'], solved=bool(now['solved']),
                                 steps=now['records'][0]['steps'], exact=old['records'][0]['status'] != 'superseded_by_solution'))
    return verified


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
            stage_checks = verify_stages(out, encoder, head, source, level, check)
            assert file_hash(out / 'model.pt') == old['model_sha256']
            saved = torch.load(out / 'model.pt', weights_only=True, map_location='cpu')
            assert tensor_hash(saved['head']) == tensor_hash(source['head'])
            encoder.set_core(saved['core'])
            now = evaluate(encoder, head, [level], 'independent', check)
            assert same_rollouts(old['after'], now)
            assert encoder.graph_digest() == encoder.base_digest and integrity(out)
            save_json(out / 'verification.json', dict(status='verified', pid=os.getpid(), training_pid=old['pid'],
                solved=bool(now['solved']), actions_and_logits_identical=True,
                stages=stage_checks, frozen_readout=True, graph_unchanged=True,
                model_sha256=old['model_sha256'], rollout_status=now['records'][0]['status'], steps=now['records'][0]['steps']))
            print(json.dumps(dict(event='independent_verified', solved=bool(now['solved']))), flush=True)
            return 0
        if (out / 'result.json').exists() or (out / 'history.jsonl').exists():
            raise RuntimeError('Attempt already exists; preserve it and use a new directory')
        result['before'] = evaluate(encoder, head, [level], 'baseline', check)
        result['critical_before'] = point_score(encoder, head, level, check)
        result['population_parity'] = population_parity(encoder, head, level, check)
        print(json.dumps(dict(event='baseline', rollout_status=result['before']['records'][0]['status'],
                              critical_margin=result['critical_before']['critical_margin'])), flush=True)
        result['training'] = fit(encoder, head, level, config, out)
        saved = torch.load(out / 'model.pt', weights_only=True, map_location='cpu')
        encoder.set_core(saved['core'])
        result['after'] = evaluate(encoder, head, [level], 'uncached-reloaded', check)
        result['critical_after'] = point_score(encoder, head, level, check)
        assert result['after']['complete']
        if result['training']['stop'] == 'autonomous_success':
            assert result['after']['solved']
        result['checks'] = dict(output_bitwise_frozen=tensor_hash(cpu_state(head)) == tensor_hash(source['head']),
            graph_unchanged=encoder.graph_digest() == encoder.base_digest,
            bounds=encoder.core.bound_audit(), old_files_unchanged=integrity(out))
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
    return 0 if result['status'] in ('learned', 'critical_budget', 'curriculum_budget') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ARCHIVE / 'lessons/20261001-branch01'))
    parser.add_argument('--critical-seconds', type=float, default=120.)
    parser.add_argument('--curriculum-seconds', type=float, default=600.)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if args.worker or args.verify:
        return worker(out, args.verify)
    if not 5 <= args.critical_seconds <= 180 or not 5 <= args.curriculum_seconds <= 900:
        parser.error('Critical budget must be 5..180 s; curriculum budget must be 5..900 s')
    prepare(out, args.critical_seconds, args.curriculum_seconds)
    lock, identity = acquire_worker_lock(), dict(supervisor_pid=os.getpid(), status='starting')
    try:
        command = [sys.executable, '-B', '-X', 'utf8', '-u', str(HERE / 'run_lessons.py'), '--out', str(out)]
        ceiling = args.critical_seconds + args.curriculum_seconds + 180
        code, state = supervise(command + ['--worker'], out / 'worker.log', out, ceiling, identity)
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
