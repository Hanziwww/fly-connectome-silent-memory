"""Bounded seed-10 joint retention experiment with independent verification."""
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
from joint_tasks import JointLessons, decorate
from joint_training import fit
from lesson_population import evaluate_many
from lesson_tasks import score
from population_gpu import PopulationGPU, logits
from progressive_worker import same_rollouts
from run_lessons import integrity as prior_integrity, point_score
from run_boundary import supervise
from run_cycle import acquire_worker_lock

PREVIOUS = ARCHIVE / 'lessons/20261001-branch01'


def prepare(out, joint_seconds, extension_seconds):
    prior_integrity(PREVIOUS)
    verification = json.loads((PREVIOUS / 'verification.json').read_text())
    assert verification['status'] == 'verified'
    for stage in ('critical', 'suffix-1', 'suffix-2', 'suffix-4'):
        assert any(s['stage'] == stage and s['exact'] for s in verification['stages'])
    out.mkdir(parents=True, exist_ok=False)
    source, explorer = PREVIOUS / 'critical.pt', PREVIOUS / 'suffix-4.pt'
    shutil.copyfile(source, out / 'source.pt')
    shutil.copyfile(explorer, out / 'explorer.pt')
    shutil.copyfile(PREVIOUS / 'level.json', out / 'level.json')
    config = dict(experiment='joint_retention', seed=10, task_id='P01',
        joint_seconds=joint_seconds, extension_seconds=extension_seconds,
        source_checkpoint=str(source.relative_to(ROOT)), source_checkpoint_sha256=file_hash(source),
        explorer_checkpoint=str(explorer.relative_to(ROOT)), explorer_checkpoint_sha256=file_hash(explorer),
        task_sha256=file_hash(out / 'level.json'),
        expected_source_critical_margin=next(s['critical_margin'] for s in verification['stages'] if s['stage']=='critical'),
        curriculum=list(range(4, 15)), initial_anchors=['critical-action', 'suffix-1', 'suffix-2'],
        promotion='Original-start solution, or current suffix solved with every required behavior retained',
        replacement='Original-start solution, or lower common loss with every required behavior retained',
        proposal='Four measured hard-forward margin-response directions, constrained fit, three real step-size checks',
        branches=4, gpu_antithetic_candidates=8, branch3_initialization='prior verified suffix-4 model',
        explorer_regret='Bounded temporary loss regression; cannot replace a retained champion until behavior recovers',
        success='Any autonomous solution from the original start within 512 moves; no path-length preference',
        old_task_retention_gate=False, frozen_readout=True, graph_unchanged=True, correction_enabled=False,
        protocol_note='Continues retained checkpoints with a combined method change, not an isolated equal-budget ablation',
        no_automatic_retry=True)
    save_json(out / 'config.json', config)
    names = ['joint_tasks.py', 'joint_training.py', 'run_joint.py']
    save_json(out / 'source-files.json', {str((HERE / p).relative_to(ROOT)):file_hash(HERE / p) for p in names})


def integrity(out):
    config = json.loads((out / 'config.json').read_text())
    for filename, key in (('source.pt', 'source_checkpoint'), ('explorer.pt', 'explorer_checkpoint')):
        assert file_hash(out / filename) == config[key+'_sha256']
        assert file_hash(ROOT / config[key]) == config[key+'_sha256']
    assert file_hash(out / 'level.json') == config['task_sha256']
    for path, digest in json.loads((out / 'source-files.json').read_text()).items():
        assert file_hash(ROOT / path) == digest, path
    prior_integrity(PREVIOUS)
    return True


def original_only(result):
    record = result['records'][0]
    success = record['status'] == 'solved'
    return dict(complete=record['status'] != 'superseded_by_solution', solved=int(success),
                total=1, success_rate=float(success), records=[record], seconds=result.get('seconds', 0.))


def gates(result, critical_margin):
    # A solved auxiliary episode must never be mistaken for original-task success.
    return decorate(dict(solved=result['records'][0]['status']=='solved', complete=result['complete'],
        lesson_solved=result['records'][1]['status']=='solved', records=result['records'],
        critical_margin=critical_margin))


def population_parity(encoder, head, source, explorer, level, check):
    lessons, expected = JointLessons(level), []
    data, levels = lessons.data(0), lessons.levels(0)
    for state in (source['core'], explorer['core']):
        encoder.set_core(state)
        features = encoder.encode(data['boards'], check)
        metrics = score(logits(head, features), data)
        actual = evaluate(encoder, head, levels, 'ordinary-parity', check)
        expected.append((metrics, actual))
    engine = PopulationGPU(encoder, head)
    observed = [decorate(o) for o in evaluate_many(engine, [source['core'], explorer['core']], levels, data, check)]
    for obs, (metrics, actual) in zip(observed, expected):
        assert obs['complete']
        assert all(obs[k] == value for k, value in metrics.items())
        assert same_rollouts(dict(complete=True, total=len(levels), records=obs['records']), actual)
        independent = gates(actual, metrics['critical_margin'])
        for key in ('retained', 'joint_passed', 'protected_results', 'solved'):
            assert obs[key] == independent[key]
    assert observed[0]['retained'] and not observed[0]['joint_passed']
    assert observed[1]['lesson_solved'] and not observed[1]['retained'] and not observed[1]['joint_passed']
    encoder.set_core(source['core'])
    return dict(passed=True, distinct_cores=2, starts_per_core=len(levels),
        scores_actions_logits_gates_exact=True, suffix4_without_critical_rejected=True)


def independent_stages(out, encoder, head, source, level, check):
    lessons, results = JointLessons(level), []
    for row in json.loads((out / 'stages.json').read_text()):
        if row['status'] not in ('joint_passed', 'original_solved'):
            continue
        saved = torch.load(out / f"joint-{row['depth']}.pt", weights_only=True, map_location='cpu')
        assert tensor_hash(saved['head']) == tensor_hash(source['head'])
        encoder.set_core(saved['core'])
        actual = evaluate(encoder, head, lessons.levels(lessons.depths.index(row['depth'])), 'stage-independent', check)
        critical = point_score(encoder, head, level, check)['critical_margin']
        observed = gates(actual, critical)
        if row['status'] == 'original_solved':
            assert same_rollouts(original_only(dict(records=row['records'])), original_only(actual))
        else:
            assert same_rollouts(dict(complete=True, total=len(row['records']), records=row['records']), actual)
        assert observed['joint_passed']
        assert critical == row['final']['critical_margin']
        results.append(dict(depth=row['depth'], joint_passed=True, retained=observed['retained'],
                            solved=observed['solved'], critical_margin=critical, actions_logits_exact=True))
    return results


def worker(out, verify=False):
    integrity(out)
    config = json.loads((out / 'config.json').read_text())
    level = json.loads((out / 'level.json').read_text())
    configure(config['seed'])
    encoder, head, source, resources = initialize(out)
    explorer = torch.load(out / 'explorer.pt', weights_only=True, map_location='cpu')
    assert tensor_hash(explorer['head']) == tensor_hash(source['head'])
    check = lambda: stop_check(out)
    started, result = time.monotonic(), dict(status='initializing', pid=os.getpid(), resources=resources)
    try:
        if verify:
            old = json.loads((out / 'result.json').read_text())
            assert old['pid'] != os.getpid()
            stages = independent_stages(out, encoder, head, source, level, check)
            assert file_hash(out / 'model.pt') == old['model_sha256']
            saved = torch.load(out / 'model.pt', weights_only=True, map_location='cpu')
            assert tensor_hash(saved['head']) == tensor_hash(source['head'])
            encoder.set_core(saved['core'])
            lessons = JointLessons(level)
            levels = lessons.levels(lessons.depths.index(old['training']['stages'][-1]['depth']))
            now = evaluate(encoder, head, levels, 'independent', check)
            assert same_rollouts(old['after_suite'], now)
            critical = point_score(encoder, head, level, check)['critical_margin']
            observed = gates(now, critical)
            assert observed['retained'] or observed['solved']
            assert critical == old['critical_after']['critical_margin']
            assert encoder.graph_digest() == encoder.base_digest and integrity(out)
            save_json(out / 'verification.json', dict(status='verified', pid=os.getpid(), training_pid=old['pid'],
                solved=observed['solved'], retained=observed['retained'], joint_passed=observed['joint_passed'],
                critical_margin=critical, protected_results=observed['protected_results'],
                actions_and_logits_identical=True, stages=stages, frozen_readout=True, graph_unchanged=True,
                model_sha256=old['model_sha256'], rollout_status=now['records'][0]['status'], steps=now['records'][0]['steps']))
            print(json.dumps(dict(event='independent_verified', solved=observed['solved'], retained=observed['retained'])), flush=True)
            return 0
        if (out / 'result.json').exists() or (out / 'history.jsonl').exists():
            raise RuntimeError('Attempt exists; do not overwrite or automatically retry')
        lessons = JointLessons(level)
        result['before_suite'] = evaluate(encoder, head, lessons.levels(0), 'baseline', check)
        result['before'] = original_only(result['before_suite'])
        result['critical_before'] = point_score(encoder, head, level, check)
        assert result['critical_before']['critical_margin'] == config['expected_source_critical_margin']
        assert gates(result['before_suite'], result['critical_before']['critical_margin'])['retained']
        result['population_parity'] = population_parity(encoder, head, source, explorer, level, check)
        print(json.dumps(dict(event='baseline', critical_margin=result['critical_before']['critical_margin'],
                              original_status=result['before']['records'][0]['status'])), flush=True)
        result['training'] = fit(encoder, head, level, config, out)
        saved = torch.load(out / 'model.pt', weights_only=True, map_location='cpu')
        encoder.set_core(saved['core'])
        depth = result['training']['stages'][-1]['depth']
        levels = lessons.levels(lessons.depths.index(depth))
        result['after_suite'] = evaluate(encoder, head, levels, 'uncached-reloaded', check)
        result['after'] = original_only(result['after_suite'])
        result['critical_after'] = point_score(encoder, head, level, check)
        observed = gates(result['after_suite'], result['critical_after']['critical_margin'])
        assert observed['retained'] or observed['solved']
        if result['training']['stop'] == 'autonomous_success':
            assert observed['solved']
        history = [json.loads(line) for line in (out / 'history.jsonl').read_text().splitlines()] if (out / 'history.jsonl').exists() else []
        assert all(row['retained'] or row['solved'] for row in history)
        result['checks'] = dict(output_bitwise_frozen=tensor_hash(cpu_state(head)) == tensor_hash(source['head']),
            graph_unchanged=encoder.graph_digest()==encoder.base_digest, bounds=encoder.core.bound_audit(),
            final_retained=observed['retained'], protected_results=observed['protected_results'],
            no_retention_failure_in_committed_history=True, old_files_unchanged=integrity(out))
        assert result['checks']['output_bitwise_frozen'] and result['checks']['graph_unchanged']
        assert result['checks']['bounds']['in_bounds']
        result['model_sha256'] = file_hash(out / 'model.pt')
        result['status'] = 'learned' if observed['solved'] else result['training']['stop']
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
            result['wall_seconds'] = time.monotonic()-started
            result['resources'].update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            save_json(out / 'result.json', result)
            save_json(out / 'progress.json', dict(status=result['status']))
            print(json.dumps(dict(event='finished', status=result['status'])), flush=True)
    return 0 if result['status'] in ('learned', 'joint_budget', 'extension_budget') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ARCHIVE / 'joint/20261001-retention01'))
    parser.add_argument('--joint-seconds', type=float, default=600.)
    parser.add_argument('--extension-seconds', type=float, default=600.)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if args.worker or args.verify:
        return worker(out, args.verify)
    if not 5 <= args.joint_seconds <= 600 or not 5 <= args.extension_seconds <= 600:
        parser.error('Each budget must be 5..600 seconds')
    prepare(out, args.joint_seconds, args.extension_seconds)
    lock, identity = acquire_worker_lock(), dict(supervisor_pid=os.getpid(), status='starting')
    try:
        command = [sys.executable, '-B', '-X', 'utf8', '-u', str(HERE / 'run_joint.py'), '--out', str(out)]
        code, state = supervise(command+['--worker'], out / 'worker.log', out,
                                args.joint_seconds+args.extension_seconds+180, identity)
        if code or state != 'running':
            identity['status'] = 'error' if state == 'running' else state
            return 1
        identity['phase'] = 'independent-verification'
        code, state = supervise(command+['--verify'], out / 'verification.log', out, 180, identity)
        identity['status'] = 'complete' if code == 0 and state == 'running' else 'error'
        return code
    finally:
        save_json(out / 'supervisor.json', identity)
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
