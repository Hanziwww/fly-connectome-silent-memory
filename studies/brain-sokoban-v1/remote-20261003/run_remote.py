"""Four independent GPU lanes for the explicitly selected follow-up experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

SESSION = Path(__file__).resolve().parent
STUDY = SESSION.parent
ROOT = STUDY.parents[1]
for directory in (ROOT / 'flybrain', ROOT / 'sokoban', STUDY / 'code'):
    sys.path.insert(0, str(directory))
from registry import read, write, digest, provenance, condition, SOURCE

GOOD = {'learned_solved', 'already_solved', 'stagnation', 'segment_limit',
        'frozen_baseline', 'untrained_baseline'}


def code_digest():
    return hashlib.sha256(''.join(digest(SESSION / f) for f in
                                 ('run_remote.py', 'memory_adapter.py')).encode()).hexdigest()


def prepare():
    if (SESSION / 'plan.json').exists():
        plan = read(SESSION / 'plan.json')
        assert plan['remote_code_sha256'] == code_digest(), 'Remote code changed after registration'
        assert plan['base_code_version'] == provenance(), 'Base method changed after registration'
        assert plan['source_sha256'] == digest(SOURCE), 'P05 source changed after registration'
        return plan
    base_version = provenance()
    source_hash = digest(SOURCE)
    jobs = []
    def add(name, mode, dependencies=(), **extra):
        jobs.append(dict(id=name, mode=mode, dependencies=list(dependencies), **extra))
    for tier in (1, 3, 5, 7):
        add(f'import-D{tier}', 'import', reference=f'difficulty-D{tier}-T0')
    for tier in (1, 3, 5, 7):
        add(f'generalization-D{tier}', 'evaluate', [f'import-D{tier}'],
            reference=f'difficulty-D{tier}-T0')
    for arm in ('head_only', 'demonstration'):
        add(f'remote-D7-{arm}', 'static', ['import-D7'], task='D7-T0', arm=arm)
    add('memory-history-preflight', 'smoke', ['import-D1'])
    for arm in ('gru', 'carry_untrained', 'reset', 'head_only', 'joint'):
        add(f'remote-dynamic-M0-{arm}', 'memory',
            ['import-D1'] if arm == 'gru' else ['memory-history-preflight'], task='M0', arm=arm)
    for job in jobs:
        out = STUDY / 'runs' / job['id'] if job['mode'] in ('static', 'memory') else SESSION / 'checks' / job['id']
        out.mkdir(parents=True, exist_ok=True)
        job['out'] = str(out.relative_to(ROOT))
        if job['mode'] in ('static', 'memory', 'smoke'):
            config = dict(id=job['id'], task=job.get('task', 'M0'), seed=10,
                          parent=None, source=str(SOURCE), source_sha256=source_hash,
                          precision='mixed', batch=16 if job['mode']=='static' else 1,
                          segments_seconds=[1200, 2400, 4800], code_version=base_version,
                          task_lock_sha256=digest(STUDY / 'tasks/lock.json'),
                          protocol_sha256=digest(STUDY / 'protocol.json'),
                          remote_code_sha256=code_digest(), remote_session=SESSION.name)
            if job['mode']=='static':
                config.update(module='controls', condition=condition(job['arm']))
            else:
                arm = job.get('arm', 'joint')
                config.update(module='dynamic_memory', condition=arm,
                              family_id='M0', state_mode='reset' if arm=='reset' else 'persistent',
                              task_sha256=digest(STUDY / 'tasks/memory.json'),
                              feature_kind='gru_hidden_28' if arm=='gru' else 'brain_activity_512',
                              internal_step_ms=20 if arm=='gru' else 1, training_batch=1,
                              evaluation_batching='at most 8 complete histories; no history truncation')
            write(out / 'config.json', config)
    plan = dict(created_unix=time.time(), seed=10, base_code_version=base_version,
                remote_code_sha256=code_digest(), local_training_remains_paused=True,
                source_sha256=source_hash, allocator_gib_per_worker=11,
                gpu_lanes=[0, 1, 2, 3], task_selection='predeclared D1/D3/D5/D7 T0 and first memory layout M0',
                no_cross_hardware_training_time_comparison=True, jobs=jobs)
    write(SESSION / 'plan.json', plan)
    return plan


def imported(job):
    import torch
    from backend import initialize, set_pair
    from behavior import rollouts
    from execution import tasks, stopped
    old = STUDY / 'runs' / job['reference']
    out = ROOT / job['out']
    config = read(old / 'config.json')
    encoder, head = initialize(config, out)
    saved = torch.load(old / 'model.pt', map_location='cpu', weights_only=True)
    set_pair(encoder, head, saved)
    observation = rollouts(encoder, head, [tasks()[config['task']]], lambda: stopped(out))['records'][0]
    write(out / 'result.json', dict(status='passed' if observation['status']=='solved' else 'reproduction_failed',
          pid=os.getpid(), independent_process=True, original_solved=observation['status']=='solved',
          original_steps=observation['steps'], imported_model_sha256=digest(old / 'model.pt'),
          origin='previously independently verified local checkpoint', rollout=observation))
    return 0 if observation['status']=='solved' else 1


def worker(job, verify=False):
    import fcntl
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    lock = (SESSION / f'gpu-{gpu}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    out = ROOT / job['out']
    try:
        if (STUDY / 'STOP').exists() or (SESSION / 'STOP').exists():
            raise KeyboardInterrupt('Remote study STOP requested')
        if job['mode']=='import':
            return imported(job)
        if job['mode']=='evaluate':
            from evaluation import evaluate_one
            evaluate_one(job['reference'])
            observations = read(STUDY / 'runs' / job['reference'] / 'generalization.json')
            held = [r for r in observations['records'] if r['split']=='heldout']
            assert len(held)==8
            write(out / 'result.json', dict(status='completed', pid=os.getpid(),
                  heldout_solved=sum(r['status']=='solved' for r in held), heldout_total=len(held),
                  reference=job['reference'], data_file=str((STUDY / 'runs' / job['reference'] / 'generalization.json').relative_to(ROOT))))
            return 0
        if job['mode'] in ('memory', 'smoke'):
            import memory_adapter
            memory_adapter.install()
            if job['mode']=='smoke':
                return memory_adapter.smoke(read(out / 'config.json'), out)
        from execution import run_worker
        return run_worker(out / 'config.json', verify)
    except BaseException as error:
        torch = sys.modules.get('torch')
        category = ('stopped' if isinstance(error, KeyboardInterrupt) else
                    'oom' if torch is not None and isinstance(error, torch.OutOfMemoryError) else
                    'numerical_error' if isinstance(error, FloatingPointError) else 'implementation_error')
        write(out / ('verification.json' if verify else 'result.json'),
              dict(status=category,
                   pid=os.getpid(), error=repr(error), traceback=traceback.format_exc()))
        traceback.print_exc()
        return 1
    finally:
        lock.close()


def alive(pid):
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
        return str(SESSION / 'run_remote.py') in command
    except (FileNotFoundError, ProcessLookupError):
        return False


def completed(job):
    out = ROOT / job['out']
    if not (out / 'result.json').exists():
        return False
    result = read(out / 'result.json')
    if job['mode'] in ('static', 'memory'):
        return result['status'] in GOOD and (out / 'verification.json').exists() and read(out / 'verification.json')['status']=='verified'
    return result['status'] in ('passed', 'completed')


def run(resume=False):
    import fcntl
    lock = (SESSION / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = prepare(); jobs = plan['jobs']
    live = {}; failed = {}; done = {j['id'] for j in jobs if completed(j)}
    if (SESSION / 'supervisor.json').exists():
        previous = read(SESSION / 'supervisor.json')
        assert not any(alive(v['pid']) for v in previous.get('active', {}).values()), 'Existing study worker is still alive'
    if resume:
        for job in jobs:
            out = ROOT / job['out']
            result_path = out / 'result.json'
            if job['id'] in done or not result_path.exists():
                continue
            result = read(result_path)
            if result['status'] in ('stopped', 'worker_exit') and (out / 'resume.pt').exists():
                config = read(out / 'config.json')
                assert config['code_version']==provenance() and config['remote_code_sha256']==code_digest()
                result_path.rename(out / f'interruption-{int(time.time())}.json')
    try:
        while True:
            stopping = (STUDY / 'STOP').exists() or (SESSION / 'STOP').exists()
            if stopping and not (STUDY / 'STOP').exists():
                (STUDY / 'STOP').write_text('Remote controller stop requested\n')
            for lane, item in list(live.items()):
                process, job, stage, log = item
                code = process.poll()
                if code is None:
                    continue
                log.close(); del live[lane]
                out = ROOT / job['out']
                if stage=='train' and code==0 and job['mode'] in ('static', 'memory'):
                    launch(live, lane, job, 'verify')
                elif code==0 and completed(job):
                    done.add(job['id'])
                else:
                    failed[job['id']] = dict(stage=stage, exit_code=code)
                    if not (out / 'result.json').exists():
                        write(out / 'result.json', dict(status='worker_exit', exit_code=code,
                              resumable=(out / 'resume.pt').exists()))
            if not stopping:
                for lane in plan['gpu_lanes']:
                    if lane in live:
                        continue
                    busy = {item[1]['id'] for item in live.values()}
                    for job in jobs:
                        if job['id'] in done | set(failed) | busy:
                            continue
                        out = ROOT / job['out']
                        if (out / 'result.json').exists() and job['mode'] in ('static','memory'):
                            if read(out / 'result.json')['status'] in GOOD and not (out / 'verification.json').exists():
                                launch(live, lane, job, 'verify'); break
                            failed[job['id']] = dict(stage='prior_attempt', reason='Preserved result requires diagnosis')
                            continue
                        if (out / 'result.json').exists():
                            failed[job['id']] = dict(stage='prior_attempt', reason='Preserved check requires diagnosis')
                            continue
                        if all(dependency in done for dependency in job['dependencies']):
                            launch(live, lane, job, 'train'); break
            state = 'stopping' if stopping and live else 'stopped' if stopping else 'running' if live else 'complete' if len(done)==len(jobs) else 'needs_attention'
            write(SESSION / 'supervisor.json', dict(status=state, supervisor_pid=os.getpid(),
                  updated_unix=time.time(), completed=sorted(done), failed=failed,
                  pending=[j['id'] for j in jobs if j['id'] not in done | set(failed) | {x[1]['id'] for x in live.values()}],
                  active={str(lane):dict(job=item[1]['id'], pid=item[0].pid, stage=item[2]) for lane,item in live.items()}))
            if not live:
                return 0 if state in ('complete', 'stopped') else 1
            time.sleep(2)
    finally:
        lock.close()


def launch(live, lane, job, stage):
    out = ROOT / job['out']
    log = (out / ('verification.log' if stage=='verify' else 'worker.log')).open('a', encoding='utf-8')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(lane), OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    command = [sys.executable, '-B', '-X', 'utf8', '-u', str(SESSION / 'run_remote.py'),
               'verify' if stage=='verify' else 'worker', '--job', job['id']]
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    write(out / f'{stage}-process.json', dict(pid=process.pid, gpu=lane, started_unix=time.time(), command=command))
    live[lane] = (process, job, stage, log)


def status():
    plan = read(SESSION / 'plan.json')
    state = read(SESSION / 'supervisor.json') if (SESSION / 'supervisor.json').exists() else {}
    state['supervisor_alive'] = alive(state.get('supervisor_pid', 0))
    rows = []
    for job in plan['jobs']:
        out = ROOT / job['out']; row = dict(id=job['id'], mode=job['mode'])
        for name in ('result', 'progress', 'verification'):
            p = out / f'{name}.json'
            if p.exists():
                value = read(p)
                row[name] = {k:value[k] for k in ('status','updates','depth','elapsed','training_seconds',
                     'final_solved','original_solved','original_steps','heldout_solved','heldout_total',
                     'solved','total','probe_correct','probe_total','resources','error') if k in value}
                row[name + '_age_seconds'] = round(time.time()-p.stat().st_mtime, 1)
        for v in state.get('active', {}).values():
            if v['job']==job['id']:
                row.update(pid=v['pid'], process_alive=alive(v['pid']), stage=v['stage'])
        for name in ('worker.log', 'verification.log', 'resume.pt'):
            p = out / name
            if p.exists(): row[name] = dict(bytes=p.stat().st_size, age_seconds=round(time.time()-p.stat().st_mtime,1))
        rows.append(row)
    print(json.dumps(dict(supervisor=state, jobs=rows), ensure_ascii=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','run','worker','verify','status'])
    parser.add_argument('--job'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command=='prepare':
        print(json.dumps(prepare(), ensure_ascii=False))
    elif args.command=='run':
        raise SystemExit(run(args.resume))
    elif args.command=='status':
        status()
    else:
        job = next(j for j in read(SESSION / 'plan.json')['jobs'] if j['id']==args.job)
        raise SystemExit(worker(job, args.command=='verify'))
