"""Adopt live workers and use the recorded verification metadata adapter."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import run_remote as runner


def alive(pid, session=runner.SESSION):
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
        names = ('run_remote.py', 'controller_v2.py', 'verify_compat.py') if session == runner.SESSION else ('run_remote.py',)
        return any(str(session / name) in command for name in names)
    except (OSError, ValueError):
        return False


runner.alive = alive


class AdoptedProcess:
    def __init__(self, pid, job, stage):
        self.pid, self.job, self.stage = pid, job, stage

    def poll(self):
        if alive(self.pid):
            command = Path(f'/proc/{self.pid}/cmdline').read_bytes().decode(errors='replace')
            if self.job['id'] in command:
                return None
        path = runner.ROOT / self.job['out'] / ('verification.json' if self.stage == 'verify' else 'result.json')
        if not path.exists():
            return 1
        status = runner.read(path).get('status')
        return 0 if status in runner.GOOD | {'verified', 'passed', 'completed'} else 1


def launch(live, lane, job, stage):
    if stage != 'verify':
        return runner.launch(live, lane, job, stage)
    out = runner.ROOT / job['out']
    log = (out / 'verification.log').open('a', encoding='utf-8')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(lane), OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED= '1')
    command = [sys.executable, '-B', '-X', 'utf8', '-u', str(runner.SESSION / 'verify_compat.py'), '--job', job['id']]
    process = subprocess.Popen(command, cwd=runner.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    runner.write(out / 'verify-process.json', dict(pid=process.pid, gpu=lane, started_unix=time.time(), command=command))
    live[lane] = (process, job, stage, log)


def run():
    import fcntl
    lock = (runner.SESSION / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = runner.prepare()
    receipt = runner.read(runner.SESSION / 'compatibility-repair.json')
    assert receipt['controller_sha256'] == runner.digest(Path(__file__))
    assert receipt['adapter_sha256'] == runner.digest(runner.SESSION / 'verify_compat.py')
    jobs = plan['jobs']; by_id = {j['id']: j for j in jobs}
    prior = runner.read(runner.SESSION / 'supervisor.json')
    assert not alive(prior.get('supervisor_pid', 0)), 'Prior controller still alive'
    live = {}; failed = dict(prior.get('failed', {}))
    done = {j['id'] for j in jobs if runner.completed(j)}
    for lane, item in prior.get('active', {}).items():
        job = by_id[item['job']]
        out = runner.ROOT / job['out']
        log = (out / ('verification.log' if item['stage'] == 'verify' else 'worker.log')).open('a', encoding='utf-8')
        live[int(lane)] = (AdoptedProcess(item['pid'], job, item['stage']), job, item['stage'], log)
    runner.write(runner.SESSION / 'compatibility-adoption.json', dict(
        adopted_unix=time.time(), controller_pid=os.getpid(), active=prior.get('active', {}),
        training_restarted=False, optimizer_or_weights_changed=False))
    try:
        while True:
            stopping = (runner.SESSION / 'STOP').exists() or (runner.STUDY / 'STOP').exists()
            for lane, (process, job, stage, log) in list(live.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close(); del live[lane]
                if stage == 'train' and code == 0 and job['mode'] == 'train':
                    launch(live, lane, job, 'verify')
                elif code == 0 and runner.completed(job):
                    done.add(job['id'])
                else:
                    failed[job['id']] = dict(stage=stage, exit_code=code)
            if not stopping:
                for lane in plan['gpu_lanes']:
                    if lane in live or not runner.lane_free(lane):
                        continue
                    busy = {v[1]['id'] for v in live.values()}
                    for job in jobs:
                        if job['id'] in done | set(failed) | busy:
                            continue
                        if any(d in failed for d in job['dependencies']):
                            failed[job['id']] = dict(stage='dependency', reason='Upstream technical failure'); continue
                        if not all(d in done for d in job['dependencies']):
                            continue
                        out = runner.ROOT / job['out']
                        if (out / 'result.json').exists():
                            result = runner.read(out / 'result.json')
                            if job['mode'] == 'train' and result['status'] in runner.GOOD and not (out / 'verification.json').exists():
                                launch(live, lane, job, 'verify'); break
                            failed[job['id']] = dict(stage='prior_attempt', reason='Preserved result requires diagnosis'); continue
                        launch(live, lane, job, 'train'); break
            pending = [j['id'] for j in jobs if j['id'] not in done | set(failed) | {v[1]['id'] for v in live.values()}]
            status = 'stopping' if stopping and live else 'stopped' if stopping else 'running' if live else 'complete' if len(done) == len(jobs) else 'waiting_gpu' if pending else 'needs_attention'
            runner.write(runner.SESSION / 'supervisor.json', dict(status=status, supervisor_pid=os.getpid(),
                updated_unix=time.time(), completed=sorted(done), failed=failed, pending=pending,
                controller_version='verification-metadata-adapter-v1',
                active={str(k): dict(job=v[1]['id'], pid=v[0].pid, stage=v[2]) for k, v in live.items()}))
            if not live and (stopping or not pending):
                return 0 if status in ('complete', 'stopped') else 1
            time.sleep(3)
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['run', 'status'])
    if parser.parse_args().command == 'status':
        import json
        print(json.dumps(runner.status()))
    else:
        raise SystemExit(run())
