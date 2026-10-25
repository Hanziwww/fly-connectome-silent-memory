"""One-time adoption of live training; does not signal any GPU worker."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path('/root/autodl-tmp/brain-sokoban-v1-20261003')
session = root / 'studies/brain-sokoban-v1/remote-supplement-20261003'
sys.path.insert(0, str(session))
import run_remote as runner
plan = runner.prepare()
assert not (session / 'compatibility-repair.json').exists(), 'One-time adoption already recorded'
prior = runner.read(session / 'supervisor.json')
pid = prior['supervisor_pid']
command = Path(f'/proc/{pid}/cmdline').read_bytes().decode()
assert str(session / 'run_remote.py') in command and '\x00run\x00' in command
assert not prior.get('failed'), 'Diagnose existing technical failures before controller adoption'
runner.write(session / 'compatibility-repair.json', dict(
    recorded_unix=time.time(), old_controller_pid=pid,
    reason='New rollout metadata duplicates legacy intervention keyword during independent verification',
    evidence='dict(intervention=kind, **row) raises TypeError when row already contains intervention',
    correction='Verification-only adapter removes duplicate metadata; original archive restores it',
    unchanged_training_sha256=plan['code_sha256'], unchanged_base_code_version=plan['base_code_version'],
    unchanged_tasks_sha256=plan['tasks_sha256'],
    adapter_sha256=runner.digest(session / 'verify_compat.py'),
    controller_sha256=runner.digest(session / 'controller_v2.py'),
    training_restarted=False, acceptance_or_model_changed=False,
    cpu_check=runner.read(session / 'compatibility-cpu-check.json')))
runner.write(session / 'compatibility-prior-supervisor.json', prior)
runner.write(session / 'compatibility-prior-launch.json', runner.read(session / 'launch.json'))
os.kill(pid, signal.SIGTERM)
for _ in range(50):
    if not runner.alive(pid):
        break
    time.sleep(.1)
assert not runner.alive(pid), 'Old controller has not exited; no replacement launched'
log = (session / 'controller.log').open('a', encoding='utf-8')
command = [str(root / '.venv/bin/python'), '-B', '-X', 'utf8', '-u', str(session / 'controller_v2.py'), 'run']
process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
    start_new_session=True, env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4'))
runner.write(session / 'launch.json', dict(pid=process.pid, started_unix=time.time(), command=command))
print(json.dumps(dict(controller_pid=process.pid, adopted_worker_pids=[v['pid'] for v in prior['active'].values()],
                     training_restarted=False, unchanged_training_sha256=plan['code_sha256'])))
