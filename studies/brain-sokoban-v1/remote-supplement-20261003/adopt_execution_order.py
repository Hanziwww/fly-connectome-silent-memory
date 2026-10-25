"""One-time change to independent-condition scheduling, without worker interruption."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import json

root = Path('/root/autodl-tmp/brain-sokoban-v1-20261003')
session = root / 'studies/brain-sokoban-v1/remote-supplement-20261003'
sys.path.insert(0, str(session))
import controller_v2 as previous
runner = previous.runner
plan = runner.prepare()
assert not (session / 'execution-order-amendment.json').exists()
prior = runner.read(session / 'supervisor.json')
pid = prior['supervisor_pid']
command = Path(f'/proc/{pid}/cmdline').read_bytes().decode()
assert str(session / 'controller_v2.py') in command and '\x00run\x00' in command
assert not prior.get('failed')
change = {j['id']: dict(original_dependencies=j['dependencies'], effective_dependencies=['supp-preflight'])
          for j in plan['jobs'] if j['id'].startswith('supp-origin-')}
assert len(change) == 8
runner.write(session / 'execution-order-amendment.json', dict(recorded_unix=time.time(),
    reason='Use free GPU lanes for independent preregistered origin/structure arms; none consumes primary-trained checkpoints',
    jobs=change, original_plan_sha256=runner.digest(session / 'plan.json'), training_sha256=plan['code_sha256'],
    controller_sha256=runner.digest(session / 'controller_v3.py'), methods_or_acceptance_changed=False,
    added_conditions=0, training_restarted=False, old_controller_pid=pid))
runner.write(session / 'execution-order-prior-supervisor.json', prior)
runner.write(session / 'compatibility-adoption-v2.json', runner.read(session / 'compatibility-adoption.json'))
os.kill(pid, signal.SIGTERM)
for _ in range(50):
    if not previous.alive(pid):
        break
    time.sleep(.1)
assert not previous.alive(pid)
log = (session / 'controller.log').open('a', encoding='utf-8')
command = [str(root / '.venv/bin/python'), '-B', '-X', 'utf8', '-u', str(session / 'controller_v3.py'), 'run']
process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
    start_new_session=True, env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4'))
runner.write(session / 'launch.json', dict(pid=process.pid, started_unix=time.time(), command=command))
print(json.dumps(dict(controller_pid=process.pid, continuing_worker_pids=[v['pid'] for v in prior['active'].values()],
                     scheduling_only=True, independent_arms=8)))
