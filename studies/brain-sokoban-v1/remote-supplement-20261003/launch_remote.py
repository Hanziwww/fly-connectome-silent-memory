"""Run through the authorized SSH transport using the remote virtual environment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('/root/autodl-tmp/brain-sokoban-v1-20261003')
session=root/'studies/brain-sokoban-v1/remote-supplement-20261003'
sys.path.insert(0,str(session))
import run_remote
plan=run_remote.prepare()
if (session/'supervisor.json').exists():
    prior=run_remote.read(session/'supervisor.json')
    assert not run_remote.alive(prior.get('supervisor_pid',0)), 'Controller already running'
log=(session/'controller.log').open('a',encoding='utf-8')
command=[str(root/'.venv/bin/python'),'-B','-X','utf8','-u',str(session/'run_remote.py'),'run']
process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,
    start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'))
run_remote.write(session/'launch.json',dict(pid=process.pid,started_unix=time.time(),command=command))
print(json.dumps(dict(controller_pid=process.pid,jobs=len(plan['jobs']),
    training_conditions=sum(j['mode']=='train' for j in plan['jobs']),gpu_lanes=plan['gpu_lanes'],
    code_sha256=plan['code_sha256'],base_code_version=plan['base_code_version'])))
