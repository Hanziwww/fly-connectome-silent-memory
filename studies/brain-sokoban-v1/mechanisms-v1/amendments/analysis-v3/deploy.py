from pathlib import Path
import sys
import shlex
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
from mechanism_base import *
from mechanism_transport import connect,execute,REMOTE,REMOTE_ROOT,snapshot

old=read(SESSION/'amendments/analysis-v3/prior-plan.json');new=read(SESSION/'plan.json')
assert old==new
assert [n for n,h in code_manifest().items() if old['code_manifest'].get(n)!=h]==['mechanism_analysis.py']
new['code_manifest']=code_manifest();new['analysis_revision']=3
client=connect();pid=None;paused=False
try:
    state=snapshot(client);pid=state['supervisor']['pid']
    assert __import__('json').loads(execute(client,'cat '+shlex.quote(REMOTE+'/plan.json')))==old
    code='import os,signal,pathlib;p='+str(pid)+';assert '+repr(REMOTE+'/code/mechanism_queue.py')+' in pathlib.Path(f"/proc/{p}/cmdline").read_bytes().decode();os.kill(p,signal.SIGSTOP)'
    execute(client,shlex.join([REMOTE_ROOT+'/.venv/bin/python','-c',code]));paused=True
    write(SESSION/'amendments/analysis-v3/updated-plan.json',new)
    with client.open_sftp() as sftp:
        for local,remote in ((CODE/'mechanism_analysis.py',REMOTE+'/code/mechanism_analysis.py'),(SESSION/'amendments/analysis-v3/updated-plan.json',REMOTE+'/plan.json')):
            sftp.put(str(local),remote+'.new');sftp.posix_rename(remote+'.new',remote)
    write(SESSION/'plan.json',new)
    receipt=dict(unix=time.time(),changed_sources=['mechanism_analysis.py'],old_manifest=old['code_manifest'],new_manifest=new['code_manifest'],
        existing_workers_interrupted=False,scientific_recording_code_or_conditions_changed=False,
        purpose='Use the actual legal-action mask for the fixed competing-action margin; four-raw-logit geometry is not equated to behavioral utilization.')
    write(SESSION/'amendments/analysis-v3/receipt.json',receipt)
    with client.open_sftp() as sftp:sftp.put(str(SESSION/'amendments/analysis-v3/receipt.json'),REMOTE+'/analysis-revision-3.json')
    print('Analysis-only legal-action correction deployed; GPU workers continued unchanged')
finally:
    if paused:execute(client,shlex.join(['kill','-CONT',str(pid)]))
    client.close()
