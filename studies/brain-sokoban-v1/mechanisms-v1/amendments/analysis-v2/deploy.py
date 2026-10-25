"""Audited CPU-analysis completion; do not interrupt any GPU worker."""
from pathlib import Path
import sys
import shlex
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
from mechanism_base import *
from mechanism_transport import connect,execute,REMOTE,REMOTE_ROOT,snapshot

def main():
    old=read(SESSION/'amendments/analysis-v2/prior-plan.json');new=read(SESSION/'plan.json')
    changed=[n for n,h in code_manifest().items() if old['code_manifest'].get(n)!=h]
    assert changed==['mechanism_analysis.py'],changed
    assert new==old,'Only apply once to the recorded prior revision'
    new['code_manifest']=code_manifest();new['analysis_revision']=2
    client=connect();pid=None;stopped=False
    try:
        state=snapshot(client);pid=state['supervisor']['pid']
        remote_plan=__import__('json').loads(execute(client,'cat '+shlex.quote(REMOTE+'/plan.json')))
        assert remote_plan==old
        command='import pathlib,os,signal;p='+str(pid)+';assert '+repr(REMOTE+'/code/mechanism_queue.py')+' in pathlib.Path(f"/proc/{p}/cmdline").read_bytes().decode();os.kill(p,signal.SIGSTOP)'
        execute(client,shlex.join([REMOTE_ROOT+'/.venv/bin/python','-c',command]));stopped=True
        with client.open_sftp() as sftp:
            target=REMOTE+'/code/mechanism_analysis.py'
            sftp.put(str(CODE/'mechanism_analysis.py'),target+'.new');sftp.posix_rename(target+'.new',target)
            write(SESSION/'amendments/analysis-v2/updated-plan.json',new)
            sftp.put(str(SESSION/'amendments/analysis-v2/updated-plan.json'),REMOTE+'/plan.json.new')
            sftp.posix_rename(REMOTE+'/plan.json.new',REMOTE+'/plan.json')
        write(SESSION/'plan.json',new)
        record=dict(unix=time.time(),changed_sources=changed,old_manifest=old['code_manifest'],new_manifest=new['code_manifest'],
            controller_pid=pid,gpu_worker_pids=[v['pid'] for v in state['supervisor']['active'].values()],
            training_or_recording_code_changed=False,model_or_intervention_conditions_changed=False,
            existing_workers_interrupted=False,purpose='Complete fixed-M0/T0 cross-family explanations and streaming cell-group extraction; count only hash-verified local datasets as complete.')
        write(SESSION/'amendments/analysis-v2/receipt.json',record)
        with client.open_sftp() as sftp:sftp.put(str(SESSION/'amendments/analysis-v2/receipt.json'),REMOTE+'/analysis-revision-2.json')
        print(__import__('json').dumps({k:record[k] for k in ('changed_sources','existing_workers_interrupted','model_or_intervention_conditions_changed')}))
    finally:
        if stopped:execute(client,shlex.join(['kill','-CONT',str(pid)]))
        client.close()

if __name__=='__main__':main()
