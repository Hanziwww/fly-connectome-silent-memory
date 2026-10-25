"""Remote-only, four exclusive lanes with no automatic retry."""
from supp_base import *
import argparse
import subprocess


def process(pid, token):
    path=Path('/proc')/str(pid)/'cmdline'
    text=path.read_bytes().replace(b'\0',b' ').decode(errors='replace') if path.exists() else ''
    return dict(alive=bool(text and token in text),command=text if token in text else None)


def status():
    supervisor=read(SESSION/'supervisor.json') if (SESSION/'supervisor.json').exists() else {}
    if os.name!='nt':supervisor.update(process(supervisor.get('pid',0),'supp_queue.py run'))
    rows=[]
    plan=read(SESSION/'plan.json') if (SESSION/'plan.json').exists() else {'jobs':[]}
    for job in plan['jobs']:
        out=SESSION/'runs'/job['id'];row=dict(id=job['id'],kind=job['kind'])
        for name in ('process','progress','result','verification','failure','manifest'):
            path=out/(name+'.json')
            if path.exists():
                row[name]=read(path);row[name+'_mtime']=path.stat().st_mtime
        if os.name!='nt':row['actual_process']=process(row.get('process',{}).get('pid',0),'supp_worker.py --job '+job['id'])
        rows.append(row)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip() if os.name!='nt' else None
    return dict(supervisor=supervisor,jobs=rows,gpu=gpu,observed_unix=time.time(),
        disk_free_gib=shutil.disk_usage(SESSION).free/2**30,stop=(SESSION/'STOP').exists())


def run():
    if os.name=='nt':raise RuntimeError('Local GPU execution is prohibited')
    check_stop();plan=read(SESSION/'plan.json')
    assert plan['code_hashes']==code_hashes()
    with file_lock(SESSION/'controller.lock',blocking=False):
        prior=status()
        for row in prior['jobs']:
            if row.get('actual_process',{}).get('alive'):raise RuntimeError('Existing worker; do not launch twice')
            if 'failure' in row or ('process' in row and 'result' not in row):raise RuntimeError('Preserved failure or process loss requires diagnosis')
        active={};started=time.time();jobs=plan['jobs']
        while True:
            completed={j['id'] for j in jobs if (SESSION/'runs'/j['id']/'result.json').exists()}
            failed={j['id'] for j in jobs if (SESSION/'runs'/j['id']/'failure.json').exists()}
            for lane,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                item['log'].close();out=SESSION/'runs'/item['job']
                if code or not (out/'result.json').exists():
                    if not (out/'failure.json').exists():write(out/'failure.json',dict(status='process_lost',exit_code=code,finished_unix=time.time()))
                    failed.add(item['job'])
                else:completed.add(item['job'])
                make_manifest(out);item['lock'].__exit__(None,None,None);del active[lane]
            stopping=(SESSION/'STOP').exists()
            if not failed and not stopping:
                free=[float(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()]
                for lane in plan['gpu_lanes']:
                    if lane in active or free[lane]<13*1024:continue
                    busy={v['job'] for v in active.values()}
                    ready=[j for j in jobs if j['id'] not in completed|failed|busy and all(d in completed for d in j['dependencies'])]
                    if not ready:continue
                    name=ready[0]['id'];out=SESSION/'runs'/name;out.mkdir(parents=True,exist_ok=True)
                    lock=file_lock(STUDY/'remote-20261003'/f'gpu-{lane}.lock',blocking=False)
                    try:handle=lock.__enter__()
                    except BlockingIOError:continue
                    log=(out/'worker.log').open('a',encoding='utf-8');env=os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=str(lane),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                    command=[sys.executable,'-B','-X','utf8','-u',str(CODE/'supp_worker.py'),'--job',name]
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(handle.fileno(),))
                    active[lane]=dict(job=name,process=child,lock=lock,log=log)
                    write(out/'process.json',dict(pid=child.pid,gpu=lane,command=command,started_unix=time.time()))
            state='running' if active else 'complete' if len(completed)==len(jobs) else 'stopped' if stopping else 'attention' if failed else 'waiting_resource'
            write(SESSION/'supervisor.json',dict(status=state,pid=os.getpid(),started_unix=started,updated_unix=time.time(),
                completed=sorted(completed),failed=sorted(failed),total=len(jobs),parameters_updated=False,
                active={str(k):dict(job=v['job'],pid=v['process'].pid) for k,v in active.items()}))
            if not active and state in ('complete','stopped','attention'):return 0 if state=='complete' else 2
            time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','status']);a=p.parse_args()
    if a.action=='status':print(json.dumps(status(),ensure_ascii=False))
    else:raise SystemExit(run())
