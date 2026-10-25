"""Four exclusive remote GPU lanes for frozen replays, with preserved failures."""
from mechanism_base import *
import argparse
import subprocess


def alive(pid,token):
    path=Path('/proc')/str(pid)/'cmdline'
    return path.exists() and token in path.read_bytes().replace(b'\0',b' ').decode(errors='replace')


def status():
    supervisor=read(SESSION/'supervisor.json') if (SESSION/'supervisor.json').exists() else {}
    if os.name!='nt':
        supervisor['process_alive']=alive(supervisor.get('pid',0),'mechanism_queue.py')
    rows=[]
    if not (SESSION/'plan.json').exists():return dict(supervisor=supervisor,jobs=[])
    for job in read(SESSION/'plan.json')['jobs']:
        out=SESSION/'runs'/job['id'];row=dict(id=job['id'],kind=job['kind'])
        for name in ('process','progress','result','failure'):
            path=out/(name+'.json')
            if path.exists():row[name]=read(path);row[name+'_mtime']=path.stat().st_mtime
        if os.name!='nt':row['process_alive']=alive(row.get('process',{}).get('pid',0),'mechanism_worker.py --job '+job['id'])
        rows.append(row)
    gpu=None
    if os.name!='nt':
        gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
    return dict(supervisor=supervisor,jobs=rows,gpu=gpu,observed_unix=time.time(),
                storage=read(SESSION/'storage.json') if (SESSION/'storage.json').exists() else None,
                disk_free_gib=shutil.disk_usage(SESSION).free/2**30,stop=(SESSION/'STOP').exists())


def manifest(out):
    files={p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
           for p in out.rglob('*') if p.is_file() and p.name not in ('manifest.json','worker.log','process.json') and not p.name.endswith(('.tmp','.partial'))}
    write(out/'manifest.json',dict(files=files,total_bytes=sum(x['bytes'] for x in files.values())))


def run(resume=False):
    if os.name=='nt':raise RuntimeError('Use run --remote. Local training remains paused; no local GPU workers.')
    check_stop();plan=read(SESSION/'plan.json')
    assert plan['code_manifest']==code_manifest() and plan['scientific_sources']==scientific_sources()
    with file_lock(SESSION/'controller.lock',blocking=False):
        previous=status()
        if any(r.get('process_alive') for r in previous['jobs']):raise RuntimeError('Existing live GPU workers require observation, not duplicate launch')
        for row in previous['jobs']:
            if 'process' in row and 'result' not in row and 'failure' not in row:
                raise RuntimeError('Unexpected previous worker loss; preserve evidence and diagnose '+row['id'])
        started=time.time();active={};completed=set();failed=set();blocked=set();count=len(plan['jobs'])
        jobs={j['id']:j for j in plan['jobs']};order=list(jobs)
        # Paired static pilot and one complete M0 recording precede broad collection.
        pilot='record-remote-dynamic-M0-head_only'
        order.remove(pilot);order.insert(1,pilot)
        while True:
            for name,job in jobs.items():
                out=SESSION/'runs'/name
                if (out/'result.json').exists():completed.add(name)
                if (out/'failure.json').exists():failed.add(name)
            for lane,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                item['log'].close();out=SESSION/'runs'/item['job']
                if code or not (out/'result.json').exists():
                    if not (out/'failure.json').exists():write(out/'failure.json',dict(status='process_lost',exit_code=code,finished_unix=time.time()))
                    failed.add(item['job'])
                else:completed.add(item['job'])
                manifest(out);item['lock'].__exit__(None,None,None);del active[lane]
            for name,job in jobs.items():
                if any(d in failed|blocked for d in job['dependencies']):blocked.add(name)
            stopping=(SESSION/'STOP').exists()
            # Any implementation/numerical/resource failure stops new dispatches.
            healthy=not failed and not stopping
            if healthy:
                free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                free=[float(v) for v in free.strip().splitlines()]
                for lane in plan['gpu_lanes']:
                    if lane in active or free[lane]<13*1024:continue
                    active_names={v['job'] for v in active.values()}
                    ready=[name for name in order if name not in completed|failed|blocked|active_names and all(d in completed for d in jobs[name]['dependencies'])]
                    if not ready:continue
                    name=ready[0];out=SESSION/'runs'/name;out.mkdir(parents=True,exist_ok=True)
                    # Share the old project lock namespace; workers inherit this held descriptor.
                    lock=file_lock(STUDY/'remote-20261003'/f'gpu-{lane}.lock',blocking=False)
                    try:handle=lock.__enter__()
                    except BlockingIOError:continue
                    log=(out/'worker.log').open('a',encoding='utf-8')
                    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(lane),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                    command=[sys.executable,'-B','-X','utf8','-u',str(CODE/'mechanism_worker.py'),'--job',name]
                    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                             start_new_session=True,pass_fds=(handle.fileno(),))
                    active[lane]=dict(job=name,process=process,lock=lock,log=log)
                    write(out/'process.json',dict(pid=process.pid,gpu=lane,started_unix=time.time(),command=command))
            state='running' if active else 'complete' if len(completed)==count else 'stopped' if stopping else 'attention' if failed else 'waiting_resource'
            write(SESSION/'supervisor.json',dict(status=state,pid=os.getpid(),started_unix=started,updated_unix=time.time(),
                completed=sorted(completed),failed=sorted(failed),blocked=sorted(blocked),total=count,
                active={str(k):dict(job=v['job'],pid=v['process'].pid) for k,v in active.items()},parameters_updated=False))
            if not active and state in ('complete','stopped','attention'):return 0 if state=='complete' else 2
            time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','status']);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.action=='status':print(json.dumps(status(),ensure_ascii=False))
    else:raise SystemExit(run(a.resume))
