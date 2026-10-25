"""Registered supplement; independent workers share the original four GPU locks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from supp_tasks import SESSION, STUDY, ROOT, prepare as prepare_tasks, validate_task
from registry import read, write, digest, provenance, SOURCE

LEGACY=STUDY/'remote-20261003'
GOOD={'learned_solved','already_solved','stagnation','segment_limit','untrained_baseline'}


def code_digest():
    return hashlib.sha256(''.join(digest(SESSION/n) for n in
        ('run_remote.py','supp_tasks.py','supp_experiments.py')).encode()).hexdigest()


def prepare():
    if (SESSION/'plan.json').exists():
        p=read(SESSION/'plan.json')
        assert p['code_sha256']==code_digest(),'Supplement code changed after registration'
        assert p['base_code_version']==provenance(),'Base code changed'
        assert p['tasks_sha256']==digest(SESSION/'tasks.json'),'Locked tasks changed'
        return p
    tasks=prepare_tasks()
    for task in tasks:
        if task['id'].startswith('S'):validate_task(task)
    source_hash=digest(SOURCE);version=provenance();jobs=[]
    warm='studies/brain-sokoban-v1/runs/remote-dynamic-M0-head_only/model.pt'
    warm_hash=digest(ROOT/warm)
    def add(name,mode,task='M0-audit',dependencies=(),**extra):
        out=SESSION/'runs'/name;out.mkdir(parents=True,exist_ok=True)
        config=dict(id=name,task=task,seed=10,condition='head_only',policy_kind='head_only',
            core_initialization='p05',head_initialization='p05',structure='real',precision='mixed',batch=1,
            source=str(SOURCE),source_sha256=source_hash,code_version=version,code_sha256=code_digest(),
            tasks_sha256=digest(SESSION/'tasks.json'),segments_seconds=[1200,2400,4800],
            training_method='Full-history demonstration CE; frozen-core features cached by complete input history',
            acceptance='Independent full hidden-goal solution on all training histories, any route within 512 actions',
            entry_margin='Fixed first demonstration action versus other legal actions; extension metric only',
            collect_interventions=True,**extra)
        write(out/'config.json',config)
        jobs.append(dict(id=name,mode=mode,task=task,out=str(out.relative_to(ROOT)),dependencies=list(dependencies)))
    add('supp-preflight','preflight',warm_start=warm,warm_start_sha256=warm_hash)
    first=[]
    for task in ('S1-ring','S2-context'):
        for arm in ('brain','gru'):
            name=f'supp-{task}-{arm}';first.append(name)
            extra=dict(policy_kind=arm if arm=='gru' else 'head_only')
            # Defaults are constructed first, then explicit policy overrides below.
            add(name,'train',task,['supp-preflight'],arm=arm)
            cfg=read(SESSION/'runs'/name/'config.json');cfg.update(extra)
            write(SESSION/'runs'/name/'config.json',cfg)
    add('supp-M0-visibility','train','M0-visibility',['supp-preflight'],
        warm_start=warm,warm_start_sha256=warm_hash)
    add('supp-M0-causal','probe',dependencies=['supp-preflight'],probe='causal',
        warm_start=warm,warm_start_sha256=warm_hash)
    add('supp-M0-delay','probe',dependencies=['supp-preflight'],probe='dense_delay',
        warm_start=warm,warm_start_sha256=warm_hash)
    add('supp-M0-visibility-causal','probe','M0-visibility',['supp-M0-visibility'],
        probe='causal',checkpoint_job='supp-M0-visibility')
    for task in ('S1-ring','S2-context'):
        add(f'supp-transfer-{task}','probe',task,['supp-preflight'],probe='transfer',
            warm_start=warm,warm_start_sha256=warm_hash)
    # Origin and graph contrasts share a new head initialization and independent
    # fixed C0/C1 normalization. Existing P05 heads are not reused for this contrast.
    for task in ('S1-ring','S2-context'):
        for arm in ('p05-fresh','neutral-real','neutral-rewired','leaky'):
            name=f'supp-origin-{task}-{arm}'
            add(name,'train',task,first,arm=arm)
            cfg=read(SESSION/'runs'/name/'config.json')
            cfg.update(head_initialization='fresh_calibrated',collect_interventions=False,
                core_initialization='p05' if arm=='p05-fresh' else 'neutral',
                structure='rewired' if arm=='neutral-rewired' else 'real',
                policy_kind='leaky' if arm=='leaky' else 'head_only')
            write(SESSION/'runs'/name/'config.json',cfg)
    plan=dict(created_unix=time.time(),seed=10,code_sha256=code_digest(),base_code_version=version,
        tasks_sha256=digest(SESSION/'tasks.json'),source_sha256=source_hash,gpu_lanes=[0,1,2,3],
        shared_gpu_lock_directory=str(LEGACY.relative_to(ROOT)),allocator_gib=11,minimum_free_gib=13,
        segments_seconds=[1200,2400,4800],local_training_remains_paused=True,jobs=jobs,
        independent_units='Two new map families; rotated contexts belong to the same family; seed 10 only',
        primary_endpoints=['autonomous hidden-goal solution','first irreversible push','paired intervention effects'],
        all_neural_clear='Neural tensor clear with clock, buffer pointer and RNG preserved; distinct from old all-state reset',
        context_scope='S2 is delayed scene reorientation plus Sokoban, not an ordinary static-board episode',
        holdouts='Untrained times and zero-update transfer are never used for model selection')
    write(SESSION/'plan.json',plan)
    return plan


def alive(pid,session=SESSION):
    try:return str(session/'run_remote.py') in Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
    except (OSError,ValueError):return False


def lane_free(lane):
    import fcntl
    if (LEGACY/'supervisor.json').exists():
        old=read(LEGACY/'supervisor.json')
        if alive(old.get('supervisor_pid',0),LEGACY) and str(lane) in old.get('active',{}):return False
    handle=(LEGACY/f'gpu-{lane}.lock').open('a')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:handle.close();return False
    handle.close();return True


def completed(job):
    out=ROOT/job['out']
    if not (out/'result.json').exists():return False
    r=read(out/'result.json')
    if job['mode']=='train':
        return r['status'] in GOOD and (out/'verification.json').exists() and read(out/'verification.json')['status']=='verified'
    return r['status'] in ('passed','completed')


def worker(job,verify=False):
    import fcntl
    lane=os.environ['CUDA_VISIBLE_DEVICES'];out=ROOT/job['out']
    lock=(LEGACY/f'gpu-{lane}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        config=read(out/'config.json')
        assert config['code_sha256']==code_digest()
        assert config['code_version']==provenance()
        assert config['tasks_sha256']==digest(SESSION/'tasks.json')
        import supp_experiments as ex
        ex.stopped(out)
        if job['mode']=='preflight':return ex.preflight(config,out)
        if job['mode']=='probe':return ex.probe_job(config,out)
        return ex.train(out/'config.json',verify)
    except BaseException as e:
        torch=sys.modules.get('torch')
        category='stopped' if isinstance(e,KeyboardInterrupt) else 'oom' if torch and isinstance(e,torch.OutOfMemoryError) else 'numerical_error' if isinstance(e,FloatingPointError) else 'implementation_error'
        write(out/('verification.json' if verify else 'result.json'),dict(status=category,pid=os.getpid(),error=repr(e),traceback=traceback.format_exc()))
        traceback.print_exc();return 1
    finally:lock.close()


def launch(live,lane,job,stage):
    out=ROOT/job['out'];log=(out/('verification.log' if stage=='verify' else 'worker.log')).open('a',encoding='utf-8')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(lane),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    command=[sys.executable,'-B','-X','utf8','-u',str(SESSION/'run_remote.py'),'verify' if stage=='verify' else 'worker','--job',job['id']]
    p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/f'{stage}-process.json',dict(pid=p.pid,gpu=lane,started_unix=time.time(),command=command))
    live[lane]=(p,job,stage,log)


def run():
    import fcntl
    handle=(SESSION/'controller.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=prepare();jobs=plan['jobs'];live={};failed={};done={j['id'] for j in jobs if completed(j)}
    if (SESSION/'supervisor.json').exists():
        assert not any(alive(v['pid']) for v in read(SESSION/'supervisor.json').get('active',{}).values()),'Existing supplement worker alive'
    try:
        while True:
            stopping=(SESSION/'STOP').exists() or (STUDY/'STOP').exists()
            for lane,(p,j,stage,log) in list(live.items()):
                code=p.poll()
                if code is None:continue
                log.close();del live[lane]
                if stage=='train' and code==0 and j['mode']=='train':launch(live,lane,j,'verify')
                elif code==0 and completed(j):done.add(j['id'])
                else:failed[j['id']]=dict(stage=stage,exit_code=code)
            if not stopping:
                for lane in plan['gpu_lanes']:
                    if lane in live or not lane_free(lane):continue
                    busy={x[1]['id'] for x in live.values()}
                    for j in jobs:
                        if j['id'] in done|set(failed)|busy:continue
                        out=ROOT/j['out']
                        if any(d in failed for d in j['dependencies']):
                            failed[j['id']]=dict(stage='dependency',reason='Upstream technical failure');continue
                        if not all(d in done for d in j['dependencies']):continue
                        if (out/'result.json').exists():
                            r=read(out/'result.json')
                            if j['mode']=='train' and r['status'] in GOOD and not (out/'verification.json').exists():
                                launch(live,lane,j,'verify');break
                            failed[j['id']]=dict(stage='prior_attempt',reason='Preserved result requires diagnosis');continue
                        launch(live,lane,j,'train');break
            pending=[j['id'] for j in jobs if j['id'] not in done|set(failed)|{x[1]['id'] for x in live.values()}]
            state='stopping' if stopping and live else 'stopped' if stopping else 'running' if live else 'complete' if len(done)==len(jobs) else 'waiting_gpu' if pending else 'needs_attention'
            write(SESSION/'supervisor.json',dict(status=state,supervisor_pid=os.getpid(),updated_unix=time.time(),
                completed=sorted(done),failed=failed,pending=pending,
                active={str(k):dict(job=v[1]['id'],pid=v[0].pid,stage=v[2]) for k,v in live.items()}))
            if not live and (stopping or not pending):return 0 if state in ('stopped','complete') else 1
            time.sleep(3)
    finally:handle.close()


def status():
    plan=read(SESSION/'plan.json');s=read(SESSION/'supervisor.json') if (SESSION/'supervisor.json').exists() else {}
    s['supervisor_alive']=alive(s.get('supervisor_pid',0));rows=[]
    for j in plan['jobs']:
        out=ROOT/j['out'];row=dict(id=j['id'],mode=j['mode'],task=j['task'])
        for name in ('result','progress','verification'):
            p=out/f'{name}.json'
            if p.exists():
                value=read(p);row[name]={k:value[k] for k in ('status','updates','elapsed','training_seconds','final_solved',
                     'original_solved','solved','total','probe_correct','probe_total','completed','resources','error') if k in value}
                if name=='verification' and 'training_history_reload' in value:
                    check=value['training_history_reload'];row[name].update(solved=check['solved'],total=check['total'])
                row[name+'_age_seconds']=round(time.time()-p.stat().st_mtime,1)
        for item in s.get('active',{}).values():
            if item['job']==j['id']:row.update(pid=item['pid'],process_alive=alive(item['pid']),stage=item['stage'])
        for name in ('worker.log','verification.log','resume.pt','probe-resume.json'):
            p=out/name
            if p.exists():row[name]=dict(bytes=p.stat().st_size,age_seconds=round(time.time()-p.stat().st_mtime,1))
        rows.append(row)
    return dict(supervisor=s,jobs=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','status','worker','verify'])
    p.add_argument('--job');a=p.parse_args()
    if a.command=='prepare':print(json.dumps(prepare()))
    elif a.command=='status':print(json.dumps(status()))
    elif a.command=='run':raise SystemExit(run())
    else:
        job=next(j for j in read(SESSION/'plan.json')['jobs'] if j['id']==a.job)
        raise SystemExit(worker(job,a.command=='verify'))
