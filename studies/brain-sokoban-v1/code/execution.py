"""Serial GPU supervisor and independent checkpoint verification."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import bootstrap
from registry import STUDY,ROOT,SOURCE,read,write,append,digest,condition,provenance


def tasks():
    return {l['id']:l for name in ('levels.json','variants.json') for l in read(STUDY/'tasks'/name)}


def stopped(out):
    if (STUDY/'STOP').exists() or (out/'STOP').exists():raise KeyboardInterrupt('Study STOP requested')


def run_worker(config_path,verify=False):
    if read(config_path)['module']=='dynamic_memory':
        from memory import worker
        return worker(config_path,verify)
    import torch
    from common import save_checkpoint
    from backend import initialize,pair,set_pair
    from behavior import rollouts
    from training import fit
    config=read(config_path);out=Path(config_path).parent
    check=lambda:stopped(out)
    t=time.perf_counter();encoder=None
    result=dict(id=config['id'],pid=os.getpid(),config=config,status='initializing')
    try:
        check();encoder,head=initialize(config,out)
        if digest(config['source'])!=config['source_sha256']:raise RuntimeError('Parent checkpoint changed')
        source=torch.load(config['source'],map_location='cpu',weights_only=True)
        level=tasks()[config['task']]
        if verify:
            saved=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
            assert saved['config']==config
            set_pair(encoder,head,saved)
            r=rollouts(encoder,head,[level],check)['records'][0]
            prior=read(out/'result.json')
            assert prior['pid']!=os.getpid()
            status='verified' if not prior.get('final_solved') or r['status']=='solved' else 'reproduction_failed'
            write(out/'verification.json',dict(status=status,pid=os.getpid(),training_pid=prior['pid'],
                model_sha256=digest(out/'model.pt'),original_solved=r['status']=='solved',
                original_steps=r['steps'],precision=config['precision'],batch=config['batch'],
                independent_process=True,rollout=r,seconds=time.perf_counter()-t))
            return 0 if status=='verified' else 1
        if (out/'result.json').exists() and not config.get('resume_authorized',False):
            raise RuntimeError('Completed/failed attempt already exists; use a new run ID for a repair')
        if config['condition']['initialization']=='p05':set_pair(encoder,head,source)
        elif encoder.core is not None:
            encoder.set_core(None)
            from game import encode,restore,state_from,step
            boards=[]
            for l in tasks().values():
                if l['id'] not in ('C0','C1'):continue
                m=restore(l);s=state_from(l['start'])
                for a in l['reference_actions']:
                    boards.append(encode(m,s));s=step(m,s,a)
            f=encoder.encode(np.stack(boards),check)
            with torch.no_grad():
                head.mean.copy_(torch.from_numpy(f.mean(0)).cuda())
                head.scale.copy_(torch.from_numpy(np.maximum(f.std(0),.01)).cuda())
        if not (out/'initial.pt').exists():save_checkpoint(out/'initial.pt',dict(**pair(encoder,head),config=config))
        result['initialization_seconds']=time.perf_counter()-t
        result['initialization_compute']=dict(encoder.counts)
        for key in encoder.counts:encoder.counts[key]=type(encoder.counts[key])(0)
        result.update(fit(encoder,head,level,config,out,check))
        saved=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
        initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=True)
        result['parameter_changes']={component:{k:dict(changed=int((v!=initial[component][k]).sum()),
            l2=float((v-initial[component][k]).double().norm())) for k,v in saved[component].items()}
            for component in ('core','head')}
        result['model_sha256']=digest(out/'model.pt')
        result['condition']=config['condition']['name']
        result['task']=config['task'];result['family']=level['family_id']
    except BaseException as e:
        if verify:
            write(out/'verification.json',dict(status='error',error=repr(e),traceback=traceback.format_exc(),pid=os.getpid()))
            return 1
        status='stopped' if isinstance(e,KeyboardInterrupt) else 'oom' if isinstance(e,torch.OutOfMemoryError) else 'numerical_error' if isinstance(e,FloatingPointError) else 'implementation_error'
        result.update(status=status,error=repr(e),traceback=traceback.format_exc())
        if (out/'resume.pt').exists():
            state=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)['state']
            result.update(updates=state['updates'],training_seconds=state['elapsed'],depth=state['depth'],phase=state['phase'])
        print(result['traceback'],flush=True)
    finally:
        if encoder is not None:encoder.release_graph()
        if not verify:
            result.update(wall_seconds=time.perf_counter()-t)
            if torch.cuda.is_initialized():result['resources']=dict(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            write(out/'result.json',result);write(out/'progress.json',dict(status=result['status']))
    return 0 if result['status'] in ('learned_solved','already_solved','stagnation','segment_limit','frozen_baseline') else 1


def child(command,log_path,out):
    from run_cycle import WorkerJob
    with Path(log_path).open('a',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        job=WorkerJob(process)
        write(STUDY/'supervisor.json',dict(status='running',supervisor_pid=os.getpid(),worker_pid=process.pid,
            run_id=out.name,phase='verify' if '--verify' in command else 'train',started_unix=time.time()))
        try:
            while process.poll() is None:
                if (STUDY/'STOP').exists():
                    # Child observes the same STOP; allow atomic resume checkpoint.
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:job.close()
                    break
                time.sleep(2)
            return process.wait()
        finally:job.close()


def execute_job(job,backend,resume=False):
    out=STUDY/'runs'/job['id'];out.mkdir(parents=True,exist_ok=True)
    stopped(out)
    if (out/'result.json').exists():
        old_result=read(out/'result.json')
        if resume and old_result['status'] in ('stopped','worker_exit') and (out/'resume.pt').exists():
            old_config=read(out/'config.json')
            if old_config['code_version']!=provenance():raise RuntimeError('Resume requires the recorded code version')
            (out/'result.json').rename(out/f'interruption-{int(time.time())}.json')
        else:
            if old_result['status'] in ('learned_solved','already_solved','stagnation','segment_limit','frozen_baseline') and not (out/'verification.json').exists():
                command=[sys.executable,'-B','-X','utf8','-u',str(ROOT/'workflow.py'),'study','worker','--config',str(out/'config.json'),'--verify']
                child(command,out/'verification.log',out)
            if (out/'verification.json').exists() and read(out/'verification.json')['status']!='verified':
                return dict(old_result,status='verification_failed',training_status=old_result['status'])
            return old_result
    source=STUDY/'runs'/job['parent']/'model.pt' if job.get('parent') else SOURCE
    if not source.exists():
        result=dict(id=job['id'],status='dependency_unavailable',parent=job.get('parent'))
        write(out/'result.json',result);return result
    config=dict(**job,source=str(source),source_sha256=digest(source),
        precision=backend['precision'],batch=backend['batch'],segments_seconds=[1200,2400,4800],
        code_version=provenance(),task_lock_sha256=digest(STUDY/'tasks/lock.json'),
        protocol_sha256=digest(STUDY/'protocol.json'))
    if (out/'config.json').exists():
        # Resumption follows the original recorded configuration, including code.
        old=read(out/'config.json')
        if old['code_version']!=config['code_version']:raise RuntimeError('Code changed before resuming '+job['id'])
        config=old
    else:write(out/'config.json',config)
    command=[sys.executable,'-B','-X','utf8','-u',str(ROOT/'workflow.py'),'study','worker','--config',str(out/'config.json')]
    code=child(command,out/'worker.log',out)
    if not (out/'result.json').exists():
        write(out/'result.json',dict(id=job['id'],status='worker_exit',exit_code=code,
            resumable=(out/'resume.pt').exists(),config=config))
    result=read(out/'result.json')
    if code==0 and (out/'model.pt').exists():
        verified=child(command+['--verify'],out/'verification.log',out)==0
        write(out/'acceptance.json',dict(accepted=verified,training_status=result['status'],
            verification=read(out/'verification.json').get('status') if (out/'verification.json').exists() else 'missing'))
        if not verified:return dict(result,status='verification_failed',training_status=result['status'])
    append(STUDY/'events.jsonl',dict(event='run_finished',run=job['id'],status=result['status'],time=time.time()))
    print(json.dumps(dict(event='run_finished',run=job['id'],status=result['status'],
        updates=result.get('updates'),training_seconds=result.get('training_seconds'))),flush=True)
    return result


def benchmark():
    from run_cycle import acquire_worker_lock
    lock=acquire_worker_lock()
    try:
        candidates=[dict(id='fp64-b16',precision='float64',batch=16),
                    dict(id='fp32-b16',precision='float32',batch=16),
                    dict(id='fp32-b8',precision='float32',batch=8),
                    dict(id='mixed-b16',precision='mixed',batch=16)]
        results=[]
        # Alternate precision within each task to avoid placing all reference
        # executions in a different time block from all optimized executions.
        for task,order in [('C0',candidates),('C1',candidates[::-1])]:
            for b in order:
                run_id='benchmark-v2-fp64-b16-C0' if b['id']=='fp64-b16' and task=='C0' else 'benchmark-v3-'+b['id']+'-'+task
                r=execute_job(dict(id=run_id,task=task,
                    condition=condition(),module='benchmark',parent=None,seed=10),b)
                results.append(r)
                if r['status'] in ('implementation_error','worker_exit','stopped','verification_failed'):
                    write(STUDY/'benchmark.json',dict(status='needs_diagnosis',results=results))
                    write(STUDY/'supervisor.json',dict(status='needs_diagnosis',run_id=r['id'],reason=r['status']))
                    return 1
        scores=[]
        for b in candidates:
            rs=[r for r in results if r['config']['precision']==b['precision'] and r['config']['batch']==b['batch']]
            for r in rs:
                if r.get('training_seconds') is None and (STUDY/'runs'/r['id']/'resume.pt').exists():
                    import torch
                    snapshot=torch.load(STUDY/'runs'/r['id']/'resume.pt',map_location='cpu',weights_only=True)
                    r['training_seconds']=snapshot['state']['elapsed'];r['updates']=snapshot['state']['updates']
                    r['timing_source']='preserved resume checkpoint after failure'
            valid=len(rs)==2 and all(r.get('status')=='learned_solved' and read(STUDY/'runs'/r['id']/'verification.json').get('original_solved') for r in rs)
            scores.append(dict(**b,eligible=valid,total_training_seconds=sum(r.get('training_seconds',0) for r in rs),
                               total_wall_seconds=sum(r.get('wall_seconds',0) for r in rs)))
        eligible=[b for b in scores if b['eligible']]
        selected=min(eligible,key=lambda b:b['total_training_seconds']) if eligible else scores[0]
        write(STUDY/'benchmark.json',dict(status='complete',candidates=scores,results=results))
        write(STUDY/'backend.json',dict(**selected,selection='fastest two-task learned+reload success' if eligible else 'reference fallback'))
        write(STUDY/'supervisor.json',dict(status='calibrated',backend=selected,finished_unix=time.time()))
        return 0
    finally:lock.close()


def train(run_id=None,module=None,resume=False):
    from run_cycle import acquire_worker_lock
    if not (STUDY/'backend.json').exists():raise RuntimeError('Run speed calibration before training')
    lock=acquire_worker_lock()
    try:
        for job in read(STUDY/'schedule.json'):
            if run_id and job['id']!=run_id:continue
            if module and job['module']!=module:continue
            r=execute_job(job,read(STUDY/'backend.json'),resume=resume)
            if r['status'] in ('implementation_error','numerical_error','oom','worker_exit','stopped','dependency_unavailable','verification_failed'):
                write(STUDY/'supervisor.json',dict(status='needs_diagnosis',run_id=r['id'],reason=r['status']))
                return 1
            from analysis import analyze
            if job['module']=='difficulty' and job['task'].endswith('T2'):analyze()
        return 0
    finally:lock.close()


def dispatch(args):
    if args.command=='worker':return run_worker(args.config,args.verify)
    if args.command=='benchmark':return benchmark()
    if args.command=='train':
        if args.module=='dynamic_memory':
            from memory import run_memory
            return run_memory(args.resume)
        return train(args.run_id,args.module,args.resume)
    if args.command=='evaluate':
        if args.module=='activity':
            from activity import run_activity
            return run_activity()
        from evaluation import evaluate_study
        return evaluate_study(args.run_id)
    if args.command=='analyze':
        from analysis import analyze
        return analyze()
    if args.command=='run':
        from tasks import prepare,separate_wall_families
        from memory_tasks import prepare_memory
        prepare();prepare_memory();separate_wall_families()
        if not (STUDY/'backend.json').exists() and benchmark():return 1
        if not (STUDY/'validation.json').exists() or read(STUDY/'validation.json')['precision']!=read(STUDY/'backend.json')['precision']:
            command=[sys.executable,'-B','-X','utf8',str(ROOT/'workflow.py'),'study','validate']
            if child(command,STUDY/'validation.log',STUDY/'validation'):return 1
        if train(resume=args.resume):return 1
        from evaluation import evaluate_study
        if evaluate_study():
            write(STUDY/'supervisor.json',dict(status='needs_diagnosis',phase='generalization'));return 1
        from memory import run_memory
        if run_memory(args.resume):
            write(STUDY/'supervisor.json',dict(status='needs_diagnosis',phase='dynamic_memory'));return 1
        from activity import run_activity
        if run_activity():
            write(STUDY/'supervisor.json',dict(status='needs_diagnosis',phase='activity'));return 1
        from analysis import analyze
        analyze();write(STUDY/'supervisor.json',dict(status='complete',finished_unix=time.time()))
        return 0
