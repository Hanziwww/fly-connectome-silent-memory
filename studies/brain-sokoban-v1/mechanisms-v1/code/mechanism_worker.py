"""GPU workers for the frozen, independently registered mechanism study."""
from mechanism_base import *
import argparse
import traceback


def task_level(task):
    return next(x for x in read(SESSION/'inputs/levels.json') if x['id']==task)


def task_memory(task):
    return next(x for x in read(SESSION/'inputs/memory-tasks.json') if x['id']==task)


def export_pair(out,phase,pair):
    arrays={'core_'+k:v.detach().cpu().numpy() for k,v in pair['core'].items()}
    arrays.update({'head_'+k:v.detach().cpu().numpy() for k,v in pair['head'].items()})
    save_npz(out/f'parameters-{phase}.npz',**arrays)


def static_record(job,out):
    from mechanism_model import Frozen, Recording, save_index, static_rollout
    from game import encode,restore,state_from,step,legal
    model=model_by_id(job['model']);level=task_level(model['task'])
    policy=Frozen(model,out,batch=16);save_index(policy)
    m=restore(level);s=state_from(level['start']);states=[];boards=[];labels=[]
    from progressive_tasks import reverse_distances
    distances,certificate=reverse_distances(level)
    write(out/'action-set-certificate.json',certificate)
    for action in level['reference_actions']:
        states.append([s[0],list(s[1])]);boards.append(encode(m,s));n=step(m,s,action)
        acceptable=[step(m,s,a) in distances for a in range(4)]
        labels.append(dict(action=action,push=n[1]!=s[1],legal=legal(m,s).tolist(),
                           solvable_actions=acceptable,remaining_certified_moves=distances[s]))
        s=n
    boards=np.stack(boards);selected={0,len(boards)//2,len(boards)-1}
    write(out/'bank.json',dict(task=level['id'],family=level['family_id'],states=states,labels=labels,
          role=model['role'],sampling='identical certified reference boards in each phase'))
    phase_files={};audits=[]
    for phase,path in unique_phases(model):
        pair=policy.load(path);export_pair(out,phase,pair)
        target=out/f'matched-{phase}.npz'
        if target.exists():phase_files[phase]=target.name;continue
        pieces=[]
        for start in range(0,len(boards),16):
            check_stop();n=min(16,len(boards)-start);x=np.zeros((16,4,10,10),np.float32);x[:n]=boards[start:start+n]
            dense=[i-start for i in selected if start<=i<start+n] if model['representative'] else []
            recorder=Recording(policy,n,dense)
            q,f=policy.advance(x,reset=True,recorder=recorder);data=recorder.finish()
            for i,values in data.pop('dense').items():
                save_npz(out/'dense'/f'{phase}-state-{start+i}.npz',values=values,
                         milliseconds=np.arange(1,41),fields=np.array(FIELDS))
            data.update(features=f[:n],logits=q[:n]);pieces.append(data)
            progress(out,stage='matched_static',phase=phase,completed_states=start+n,total_states=len(boards))
        save_npz(target,**{k:np.concatenate([x[k] for x in pieces]) for k in ('mean','std','group','instantaneous','features','logits')})
        phase_files[phase]=target.name;audits.append(policy.frozen_audit())
    old=load_pair(SESSION/model['checkpoints']['before']['path'])
    new=load_pair(SESSION/model['checkpoints']['after']['path'])
    with np.load(out/'matched-before.npz') as b,np.load(out/'matched-after.npz') as a:
        dec=contribution_decomposition(b['features'],a['features'],head_arrays(old),head_arrays(new))
    if dec['reconstruction_error']>1e-4:raise ArithmeticError('Decomposition reconstruction failed')
    save_npz(out/'decomposition.npz',**dec)
    behaviors=[]
    import torch
    from backend import set_pair
    for brain,head in ((0,0),(1,0),(0,1),(1,1)):
        pair=dict(core=(new if brain else old)['core'],head=(new if head else old)['head'])
        set_pair(policy.encoder,policy.head,pair)
        behaviors.append(dict(brain_phase=brain,head_phase=head,**static_rollout(policy,level)))
    expected=model['source_acceptance']['original_solved']
    actual=behaviors[-1]['status']=='solved'
    if expected is not None and actual!=expected:raise AssertionError('Saved original-start acceptance changed')
    write(out/'behaviors.json',behaviors)
    return dict(status='complete',model=model['id'],kind='static',phases=phase_files,
        original_solved=actual,source_acceptance_matches=True,states=len(boards),frozen_audits=audits,
        decomposition_error=dec['reconstruction_error'],role=model['role'])


class MemoryCapture:
    def __init__(self,policy,out,stem,cases,dense=False):
        self.policy=policy;self.out=out;self.stem=stem;self.cases=cases;self.dense=dense
        self.pushed=set();self.parts=[];self.events=[];self.dense_files=[]
    def begin(self,kind,ms,elapsed,decision_index):
        from mechanism_model import Recording
        if decision_index>=8:return None
        rows=[i for i in range(len(self.cases)) if i not in self.pushed] if self.dense else []
        return Recording(self.policy,dense_rows=rows)
    def end(self,recorder,q,f,kind,ms,elapsed,decision_index):
        if recorder is None:return
        data=recorder.finish();index=len(self.parts)
        for row,values in data.pop('dense').items():
            name=f'{self.stem}-segment{index:02d}-case{row}.npz'
            save_npz(self.out/'dense'/name,values=values,milliseconds=np.arange(elapsed+1,elapsed+ms+1),fields=np.array(FIELDS))
            self.dense_files.append(dict(file='dense/'+name,case=row,kind=kind,start_ms=elapsed,ms=ms))
        data.update(features=f,logits=q);self.parts.append(data)
        self.events.append(dict(kind=kind,start_ms=elapsed,ms=ms,decision_index=decision_index))
    def finish(self):
        target=self.out/(self.stem+'.npz')
        save_npz(target,mean=np.stack([p['mean'] for p in self.parts]),std=np.stack([p['std'] for p in self.parts]),
            group=np.concatenate([p['group'] for p in self.parts],axis=1),
            instantaneous=np.concatenate([p['instantaneous'] for p in self.parts],axis=1),
            features=np.stack([p['features'] for p in self.parts]),logits=np.stack([p['logits'] for p in self.parts]),
            segment_start=np.array([x['start_ms'] for x in self.events]),segment_ms=np.array([x['ms'] for x in self.events]))
        write(self.out/(self.stem+'.json'),dict(file=target.name,events=self.events,cases=[{k:c[k] for k in ('id','cue','context','visible')} for c in self.cases],
            dense_files=self.dense_files,neural_recording_decision_cap=8,full_behavior_cap=512))
        return target.name


def memory_record(job,out):
    from mechanism_model import Frozen,save_index,tasklib,memory_rollout
    model=model_by_id(job['model']);task=task_memory(model['task']);cases=task['cases'];lib=tasklib()
    policy=Frozen(model,out,batch=len(cases));save_index(policy)
    records=[];activity_cache={};phase_hashes={};start=time.monotonic()
    for phase,path in unique_phases(model):
        pair=policy.load(path);export_pair(out,phase,pair);core_hash=tensors_hash(pair['core']);phase_hashes[phase]=core_hash
        for delay in (0,20,40,80,160,320,640):
            for prefix in (0,1,2,4,8):
                check_stop();stem=f'matched-{phase}-d{delay}-p{prefix}'
                cache_key=(core_hash,delay,prefix)
                if cache_key in activity_cache:
                    source=activity_cache[cache_key]
                    with np.load(out/source,allow_pickle=False) as f:features=f['features']
                    q=logits(features,head_arrays(pair))
                    save_npz(out/(stem+'-readout.npz'),features=features,logits=q)
                    records.append(dict(phase=phase,delay=delay,prefix=prefix,activity=source,
                        readout=stem+'-readout.npz',activity_reused_same_core_and_complete_history=True))
                    continue
                policy.reset(len(cases));dense=phase!='after' and delay in (160,640) and prefix==4
                capture=MemoryCapture(policy,out,stem,cases,dense=dense)
                seq=[lib.frames(c,delay,prefix,False) for c in cases];elapsed=0
                for i,frame in enumerate(seq[0]):
                    rec=capture.begin(frame['kind'],frame['ms'],elapsed,0 if frame['kind']=='decision' else -1)
                    q,f=policy.advance(np.stack([s[i]['board'] for s in seq]),frame['ms'],recorder=rec)
                    capture.end(rec,q,f,frame['kind'],frame['ms'],elapsed,0 if frame['kind']=='decision' else -1);elapsed+=frame['ms']
                file=capture.finish();activity_cache[cache_key]=file
                records.append(dict(phase=phase,delay=delay,prefix=prefix,activity=file,readout=file,
                                    activity_reused_same_core_and_complete_history=False))
                progress(out,stage='matched_memory',phase=phase,delay=delay,prefix=prefix,
                         completed_histories=len(records),elapsed_seconds=time.monotonic()-start)
        policy.frozen_audit()
    policy.load(SESSION/model['checkpoints']['after']['path']);free=[]
    for delay in (0,20,40,80,160,320,640):
        for prefix in (0,1,2,4,8):
            stem=f'free-after-d{delay}-p{prefix}'
            if (out/(stem+'-behavior.json')).exists():
                rows=read(out/(stem+'-behavior.json'));free.extend(rows);continue
            cap=MemoryCapture(policy,out,stem,cases,dense=delay in (160,640) and prefix==4)
            rows=memory_rollout(policy,task,delay,prefix,capture=cap);cap.finish()
            write(out/(stem+'-behavior.json'),rows);free.extend(rows)
            progress(out,stage='free_memory',phase='after',delay=delay,prefix=prefix,
                completed_histories=len(free),elapsed_seconds=time.monotonic()-start)
    train=[r for r in free if r['history_split']=='train'];held=[r for r in free if r['history_split']=='heldout']
    count=sum(r['status']=='solved' for r in train);expected=model['source_acceptance']
    if expected['solved'] is not None and count!=expected['solved']:
        raise AssertionError(f'Original memory training-history acceptance changed: {count} vs {expected["solved"]}')
    if expected['original_solved'] is True and count!=len(train):raise AssertionError('Previously solved model lost a trained history')
    write(out/'matched-index.json',records)
    return dict(status='complete',kind='memory',model=model['id'],matched_records=len(records),
        phase_core_hashes=phase_hashes,training_solved=count,training_total=len(train),
        heldout_solved=sum(r['status']=='solved' for r in held),heldout_total=len(held),
        source_acceptance_matches=True,frozen_audit=policy.frozen_audit(),role=model['role'])


def preflight(job,out):
    from mechanism_model import Frozen,Recording,save_index,memory_rollout,state_intervention,tasklib
    from game import encode,restore,state_from
    import torch
    model=model_by_id('difficulty-D1-T0');policy=Frozen(model,out,batch=1)
    policy.load(SESSION/model['checkpoints']['after']['path']);save_index(policy)
    level=task_level(model['task']);x=encode(restore(level),state_from(level['start']))
    started=time.monotonic();q0,f0=policy.advance(x,reset=True);plain_seconds=time.monotonic()-started;rec=Recording(policy)
    started=time.monotonic()
    q1,f1=policy.advance(x,reset=True,recorder=rec)
    recorded_seconds=time.monotonic()-started
    np.testing.assert_allclose(f0,f1,atol=1e-5,rtol=1e-5);np.testing.assert_allclose(q0,q1,atol=1e-5,rtol=1e-5)
    static_error=float(np.max(np.abs(f0-f1)));policy.frozen_audit()
    sample=rec.finish();started=time.monotonic()
    save_npz(out/'recorder-sample.npz',mean=sample['mean'],std=sample['std'],group=sample['group'],instantaneous=sample['instantaneous'])
    pilot_storage=dict(sample_bytes=(out/'recorder-sample.npz').stat().st_size,
        simulated_neuron_ms=40*policy.encoder.brain.n,write_seconds=time.monotonic()-started,
        fields=FIELDS,float_dtype='float32',lossless=True,
        forward_seconds=plain_seconds,recorded_forward_seconds=recorded_seconds)
    # Release the first graph before constructing a memory model on the same lane.
    del rec,policy;torch.cuda.empty_cache()
    model=model_by_id('remote-dynamic-M0-head_only');policy=Frozen(model,out/'memory',batch=2)
    pair=policy.load(SESSION/model['checkpoints']['after']['path']);task=task_memory('M0-audit');cases=task['cases'];lib=tasklib()
    def prefix(record=False):
        policy.reset(2);features=[]
        xs=[lib.frames(c,160,4,False) for c in cases]
        for i,frame in enumerate(xs[0]):
            r=Recording(policy) if record else None
            q,f=policy.advance(np.stack([s[i]['board'] for s in xs]),frame['ms'],recorder=r);features.append(f)
        return np.stack(features),q
    started=time.monotonic();a,qa=prefix();memory_plain_seconds=time.monotonic()-started
    started=time.monotonic();b,qb=prefix(True);memory_recorded_seconds=time.monotonic()-started
    np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-5);np.testing.assert_allclose(qa,qb,atol=1e-5,rtol=1e-5)
    snapshot=policy.encoder.capture_state(detach=True)
    state_intervention(policy,'restore',cases)
    now=policy.encoder.capture_state(detach=True)
    assert all(torch.equal(v,now['tensors'][k]) for k,v in snapshot['tensors'].items())
    rows=memory_rollout(policy,task,160,4)
    sham=memory_rollout(policy,task,160,4,intervention=dict(kind='sham',window='read',part='all'))
    restored=memory_rollout(policy,task,160,4,intervention=dict(kind='restore',window='read',part='all'))
    for other in (sham,restored):
        assert [r['status'] for r in rows]==[r['status'] for r in other]
        for r,s in zip(rows,other):
            assert [d['action'] for d in r['decisions']]==[d['action'] for d in s['decisions']]
            np.testing.assert_array_equal([d['logits'] for d in r['decisions']],[d['logits'] for d in s['decisions']])
    assert all(r['status']=='solved' for r in rows)
    h=head_arrays(pair);basis,_=decision_basis(h);z=standardized(a,h)
    null=z-(z@basis)@basis.T
    centered=h['proposal.weight'].astype(np.float64);centered-=centered.mean(0)
    null_error=float(np.max(np.abs(null@centered.T)))
    assert null_error<1e-4
    for name,sha in read(SESSION/'plan.json')['backend_data'].items():assert digest(ROOT/name)==sha
    return dict(status='complete',static_feature_error=static_error,memory_feature_error=float(np.max(np.abs(a-b))),
        centered_readout_rank=basis.shape[1],nullspace_error=null_error,sham_exact=True,restore_exact=True,
        memory_original_solved=True,frozen_audit=policy.frozen_audit(),clock_rng_preserved=True,recorder_pilot=pilot_storage,
        memory_plain_seconds=memory_plain_seconds,memory_recorded_seconds=memory_recorded_seconds,
        backend_mapping_hashes_verified=True)


def main(job_id):
    plan=read(SESSION/'plan.json')
    assert plan['code_manifest']==code_manifest(),'Frozen mechanism code changed'
    assert plan['scientific_sources']==scientific_sources(),'Scientific source drift'
    job=next(j for j in plan['jobs'] if j['id']==job_id);out=SESSION/'runs'/job_id
    out.mkdir(parents=True,exist_ok=True);started=time.time()
    if (out/'result.json').exists():raise RuntimeError('A preserved result already exists; do not overwrite it')
    try:
        for model in ([model_by_id(job['model'])] if 'model' in job else []):
            for phase,c in model['checkpoints'].items():assert digest(SESSION/c['path'])==c['sha256']
        if job['kind']=='preflight':value=preflight(job,out)
        elif job['kind']=='static':value=static_record(job,out)
        elif job['kind']=='memory':value=memory_record(job,out)
        else:
            from mechanism_causal import dispatch
            value=dispatch(job,out)
        value.update(job=job_id,wall_seconds=time.time()-started,pid=os.getpid(),finished_unix=time.time(),
                     parameters_updated=False,code_sha256=object_hash(code_manifest()))
        write(out/'result.json',value);progress(out,stage='complete',elapsed_seconds=time.time()-started)
        return 0
    except BaseException as exc:
        reason='stopped' if isinstance(exc,InterruptedError) else 'oom' if 'out of memory' in str(exc).lower() else 'numerical' if isinstance(exc,FloatingPointError) else 'error'
        write(out/'failure.json',dict(status=reason,error_type=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(),pid=os.getpid(),wall_seconds=time.time()-started))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    raise SystemExit(main(a.job))
