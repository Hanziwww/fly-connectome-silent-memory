"""Discovery, held-out confirmation and sparse pathway summaries; no fitting of policies."""
from mechanism_base import *
import itertools


def groups_from_scores(scores,baseline,line,out):
    import pandas as pd
    with np.load(SESSION/'neurons/index.npz') as f:
        ids=f['body_ids'];groups=f['groups'];inputs=f['inputs'];observed=f['observed'];degree=f['degree']
    annotation=pd.read_parquet(SESSION/'neurons/annotations.parquet')
    selected=[];eligible=[]
    for g in np.unique(groups):
        rows=np.flatnonzero((groups==g)&~inputs)
        if len(rows)>=8:eligible.append((float(scores[rows].mean()),int(g),rows))
    eligible.sort(key=lambda x:(-x[0],x[1]))
    targets=eligible[:3];excluded=inputs.copy()
    for _,_,rows in targets:excluded[rows]=True
    covariates=np.stack((baseline,np.log1p(degree)),1).astype(float)
    scale=covariates[~inputs].std(0);scale[scale<1e-8]=1.
    normalized=(covariates-covariates[~inputs].mean(0))/scale
    for score,g,rows in targets:
        pool=~excluded;controls=[];distances=[]
        # Deterministic nearest-neighbour matching, exact policy-observation stratum.
        for i in sorted(rows,key=lambda k:(not observed[k],-degree[k],int(k))):
            candidates=np.flatnonzero(pool&(observed==observed[i]))
            if not len(candidates):raise RuntimeError('Insufficient matched control neurons')
            d=((normalized[candidates]-normalized[i])**2).sum(1)
            chosen=int(candidates[int(np.argmin(d))]);pool[chosen]=False
            controls.append(chosen);distances.append(float(d.min()**.5))
        labels=annotation.iloc[rows]['type'].dropna().astype(str).unique().tolist()
        selected.append(dict(id=f'{line}-g{g}',parameter_group=g,type_labels=labels,
            score=score,indices=rows.tolist(),body_ids=ids[rows].tolist(),
            controls=controls,control_body_ids=ids[controls].tolist(),
            count=len(rows),noninput=True,observed_count=int(observed[rows].sum()),
            match_mean_distance=float(np.mean(distances)),
            target_covariate_mean=covariates[rows].mean(0).tolist(),
            control_covariate_mean=covariates[controls].mean(0).tolist()))
    save_npz(out/'neuron-scores.npz',scores=scores.astype('float32'),baseline=baseline.astype('float32'))
    payload=dict(line=line,locked_unix=time.time(),selection_data='T0 maps' if line=='learning' else 'trained M0 histories',
                 validation_data='T1 maps' if line=='learning' else 'S1 family; S2 scope',groups=selected,
                 matching_covariates=['spike_probability','log1p_total_degree'],observed_membership_exact=True)
    write(out/'groups.json',payload)
    return dict(status='complete',groups=len(selected),selection_sha256=object_hash(payload),line=line)


def discover_learning(job,out):
    scores=[];baseline=[]
    for d in range(8):
        folder=SESSION/'runs'/f'record-difficulty-D{d}-T0'
        with np.load(folder/'matched-before.npz') as b,np.load(folder/'matched-after.npz') as a:
            scores.append(np.abs(a['mean'][:,2]-b['mean'][:,2]).mean(0))
            baseline.append(b['mean'][:,0].mean(0))
    return groups_from_scores(np.mean(scores,0),np.mean(baseline,0),'learning',out)


def discover_memory(job,out):
    folder=SESSION/'runs/record-remote-dynamic-M0-head_only'
    rows=[r for r in read(folder/'matched-index.json') if r['phase']=='after' and r['delay'] in (0,40,160) and r['prefix'] in (0,1,4)]
    # Scale each state component across discovery cue/history observations only.
    states=[];baseline=[]
    for r in rows:
        with np.load(folder/r['activity']) as f:
            states.append(f['mean'][-1,:,1:]);baseline.append(f['mean'][-1,:,0].mean(0))
    states=np.stack(states);scale=states.reshape(-1,4,states.shape[-1]).std(0)
    floors=np.maximum(np.median(scale,axis=1)*.01,1e-6)
    scale=np.maximum(scale,floors[:,None])
    score=np.sqrt(np.mean(((states[:,0]-states[:,1])/scale)**2,axis=(0,1)))
    save_npz(out/'discovery-scale.npz',scale=scale.astype('float32'))
    return groups_from_scores(score,np.mean(baseline,0),'memory',out)


def comparison(rows,baseline):
    answer=[]
    for a,b in zip(rows,baseline):
        answer.append(dict(case=a['case'],cue=a['cue'],context=a['context'],visible=a['visible'],
            status=a['status'],baseline_status=b['status'],solved_effect=int(a['status']=='solved')-int(b['status']=='solved'),
            first_push_correct=a['first_push_correct'],steps=a['steps'],baseline_steps=b['steps'],
            first_logits=a['decisions'][0]['logits'] if a['decisions'] else None,
            baseline_logits=b['decisions'][0]['logits'] if b['decisions'] else None))
    return answer


def assert_same(rows,baseline):
    for a,b in zip(rows,baseline):
        assert a['status']==b['status'] and len(a['decisions'])==len(b['decisions'])
        assert [r['action'] for r in a['decisions']]==[r['action'] for r in b['decisions']]
        np.testing.assert_allclose([r['logits'] for r in a['decisions']],
            [r['logits'] for r in b['decisions']],rtol=1e-5,atol=1e-5)


def memory_interventions(job,out):
    from mechanism_model import Frozen,memory_rollout
    from mechanism_worker import task_memory
    model=model_by_id(job['model']);task=task_memory(model['task'])
    policy=Frozen(model,out,batch=len(task['cases']));policy.load(SESSION/model['checkpoints']['after']['path'])
    specifications=[]
    if job['kind']=='components':
        for window,part,kind in itertools.product(('write','hold','read'),('voltage','current','adaptation','resource','delay','all'),('clear','swap')):
            specifications.append(dict(id=f'{window}-{part}-{kind}',intervention=dict(window=window,part=part,kind=kind)))
        for window,kind in itertools.product(('write','hold','read'),('sham','restore','same_cue')):
            specifications.append(dict(id=f'{window}-{kind}',intervention=dict(window=window,part='all',kind=kind),control=True))
    else:
        groups=read(SESSION/'runs/discover-memory/groups.json')
        write(out/'discovery-reference.json',dict(sha256=object_hash(groups),role=model['role']))
        for g,side,strength,window in itertools.product(groups['groups'],('indices','controls'),(.25,.5,1.),('write','hold','read')):
            specifications.append(dict(id=f'{g["id"]}-{side}-{strength}-{window}',
                gate_spec=dict(indices=g[side],strength=strength,window=window),group=g['id'],side=side))
        specifications += [dict(id='sham-gate',gate_spec=dict(indices=groups['groups'][0]['indices'],strength=0.,window='read'),control=True),
                           dict(id='restore',intervention=dict(kind='restore',part='all',window='read'),control=True)]
    outcomes=[];total=6*(len(specifications)+1);done=0
    for delay,prefix in itertools.product((40,160,640),(0,4)):
        check_stop();basepath=out/'conditions'/f'd{delay}-p{prefix}-baseline.json'
        if basepath.exists():baseline=read(basepath)['rows']
        else:
            baseline=memory_rollout(policy,task,delay,prefix)
            write(basepath,dict(delay=delay,prefix=prefix,id='baseline',rows=baseline))
        done+=1
        for spec in specifications:
            check_stop();target=out/'conditions'/f'd{delay}-p{prefix}-{spec["id"]}.json'
            if target.exists():value=read(target)
            else:
                rows=memory_rollout(policy,task,delay,prefix,intervention=spec.get('intervention'),gate_spec=spec.get('gate_spec'))
                if spec.get('control'):assert_same(rows,baseline)
                value=dict(delay=delay,prefix=prefix,**spec,rows=rows,paired=comparison(rows,baseline))
                write(target,value)
            outcomes.append(dict(delay=delay,prefix=prefix,id=spec['id'],paired=value['paired']))
            done+=1;progress(out,stage=job['kind'],completed_conditions=done,total_conditions=total,delay=delay,prefix=prefix,condition=spec['id'])
    write(out/'effects.json',outcomes)
    return dict(status='complete',model=model['id'],conditions=done,frozen_audit=policy.frozen_audit(),
        discovery_role=model['role'],sham_and_restore_passed=True,parameters_updated=False)


def static_interventions(job,out):
    from mechanism_model import Frozen,static_rollout
    from mechanism_worker import task_level
    model=model_by_id(job['model']);level=task_level(model['task'])
    policy=Frozen(model,out,batch=16);policy.load(SESSION/model['checkpoints']['after']['path'])
    groups=read(SESSION/'runs/discover-learning/groups.json')
    write(out/'discovery-reference.json',dict(sha256=object_hash(groups),role='T1 confirmation'))
    baseline=static_rollout(policy,level);write(out/'baseline.json',baseline);values=[]
    for g,side,strength,window in itertools.product(groups['groups'],('indices','controls'),(.25,.5,1.),('write','hold','read')):
        check_stop();name=f'{g["id"]}-{side}-{strength}-{window}'
        target=out/'conditions'/(name+'.json')
        if target.exists():value=read(target)
        else:
            result=static_rollout(policy,level,dict(indices=g[side],strength=strength,window=window))
            value=dict(id=name,group=g['id'],side=side,strength=strength,window=window,
                baseline_solved=baseline['status']=='solved',solved=result['status']=='solved',behavior=result)
            write(target,value)
        values.append({k:v for k,v in value.items() if k!='behavior'})
        progress(out,stage='group_confirmation',completed_conditions=len(values),total_conditions=len(groups['groups'])*18)
    restored=static_rollout(policy,level)
    assert restored==baseline,'Removal of outgoing gate did not restore exact baseline'
    sham=static_rollout(policy,level,dict(indices=groups['groups'][0]['indices'],strength=0.,window='read'))
    assert sham==baseline,'Sham outgoing gate changed behavior'
    write(out/'effects.json',values)
    return dict(status='complete',model=model['id'],conditions=len(values),sham_exact=True,restore_exact=True,
        frozen_audit=policy.frozen_audit(),role='confirmation')


def pathways(job,out):
    from mechanism_model import Frozen
    import scipy.sparse as sp
    model=model_by_id('remote-dynamic-M0-head_only');p=Frozen(model,out,batch=1)
    pair=p.load(SESSION/model['checkpoints']['after']['path']);groups=p.encoder.neuron_group.cpu().numpy()
    # Stored matrix is postsynaptic x presynaptic; use the trained effective gains.
    w=p.encoder.brain._W_cpu.astype(np.float64).tocsr()
    pre=pair['core']['pre_log_gain'].exp().numpy()[groups]
    post=pair['core']['post_log_gain'].exp().numpy()[groups]
    w=sp.diags(post)@w@sp.diags(pre);obs=p.encoder.observed.cpu().numpy();answers=[]
    for line in ('learning','memory'):
        selected=read(SESSION/f'runs/discover-{line}/groups.json')
        for g in selected['groups']:
            for side in ('indices','controls'):
                indices=g[side];drive=np.asarray(w[:,indices].sum(1)).ravel()
                unsigned=np.asarray(abs(w[:,indices]).sum(1)).ravel()
                two=np.asarray(w[obs]@drive).ravel();twomass=np.asarray(abs(w[obs])@unsigned).ravel()
                answers.append(dict(line=line,group=g['id'],side=side,model=model['id'],
                    direct_signed=float(drive[obs].sum()),direct_absolute=float(unsigned[obs].sum()),
                    two_hop_signed=float(two.sum()),two_hop_absolute=float(twomass.sum()),
                    observed_reached_one_hop=int(np.count_nonzero(unsigned[obs])),
                    observed_reached_two_hop=int(np.count_nonzero(twomass))))
                save_npz(out/f'{g["id"]}-{side}.npz',direct_signed=drive.astype('float32'),
                         direct_absolute=unsigned.astype('float32'),observed_two_hop=two.astype('float32'))
    # Observe the actual delayed, resource-scaled input to the unchanged sparse operator.
    # This is an exact direct-synaptic contribution, not a product of post-step averages.
    from mechanism_worker import task_memory
    from mechanism_model import tasklib
    import torch
    selected=read(SESSION/'runs/discover-memory/groups.json')['groups']
    union=np.unique(np.concatenate([g[side] for g in selected for side in ('indices','controls')]))
    coefficients=[];labels=[]
    for g in selected:
        for side in ('indices','controls'):
            coefficient=np.zeros(len(union));places=np.searchsorted(union,g[side])
            coefficient[places]=np.asarray(w[obs][:,g[side]].sum(0)).ravel()
            coefficients.append(coefficient);labels.append(g['id']+'-'+side)
    coefficients=np.stack(coefficients)
    class FluxObserver:
        def __init__(self,delegate):
            object.__setattr__(self,'delegate',delegate);object.__setattr__(self,'frames',[])
            object.__setattr__(self,'indices',torch.as_tensor(union,device='cuda'))
        def __getattr__(self,name):return getattr(self.delegate,name)
        def __setattr__(self,name,value):setattr(self.delegate,name,value)
        def __matmul__(self,x):
            result=self.delegate@x
            self.frames.append(x[self.indices].detach().cpu().numpy())
            return result
    task=task_memory('M0-audit');cases=task['cases'];lib=tasklib()
    def replay():
        p.reset(len(cases));seq=[lib.frames(c,160,4,False) for c in cases]
        for i,f in enumerate(seq[0]):q,features=p.advance(np.stack([s[i]['board'] for s in seq]),f['ms'])
        return q,features
    original=replay();observer=FluxObserver(p.encoder.matrix);p.encoder.matrix=observer
    recorded=replay();np.testing.assert_array_equal(original[0],recorded[0]);np.testing.assert_array_equal(original[1],recorded[1])
    transmitted=np.stack(observer.frames)
    direct=np.einsum('gk,tkb->tbg',coefficients,transmitted)
    save_npz(out/'actual-direct-transmission.npz',transmitted=transmitted.astype('float32'),
        neuron_indices=union,direct_current_into_observed=direct.astype('float32'),group_labels=np.array(labels),
        milliseconds=np.arange(1,len(transmitted)+1))
    write(out/'pathways.json',dict(rows=answers,orientation='post x pre',
        interpretation='Signed structural transfer capacity, not a measured causal current or biological pathway.',
        direct_transmission='Separate exact observed operator inputs, including presynaptic resources and delay; M0 160 ms, prefix 4.',
        operator_observer_exact=True,
        biological_confirmation='Outgoing signal interventions and held-out behavior are evaluated separately.'))
    return dict(status='complete',path_summaries=len(answers),frozen_audit=p.frozen_audit())


def dispatch(job,out):
    return {'discover_learning':discover_learning,'discover_memory':discover_memory,
            'components':memory_interventions,'intervene_memory':memory_interventions,
            'intervene_static':static_interventions,'pathways':pathways}[job['kind']](job,out)
