"""Scientific activity recording and discovery/confirmation synaptic interventions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'sokoban'))
import json
import os
import time
import subprocess
import numpy as np
import torch
import bootstrap
from game import encode,restore,state_from
from common import configure
from registry import STUDY,ROOT,read,write
from backend import initialize,set_pair,Matrix
from behavior import rollouts


def record_states(encoder,boards,out,stem,full_indices=(),discovery_mask=None):
    """Float32 scientific records; summaries include every connectome neuron."""
    groups=encoder.neuron_group;ng=int(groups.max())+1
    sizes=torch.bincount(groups,minlength=ng).float()
    fields=('spikes','g','v','adapt');counts=0
    totals=torch.zeros((4,encoder.brain.n),device='cuda');squares=torch.zeros_like(totals)
    group_means=[];group_times=torch.zeros((40,4,ng));features=[];time_features=[];full={}
    discovery_total=torch.zeros_like(totals);discovery_count=0
    with torch.no_grad():
        for start in range(0,len(boards),encoder.batch):
            part=boards[start:start+encoder.batch];n=len(part)
            x=np.zeros((encoder.batch,4,10,10),np.float32);x[:n]=part
            by_group=[];instant=[]
            selected={i-start:[] for i in full_indices if start<=i<start+n}
            def hook(brain,t):
                nonlocal counts,discovery_count
                values=torch.stack([getattr(brain,k)[:n] for k in fields])
                totals.add_(values.sum(1));squares.add_(values.square().sum(1));counts+=n
                selected_rows=np.ones(n,bool) if discovery_mask is None else discovery_mask[start:start+n]
                if np.any(selected_rows):
                    discovery_total.add_(values[:,selected_rows].sum(1));discovery_count+=int(np.sum(selected_rows))
                group=torch.zeros((4,n,ng),device='cuda')
                group.scatter_add_(2,groups[None,None].expand(4,n,-1),values)
                by_group.append((group/sizes[None,None]).cpu())
                current=brain.g[:n,encoder.observed].reshape(n,256,16).mean(2)
                voltage=(brain.v[:n,encoder.observed]-encoder.params.v_rest).reshape(n,256,16).mean(2)
                instant.append(torch.cat((current,voltage),1).cpu())
                for i in selected:selected[i].append(values[:,i].cpu())
            f=encoder.advance(torch.as_tensor(x,device='cuda'),reset=True,hook=hook)[:n].cpu().numpy()
            group=torch.stack(by_group).permute(2,0,1,3)
            features.append(f);group_means.append(group.mean(1).numpy());group_times.add_(group.sum(0))
            time_features.append(torch.stack(instant).permute(1,0,2).numpy())
            for i,values in selected.items():full[start+i]=torch.stack(values).numpy()
    mean=(totals/max(counts,1)).cpu().numpy();second=(squares/max(counts,1)).cpu().numpy()
    np.savez_compressed(out/(stem+'-neuron-stats.npz'),mean=mean,std=np.sqrt(np.maximum(0,second-mean**2)),
        discovery_mean=(discovery_total/max(1,discovery_count)).cpu().numpy(),
        fields=np.array(fields),observations=counts,body_ids=encoder.c.neurons.bodyId.to_numpy())
    # Group summaries retain per-state and per-ms axes, never count neurons as
    # independent trained models in downstream statistics.
    np.savez_compressed(out/(stem+'-group-traces.npz'),values=(group_times/len(boards)).numpy(),
        state_means=np.concatenate(group_means),fields=np.array(fields))
    np.savez_compressed(out/(stem+'-features.npz'),features=np.concatenate(features),instantaneous=np.concatenate(time_features))
    for index,values in full.items():
        np.savez_compressed(out/f'{stem}-state-{index}-full.npz',values=values,fields=np.array(fields),
            body_ids=encoder.c.neurons.bodyId.to_numpy(),milliseconds=np.arange(1,41))
    return np.concatenate(features)


def checkpoint_phases(out):
    phases=[('before',out/'initial.pt')]
    if (out/'mid.pt').exists():phases.append(('mid',out/'mid.pt'))
    else:
        options=list(out.glob('suffix-*.pt'))
        if options:
            levels=read(out/'config.json')['task']
            target=next(l['optimal_steps'] for name in ('levels.json','variants.json') for l in read(STUDY/'tasks'/name) if l['id']==levels)/2
            phases.append(('mid',min(options,key=lambda p:abs(int(p.stem.split('-')[1])-target))))
    phases.append(('after',out/'model.pt'))
    return phases


def record_run(run_id):
    from execution import tasks,stopped
    run=STUDY/'runs'/run_id;out=STUDY/'activity'/run_id;out.mkdir(parents=True,exist_ok=True)
    if (out/'recording.json').exists():return
    config=read(run/'config.json')
    if config['condition']['model']!='brain_linear':return
    encoder,head=initialize(config,out);levels=tasks()
    allbank=read(STUDY/'tasks/matched-bank.json')
    representative=config['module']=='difficulty' and config['task'] in read(STUDY/'protocol.json')['representative_tasks']
    own=[r for r in allbank if r['task']==config['task']]
    if not own:
        l=levels[config['task']]
        from lesson_tasks import trace
        rows,_=trace(restore(l),state_from(l['start']),l['reference_actions'])
        own=[dict(task=l['id'],state=[s[0],list(s[1])],index=i,role='descriptive',family=l['family_id']) for i,(s,a) in enumerate(rows)]
    selected=[own[0],own[len(own)//2],own[-1]]
    # Shared across all phases; representative models additionally observe both
    # discovery and confirmation families, regardless of their learning outcome.
    bank=[r for r in allbank if r['role'] in ('discovery','confirmation')] if representative else selected
    boards=np.stack([encode(restore(levels[r['task']]),state_from(r['state'])) for r in bank])
    full_ids=[bank.index(r) for r in selected if r in bank] if representative else []
    write(out/'bank.json',bank)
    if not (STUDY/'activity/neuron-index.parquet').exists():
        encoder.c.neurons.assign(type_group=encoder.neuron_group.cpu().numpy()).to_parquet(STUDY/'activity/neuron-index.parquet',index=False)
    phases=[]
    for phase,path in checkpoint_phases(run):
        stopped(run);saved=torch.load(path,map_location='cpu',weights_only=True);set_pair(encoder,head,saved)
        if not (out/(phase+'-features.npz')).exists():
            f=record_states(encoder,boards,out,phase,full_ids,np.array([r['role']=='discovery' for r in bank]))
        else:f=np.load(out/(phase+'-features.npz'))['features']
        with torch.no_grad():q=head(torch.as_tensor(f,device='cuda'))['final_logits'].cpu().numpy()
        np.savez_compressed(out/(phase+'-logits.npz'),logits=q)
        phases.append(dict(phase=phase,checkpoint=str(path.relative_to(ROOT)),states=len(boards)))
    # Actual free behavior is a separate sample, not pooled into matched states.
    set_pair(encoder,head,torch.load(run/'model.pt',map_location='cpu',weights_only=True))
    actual=read(run/'after.json')['decisions']
    if actual:
        indices=sorted({0,len(actual)//2,len(actual)-1})
        freeboards=np.stack([encode(restore(levels[config['task']]),state_from(actual[i]['state'])) for i in indices])
        record_states(encoder,freeboards,out,'free-events',range(len(indices)) if representative else [])
        write(out/'free-events.json',dict(indices=indices,decisions=[actual[i] for i in indices],
            status=read(run/'after.json')['status'],matched_analysis=False))
    if representative:
        opposite='float64' if config['precision']=='float32' else 'float32'
        encoder.matrix=Matrix(encoder.matrix.base,{'float64':torch.float64,'float32':torch.float32}[opposite]);encoder.brain.W=encoder.matrix
        f=encoder.encode(boards)
        with torch.no_grad():q=head(torch.as_tensor(f,device='cuda'))['final_logits'].cpu().numpy()
        behavior=rollouts(encoder,head,[levels[config['task']]])
        np.savez_compressed(out/'precision-sensitivity.npz',features=f,logits=q)
        write(out/'precision-sensitivity.json',dict(precision=opposite,behavior=behavior,checkpoint_unchanged=True))
    write(out/'recording.json',dict(status='complete',run_id=run_id,precision=config['precision'],
        batch=config['batch'],phases=phases,scientific_storage='float32, lossless npz compression',
        full_series_indices=full_ids,representative=representative,
        anatomical_groups='annotated type/class/superclass/somaNeuromere; no mesh-derived neuropil labels'))


def discover_groups(out,encoder):
    bank=read(out/'bank.json');select=np.array([r['role']=='discovery' for r in bank])
    before=np.load(out/'before-group-traces.npz')['state_means'][select]
    after=np.load(out/'after-group-traces.npz')['state_means'][select]
    # Predeclared discovery score: mean absolute within-state learning-induced
    # change in group membrane potential; confirmation data never enter ranking.
    scores=np.abs(after[:,2]-before[:,2]).mean(0)
    groups=encoder.neuron_group.cpu().numpy();sizes=np.bincount(groups)
    candidates=np.flatnonzero(sizes>=8)
    chosen=candidates[np.argsort(scores[candidates])[-3:][::-1]]
    stats=np.load(out/'before-neuron-stats.npz')['discovery_mean'][0]
    w=encoder.brain._W_cpu
    degree=np.diff(w.indptr)+np.bincount(w.indices,minlength=w.shape[0])
    # Neuron-level matching jointly uses activity and log degree, with replacement
    # prohibited. Retain achieved balance so quality is inspectable.
    cov=np.stack([stats,np.log1p(degree)],axis=1);scale=np.maximum(cov.std(0),1e-6);z=(cov-cov.mean(0))/scale
    selected=[];rng=np.random.default_rng(10)
    for g in chosen:
        ids=np.flatnonzero(groups==g);available=groups!=g;controls=[]
        for i in ids:
            distance=((z-z[i])**2).sum(1);distance[~available]=np.inf
            j=int(np.argmin(distance));controls.append(j);available[j]=False
        controls=np.array(controls)
        selected.append(dict(group=int(g),score=float(scores[g]),indices=ids.tolist(),controls=controls.tolist(),
            neurons=len(ids),covariates=['baseline_spike_probability','log1p_total_degree'],
            target_mean=cov[ids].mean(0).tolist(),control_mean=cov[controls].mean(0).tolist()))
    return selected


def intervene_run(run_id):
    from execution import tasks,stopped
    run=STUDY/'runs'/run_id;out=STUDY/'activity'/run_id
    if (out/'interventions.json').exists():return
    config=read(run/'config.json');encoder,head=initialize(config,out)
    set_pair(encoder,head,torch.load(run/'model.pt',map_location='cpu',weights_only=True))
    selected=discover_groups(out,encoder);write(out/'discovery-groups.json',selected)
    levels=tasks();bank=read(out/'bank.json')
    confirmation=[]
    for task_id in sorted({r['task'] for r in bank if r['role']=='confirmation'}):
        rows=[r for r in bank if r['task']==task_id]
        for r in (rows[0],rows[len(rows)//2],rows[-1]):
            confirmation.append(dict(levels[task_id],id=task_id+f"-confirm-{r['index']}",start=r['state']))
    baseline=rollouts(encoder,head,confirmation,lambda:stopped(run))
    records=[]
    for group in selected:
        for matched in (False,True):
            for strength in (.5,1.):
                for window in ((0,14),(14,40)):
                    encoder.intervention=dict(indices=group['controls'] if matched else group['indices'],strength=strength,window=window)
                    result=rollouts(encoder,head,confirmation,lambda:stopped(run))
                    records.append(dict(group=group['group'],matched_control=matched,strength=strength,window_ms=window,**result))
        encoder.intervention=None
    restored=rollouts(encoder,head,confirmation,lambda:stopped(run))
    write(out/'interventions.json',dict(mechanism='attenuate outgoing synaptic transmission of selected cells during the specified window',
        discovery_families=sorted({r['family'] for r in bank if r['role']=='discovery'}),
        confirmation_families=sorted({r['family'] for r in bank if r['role']=='confirmation'}),
        baseline=baseline,records=records,restored=restored))


def run_activity():
    from run_cycle import acquire_worker_lock
    lock=acquire_worker_lock()
    try:
        for j in read(STUDY/'schedule.json'):
            out=STUDY/'runs'/j['id']
            if not (out/'model.pt').exists() or j['condition']['model']!='brain_linear':continue
            for mode in ['record']+(['intervene'] if j['module']=='difficulty' and j['task'] in read(STUDY/'protocol.json')['representative_tasks'] else []):
                with (out/'activity.log').open('a',encoding='utf-8') as log:
                    code=subprocess.call([sys.executable,'-B','-X','utf8',str(Path(__file__)),mode,j['id']],cwd=ROOT,
                        stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                if code:return code
        for task in ('M0','M1','M2'):
            run_id=f'dynamic-{task}-joint';run=STUDY/'runs'/run_id
            if not (run/'model.pt').exists():continue
            with (run/'activity.log').open('a',encoding='utf-8') as log:
                code=subprocess.call([sys.executable,'-B','-X','utf8',str(Path(__file__)),'memory',run_id],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if code:return code
        return 0
    finally:lock.close()


def record_memory(run_id):
    from memory import MemoryPolicy
    from memory_tasks import prepare_memory,frames
    run=STUDY/'runs'/run_id;out=STUDY/'activity'/run_id;out.mkdir(parents=True,exist_ok=True)
    if (out/'recording.json').exists():return
    config=read(run/'config.json');policy=MemoryPolicy(config,out)
    task=next(t for t in prepare_memory() if t['id']==config['task']);phases=[]
    for phase,name in [('before','initial.pt'),('mid','mid.pt'),('after','model.pt')]:
        if not (run/name).exists():continue
        policy.load(torch.load(run/name,map_location='cpu',weights_only=True)['pair'])
        for cue in (0,1):
            policy.reset();elapsed=0;total=None;count=0
            for index,frame in enumerate(frames(task,cue,160,4,False)):
                values=[]
                def hook(brain,t):
                    values.append(torch.stack([getattr(brain,k)[0] for k in ('spikes','g','v','adapt')]).cpu())
                with torch.no_grad():
                    f=policy.encoder.advance(torch.as_tensor(frame['board'][None],device='cuda'),frame['ms'],hook=hook)
                    q=policy.head(f)['final_logits'][0].cpu().tolist()
                value=torch.stack(values).numpy()
                np.savez_compressed(out/f'{phase}-cue{cue}-segment{index}.npz',values=value,
                    milliseconds=np.arange(elapsed+1,elapsed+frame['ms']+1),
                    fields=np.array(['spikes','g','v','adapt']),features=f.cpu().numpy())
                total=value.sum(0) if total is None else total+value.sum(0);count+=len(value)
                phases.append(dict(phase=phase,cue=cue,segment=index,kind=frame['kind'],
                    start_ms=elapsed,duration_ms=frame['ms'],logits=q));elapsed+=frame['ms']
            np.savez_compressed(out/f'{phase}-cue{cue}-neuron-stats.npz',mean=total/count,
                body_ids=policy.encoder.c.neurons.bodyId.to_numpy())
    write(out/'recording.json',dict(status='complete',run_id=run_id,precision=config['precision'],
        scientific_storage='float32, lossless npz compression',event=dict(delay_ms=160,prefix_windows=4),segments=phases))


if __name__=='__main__':
    {'record':record_run,'intervene':intervene_run,'memory':record_memory}[sys.argv[1]](sys.argv[2])
