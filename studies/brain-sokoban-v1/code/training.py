"""Paired training arms with resumable Adam, finite extensions and full observations."""
import json
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import bootstrap
from common import cpu_state,save_checkpoint
from game import encode,legal,restore,state_from
from linear_joint_tasks import SuffixCurriculum
from linear_joint_training import can_replace
from hybrid_tasks import objective as recovery
from outcome_objective import losses_and_margins
from registry import STUDY,read,write,append
from backend import pair,set_pair
from behavior import Features,rollouts


class Lessons(SuffixCurriculum):
    def data(self,depth):
        bridge,action=self.witness[-depth]
        targets={s:set(a) for s,a in self.learned.items()};targets.setdefault(bridge,set()).add(action)
        states=sorted(targets)
        return dict(states=states,boards=np.stack([encode(self.game,s) for s in states]).astype('uint8'),
            legal=np.stack([legal(self.game,s) for s in states]),
            clauses=[dict(kind='positive_witness',atoms=[(i,tuple(sorted(targets[s])))]) for i,s in enumerate(states)],
            bridge_clause=states.index(bridge),protected_clauses=[i for i,s in enumerate(states) if s in self.learned],depth=depth)


def objective(q,data,kind):
    ce,margins,correct=losses_and_margins(q,data)
    if kind=='recovery':value=recovery(margins,data)
    elif kind=='entry_only':value=.25*F.softplus((.1-margins[data['bridge_clause']])/.25)
    elif kind=='entry_ce':
        entry=.25*F.softplus((.1-margins[data['bridge_clause']])/.25)
        value=recovery(margins,data)-entry+ce[data['bridge_clause']]
    elif kind=='demonstration_ce':value=ce.mean()
    else:raise ValueError(kind)
    return value,margins,correct


def loss_metrics(q,data,kind):
    loss,margins,correct=objective(q,data,kind)
    return dict(loss=float(loss),bridge_margin=float(margins[data['bridge_clause']]),
        margins=margins.detach().cpu().tolist(),correct=int(correct.sum()),total=len(margins))


def gradients(encoder,head,data,condition,update,check):
    core_on=condition['train_core'];head_on=condition['train_head']
    if condition.get('update_mode')=='alternating':
        core_on=update%2==0;head_on=not core_on
    core_params=list(encoder.core.parameters()) if encoder.core is not None else []
    for p in [*core_params,*head.parameters()]:p.grad=None
    t=time.perf_counter()
    features=torch.as_tensor(encoder.encode(data['boards'],check),device='cuda').detach().requires_grad_(core_on)
    encode_seconds=time.perf_counter()-t
    t=time.perf_counter();value=objective(head(features)['final_logits'],data,condition['loss'])[0]
    value.backward();head_seconds=time.perf_counter()-t
    if not head_on:
        for p in head.parameters():p.grad=None
    discrepancy=0.;core_seconds=0.
    if core_on:
        desired=features.grad.detach();totals=[torch.zeros_like(p) for p in core_params]
        t=time.perf_counter()
        for start in range(0,len(features),encoder.batch):
            check();n=min(encoder.batch,len(features)-start)
            x=np.zeros((encoder.batch,4,10,10),dtype='float32');x[:n]=data['boards'][start:start+n]
            encoded=encoder.forward(torch.from_numpy(x).cuda())[:n]
            discrepancy=max(discrepancy,float((encoded.detach()-features[start:start+n].detach()).abs().max()))
            (encoded*desired[start:start+n]).sum().backward()
            for total,p in zip(totals,core_params):
                if p.grad is not None:total.add_(p.grad)
            encoder.release_graph()
        for p,total in zip(core_params,totals):p.grad=total
        core_seconds=time.perf_counter()-t
        # Same-config recomputation is required for a valid chain rule; this does
        # not compare FP32 behavior to FP64 behavior.
        tolerance=1e-3 if getattr(encoder,'precision','float64')=='float32' else 1e-5
        if discrepancy>tolerance:raise FloatingPointError(f'Same-config feature recomputation mismatch: {discrepancy} > {tolerance}')
    norm=lambda ps:float(torch.sqrt(sum((p.grad.square().sum() for p in ps if p.grad is not None),torch.zeros((),device='cuda'))))
    norms=dict(core=norm(core_params),head=norm(head.parameters()))
    if not all(np.isfinite(v) for v in [float(value),*norms.values()]):raise FloatingPointError('Nonfinite gradients/loss')
    return dict(loss=float(value),gradient_norms=norms,recompute_max_error=discrepancy,
        recompute_tolerance=1e-3 if getattr(encoder,'precision','float64')=='float32' else 1e-5,
        core_updated=core_on,head_updated=head_on,
        timings=dict(feature_forward=encode_seconds,head_backward=head_seconds,core_recompute_backward=core_seconds))


def assess(encoder,head,lessons,depth,condition,check):
    bank=Features(encoder);data=lessons.data(depth)
    with torch.no_grad():metrics=loss_metrics(head(torch.as_tensor(bank.encode(data['boards'],check),device='cuda'))['final_logits'],data,condition['loss'])
    levels=lessons.levels(depth) if condition['schedule']=='suffix' else [lessons.level,lessons.level]
    suite=rollouts(encoder,head,levels,check,features=bank);records=suite['records']
    obs=dict(**metrics,complete=True,solved=records[0]['status']=='solved',
        original_status=records[0]['status'],original_steps=records[0]['steps'],
        lesson_solved=records[1]['status']=='solved',retained=all(r['status']=='solved' for r in records[2:]))
    obs['stage_passed']=obs['solved'] or (obs['lesson_solved'] and obs['retained'])
    return obs,suite,data


def rng_state():
    n=np.random.get_state()
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),
        python=random.getstate(),numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]])


def restore_rng(r):
    torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda']);random.setstate(r['python'])
    n=r['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype='uint32'),n[2],n[3],n[4]))


def extension_allowed(state,obs):
    return state['promotions']>state['phase_promotions'] or (
        state['depth']==state['phase_depth'] and obs['retained'] and
        obs['bridge_margin']>=state['phase_margin']+.02)


def fit(encoder,head,level,config,out,check):
    cond=config['condition'];resume=out/'resume.pt'
    if not cond['train_core']:
        from behavior import FrozenEncoder
        encoder=FrozenEncoder(encoder)
    head.requires_grad_(cond['train_head'])
    lessons=Lessons(level);depth=1
    probe_levels=[];probe_matched=[]
    if config.get('probe_tasks'):
        from execution import tasks
        lookup=tasks();probe_levels=[lookup[t] for t in config['probe_tasks']]
        for task_id in config['probe_tasks']:
            origin=STUDY/'runs'/('difficulty-'+task_id)
            if not (origin/'after.json').exists():continue
            reference=read(origin/'after.json')
            if not reference['decisions']:continue
            ds=reference['decisions'];game=restore(lookup[task_id])
            probe_matched.append(dict(task=task_id,reference_solved=reference['status']=='solved',
                boards=np.stack([encode(game,state_from(d['state'])) for d in ds]),
                legal=np.stack([legal(game,state_from(d['state'])) for d in ds]),
                actions=np.asarray([d['action'] for d in ds])))
    def assessment(depth):
        obs,suite,data=assess(encoder,head,lessons,depth,cond,check)
        if probe_levels:
            extra=rollouts(encoder,head,probe_levels,check)
            obs['probe_results']=[dict(task=r['task'],status=r['status'],steps=r['steps']) for r in extra['records']]
            suite['probe_records']=extra['records']
            obs['probe_margins']=[]
            with torch.no_grad():
                for p in probe_matched:
                    q=head(torch.as_tensor(encoder.encode(p['boards'],check),device='cuda'))['final_logits'].cpu().numpy()
                    opponents=p['legal'].copy();opponents[np.arange(len(q)),p['actions']]=False
                    valid=opponents.any(1)
                    margin=q[np.arange(len(q)),p['actions']]-np.where(opponents,q,-np.inf).max(1)
                    obs['probe_margins'].append(dict(task=p['task'],reference_solved=p['reference_solved'],
                        margins=margin[valid].tolist(),mean_margin=float(margin[valid].mean()) if valid.any() else None))
        return obs,suite,data
    if cond['schedule']=='demonstration':
        lessons.learned={s:{a} for s,a in lessons.witness};depth=len(lessons.witness)
    state=dict(depth=depth,updates=0,assessments=0,promotions=0,phase=0,elapsed=0.,phase_start=0.,
        phase_promotions=0,phase_depth=depth,phase_margin=None,stage_initialized=False)
    champion=pair(encoder,head);champion_obs=None;optimizer=None;last_gradient={}
    def make_optimizer():
        groups=[]
        if cond['train_core'] and encoder.core is not None:groups.append(dict(params=list(encoder.core.parameters()),lr=.0001))
        if cond['train_head']:groups.append(dict(params=list(head.parameters()),lr=.0003))
        return torch.optim.Adam(groups) if groups else None
    optimizer=make_optimizer()
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        if saved['config']!=config:raise RuntimeError('Resume configuration changed')
        state=saved['state'];lessons=Lessons(level,saved['lessons']);champion=saved['champion'];champion_obs=saved['champion_obs']
        set_pair(encoder,head,saved['current'])
        if optimizer is not None and saved['optimizer'] is not None:optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'])
    started=time.perf_counter();previous_elapsed=state['elapsed'];last_io=0.
    def elapsed():return previous_elapsed+time.perf_counter()-started
    def save():
        nonlocal last_io
        t=time.perf_counter();state['elapsed']=elapsed()
        save_checkpoint(out/'model.pt',dict(**champion,config=config,architecture=head.architecture,
            lesson_depth=state['depth'],feedback_snapshot=lessons.snapshot(),updates=state['updates']))
        save_checkpoint(resume,dict(config=config,state=state,current=pair(encoder,head),champion=champion,
            champion_obs=champion_obs,optimizer=optimizer.state_dict() if optimizer is not None else None,
            lessons=lessons.snapshot(),rng=rng_state()))
        last_io=time.perf_counter()-t
    def log(obs,suite,data,event):
        state['assessments']+=1
        row=dict(event=event,depth=state['depth'],updates=state['updates'],elapsed=elapsed(),
                 phase=state['phase'],training_states=len(data['states']),checkpoint_io_seconds=last_io,**obs)
        append(out/'history.jsonl',row)
        from behavior import archive_behavior
        archive_behavior(out,dict(assessment=state['assessments'],depth=state['depth'],updates=state['updates'],**suite),state['assessments'])
        append(out/'exposure.jsonl',dict(assessment=state['assessments'],updates=state['updates'],states=data['states'],
            clauses=data['clauses'],bridge_clause=data['bridge_clause'],protected_clauses=data['protected_clauses']))
        write(out/'progress.json',dict(status='training',**row));save()
        if state['assessments']==1 or state['assessments']%5==0 or obs['stage_passed']:
            print(json.dumps(dict(run=config['id'],**row)),flush=True)
    stop='stagnation';before=None
    def finish():
        save()
        set_pair(encoder,head,champion)
        final=rollouts(encoder,head,[level],check)['records'][0]
        write(out/'after.json',final)
        return dict(status=stop,training_seconds=elapsed(),updates=state['updates'],
            promotions=state['promotions'],depth=state['depth'],phase=state['phase'],
            final_solved=final['status']=='solved',final_steps=final['steps'],
            training_compute=dict(encoder.counts),assessments=state['assessments'])
    try:
        while state['depth']<=len(lessons.witness):
            check();depth=state['depth']
            if not state['stage_initialized']:
                set_pair(encoder,head,champion)
                obs,suite,data=assessment(depth)
                if before is None and state['updates']==0:
                    before=suite['records'][0];write(out/'before.json',before)
                champion_obs=obs
                if state['phase_margin'] is None:state['phase_margin']=obs['bridge_margin']
                optimizer=make_optimizer();state['stage_initialized']=True
                log(obs,suite,data,'stage_baseline')
            else:
                obs,suite,data=assessment(depth)
            if obs['solved']:
                champion=pair(encoder,head);champion_obs=obs
                stop='already_solved' if state['updates']==0 else 'learned_solved';break
            if optimizer is None:
                stop='frozen_baseline';break
            if obs['stage_passed'] and cond['schedule']=='suffix':
                champion=pair(encoder,head);champion_obs=obs
                save_checkpoint(out/f'suffix-{depth}.pt',dict(**champion,config=config,depth=depth))
                for task,r in zip(lessons.levels(depth),suite['records']):
                    if r['status']=='solved':lessons.observe_success(task,r)
                state['promotions']+=1
                state['depth']+=1;state['stage_initialized']=False
                save();continue
            while True:
                check()
                if elapsed()-state['phase_start']>=config['segments_seconds'][state['phase']]:
                    progress=extension_allowed(state,champion_obs)
                    append(out/'extensions.jsonl',dict(phase=state['phase'],progress=progress,
                        elapsed=elapsed(),depth=depth,promotions=state['promotions'],
                        bounds=encoder.core.bound_audit() if encoder.core is not None else None,
                        gradients=last_gradient,loop=suite['records'][0],states=len(data['states'])))
                    if not progress or state['phase']==2:
                        stop='stagnation' if not progress else 'segment_limit';return finish()
                    state.update(phase=state['phase']+1,phase_start=elapsed(),phase_promotions=state['promotions'],
                        phase_depth=depth,phase_margin=champion_obs['bridge_margin'])
                    save()  # Preserve Adam and the current exploration parameters at a time extension.
                for _ in range(4):
                    check();t=time.perf_counter()
                    last_gradient=gradients(encoder,head,data,cond,state['updates'],check)
                    if encoder.core is not None:torch.nn.utils.clip_grad_norm_(encoder.core.parameters(),1.)
                    torch.nn.utils.clip_grad_norm_(head.parameters(),5.)
                    optimizer.step()
                    if encoder.core is not None:encoder.core.project()
                    encoder.release_graph();state['updates']+=1
                    append(out/'gradients.jsonl',dict(update=state['updates'],depth=depth,
                        total_seconds=time.perf_counter()-t,**last_gradient))
                obs,suite,data=assessment(depth)
                if can_replace(obs,champion_obs):champion=pair(encoder,head);champion_obs=obs
                log(obs,suite,data,'update')
                if not (out/'mid.pt').exists() and (state['depth']>=max(1,len(lessons.witness)//2) or state['elapsed']>=600):
                    save_checkpoint(out/'mid.pt',dict(**pair(encoder,head),config=config,phase='mid',updates=state['updates']))
                if obs['stage_passed']:break
            if obs['solved']:
                stop='learned_solved';break
        return finish()
    except BaseException:
        save();raise

