"""Persistent-state experiments, full-history gradients, lesions and cue swaps."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
import bootstrap
from common import configure,save_checkpoint,cpu_state
from game import encode,legal,restore,solved,state_from,step
from registry import STUDY,ROOT,SOURCE,read,write,append,digest,provenance
from backend import StudyBrain,pair,set_pair
from readout_models import ActivityLinear
from memory_tasks import prepare_memory,frames,hidden_board,prefix_states
from training import rng_state,restore_rng


class GRU(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(400,28,batch_first=True);self.head=nn.Linear(28,4);self.state=None
    def reset(self):self.state=None
    def advance(self,board,ms):
        if ms%20:raise ValueError('GRU baseline uses predeclared 20-ms observation ticks')
        x=torch.as_tensor(board,dtype=torch.float32,device='cuda').reshape(1,1,400).expand(1,ms//20,400)
        _,self.state=self.gru(x,self.state);return self.head(self.state[-1])


class MemoryPolicy:
    def __init__(self,config,out):
        self.config=config;self.kind=config['condition'];self.encoder=None;self.head=None
        configure(10);free,total=torch.cuda.mem_get_info()
        if free<13*2**30:raise RuntimeError('resource_gate: less than 13 GiB free')
        torch.cuda.set_per_process_memory_fraction(11*2**30/total)
        if self.kind=='gru':self.gru=GRU().cuda()
        else:
            self.encoder=StudyBrain(batch=1,precision=config['precision'],metadata_dir=out)
            self.head=ActivityLinear().cuda()
            saved=torch.load(SOURCE,map_location='cpu',weights_only=True);set_pair(self.encoder,self.head,saved)
            if self.kind in ('carry_untrained','head_only'):self.encoder.core.requires_grad_(False)
            if self.kind=='carry_untrained':self.head.requires_grad_(False)
        self.train_core=self.kind in ('joint','reset')
        self.frozen_features={}

    def parameters(self):
        if self.kind=='gru':return list(self.gru.parameters())
        return list(self.head.parameters())+(list(self.encoder.core.parameters()) if self.train_core else [])

    def reset(self):
        if self.kind=='gru':self.gru.reset()
        else:self.encoder.reset()

    def advance(self,board,ms=40,grad=False):
        if self.kind=='gru':return self.gru.advance(board,ms),self.gru.state[-1]
        x=torch.as_tensor(board,dtype=torch.float32,device='cuda')[None]
        reset=self.kind=='reset'
        if grad and self.train_core:f=self.encoder.checkpointed_advance(x,ms,reset=reset)
        else:f=self.encoder.advance(x,ms,reset=reset)
        return self.head(f)['final_logits'],f

    def saved(self):
        return dict(gru=cpu_state(self.gru)) if self.kind=='gru' else pair(self.encoder,self.head)

    def load(self,saved):
        if self.kind=='gru':self.gru.load_state_dict(saved['gru'])
        else:set_pair(self.encoder,self.head,saved)


def replay_prefix(policy,task,cue,delay,prefix,grad=False):
    policy.reset()
    for frame in frames(task,cue,delay,prefix,False)[:-1]:policy.advance(frame['board'],frame['ms'],grad)


def probe(policy,task,delay,prefix,intervention=None):
    result=[];prefix_snapshots=[]
    with torch.no_grad():
        if intervention=='swap':
            for cue in (0,1):
                replay_prefix(policy,task,cue,delay,prefix)
                prefix_snapshots.append(policy.encoder.capture_state(detach=True))
        for cue in (0,1):
            replay_prefix(policy,task,cue,delay,prefix)
            if intervention=='swap':policy.encoder.restore_state(prefix_snapshots[1-cue])
            elif intervention and intervention!='restore':policy.encoder.intervene(intervention)
            elif intervention=='restore':
                state=policy.encoder.capture_state(detach=True);policy.encoder.intervene('all');policy.encoder.restore_state(state)
            level=task['levels'][cue];q,f=policy.advance(hidden_board(level,state_from(task['query_state'])))
            # The short probe has a fixed, cue-independent two-response alphabet.
            choice=int(q[0,2:].argmax())+2
            result.append(dict(cue=cue,choice=choice,correct=choice==task['choice_targets'][cue],
                logits=q[0].cpu().tolist(),features=f[0].cpu().tolist() if f is not None else None))
    return dict(delay_ms=delay,prefix_windows=prefix,intervention=intervention,
        correct=sum(r['correct'] for r in result),total=2,records=result)


def hidden_rollout(policy,task,cue,delay,prefix,max_steps=512):
    level=task['levels'][cue];m=restore(level);s=state_from(task['query_state']);decisions=[]
    with torch.no_grad():
        replay_prefix(policy,task,cue,delay,prefix)
        status='action_limit'
        for t in range(max_steps):
            mask=legal(m,s)
            if not mask.any():status='blocked';break
            q,f=policy.advance(hidden_board(level,s));q=q[0].cpu().numpy()
            a=int(np.where(mask,q,-np.inf).argmax());nxt=step(m,s,a)
            decisions.append(dict(state=[s[0],list(s[1])],action=a,logits=q.tolist(),
                features=f[0].cpu().tolist() if f is not None else None))
            if solved(m,nxt):status='solved';s=nxt;break
            # Both terminal corridors are geometrically dead ends. Only a box
            # irreversibly committed to the *wrong* corridor is a proven failure.
            if nxt[1]!=s[1] and ((cue==0 and nxt[1][0]>34) or (cue==1 and nxt[1][0]<34)):
                status='wrong_irreversible_push';s=nxt;break
            s=nxt
            # Deliberately no board-only cache and no repeated-board loop stop.
    return dict(cue=cue,delay_ms=delay,prefix_windows=prefix,status=status,steps=len(decisions),decisions=decisions)


def assess(policy,task,full=False):
    cfg=read(STUDY/'protocol.json')
    delays=cfg['delay_ms'] if full else cfg['trained_delay_ms']
    prefixes=cfg['prefix_windows'] if full else cfg['trained_prefix_windows']
    probes,rolls=batched_assessment(policy,task,delays,prefixes)
    for r in probes+rolls:r['history_split']='train' if r['delay_ms'] in cfg['trained_delay_ms'] and r['prefix_windows'] in cfg['trained_prefix_windows'] else 'heldout'
    return dict(probes=probes,rollouts=rolls,probe_correct=sum(r['correct'] for r in probes),
        probe_total=2*len(probes),solved=sum(r['status']=='solved' for r in rolls),total=len(rolls),
        min_choice_margin=min(r['logits'][task['choice_targets'][r['cue']]]-r['logits'][5-task['choice_targets'][r['cue']]]
            for p in probes for r in p['records']))


def batched_assessment(policy,task,delays,prefixes,max_steps=512,intervention=None):
    """Batch histories and free rollouts; same board may retain different states."""
    from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
    cases=[(cue,d,n) for d in delays for n in prefixes for cue in (0,1)]
    batch=len(cases);sequences=[];boundaries=[]
    for cue,d,n in cases:
        fs=frames(task,cue,d,n,False)[:-1]
        sequences.append(np.concatenate([np.repeat(f['board'][None],f['ms'],axis=0) for f in fs]))
        boundaries.append(np.cumsum([0]+[f['ms'] for f in fs[:-1]]).tolist())
    with torch.no_grad():
        if policy.kind=='gru':
            xs=[torch.as_tensor(s[::20].reshape(-1,400),device='cuda') for s in sequences]
            packed=pack_padded_sequence(pad_sequence(xs,batch_first=True),[len(x) for x in xs],
                batch_first=True,enforce_sorted=False)
            _,policy.gru.state=policy.gru.gru(packed)
        else:
            e=policy.encoder;e.reset_batch(batch);e.bind();b=e.brain
            length=max(map(len,sequences));data=np.zeros((length,batch,4,10,10),np.float32);resets={}
            for row,s in enumerate(sequences):
                offset=length-len(s);data[offset:,row]=s
                for begin in boundaries[row] if policy.kind=='reset' else [0]:
                    resets.setdefault(offset+begin,[]).append(row)
            inputs=torch.as_tensor(data,device='cuda')
            for tick in range(length):
                if tick in resets:b.reset(rows=resets[tick])
                if tick==0 or tick%20==0:
                    b.set_drive(e.input_index,inputs[tick].reshape(batch,400).repeat_interleave(8,1)*18)
                b.step()
            e.counts['neuron_ms']+=length*batch*167106
            if intervention=='swap':
                # Adjacent rows are cue pairs with exactly the same delay/prefix.
                indices=torch.arange(batch,device='cuda')^1
                snapshot=e.capture_state(detach=True)
                for name,value in snapshot['tensors'].items():
                    snapshot['tensors'][name]=value.index_select(1 if name in ('spike_buf','g_slow_cls') else 0,indices)
                e.restore_state(snapshot)
            elif intervention=='restore':
                snapshot=e.capture_state(detach=True);e.intervene('all');e.restore_state(snapshot)
            elif intervention:e.intervene(intervention)
        states=[state_from(task['query_state']) for _ in cases]
        maps=[restore(task['levels'][cue]) for cue,_,_ in cases]
        records=[dict(cue=cue,delay_ms=d,prefix_windows=n,status='running',decisions=[]) for cue,d,n in cases]
        probe_records=[]
        for index in range(max_steps):
            if (STUDY/'STOP').exists():raise KeyboardInterrupt('Study STOP requested')
            boards=np.stack([hidden_board(task['levels'][cue],s) for (cue,_,_),s in zip(cases,states)])
            if policy.kind=='gru':
                x=torch.as_tensor(boards,device='cuda').reshape(batch,1,400).expand(batch,2,400)
                _,policy.gru.state=policy.gru.gru(x,policy.gru.state)
                q=policy.gru.head(policy.gru.state[-1]).cpu().numpy();features=policy.gru.state[-1].cpu().numpy()
            else:
                f=policy.encoder.advance(torch.as_tensor(boards,device='cuda'),reset=policy.kind=='reset')
                q=policy.head(f)['final_logits'].cpu().numpy();features=f.cpu().numpy()
            if not np.isfinite(q).all():raise FloatingPointError('Nonfinite memory policy logits')
            for row,((cue,d,n),s,m,r) in enumerate(zip(cases,states,maps,records)):
                if index==0:
                    choice=int(q[row,2:].argmax())+2
                    probe_records.append(dict(cue=cue,choice=choice,correct=choice==task['choice_targets'][cue],
                        logits=q[row].tolist(),features=features[row].tolist() if features is not None else None))
                if r['status']!='running':continue
                mask=legal(m,s)
                if not mask.any():r['status']='blocked';continue
                a=int(np.where(mask,q[row],-np.inf).argmax());nxt=step(m,s,a)
                r['decisions'].append(dict(state=[s[0],list(s[1])],action=a,logits=q[row].tolist(),
                    features=features[row].tolist() if features is not None else None))
                if solved(m,nxt):r['status']='solved'
                elif nxt[1]!=s[1] and ((cue==0 and nxt[1][0]>34) or (cue==1 and nxt[1][0]<34)):
                    r['status']='wrong_irreversible_push'
                states[row]=nxt
            if all(r['status']!='running' for r in records):break
        for r in records:
            if r['status']=='running':r['status']='action_limit'
            r['steps']=len(r['decisions'])
        probes=[]
        for i in range(0,len(cases),2):
            _,d,n=cases[i];rs=probe_records[i:i+2]
            probes.append(dict(delay_ms=d,prefix_windows=n,intervention=intervention,correct=sum(r['correct'] for r in rs),total=2,records=rs))
        if policy.encoder is not None:policy.encoder.reset_batch(1)
        else:policy.gru.reset()
    return probes,records


def training_step(policy,task,optimizer,iteration):
    cfg=read(STUDY/'protocol.json');delays=cfg['trained_delay_ms'];prefixes=cfg['trained_prefix_windows']
    delay=delays[iteration%len(delays)];prefix=prefixes[(iteration//len(delays))%len(prefixes)]
    optimizer.zero_grad(set_to_none=True);value=0.
    for cue in (0,1):
        policy.reset();losses=[]
        sequence=frames(task,cue,delay,prefix,True)
        key=tuple((frame['board'].astype('uint8').tobytes(),frame['ms']) for frame in sequence)
        cached=policy.frozen_features.get(key) if policy.kind=='head_only' else None
        if policy.kind=='head_only' and cached is None:
            cached=[]
            with torch.no_grad():
                for frame in sequence:
                    _,f=policy.advance(frame['board'],frame['ms'])
                    cached.append(f.detach().cpu())
            policy.frozen_features[key]=cached
        for index,frame in enumerate(sequence):
            if cached is not None:q=policy.head(cached[index].cuda())['final_logits']
            else:q,_=policy.advance(frame['board'],frame['ms'],True)
            if frame['target'] is not None:
                q=q.masked_fill(~torch.as_tensor(frame['legal'],device='cuda')[None],-1e9)
                losses.append(nn.functional.cross_entropy(q,torch.tensor([frame['target']],device='cuda')))
        loss=torch.stack(losses).mean()/2;loss.backward();value+=float(loss.detach())
        # Detach mutable simulator history only AFTER the entire cue-to-action
        # backward pass. Preserve accumulated parameter gradients for the pair.
        if policy.encoder is not None:
            policy.encoder.brain.detach_state()
    if policy.encoder is not None:
        head_norm=float(torch.nn.utils.clip_grad_norm_(policy.head.parameters(),5.))
        core_norm=float(torch.nn.utils.clip_grad_norm_(policy.encoder.core.parameters(),1.)) if policy.train_core else 0.
        norm=float(np.hypot(head_norm,core_norm))
    else:
        norm=float(torch.nn.utils.clip_grad_norm_(policy.parameters(),5.));head_norm=norm;core_norm=0.
    if not np.isfinite(value+norm):raise FloatingPointError('Nonfinite dynamic-memory gradient')
    optimizer.step()
    if policy.encoder is not None:policy.encoder.core.project()
    return dict(loss=value,gradient_norm=norm,head_gradient_norm=head_norm,core_gradient_norm=core_norm,
        delay_ms=delay,prefix_windows=prefix)


def worker(path,verify=False):
    from execution import stopped
    config=read(path);out=Path(path).parent;task=next(t for t in prepare_memory() if t['id']==config['task'])
    t=time.perf_counter();policy=None;result=dict(id=config['id'],config=config,pid=os.getpid(),status='initializing')
    try:
        policy=MemoryPolicy(config,out)
        if digest(SOURCE)!=config['source_sha256']:raise RuntimeError('Memory parent checkpoint changed')
        if verify:
            saved=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
            assert saved['config']==config
            assert read(out/'result.json')['pid']!=os.getpid()
            policy.load(saved['pair'])
            train_check=assess(policy,task,False);final=assess(policy,task,True)
            interventions=[];intervention_rollouts=[]
            if policy.encoder is not None:
                for kind in ('voltage','current','adaptation','delay','all','swap','restore'):
                    probes,rolls=batched_assessment(policy,task,read(STUDY/'protocol.json')['delay_ms'],[4],intervention=kind)
                    interventions.extend(probes);intervention_rollouts.extend(dict(intervention=kind,**r) for r in rolls)
            trained=train_check['rollouts']
            observed=all(r['status']=='solved' for r in trained)
            expected=read(out/'result.json').get('final_solved',False)
            write(out/'verification.json',dict(status='verified' if observed or not expected else 'reproduction_failed',
                pid=os.getpid(),training_pid=read(out/'result.json')['pid'],original_solved=observed,
                results=final,training_history_reload=train_check,interventions=interventions,
                intervention_rollouts=intervention_rollouts,model_sha256=digest(out/'model.pt')))
            return 0 if observed or not expected else 1
        optimizer=None
        if policy.kind!='carry_untrained':
            if policy.encoder is not None:
                groups=[dict(params=policy.head.parameters(),lr=.0003)]
                if policy.train_core:groups.append(dict(params=policy.encoder.core.parameters(),lr=.0001))
                optimizer=torch.optim.Adam(groups)
            else:optimizer=torch.optim.Adam(policy.parameters(),lr=.0003)
        initial=policy.saved()
        if not (out/'initial.pt').exists():save_checkpoint(out/'initial.pt',dict(pair=initial,config=config))
        state=dict(updates=0,assessments=0,elapsed=0.,phase=0,phase_start=0.,phase_score=None,phase_loss=None)
        champion=initial;best=(-1,-1,-float('inf'));last_loss=None
        if (out/'resume.pt').exists():
            s=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)
            assert s['config']==config
            state=s['state'];champion=s['champion'];best=s['best'];last_loss=s['last_loss'];policy.load(s['current'])
            if optimizer is not None:optimizer.load_state_dict(s['optimizer'])
            restore_rng(s['rng'])
        train_start=time.perf_counter();elapsed_before=state['elapsed']
        elapsed=lambda:elapsed_before+time.perf_counter()-train_start
        def save():
            state['elapsed']=elapsed()
            save_checkpoint(out/'model.pt',dict(pair=champion,config=config))
            save_checkpoint(out/'resume.pt',dict(config=config,state=state,current=policy.saved(),champion=champion,
                best=best,last_loss=last_loss,optimizer=optimizer.state_dict() if optimizer else None,rng=rng_state()))
        while True:
            stopped(out);obs=assess(policy,task)
            state['assessments']=state.get('assessments',0)+1
            score=(obs['solved'],obs['probe_correct'],-last_loss if last_loss is not None else -1e9)
            if score>best:champion=policy.saved();best=score
            append(out/'history.jsonl',dict(updates=state['updates'],elapsed=elapsed(),
                solved=obs['solved'],total=obs['total'],probe_correct=obs['probe_correct'],probe_total=obs['probe_total'],loss=last_loss))
            write(out/'progress.json',dict(status='training',updates=state['updates'],elapsed=elapsed(),
                solved=obs['solved'],total=obs['total'],probe_correct=obs['probe_correct']))
            from behavior import archive_behavior
            archive_behavior(out,dict(assessment=state['assessments'],updates=state['updates'],**obs),state['assessments']);save()
            solved_keys=[f"{r['cue']}:{r['delay_ms']}:{r['prefix_windows']}" for r in obs['rollouts'] if r['status']=='solved']
            if state['phase_score'] is None:
                state.update(phase_score=list(score),phase_loss=last_loss,phase_margin=obs['min_choice_margin'],phase_solved=solved_keys)
            if obs['solved']==obs['total'] and obs['probe_correct']==obs['probe_total']:
                result['status']='learned_solved' if state['updates'] else 'already_solved';break
            if optimizer is None:result['status']='untrained_baseline';break
            if elapsed()-state['phase_start']>=[1200,2400,4800][state['phase']]:
                progress=set(state['phase_solved'])<=set(solved_keys) and obs['min_choice_margin']>=state['phase_margin']+.02
                append(out/'extensions.jsonl',dict(phase=state['phase'],progress=progress,score=score,
                    min_choice_margin=obs['min_choice_margin'],phase_margin=state['phase_margin'],
                    solved_histories=solved_keys,required_histories=state['phase_solved'],
                    gradients=g if 'g' in locals() else None,
                    bounds=policy.encoder.core.bound_audit() if policy.encoder is not None else None,
                    data_coverage=dict(trained_delays=read(STUDY/'protocol.json')['trained_delay_ms'],
                        trained_prefixes=read(STUDY/'protocol.json')['trained_prefix_windows'],updates=state['updates']),
                    behavior_assessment=state['assessments']))
                if not progress or state['phase']==2:result['status']='stagnation' if not progress else 'segment_limit';break
                state.update(phase=state['phase']+1,phase_start=elapsed(),phase_score=list(score),phase_loss=last_loss,
                    phase_margin=obs['min_choice_margin'],phase_solved=solved_keys)
            for _ in range(4):
                stopped(out);g=training_step(policy,task,optimizer,state['updates']);state['updates']+=1;last_loss=g['loss']
                append(out/'gradients.jsonl',dict(update=state['updates'],**g))
            if not (out/'mid.pt').exists() and state['updates']>=20:save_checkpoint(out/'mid.pt',dict(pair=policy.saved(),config=config))
        policy.load(champion);final=assess(policy,task)
        result.update(updates=state['updates'],training_seconds=elapsed(),final_solved=final['solved']==final['total'],
            final=final,model_sha256=digest(out/'model.pt'),
            training_compute=dict(policy.encoder.counts) if policy.encoder is not None else {},
            parameter_changes={component:{name:dict(changed=int((value!=initial[component][name]).sum()),
                l2=float((value-initial[component][name]).double().norm())) for name,value in values.items()}
                for component,values in champion.items()},
            trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad))
    except BaseException as e:
        if 'save' in locals():save()
        result.update(status='stopped' if isinstance(e,KeyboardInterrupt) else 'oom' if isinstance(e,torch.OutOfMemoryError) else 'numerical_error' if isinstance(e,FloatingPointError) else 'implementation_error',
            error=repr(e),traceback=traceback.format_exc());print(result['traceback'],flush=True)
        if verify:write(out/'verification.json',dict(status='error',error=repr(e),traceback=traceback.format_exc()))
    finally:
        if not verify:
            result['wall_seconds']=time.perf_counter()-t
            if torch.cuda.is_initialized():result['resources']=dict(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            write(out/'result.json',result)
            write(out/'progress.json',dict(status=result['status'],updates=result.get('updates')))
    return 0 if result['status'] in ('learned_solved','already_solved','untrained_baseline','stagnation','segment_limit') else 1


def run_memory(resume=False):
    from execution import child
    from run_cycle import acquire_worker_lock
    lock=acquire_worker_lock()
    try:
        for task in prepare_memory():
            for kind in ('carry_untrained','reset','head_only','joint','gru'):
                run_id=f"dynamic-{task['id']}-{kind}";out=STUDY/'runs'/run_id;out.mkdir(parents=True,exist_ok=True)
                config=dict(id=run_id,task=task['id'],condition=kind,module='dynamic_memory',seed=10,
                    family_id=task['id'],
                    precision=read(STUDY/'backend.json')['precision'],batch=1,source=str(SOURCE),
                    source_sha256=digest(SOURCE),code_version=provenance(),state_mode='reset' if kind=='reset' else 'persistent',
                    protocol_sha256=digest(STUDY/'protocol.json'),task_sha256=digest(STUDY/'tasks/memory.json'),
                    feature_kind='gru_hidden_28' if kind=='gru' else 'brain_activity_512',
                    internal_step_ms=20 if kind=='gru' else 1,
                    training_batch=1,evaluation_batching='independent history episodes; own training-history batch used for reload acceptance')
                if not (out/'config.json').exists():write(out/'config.json',config)
                command=[sys.executable,'-B','-X','utf8',str(ROOT/'workflow.py'),'study','worker','--config',str(out/'config.json')]
                if (out/'result.json').exists():
                    prior=read(out/'result.json')
                    if resume and prior['status'] in ('stopped','worker_exit') and (out/'resume.pt').exists():
                        assert read(out/'config.json')['code_version']==provenance(),'Resume requires the recorded code version'
                        (out/'result.json').rename(out/f'interruption-{int(time.time())}.json')
                    elif prior['status'] not in ('learned_solved','already_solved','untrained_baseline','stagnation','segment_limit'):
                        return 1
                elif read(out/'config.json')['code_version']!=provenance():
                    raise RuntimeError('Code changed before memory run/resume '+run_id)
                if not (out/'result.json').exists() and child(command,out/'worker.log',out):return 1
                if not (out/'verification.json').exists() and child(command+['--verify'],out/'verification.log',out):return 1
                if read(out/'verification.json')['status']!='verified':return 1
        return 0
    finally:lock.close()
