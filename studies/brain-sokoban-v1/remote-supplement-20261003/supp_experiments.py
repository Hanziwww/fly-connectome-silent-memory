"""Frozen-core memory learning, paired causal audits, and dense delay probes."""
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import supp_tasks as tasklib
from supp_tasks import SESSION, STUDY, ROOT, board, frames, state_key
from game import encode, restore, state_from, step, legal, solved
from registry import read, write, append, digest, SOURCE
from common import configure, cpu_state
from backend import StudyBrain, STATE_FIELDS
from readout_models import ActivityLinear
import memory as legacy

VERIFY=False
PROTOCOL=dict(trained_delay_ms=[0,40,160],trained_prefix_windows=[0,1,4],
              delay_ms=[0,20,40,80,160,320,640],prefix_windows=[0,1,2,4,8])
NEURAL_FIELDS=('v','g','g_slow','g_slow_cls','refrac','spikes','adapt','res','spike_buf','spike_counts')


def stopped(out=None):
    for p in (STUDY/'STOP',SESSION/'STOP',Path(out)/'STOP' if out else SESSION/'STOP'):
        if p.exists():raise KeyboardInterrupt('Supplement STOP requested')


class Policy:
    def __init__(self,config,out):
        self.config=config;self.out=out;self.encoder=None;self.head=None
        self.train_core=False;self.frozen_features={};self.state=None
        configure(10)
        free,total=torch.cuda.mem_get_info()
        if free<13*2**30:raise RuntimeError('resource_gate: less than 13 GiB free')
        torch.cuda.set_per_process_memory_fraction(11*2**30/total)
        self.kind=config.get('policy_kind','head_only')
        write(out/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),free_before_bytes=free,
            total_bytes=total,allocator_gib=11,seed=10,precision='mixed',pid=os.getpid()))
        if self.kind=='gru':
            self.gru=legacy.GRU().cuda()
        else:
            if self.kind!='leaky':
                self.encoder=StudyBrain(batch=1,precision='mixed',structure=config.get('structure','real'),metadata_dir=out)
                saved=torch.load(SOURCE,map_location='cpu',weights_only=True)
                self.encoder.set_core(saved['core'] if config.get('core_initialization','p05')=='p05' else None)
                self.encoder.core.requires_grad_(False)
            else:
                rng=np.random.default_rng(10)
                self.projection=torch.tensor(rng.normal(0,.05,(400,512)),dtype=torch.float32,device='cuda')
                self.taus=torch.tensor(np.resize([20.,40.,80.,160.,320.,640.],512),device='cuda')
            torch.manual_seed(10)
            self.head=ActivityLinear().cuda()
            if config.get('head_initialization','p05')=='p05' and self.encoder is not None:
                self.head.load_state_dict(saved['head'])
            if config.get('warm_start'):
                path=ROOT/config['warm_start']
                assert digest(path)==config['warm_start_sha256']
                checkpoint=torch.load(path,map_location='cpu',weights_only=True)
                self.load(checkpoint.get('pair',checkpoint))
            if config.get('head_initialization')=='fresh_calibrated' and not VERIFY:self.calibrate()

    def reset(self,batch=1):
        self.batch=batch
        if self.encoder is not None:self.encoder.reset_batch(batch)
        elif self.kind=='gru':self.gru.reset()
        else:self.state=torch.zeros((batch,512),device='cuda')

    def advance(self,x,ms=40,grad=False):
        x=torch.as_tensor(x,dtype=torch.float32,device='cuda')
        if x.ndim==3:x=x[None]
        if self.encoder is not None:
            with torch.no_grad():f=self.encoder.advance(x,ms)
            return self.head(f)['final_logits'],f
        if self.kind=='gru':
            assert ms%20==0
            _,self.gru.state=self.gru.gru(x.reshape(len(x),1,400).expand(len(x),ms//20,400),self.gru.state)
            f=self.gru.state[-1];return self.gru.head(f),f
        decay=torch.exp(-ms/self.taus).float()
        self.state=self.state*decay+(x.reshape(len(x),400)@self.projection)*(1-decay)
        return self.head(self.state)['final_logits'],self.state

    def parameters(self):
        return list(self.gru.parameters()) if self.kind=='gru' else list(self.head.parameters())

    def saved(self):
        if self.kind=='gru':return dict(gru=cpu_state(self.gru))
        result=dict(head=cpu_state(self.head))
        if self.encoder is not None:result['core']=cpu_state(self.encoder.core)
        return result

    def load(self,s):
        if self.kind=='gru':self.gru.load_state_dict(s['gru'])
        else:
            self.head.load_state_dict(s['head'])
            if self.encoder is not None:self.encoder.set_core(s['core'])

    def calibrate(self):
        features=[];examples=[]
        # Same fixed C0/C1 histories for all fresh-readout arms. No supplement test data.
        with torch.no_grad():
            for level in read(STUDY/'tasks/levels.json'):
                if level['id'] not in ('C0','C1'):continue
                m=restore(level);s=state_from(level['start']);states=[]
                for a in level['reference_actions']:
                    states.append(s);s=step(m,s,a)
                for index in sorted(set([0,len(states)//3,2*len(states)//3,len(states)-1])):
                    for delay in (0,40,160):
                        self.reset();x=encode(m,states[index]);self.advance(x,40)
                        if delay:self.advance(np.zeros_like(x),delay)
                        hidden=x.copy();hidden[1]=0
                        _,f=self.advance(hidden,40);features.append(f.cpu().numpy()[0])
                        examples.append(dict(task=level['id'],state_index=index,delay_ms=delay))
            f=np.stack(features)
            self.head.mean.copy_(torch.from_numpy(f.mean(0)).cuda())
            self.head.scale.copy_(torch.from_numpy(np.maximum(f.std(0),.01)).cuda())
        write(self.out/'normalization.json',dict(source='C0/C1 only, fixed before supplement training',
             examples=examples,mean=f.mean(0).tolist(),scale=np.maximum(f.std(0),.01).tolist(),
             fixed_after_calibration=True))


def apply_intervention(policy,kind,cases,donor=None):
    if kind in (None,'none'):return
    if kind=='reset_each_window':return
    assert policy.encoder is not None
    e=policy.encoder;b=e.brain
    snapshot=e.capture_state(detach=True)
    if kind in ('sham','restore'):
        if kind=='restore':apply_intervention(policy,'neural_clear',cases)
        e.restore_state(snapshot);return
    if kind=='same_cue':
        assert donor is not None;e.restore_state(donor);return
    if kind=='swap':
        indices=torch.tensor([next(i for i,d in enumerate(cases) if d['cue']==1-c['cue'] and
             d['context']==c['context'] and d.get('visible')==c.get('visible')) for c in cases],device='cuda')
        for name,value in snapshot['tensors'].items():
            snapshot['tensors'][name]=value.index_select(1 if name in ('g_slow_cls','spike_buf') else 0,indices)
        e.restore_state(snapshot);return
    fields={'all':NEURAL_FIELDS,'neural_clear':NEURAL_FIELDS,'voltage':['v'],
            'current':['g','g_slow','g_slow_cls'],'adaptation':['adapt'],'delay':['spike_buf']}[kind]
    clocks=(b.t,b.step_count,b.buf_pos);rng=b.gen.get_state().clone()
    for name in fields:
        value=getattr(b,name).clone()
        value.fill_(e.params.v_rest if name=='v' else 1. if name=='res' else 0.)
        setattr(b,name,value)
    b._rate_np_key=None
    assert clocks==(b.t,b.step_count,b.buf_pos) and torch.equal(rng,b.gen.get_state())


def prelude(policy,cases,delay,prefix,kind=None,window='decision',variant=0):
    policy.reset(len(cases))
    xs=[frames(c,delay,prefix,False,variant)[:-1] for c in cases]
    for index,frame in enumerate(xs[0]):
        x=np.stack([s[index]['board'] for s in xs])
        if kind=='reset_each_window':policy.reset(len(cases))
        if frame['kind']=='delay' and window=='delay_mid' and kind and frame['ms']>=40:
            half=(frame['ms']//40)*20
            policy.advance(x,half);apply_intervention(policy,kind,cases)
            policy.advance(x,frame['ms']-half)
        else:policy.advance(x,frame['ms'])
        if window=='cue_offset' and frame['kind']=='cue':apply_intervention(policy,kind,cases)


def one_assessment(policy,task,delay,prefix,intervention=None,window='decision'):
    cases=task['cases'];maps=[restore(c['level']) for c in cases]
    safe=[set(c['solvable_states']) for c in cases]
    states=[state_from(c['level']['start']) for c in cases]
    records=[dict(case=c['id'],cue=c['cue'],context=c['context'],visible=c.get('visible',False),
                  delay_ms=delay,prefix_windows=prefix,intervention=intervention,window=window,
                  status='running',first_push_correct=None,decisions=[]) for c in cases]
    donor=None
    with torch.no_grad():
        if intervention=='same_cue':
            prelude(policy,cases,delay,prefix,variant=1)
            donor=policy.encoder.capture_state(detach=True)
        prelude(policy,cases,delay,prefix,intervention,window)
        if window=='decision':apply_intervention(policy,intervention,cases,donor)
        for index in range(512):
            stopped(policy.out)
            if intervention=='reset_each_window':policy.reset(len(cases))
            q,f=policy.advance(np.stack([board(c,s) for c,s in zip(cases,states)]))
            q=q.cpu().numpy();f=f.cpu().numpy()
            if not np.isfinite(q).all():raise FloatingPointError('Nonfinite policy outputs')
            for row,(r,c,m,s) in enumerate(zip(records,cases,maps,states)):
                if r['status']!='running':continue
                mask=legal(m,s)
                if not mask.any():r['status']='blocked';continue
                action=int(np.where(mask,q[row],-np.inf).argmax());n=step(m,s,action)
                r['decisions'].append(dict(state=[s[0],list(s[1])],action=action,
                      logits=q[row].tolist(),features=f[row].tolist()))
                if n[1]!=s[1] and r['first_push_correct'] is None:
                    r['first_push_correct']=state_key(n) in safe[row]
                    r['first_push_action']=action
                if solved(m,n):r['status']='solved'
                elif n[1]!=s[1] and state_key(n) not in safe[row]:r['status']='irreversible_deadlock'
                states[row]=n
            if all(r['status']!='running' for r in records):break
    for r in records:
        if r['status']=='running':r['status']='action_limit'
        r['steps']=len(r['decisions'])
    probes=[]
    for r,c in zip(records,cases):
        first=r['decisions'][0] if r['decisions'] else dict(logits=[0.]*4,features=[])
        target=c['level']['reference_actions'][0]
        alternatives=legal(restore(c['level']),state_from(c['level']['start']))
        alternatives[target]=False
        competitors=np.array(first['logits'])[alternatives]
        margin=first['logits'][target]-float(competitors.max()) if len(competitors) else 0.
        probes.append(dict(delay_ms=delay,prefix_windows=prefix,intervention=intervention,
             correct=int(r['first_push_correct'] is True),total=1,
             records=[dict(cue=r['cue'],case=r['case'],context=r['context'],visible=r['visible'],
                           choice=r.get('first_push_action'),correct=r['first_push_correct'] is True,
                           logits=first['logits'],features=first['features'],entry_margin=margin)],
             measure='first irreversible push, not first movement'))
    return probes,records


def batched_assessment(policy,task,delays,prefixes,max_steps=512,intervention=None):
    assert max_steps==512
    if intervention and not policy.config.get('collect_interventions',True):return [],[]
    p=[];r=[]
    for delay in delays:
        for prefix in prefixes:
            a,b=one_assessment(policy,task,delay,prefix,intervention)
            p.extend(a);r.extend(b)
    return p,r


def assess(policy,task,full=False):
    delays=PROTOCOL['delay_ms'] if full else PROTOCOL['trained_delay_ms']
    prefixes=PROTOCOL['prefix_windows'] if full else PROTOCOL['trained_prefix_windows']
    probes,rolls=batched_assessment(policy,task,delays,prefixes)
    for row in probes+rolls:
        row['history_split']='train' if row['delay_ms'] in PROTOCOL['trained_delay_ms'] and row['prefix_windows'] in PROTOCOL['trained_prefix_windows'] else 'heldout'
    # Legacy extension machinery keys histories by cue/delay/prefix. A stable case
    # index incorporates context/visibility so retaining one cannot hide losing another.
    for r in rolls:
        r['cue_label']=r['cue'];r['cue']=next(i for i,c in enumerate(task['cases']) if c['id']==r['case'])
    return dict(probes=probes,rollouts=rolls,probe_correct=sum(x['correct'] for x in probes),
          probe_total=len(rolls),solved=sum(x['status']=='solved' for x in rolls),total=len(rolls),
          min_choice_margin=min(r['entry_margin'] for p in probes for r in p['records']),
          choice_measure='first irreversible push; arbitrary successful movement routes accepted')


def training_step(policy,task,optimizer,iteration):
    stopped(policy.out)
    delays=PROTOCOL['trained_delay_ms'];prefixes=PROTOCOL['trained_prefix_windows']
    delay=delays[iteration%len(delays)];prefix=prefixes[(iteration//len(delays))%len(prefixes)]
    optimizer.zero_grad(set_to_none=True);total=0.
    for case in task['cases']:
        sequence=frames(case,delay,prefix,True)
        key=tuple((f['board'].astype('uint8').tobytes(),f['ms']) for f in sequence)
        cached=policy.frozen_features.get(key) if policy.kind!='gru' else None
        if policy.kind!='gru' and cached is None:
            policy.reset();cached=[]
            with torch.no_grad():
                for frame in sequence:
                    _,f=policy.advance(frame['board'],frame['ms']);cached.append(f.detach().cpu())
            policy.frozen_features[key]=cached
        policy.reset();losses=[]
        for index,frame in enumerate(sequence):
            if cached is None:q,_=policy.advance(frame['board'],frame['ms'],True)
            else:q=policy.head(cached[index].cuda())['final_logits']
            if frame['target'] is not None:
                q=q.masked_fill(~torch.as_tensor(frame['legal'],device='cuda')[None],-1e9)
                losses.append(nn.functional.cross_entropy(q,torch.tensor([frame['target']],device='cuda')))
        loss=torch.stack(losses).mean()/len(task['cases']);loss.backward();total+=float(loss.detach())
    norm=float(torch.nn.utils.clip_grad_norm_(policy.parameters(),5.))
    if not np.isfinite(total+norm):raise FloatingPointError('Nonfinite gradient')
    optimizer.step()
    return dict(loss=total,gradient_norm=norm,head_gradient_norm=norm,core_gradient_norm=0.,
                delay_ms=delay,prefix_windows=prefix,training_cases=len(task['cases']))


def train(config_path,verify=False):
    global VERIFY
    VERIFY=verify
    legacy.MemoryPolicy=Policy
    legacy.prepare_memory=tasklib.prepare
    legacy.assess=assess
    legacy.batched_assessment=batched_assessment
    legacy.training_step=training_step
    return legacy.worker(config_path,verify)


def preflight(config,out):
    policy=Policy(config,out)
    task=next(t for t in tasklib.prepare() if t['id']=='M0-audit')
    observation=assess(policy,task)
    assert observation['solved']==18
    old=read(STUDY/'runs/remote-dynamic-M0-head_only/verification.json')['training_history_reload']
    prior={(r['cue'],r['delay_ms'],r['prefix_windows']):r for r in old['rollouts']}
    differences=[]
    for r in observation['rollouts']:
        o=prior[(r['cue_label'],r['delay_ms'],r['prefix_windows'])]
        assert [d['action'] for d in r['decisions']]==[d['action'] for d in o['decisions']]
        differences.append(float(np.max(np.abs(np.array(r['decisions'][0]['logits'])-o['decisions'][0]['logits']))))
    assert max(differences)<1e-5
    baseline=one_assessment(policy,task,40,4)[1]
    sham=one_assessment(policy,task,40,4,'sham')[1]
    restored=one_assessment(policy,task,40,4,'restore')[1]
    for rows in (sham,restored):
        for a,b in zip(baseline,rows):
            assert a['decisions']==b['decisions']
    # Verify paired context histories are identical before the reveal.
    for t in tasklib.prepare():
        if t['id'].startswith('S'):tasklib.validate_task(t)
    temporary_gru=legacy.GRU().cuda()
    cue=torch.as_tensor(frames(task['cases'][0],40,0,False)[0]['board'],device='cuda').reshape(1,1,400).clone().requires_grad_()
    _,hidden=temporary_gru.gru(cue.expand(1,2,400))
    _,hidden=temporary_gru.gru(torch.zeros((1,2,400),device='cuda'),hidden)
    objective=temporary_gru.head(hidden[-1])[0,2]
    gradient=torch.autograd.grad(objective,cue)[0]
    assert torch.isfinite(gradient).all() and float(gradient.norm())>0
    write(out/'result.json',dict(status='passed',pid=os.getpid(),original_solved=True,
        imported_M0_train_solved=18,original_logits_max_difference=max(differences),
        sham_and_restore_exact=True,core_frozen=all(not p.requires_grad for p in policy.encoder.core.parameters()),
        tasks_certified=True,local_training_started=False,gru_cue_gradient_norm=float(gradient.norm()),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    return 0


def probe_job(config,out):
    policy=Policy(config,out)
    task=next(t for t in tasklib.prepare() if t['id']==config['task'])
    if config.get('checkpoint_job'):
        source=SESSION/'runs'/config['checkpoint_job']/'model.pt'
        policy.load(torch.load(source,map_location='cpu',weights_only=True)['pair'])
    else:source=ROOT/config['warm_start']
    entries=[]
    if config['probe']=='dense_delay':
        entries=[(d,p,None,'decision') for d in range(0,641,20) for p in (0,1,2,4,8)]
    elif config['probe']=='transfer':
        entries=[(d,p,None,'decision') for d in PROTOCOL['delay_ms'] for p in PROTOCOL['prefix_windows']]
    else:
        for d in (0,40,160,320,640):
            for kind in (None,'sham','same_cue','swap','neural_clear','voltage','current','restore','reset_each_window'):
                entries.append((d,4,kind,'decision'))
        for d in (40,160):
            for window in ('cue_offset','delay_mid'):
                for kind in ('sham','neural_clear','voltage','current','swap','restore'):
                    entries.append((d,4,kind,window))
    checkpoint=out/'probe-resume.json'
    completed=read(checkpoint).get('completed',0) if checkpoint.exists() else 0
    t=time.perf_counter()
    from behavior import archive_behavior
    for index,(d,p,kind,window) in enumerate(entries):
        if index<completed:continue
        stopped(out)
        probes,rolls=one_assessment(policy,task,d,p,kind,window)
        value=dict(index=index,delay_ms=d,prefix_windows=p,intervention=kind,window=window,
                   probes=probes,rollouts=rolls)
        archive_behavior(out,value,index)
        # Compact summaries retain first-decision features for explanatory decoding.
        append(out/'observations.jsonl',dict(index=index,delay_ms=d,prefix_windows=p,intervention=kind,window=window,
            records=[dict(case=r['case'],cue=r['cue'],context=r['context'],visible=r['visible'],
                solved=r['status']=='solved',status=r['status'],first_push_correct=r['first_push_correct'],
                first_push_action=r.get('first_push_action'),steps=r['steps'],
                logits=r['decisions'][0]['logits'],features=r['decisions'][0]['features']) for r in rolls]))
        write(checkpoint,dict(completed=index+1,total=len(entries),model_sha256=digest(source)))
        write(out/'progress.json',dict(status='evaluating',completed=index+1,total=len(entries),elapsed=time.perf_counter()-t))
    write(out/'result.json',dict(status='completed',pid=os.getpid(),completed=len(entries),
        evaluation_only=True,model_sha256=digest(source),wall_seconds=time.perf_counter()-t,
        resource=dict(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30),
        acceptance='Evaluation completed; does not imply successful memory or task learning'))
    return 0
