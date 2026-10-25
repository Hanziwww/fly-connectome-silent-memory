"""Untrained adapters, 1-ms recording, and clock-preserving state interventions."""
from mechanism_base import *
import torch
import bootstrap
from backend import StudyBrain, set_pair
from readout_models import ActivityLinear
from game import encode, legal, restore, solved, state_from, step
from curriculum import static_dead


class Frozen:
    def __init__(self, model, out, batch=1):
        if os.name=='nt':
            raise RuntimeError('GPU mechanism workers run on the authorized remote host only')
        from common import configure
        configure(10)
        free,total=torch.cuda.mem_get_info()
        if free < 13*2**30: raise RuntimeError('resource_gate: less than 13 GiB GPU memory free')
        torch.cuda.set_per_process_memory_fraction(11*2**30/total)
        torch.set_num_threads(4)
        self.model=model; self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        write(self.out/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),seed=10,tf32=False,allocator_gib=11,precision=model['precision']))
        cfg=model['config'];cond=cfg.get('condition')
        structure=cond.get('structure','real') if isinstance(cond,dict) else cfg.get('structure','real')
        self.encoder=StudyBrain(batch=batch,precision=model['precision'],structure=structure,metadata_dir=self.out)
        self.head=ActivityLinear().cuda();self.head.requires_grad_(False)
        self.encoder.core.requires_grad_(False)
        self.initial_hash=None
        labels=self.encoder.c.neurons.superclass.fillna('unannotated').astype(str).to_numpy()
        names,index=np.unique(labels,return_inverse=True)
        self.coarse_names=names;self.coarse=torch.as_tensor(index,device='cuda')
        self.coarse_sizes=torch.bincount(self.coarse).clamp_min(1).float()
        self.reset_each=cond=='reset'

    def load(self,path):
        saved=load_pair(path);set_pair(self.encoder,self.head,saved)
        self.encoder.core.requires_grad_(False);self.head.requires_grad_(False)
        self.initial_hash=(tensors_hash(saved['core']),tensors_hash(saved['head']))
        return saved

    def frozen_audit(self):
        current=(tensors_hash(self.encoder.core.state_dict()),tensors_hash(self.head.state_dict()))
        if self.initial_hash != current: raise AssertionError('Frozen parameters changed')
        return dict(core_sha256=current[0],head_sha256=current[1],parameters_unchanged=True)

    def reset(self,batch=1):
        self.encoder.reset_batch(batch)

    def advance(self,x,ms=40,recorder=None,reset=False,gate=None):
        x=torch.as_tensor(np.asarray(x),dtype=torch.float32,device='cuda')
        if x.ndim==3:x=x[None]
        if self.reset_each:reset=True
        if gate is not None:self.encoder.matrix.gate=gate
        with torch.no_grad():
            f=self.encoder.advance(x,ms,reset=reset,hook=recorder.hook if recorder else None)
            q=self.head(f)['final_logits']
        return q.cpu().numpy(),f.cpu().numpy()


class Recording:
    def __init__(self, policy, valid=None, dense_rows=()):
        self.policy=policy;self.e=policy.encoder
        self.valid=valid or self.e.batch;self.steps=0
        self.sum=torch.zeros((5,self.valid,self.e.brain.n),dtype=torch.float64,device='cuda')
        self.square=torch.zeros_like(self.sum)
        self.group=[];self.instant=[];self.dense={int(i):[] for i in dense_rows}

    def hook(self,brain,t):
        value=torch.stack([getattr(brain,k)[:self.valid] for k in FIELDS])
        self.sum.add_(value);self.square.add_(value.double().square());self.steps+=1
        group=torch.zeros((5,self.valid,len(self.policy.coarse_names)),device='cuda')
        group.scatter_add_(2,self.policy.coarse[None,None].expand(5,self.valid,-1),value)
        self.group.append((group/self.policy.coarse_sizes[None,None]).cpu().numpy())
        g=brain.g[:self.valid,self.e.observed].reshape(self.valid,256,16).mean(2)
        v=(brain.v[:self.valid,self.e.observed]-self.e.params.v_rest).reshape(self.valid,256,16).mean(2)
        self.instant.append(torch.cat((g,v),1).cpu().numpy())
        for i in self.dense:self.dense[i].append(value[:,i].cpu().numpy())

    def finish(self):
        mean=self.sum/self.steps
        std=(self.square/self.steps-mean.square()).clamp_min(0).sqrt()
        return dict(mean=mean.float().cpu().numpy().transpose(1,0,2),
                    std=std.float().cpu().numpy().transpose(1,0,2),
                    group=np.stack(self.group).transpose(2,0,1,3),
                    instantaneous=np.stack(self.instant).transpose(1,0,2),
                    dense={k:np.stack(v).astype(np.float32) for k,v in self.dense.items()},steps=self.steps)


def save_index(policy):
    folder=SESSION/'neurons';folder.mkdir(exist_ok=True)
    e=policy.encoder
    with file_lock(SESSION/'index.lock'):
        if (folder/'index.npz').exists():
            with np.load(folder/'index.npz',allow_pickle=False) as old:
                np.testing.assert_array_equal(old['body_ids'],e.c.neurons.bodyId.to_numpy())
            return
        groups=e.neuron_group.cpu().numpy();observed=e.observed.cpu().numpy();inputs=e.inputs.ravel()
        input_mask=np.zeros(e.brain.n,bool);input_mask[inputs]=True
        output_mask=np.zeros(e.brain.n,bool);output_mask[observed]=True
        w=e.brain._W_cpu
        degree=np.diff(w.indptr)+np.bincount(w.indices,minlength=w.shape[0])
        save_npz(folder/'index.npz',body_ids=e.c.neurons.bodyId.to_numpy(),groups=groups,
            inputs=input_mask,observed=output_mask,observed_order=observed,
            populations=observed.reshape(256,16),degree=degree,coarse=policy.coarse.cpu().numpy(),
            coarse_names=policy.coarse_names.astype(str))
        # The exact model subset is the join spine; missing annotations stay missing.
        frame=e.c.neurons.copy();frame['parameter_group']=groups
        frame['direct_input']=input_mask;frame['observed_by_policy']=output_mask
        frame.to_parquet(folder/'annotations.parquet',index=False)
        write(folder/'metadata.json',dict(neurons=e.brain.n,effective_edges=w.nnz,
            fields=FIELDS,simulated=True,annotation_source='MaleCNS v1.0 model subset',
            soma_location_is_not_synaptic_neuropil=True,coarse_labels=policy.coarse_names.tolist()))


NEURAL=('v','g','g_slow','g_slow_cls','refrac','spikes','adapt','res','spike_buf','spike_counts')
PARTS={'voltage':('v',),'current':('g','g_slow','g_slow_cls'),
       'adaptation':('adapt',),'resource':('res',),'delay':('spike_buf',),'all':NEURAL}


def state_intervention(policy,kind,cases,part='all',donor=None):
    e=policy.encoder;b=e.brain;snapshot=e.capture_state(detach=True)
    if kind=='none':return
    clocks=(b.t,b.step_count,b.buf_pos);rng=b.gen.get_state().clone()
    if kind in ('sham','restore'):
        if kind=='restore':state_intervention(policy,'clear',cases,part)
        e.restore_state(snapshot)
    elif kind in ('swap','same_cue'):
        if kind=='same_cue':
            if donor is None:raise ValueError('Same-cue control requires an independently replayed donor')
            other=donor
        else:
            index=torch.tensor([next(j for j,d in enumerate(cases) if d['cue']==1-c['cue'] and
                    d['context']==c['context'] and d.get('visible',False)==c.get('visible',False))
                    for c in cases],device='cuda')
            other={'tensors':{k:v.index_select(1 if k in ('g_slow_cls','spike_buf') else 0,index)
                              for k,v in snapshot['tensors'].items()}}
        for name in PARTS[part]:snapshot['tensors'][name]=other['tensors'][name].clone()
        e.restore_state(snapshot)
    elif kind=='clear':
        for name in PARTS[part]:
            v=getattr(b,name).clone()
            v.fill_(e.params.v_rest if name=='v' else 1. if name=='res' else 0.)
            setattr(b,name,v)
        b._rate_np_key=None
    else:raise ValueError(kind)
    assert clocks==(b.t,b.step_count,b.buf_pos) and torch.equal(rng,b.gen.get_state())


def tasklib():
    path=STUDY/'remote-supplement-20261003'
    if str(path) not in sys.path:sys.path.append(str(path))
    import supp_tasks
    return supp_tasks


def memory_rollout(policy, task, delay, prefix, capture=None, intervention=None, gate_spec=None, variant=0):
    """Original hidden/visible inputs and 512-step acceptance; no board-only cache."""
    lib=tasklib();cases=task['cases'];policy.reset(len(cases))
    xs=[lib.frames(c,delay,prefix,False,variant)[:-1] for c in cases]
    spec=intervention or dict(kind='none',window='read',part='all')
    donor=None
    if spec['kind']=='same_cue':
        # Independent replay reaches exactly the recipient's intervention time.
        for i,frame in enumerate(xs[0]):
            length=frame['ms']//2 if spec['window']=='hold' and frame['kind']=='delay' else frame['ms']
            policy.advance(np.stack([x[i]['board'] for x in xs]),length)
            if spec['window']=='write' and frame['kind']=='cue':break
            if spec['window']=='hold' and frame['kind']=='delay':break
        donor=policy.encoder.capture_state(detach=True);policy.reset(len(cases))
    gate=None
    if gate_spec:
        gate=torch.ones(policy.encoder.brain.n,device='cuda')
        gate[gate_spec['indices']]=1.-gate_spec['strength']
    elapsed=0
    def advance(boards,ms,kind,decision_index=-1):
        nonlocal elapsed
        rec=capture.begin(kind,ms,elapsed,decision_index) if capture else None
        window='write' if kind=='cue' else 'hold' if kind in ('delay','prefix') else 'read'
        active=gate if gate_spec and gate_spec['window']==window and (window!='read' or decision_index==0) else None
        q,f=policy.advance(boards,ms,recorder=rec,gate=active)
        if capture:capture.end(rec,q,f,kind,ms,elapsed,decision_index)
        elapsed+=ms;return q,f
    for i,frame in enumerate(xs[0]):
        check_stop();boards=np.stack([x[i]['board'] for x in xs]);kind=frame['kind'];ms=frame['ms']
        if spec['window']=='hold' and kind=='delay' and ms>=2 and spec['kind']!='none':
            half=ms//2;advance(boards,half,kind)
            state_intervention(policy,spec['kind'],cases,spec['part'],donor)
            advance(boards,ms-half,kind)
        else:advance(boards,ms,kind)
        if spec['window']=='write' and kind=='cue':state_intervention(policy,spec['kind'],cases,spec['part'],donor)
    if spec['window']=='read':state_intervention(policy,spec['kind'],cases,spec['part'],donor)
    maps=[restore(c['level']) for c in cases];states=[state_from(c['level']['start']) for c in cases]
    safe=[set(c['solvable_states']) for c in cases]
    records=[dict(case=c['id'],cue=c['cue'],context=c['context'],visible=c.get('visible',False),
        delay_ms=delay,prefix_windows=prefix,status='running',first_push_correct=None,decisions=[]) for c in cases]
    for index in range(512):
        check_stop()
        q,f=advance(np.stack([lib.board(c,s) for c,s in zip(cases,states)]),40,'decision',index)
        if not np.isfinite(q).all():raise FloatingPointError('Nonfinite logits')
        for row,(r,c,m,s) in enumerate(zip(records,cases,maps,states)):
            if r['status']!='running':continue
            mask=legal(m,s)
            if not mask.any():r['status']='blocked';continue
            a=int(np.where(mask,q[row],-np.inf).argmax());n=step(m,s,a)
            r['decisions'].append(dict(state=[s[0],list(s[1])],action=a,legal=mask.tolist(),logits=q[row].tolist(),features=f[row].tolist()))
            if n[1]!=s[1] and r['first_push_correct'] is None:
                r['first_push_correct']=lib.state_key(n) in safe[row];r['first_push_action']=a
                if capture:capture.pushed.add(row)
            if solved(m,n):r['status']='solved'
            elif n[1]!=s[1] and lib.state_key(n) not in safe[row]:r['status']='irreversible_deadlock'
            states[row]=n
        if all(r['status']!='running' for r in records):break
    for r in records:
        if r['status']=='running':r['status']='action_limit'
        r['steps']=len(r['decisions'])
        r['history_split']='train' if delay in (0,40,160) and prefix in (0,1,4) else 'heldout'
    return records


def static_rollout(policy, level, gate_spec=None):
    m=restore(level);s=state_from(level['start']);seen={s};decisions=[];status='running'
    from types import MethodType
    e=policy.encoder
    for index in range(512):
        check_stop()
        if solved(m,s):status='solved';break
        if static_dead(m,s):status='proven_corner_deadlock';break
        mask=legal(m,s)
        if not mask.any():status='blocked';break
        if gate_spec:
            e.intervention=dict(indices=gate_spec['indices'],strength=gate_spec['strength'],
                               window={'write':(0,14),'hold':(14,26),'read':(26,40)}[gate_spec['window']])
        x=np.zeros((e.batch,4,10,10),np.float32);x[0]=encode(m,s)
        q,f=policy.advance(x,reset=True);e.intervention=None
        a=int(np.where(mask,q[0],-np.inf).argmax());n=step(m,s,a)
        decisions.append(dict(state=[s[0],list(s[1])],action=a,legal=mask.tolist(),logits=q[0].tolist(),features=f[0].tolist()))
        if solved(m,n):status='solved';s=n;break
        if n in seen:status='repeated_state';s=n;break
        seen.add(n);s=n
    if status=='running':status='action_limit'
    return dict(task=level['id'],family=level.get('family_id'),status=status,steps=len(decisions),decisions=decisions)
