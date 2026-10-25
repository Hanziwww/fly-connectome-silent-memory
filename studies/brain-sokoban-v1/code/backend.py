"""Full-connectome adapter; precision, state and structural controls are explicit."""
import dataclasses
import time
import numpy as np
import torch
from torch import nn
import bootstrap
from brain_model import FullBrain
from common import cpu_state
from readout_models import ActivityLinear, output
from registry import ROOT, STUDY, write

STATE_FIELDS=('v','g','g_slow','g_slow_cls','refrac','drive','poisson_p','rate',
              'spikes','adapt','res','spike_buf','spike_counts')


class Matrix:
    def __init__(self, base, dtype):
        self.base=base.to(dtype=dtype); self.dtype=dtype; self.pre=self.post=None
        self.profile=False; self.events=[]
        self.column_major=False
        self.gate=None

    def bind(self, core, groups):
        self.pre=core.pre_log_gain.exp()[groups].to(self.dtype)
        self.post=core.post_log_gain.exp()[groups].to(self.dtype)

    def __matmul__(self,x):
        if self.profile:
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
        pre=self.pre if self.gate is None else self.pre*self.gate.to(self.dtype)
        dense=x.to(self.dtype)*pre[:,None]
        if self.column_major:dense=dense.T.contiguous().T
        y=((self.base @ dense)*self.post[:,None]).float()
        if self.profile: b.record();self.events.append((a,b))
        return y


class _SegmentMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,matrix):
        ctx.matrix=matrix
        return matrix.multiply(x,False)
    @staticmethod
    def backward(ctx,gradient):
        return ctx.matrix.multiply(gradient,True),None


class SegmentedMatrix(Matrix):
    """Fixed-row FP32 reduction with an explicit transpose gradient.

    Avoid atomic accumulation order in cuSPARSE. Temporary storage is bounded to
    four dense columns of edge messages, with no gradient for constant edge weights.
    """
    def __init__(self,base):
        super().__init__(base,torch.float32)
        self.transposed=self.base.transpose(0,1).to_sparse_csr()
        self.lengths=[m.crow_indices().diff().to(torch.int64) for m in (self.base,self.transposed)]
    def multiply(self,x,transpose):
        m=self.transposed if transpose else self.base
        parts=[]
        for start in range(0,x.shape[1],4):
            messages=x[:,start:start+4].index_select(0,m.col_indices())*m.values()[:,None]
            parts.append(torch.segment_reduce(messages,'sum',lengths=self.lengths[int(transpose)],axis=0))
        return torch.cat(parts,1)
    def __matmul__(self,x):
        pre=self.pre if self.gate is None else self.pre*self.gate
        return _SegmentMultiply.apply(x.float()*pre[:,None],self)*self.post[:,None]


class _MixedMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,matrix):
        ctx.matrix=matrix
        return matrix.base@x
    @staticmethod
    def backward(ctx,gradient):
        return (ctx.matrix.backward_matrix@gradient.float()).double(),None


class MixedMatrix(Matrix):
    """FP64 deterministic forward sums, FP32 transpose sums for the surrogate gradient."""
    def __init__(self,base):
        super().__init__(base,torch.float64)
        self.backward_matrix=self.base.transpose(0,1).to_sparse_csr().float()
    def __matmul__(self,x):
        pre=self.pre if self.gate is None else self.pre*self.gate.double()
        return (_MixedMultiply.apply(x.double()*pre[:,None],self)*self.post[:,None]).float()


def structure_matrix(matrix, kind, seed=10):
    """Controls operate on the effective graph; preserve labelled neurons and signs.

    Rewire by directed double-edge swaps within presynaptic-sign strata. This
    exactly preserves binary in/out degrees, each source's sign and weight multiset.
    It does not claim to preserve target weighted strengths or spatial locality.
    """
    import scipy.sparse as sp
    rng=np.random.default_rng(seed)
    w=matrix.tocoo(copy=True); mask=w.data!=0
    row=w.row[mask].copy();col=w.col[mask].copy();val=w.data[mask].copy()
    original_row=row.copy()
    before_out=np.bincount(col,minlength=w.shape[0]);before_in=np.bincount(row,minlength=w.shape[0])
    before_sign=np.sign(val)
    audit=dict(kind=kind,nonzero_edges=len(val),seed=seed)
    if kind=='weight_shuffle':
        for sign in (-1,1):
            ids=np.flatnonzero(before_sign==sign);val[ids]=val[rng.permutation(ids)]
        audit.update(topology_unchanged=True,global_sign_specific_weights_preserved=True)
    elif kind=='rewired':
        # Disjoint double-edge proposals cover the full edge set. Reject edges
        # present before the round as well as duplicate proposed edges, so restoring
        # a rejected pair cannot create a new collision with an accepted pair.
        swaps=0
        for _ in range(3):
            a=rng.permutation(len(val))
            a=a[:len(a)//2*2].reshape(-1,2);i,j=a[:,0],a[:,1]
            ok=(before_sign[i]==before_sign[j])&(col[i]!=col[j])&(row[i]!=row[j])
            i,j=i[ok],j[ok]
            oldi,oldj=row[i].copy(),row[j].copy()
            existing=np.sort(row.astype('int64')*w.shape[0]+col)
            proposed=np.concatenate([oldj.astype('int64')*w.shape[0]+col[i],oldi.astype('int64')*w.shape[0]+col[j]])
            positions=np.searchsorted(existing,proposed);inside=positions<len(existing)
            collision=np.zeros(len(proposed),bool);collision[inside]=existing[positions[inside]]==proposed[inside]
            _,inv,counts=np.unique(proposed,return_inverse=True,return_counts=True)
            collision|=counts[inv]>1;n=len(i)
            bad=collision[:n]|collision[n:]|(oldj==col[i])|(oldi==col[j])
            row[i[~bad]],row[j[~bad]]=oldj[~bad],oldi[~bad];swaps+=int((~bad).sum())
        audit.update(successful_edge_swaps=swaps,presynaptic_weight_multisets_preserved=True,
                     target_weighted_strength_preserved=False,spatial_locality_preserved=False,
                     changed_edge_endpoint_fraction=float(np.mean(row!=original_row)))
    else: raise ValueError(kind)
    assert np.array_equal(before_out,np.bincount(col,minlength=w.shape[0]))
    assert np.array_equal(before_in,np.bincount(row,minlength=w.shape[0]))
    assert np.array_equal(before_sign,np.sign(val))
    audit.update(in_degree_exact=True,out_degree_exact=True,edge_signs_exact=True,
        absolute_weight_quantiles=np.quantile(np.abs(val),[0,.25,.5,.75,1]).tolist())
    result=sp.coo_matrix((val,(row,col)),shape=w.shape).tocsr()
    assert result.nnz==len(val)
    return result,audit


class StudyBrain(FullBrain):
    def __init__(self,batch=16,precision='float64',structure='real',metadata_dir=None):
        t=time.perf_counter()
        super().__init__(ROOT/'archive/experiments/campaign/run',batch=batch,persist_mapping=False)
        base=self.matrix.base
        if structure!='real':
            w,audit=structure_matrix(self.brain._W_cpu,structure)
            import hashlib
            h=hashlib.sha256()
            for a in (w.indptr,w.indices,w.data):h.update(a.tobytes())
            audit['effective_graph_sha256']=h.hexdigest()
            base=self.brain._sparse(w,torch.float32)
            if metadata_dir: write(metadata_dir/'structure.json',audit)
        self.precision=precision
        self.intervention=None
        if precision=='float32_segmented':self.matrix=SegmentedMatrix(base)
        elif precision=='mixed':self.matrix=MixedMatrix(base)
        else:self.matrix=Matrix(base,{'float32':torch.float32,'float64':torch.float64}[precision])
        self.brain.W=self.matrix
        # Drop the unused double-precision reference when the candidate uses FP32.
        self.reference=None
        self.counts=dict(forward_calls=0,neuron_ms=0,gradient_forward_calls=0,transfer_seconds=0.)
        self.initialization_seconds=time.perf_counter()-t
        if metadata_dir:
            metadata=dict(self.metadata,precision=precision,structure=structure,batch=batch,
                sparse_accumulation=('float64 forward; float32 gradient sums' if precision=='mixed' else precision),
                lif=dataclasses.asdict(self.params),slow_active=self.brain._slow_active,
                std_active=self.brain._std_on,K=self.brain.K,n_delay=self.brain.n_delay,
                adaptation_jump_range=[float(self.brain.adapt_jump_vec.min()),float(self.brain.adapt_jump_vec.max())]
                    if self.brain.adapt_jump_vec is not None else None,
                initialization_seconds=self.initialization_seconds)
            write(metadata_dir/'brain.json',metadata)
            np.savez_compressed(metadata_dir/'resolved-dynamics.npz',
                adapt_jump=self.brain.adapt_jump_vec.cpu().numpy() if self.brain.adapt_jump_vec is not None else [],
                std_u=self.brain.std_u_vec.cpu().numpy(),active=self.brain.active.cpu().numpy())

    def bind(self):
        self.matrix.bind(self.core,self.neuron_group);self.brain.W=self.matrix
        self.params.v_th=self.base_threshold+self.core.threshold_mv[self.neuron_group]

    def reset(self):
        self.brain.reset();self.brain.t=0.;self.brain.step_count=self.brain.buf_pos=0

    def reset_batch(self,batch):
        """Change independent episode count only at an episode boundary."""
        if batch==self.batch:
            self.reset();return
        self.release_graph();b=self.brain
        for name in STATE_FIELDS:
            old=getattr(b,name);shape=list(old.shape)
            axis=1 if name in ('g_slow_cls','spike_buf') else 0
            shape[axis]=batch
            value=torch.zeros(shape,dtype=old.dtype,device=old.device)
            if name=='v':value.fill_(self.params.v_rest)
            if name=='res':value.fill_(1.)
            setattr(b,name,value)
        b._syn_input=torch.zeros((b.n,batch),device=b.device)
        b.B=self.batch=batch;self.reset()

    def advance(self,boards,ms=40,reset=False,hook=None):
        if boards.shape!=(self.batch,4,10,10): raise ValueError('Incorrect brain batch')
        self.bind()
        if reset: self.reset()
        self.brain.set_drive(self.input_index,boards.reshape(self.batch,400).repeat_interleave(8,1)*18)
        if self.intervention is not None:
            user_hook=hook;spec=self.intervention
            gate=torch.ones(self.brain.n,device='cuda');gate[spec['indices']]=1.-spec['strength']
            self.matrix.gate=gate if spec['window'][0]==0 else None
            def lesion_hook(brain,t):
                if user_hook:user_hook(brain,t)
                self.matrix.gate=gate if spec['window'][0]<=t<spec['window'][1] else None
            hook=lesion_hook
        self.counts['forward_calls']+=1;self.counts['neuron_ms']+=self.batch*167106*ms
        self.counts['gradient_forward_calls']+=int(torch.is_grad_enabled())
        if ms==40 and hook is None:
            self.brain.run_ms(14);current=self.brain.g[:,self.observed]
            self.brain.run_ms(26)
        else:
            current=None
            for i in range(ms):
                self.brain.step()
                if i==13: current=self.brain.g[:,self.observed]
                if hook: hook(self.brain,i+1)
            if current is None: current=self.brain.g[:,self.observed]
        voltage=self.brain.v[:,self.observed]-self.params.v_rest
        self.matrix.gate=None
        return torch.cat((current.reshape(self.batch,256,16).mean(2),voltage.reshape(self.batch,256,16).mean(2)),1)

    def forward(self,boards,**kwargs):
        return self.advance(boards,reset=True)

    def checkpointed_advance(self,boards,ms=40,reset=False):
        """Recompute 14/26-ms or 20-ms chunks without truncating history gradients.

        The complete mutable state is an explicit input/output of each checkpoint;
        clock values are captured by value so backward does not depend on whatever
        state happens to remain in the simulator object.
        """
        from torch.utils.checkpoint import checkpoint,set_checkpoint_early_stop
        if reset:self.reset()
        current=None
        chunks=[14,26] if ms==40 else [min(20,ms-i) for i in range(0,ms,20)]
        for index,length in enumerate(chunks):
            clocks=(self.brain.t,self.brain.step_count,self.brain.buf_pos)
            states=tuple(getattr(self.brain,n) for n in STATE_FIELDS)
            calls=[0]
            def block(x,*incoming,length=length,clocks=clocks,calls=calls):
                for n,v in zip(STATE_FIELDS,incoming):setattr(self.brain,n,v)
                self.brain.t,self.brain.step_count,self.brain.buf_pos=clocks
                self.bind()
                self.brain.set_drive(self.input_index,x.reshape(self.batch,400).repeat_interleave(8,1)*18)
                self.brain.run_ms(length)
                self.counts['neuron_ms']+=self.batch*167106*length
                if calls[0]:self.counts['recomputed_neuron_ms']=self.counts.get('recomputed_neuron_ms',0)+self.batch*167106*length
                calls[0]+=1
                return tuple(getattr(self.brain,n) for n in STATE_FIELDS)
            with set_checkpoint_early_stop(False):
                values=checkpoint(block,boards,*states,use_reentrant=False,preserve_rng_state=True)
            for n,v in zip(STATE_FIELDS,values):setattr(self.brain,n,v)
            self.brain.t=clocks[0]+length
            self.brain.step_count=clocks[1]+length
            self.brain.buf_pos=(clocks[2]+length)%self.brain.n_delay
            if ms==40 and index==0:current=self.brain.g[:,self.observed]
        if current is None:current=self.brain.g[:,self.observed]
        voltage=self.brain.v[:,self.observed]-self.params.v_rest
        self.counts['forward_calls']+=1
        self.counts['gradient_forward_calls']+=1
        return torch.cat((current.reshape(self.batch,256,16).mean(2),voltage.reshape(self.batch,256,16).mean(2)),1)

    def encode(self,boards,check=lambda:None):
        features=[]
        with torch.no_grad():
            for start in range(0,len(boards),self.batch):
                check();part=boards[start:start+self.batch]
                x=np.zeros((self.batch,4,10,10),dtype='float32');x[:len(part)]=part
                t=time.perf_counter();x=torch.from_numpy(x).cuda();self.counts['transfer_seconds']+=time.perf_counter()-t
                features.append(self.forward(x)[:len(part)].cpu().numpy())
        return np.concatenate(features)

    def capture_state(self,detach=False):
        take=lambda t:t.detach().clone() if detach else t.clone()
        return dict(tensors={n:take(getattr(self.brain,n)) for n in STATE_FIELDS},
            acc={k:take(v) for k,v in self.brain._acc.items()},
            t=self.brain.t,step_count=self.brain.step_count,buf_pos=self.brain.buf_pos,
            poisson_on=self.brain._poisson_on,generator=self.brain.gen.get_state().clone())

    def restore_state(self,s):
        for name,value in s['tensors'].items(): setattr(self.brain,name,value.clone())
        self.brain._acc={k:v.clone() for k,v in s['acc'].items()}
        for name in ('t','step_count','buf_pos'):setattr(self.brain,name,s[name])
        self.brain._poisson_on=s['poisson_on'];self.brain.gen.set_state(s['generator'])
        self.brain._rate_np_key=None

    def intervene(self,part,rows=None):
        b=self.brain
        fields={'voltage':['v'],'current':['g','g_slow','g_slow_cls'],
                'adaptation':['adapt'],'delay':['spike_buf'],'all':list(STATE_FIELDS)}[part]
        for name in fields:
            value=getattr(b,name).clone()
            fill=self.params.v_rest if name=='v' else 1. if name=='res' else 0.
            if rows is None:value.fill_(fill)
            elif name in ('g_slow_cls','spike_buf'):value[:,rows]=fill
            else:value[rows]=fill
            setattr(b,name,value)
        if part=='all':
            b._acc={k:torch.zeros_like(v) for k,v in b._acc.items()}
            b._poisson_on=False
            b.t=0.;b.step_count=0;b.buf_pos=0;b.gen.manual_seed(0)
        b._rate_np_key=None


class BoardMLP(nn.Module):
    """400 -> 92 -> 4 = 37,264 parameters vs brain+readout 37,362."""
    architecture='board_mlp_400_92_4'
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(400,92),nn.ReLU(),nn.Linear(92,4))
    def forward(self,x,**kwargs):return output(self.net(x.reshape(len(x),400)))


class BoardEncoder:
    batch=128
    core=None
    counts={}
    def encode(self,boards,check=lambda:None):check();return np.asarray(boards,dtype='float32').reshape(-1,400)
    def release_graph(self):pass


def initialize(config,out):
    from common import configure
    import platform
    import sys
    configure(10)
    free,total=torch.cuda.mem_get_info()
    if free<13*2**30:raise RuntimeError('resource_gate: less than 13 GiB free GPU memory')
    torch.cuda.set_per_process_memory_fraction(11*2**30/total)
    write(out/'environment.json',dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,
        cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(),gpu_total_bytes=total,
        gpu_free_before_bytes=free,platform=platform.platform(),tf32=False,seed=10))
    if config['condition']['model']=='board_mlp':
        return BoardEncoder(),BoardMLP().cuda()
    encoder=StudyBrain(config['batch'],config['precision'],config['condition']['structure'],out)
    head=ActivityLinear().cuda()
    return encoder,head


def set_pair(encoder,head,pair):
    if encoder.core is not None:encoder.set_core(pair['core'])
    head.load_state_dict(pair['head'])


def pair(encoder,head):
    return dict(core=cpu_state(encoder.core) if encoder.core is not None else {},head=cpu_state(head))
