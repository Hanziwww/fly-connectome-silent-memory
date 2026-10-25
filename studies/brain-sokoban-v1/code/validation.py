"""Real-GPU state, gradient, batch and resume-artifact validation before the study."""
import numpy as np
import torch
import bootstrap
from registry import STUDY,SOURCE,read,write
from backend import StudyBrain,set_pair
from common import configure
from readout_models import ActivityLinear
from memory_tasks import prepare_memory,hidden_board
from memory import MemoryPolicy,probe,batched_assessment


def validate():
    from run_cycle import acquire_worker_lock
    lock=acquire_worker_lock()
    try:
        configure(10);free,total=torch.cuda.mem_get_info()
        assert free>=13*2**30;torch.cuda.set_per_process_memory_fraction(11*2**30/total)
        precision=read(STUDY/'backend.json')['precision'] if (STUDY/'backend.json').exists() else 'float64'
        encoder=StudyBrain(batch=1,precision=precision);head=ActivityLinear().cuda()
        saved=torch.load(SOURCE,map_location='cpu',weights_only=True);set_pair(encoder,head,saved)
        task=prepare_memory()[0];level=task['levels'][0]
        from game import encode,restore,state_from
        cue=np.array(encode(restore(level),state_from(level['start'])),dtype='float32')
        hidden=hidden_board(level,state_from(level['start']))
        outcomes=[]
        for recompute in (False,True):
            encoder.release_graph();encoder.reset()
            x=torch.as_tensor(cue[None],device='cuda').requires_grad_(True)
            forward=encoder.checkpointed_advance if recompute else encoder.advance
            forward(x,40)
            f=forward(torch.as_tensor(hidden[None],device='cuda'),40)
            loss=head(f)['final_logits'][0,3];loss.backward()
            cue_grad=x.grad.detach().cpu().numpy().copy()
            core_grad=np.concatenate([p.grad.detach().cpu().numpy() for p in encoder.core.parameters()])
            outcomes.append((f.detach().cpu().numpy(),cue_grad,core_grad))
        assert np.linalg.norm(outcomes[0][1])>0,'Cue-to-decision gradient is severed'
        for i in range(3):np.testing.assert_allclose(outcomes[0][i],outcomes[1][i],rtol=.01,atol=1e-3 if precision=='float32' else 1e-5)
        encoder.release_graph()
        policy=MemoryPolicy.__new__(MemoryPolicy)
        policy.encoder=encoder;policy.head=head;policy.kind='joint';policy.train_core=True;policy.config={}
        sequential=probe(policy,task,40,4)
        batched,_=batched_assessment(policy,task,[40],[4],max_steps=2)
        for a,b in zip(sequential['records'],batched[0]['records']):
            np.testing.assert_allclose(a['features'],b['features'],rtol=.001,atol=.001)
            assert a['choice']==b['choice']
        with torch.no_grad():
            state=encoder.capture_state(detach=True);encoder.intervene('all');encoder.restore_state(state)
            assert all(torch.equal(getattr(encoder.brain,n),v) for n,v in state['tensors'].items())
        from behavior import FrozenEncoder
        from common import cpu_state
        cache=FrozenEncoder(encoder);boards=np.stack([cue,hidden]);cache.encode(boards)
        calls=encoder.counts['forward_calls'];cache.encode(boards)
        assert calls==encoder.counts['forward_calls']
        original=cpu_state(encoder.core);changed={k:v.clone() for k,v in original.items()}
        changed['threshold_mv'][0]+=.0001;cache.set_core(changed);cache.encode(boards)
        assert encoder.counts['forward_calls']>calls;encoder.set_core(original)
        from activity import record_states
        records=STUDY/'validation-artifacts';records.mkdir(exist_ok=True)
        record_states(encoder,boards,records,'recording-smoke',full_indices=[0],discovery_mask=np.array([True,False]))
        trace=np.load(records/'recording-smoke-state-0-full.npz')['values']
        assert trace.shape==(40,4,167106) and trace.dtype==np.float32
        encoder.intervention=dict(indices=[0,1,2],strength=.5,window=(0,14));encoder.encode(boards)
        assert encoder.matrix.gate is None;encoder.intervention=None
        from memory import training_step,GRU
        training_checks=[]
        for kind in ('joint','reset','head_only','gru'):
            encoder.reset_batch(1);encoder.release_graph();set_pair(encoder,head,saved)
            encoder.core.requires_grad_(kind!='head_only');head.requires_grad_(True)
            policy.kind=kind;policy.train_core=kind in ('joint','reset');policy.frozen_features={}
            policy.encoder=None if kind=='gru' else encoder
            if kind=='gru':policy.gru=GRU().cuda()
            optimizer=torch.optim.Adam(policy.parameters(),lr=.0001)
            result=training_step(policy,task,optimizer,0)
            assert np.isfinite(result['loss']) and result['head_gradient_norm']>0
            if kind in ('joint','reset'):assert result['core_gradient_norm']>0
            if kind=='head_only':
                calls=encoder.counts['forward_calls'];training_step(policy,task,optimizer,0)
                assert calls==encoder.counts['forward_calls'],'Frozen-history cache missed an identical history'
            training_checks.append(dict(condition=kind,**result))
            del optimizer
        encoder.core.requires_grad_(True);encoder.release_graph();set_pair(encoder,head,saved)
        resumes=[]
        for p in (STUDY/'runs').glob('benchmark-*/resume.pt'):
            item=torch.load(p,map_location='cpu',weights_only=True)
            assert all(k in item for k in ('optimizer','rng','lessons','champion','current','state'))
            assert item['rng']['torch'].dtype==torch.uint8
            resumes.append(str(p.relative_to(STUDY)))
        receipt=dict(status='passed',precision=precision,cue_gradient_norm=float(np.linalg.norm(outcomes[0][1])),
            full_vs_recomputed_gradient_max_abs=float(np.max(np.abs(outcomes[0][2]-outcomes[1][2]))),
            independent_batch_probe_equivalent=True,full_state_restore_exact=True,resume_records=resumes,
            frozen_cache_invalidation=True,scientific_recording_shape=list(trace.shape),transient_intervention_restored=True,
            dynamic_training_steps=training_checks,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        if (STUDY/'validation.json').exists():
            old=read(STUDY/'validation.json');write(STUDY/f"validation-{old['precision']}.json",old)
        write(STUDY/f'validation-{precision}.json',receipt)
        write(STUDY/'validation.json',receipt);print(receipt);return 0
    finally:lock.close()
