"""Batched autonomous evaluation with complete per-decision scientific records."""
import time
import numpy as np
import torch
import bootstrap
from game import encode,legal,restore,solved,state_from,step
from curriculum import static_dead


def archive_behavior(out,value,index):
    """Lossless FP32 feature table plus per-decision references avoids duplication."""
    from registry import append
    features=[];indices={}
    def pack(obj):
        if isinstance(obj,dict):
            result={k:pack(v) for k,v in obj.items() if k!='features'}
            if obj.get('features') is not None:
                feature=np.asarray(obj['features'],dtype='float32');key=feature.tobytes()
                if key not in indices:indices[key]=len(features);features.append(feature)
                result['feature_row']=indices[key]
            return result
        if isinstance(obj,(list,tuple)):return [pack(v) for v in obj]
        return obj
    result=pack(value);folder=out/'behavior-features';folder.mkdir(exist_ok=True)
    name=f'{index:06d}.npz'
    retry=0
    while (folder/name).exists():
        retry+=1;name=f'{index:06d}-continuation-{retry}.npz'
    np.savez_compressed(folder/name,features=np.stack(features) if features else np.empty((0,512),np.float32))
    result['feature_table']='behavior-features/'+name
    result['feature_storage']='float32 lossless; each decision references feature_row'
    append(out/'behavior.jsonl',result)
    return result


def restore_behavior(out,value):
    f=np.load(out/value['feature_table'])['features']
    def unpack(obj):
        if isinstance(obj,dict):
            result={k:unpack(v) for k,v in obj.items() if k!='feature_row'}
            if 'feature_row' in obj:result['features']=f[obj['feature_row']].tolist()
            return result
        if isinstance(obj,list):return [unpack(v) for v in obj]
        return obj
    return unpack(value)


class Features:
    """An assessment-local cache; never survives a parameter update."""
    def __init__(self,encoder):self.encoder=encoder;self.cache={}
    def encode(self,boards,check=lambda:None):
        keys=[b.astype('uint8').tobytes() for b in boards]
        missing={k:b for k,b in zip(keys,boards) if k not in self.cache}
        if missing:self.cache.update(zip(missing,self.encoder.encode(np.stack(list(missing.values())),check)))
        return np.stack([self.cache[k] for k in keys])


class FrozenEncoder(Features):
    """Reuse reset-state features only while the frozen core hash is unchanged."""
    def __init__(self,encoder):
        super().__init__(encoder)
        from common import tensor_hash,cpu_state
        self.core_hash=tensor_hash(cpu_state(encoder.core)) if encoder.core is not None else None
    def __getattr__(self,name):return getattr(self.encoder,name)
    def set_core(self,state):
        from common import tensor_hash
        key=tensor_hash(state)
        if key!=self.core_hash:self.cache.clear();self.core_hash=key
        self.encoder.set_core(state)


def rollouts(encoder,head,levels,check=lambda:None,max_steps=512,features=None):
    t=time.perf_counter();bank=features or Features(encoder)
    maps=[restore(l) for l in levels];states=[state_from(l['start']) for l in levels]
    seen=[{s} for s in states];records=[dict(task=l['id'],family=l.get('family_id'),status='running',decisions=[]) for l in levels]
    for _ in range(max_steps):
        check();active=[]
        for i,r in enumerate(records):
            if r['status']!='running':continue
            if solved(maps[i],states[i]):r['status']='solved'
            elif static_dead(maps[i],states[i]):r['status']='proven_corner_deadlock'
            elif not legal(maps[i],states[i]).any():r['status']='blocked'
            else:active.append(i)
        if not active:break
        boards=np.stack([encode(maps[i],states[i]) for i in active]);f=bank.encode(boards,check)
        with torch.no_grad():q=head(torch.as_tensor(f,device='cuda'))['final_logits'].cpu().numpy()
        if not np.isfinite(q).all():raise FloatingPointError('Nonfinite policy logits')
        for k,i in enumerate(active):
            mask=legal(maps[i],states[i]);a=int(np.where(mask,q[k],-np.inf).argmax())
            records[i]['decisions'].append(dict(state=[states[i][0],list(states[i][1])],
                action=a,logits=q[k].tolist(),features=f[k].tolist(),legal=mask.tolist()))
            states[i]=step(maps[i],states[i],a)
            if solved(maps[i],states[i]):records[i]['status']='solved'
            elif states[i] in seen[i]:records[i]['status']='repeated_state'
            else:seen[i].add(states[i])
    for r in records:
        if r['status']=='running':r['status']='action_limit'
        r['actions']=[d['action'] for d in r['decisions']];r['steps']=len(r['actions'])
    return dict(records=records,seconds=time.perf_counter()-t,complete=True)
