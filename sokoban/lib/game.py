"""Sokoban on a common 10x10 canvas, including centered smaller rooms."""
from collections import deque
import numpy as np
SIZE=10
DELTAS=(-10,10,-1,1)

def restore(l):return {'walls':set(l['walls']),'goals':tuple(l['goals'])}
def state_from(s):return int(s[0]),tuple(sorted(s[1]))
def step(l,s,a):
    p,b=s;d=DELTAS[a];q=p+d
    if q<0 or q>=100 or q in l['walls']:return None
    if q in b:
        z=q+d
        if z<0 or z>=100 or z in l['walls'] or z in b:return None
        return q,tuple(sorted((set(b)-{q})|{z}))
    return q,b
def solved(l,s):return set(s[1])==set(l['goals'])
def legal(l,s):return np.array([step(l,s,a) is not None for a in range(4)])
def encode(l,s):
    x=np.zeros((4,100),dtype=np.float32);x[0,list(l['walls'])]=1;x[1,list(l['goals'])]=1;x[2,list(s[1])]=1;x[3,s[0]]=1
    return x.reshape(4,10,10)
def canonical(l):
    x=np.zeros((10,10),dtype=np.uint8);x.ravel()[list(l['walls'])]=1;x.ravel()[list(l['goals'])]=2
    return min(np.rot90(g,k).tobytes() for g in (x,x[:,::-1]) for k in range(4)).hex()
def distances(l,cap=20000):
    floor=set(range(100))-l['walls'];g=tuple(sorted(l['goals']));ds={(p,g):0 for p in sorted(floor-set(g))};q=deque(ds)
    while q and len(ds)<cap:
        current=q.popleft();p,b=current
        for delta in DELTAS:
            prev=p-delta
            if prev not in floor or prev in b:continue
            candidates=[(prev,b)];z=p+delta
            if z in b:candidates.append((prev,tuple(sorted((set(b)-{z})|{p}))))
            for s in candidates:
                if s not in ds:ds[s]=ds[current]+1;q.append(s)
    return ds
def optimal(l,s,ds):return np.array([n is not None and ds.get(n,10**9)==ds[s]-1 for n in (step(l,s,a) for a in range(4))])

def embed(old):
    size=6;offset=2;convert=lambda p:(p//size+offset)*10+p%size+offset
    floor={convert(p) for p in range(36) if p not in old['walls']}
    result=dict(old);result.update(walls=sorted(set(range(100))-floor),goals=tuple(convert(p) for p in old['goals']),
                                  start=(convert(old['start'][0]),tuple(sorted(convert(p) for p in old['start'][1]))),size=6)
    result['canonical_map']=canonical(restore(result));return result
