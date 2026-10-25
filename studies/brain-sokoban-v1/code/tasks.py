"""Model-blind map generation, exact certification, family-aware variants."""
from collections import deque
import hashlib
import json
import time
import numpy as np
import bootstrap
from game import canonical, encode, legal, restore, solved, state_from, step, DELTAS
from progressive_tasks import ROOM8, ROOM10, reverse_distances, forward_certificate, room
from registry import STUDY, ROOT, TIERS, append, read, write, protocol, make_schedule


def connected(floor):
    if not floor: return False
    seen, q = set(), [min(floor)]
    while q:
        p=q.pop()
        if p in seen: continue
        seen.add(p); q.extend(p+d for d in DELTAS if p+d in floor and p+d not in seen)
    return seen==floor


def family(level):
    return hashlib.sha256(bytes.fromhex(canonical(restore(level)))).hexdigest()[:20]


def witness(level, ds, start):
    m=restore(level); s=start; actions=[]; states=[]; branching=[]
    while ds[s]:
        choices=[a for a in range(4) if (n:=step(m,s,a)) is not None and ds.get(n)==ds[s]-1]
        states.append(s); branching.append(len(choices)); actions.append(choices[0]); s=step(m,s,choices[0])
    assert solved(m,s)
    return actions,states,branching


def describe(level, ds):
    m=restore(level); start=state_from(level['start']); actions,states,branching=witness(level,ds,start)
    # Exact shortest-path multiplicity on the reachable shortest-path DAG; saturation
    # is explicit and prevents astronomical Python integers from dominating records.
    memo={}; cap=10**12
    def paths(s):
        if ds[s]==0: return (1,0)
        if s in memo: return memo[s]
        choices=[n for a in range(4) if (n:=step(m,s,a)) is not None and ds.get(n)==ds[s]-1]
        children=[(paths(n),int(n[1]!=s[1])) for n in choices]
        value=(min(cap,sum(v[0] for v,_ in children)),min(v[1]+push for v,push in children))
        memo[s]=value; return value
    multiplicity,minpush=paths(start)
    floor=set(range(100))-m['walls']
    articulation=[p for p in sorted(floor) if not connected(floor-{p})]
    risk=[sum((n:=step(m,s,a)) is not None and n not in ds for a in range(4)) for s in states]
    return dict(optimal_steps=ds[start],reference_actions=actions,floor_cells=len(floor),
        minimum_pushes_among_shortest_solutions=minpush,
        witness_pushes=sum(step(m,s,a)[1]!=s[1] for s,a in zip(states,actions)),
        shortest_path_count_capped=multiplicity,shortest_path_count_cap=cap,
        shortest_dag_states=len(memo),witness_branch_counts=branching,
        floor_branch_cells=sum(sum(p+d in floor for d in DELTAS)>=3 for p in floor),
        articulation_cells=articulation,deadlocking_alternatives_on_witness=risk,
        deadlock_risk_definition='legal one-step transitions outside the exact solvable-state set')


def candidate_geometry(size, boxes, rng):
    if size==6:
        rows=['######','#....#','#....#','#....#','#....#','######']
    else: rows=list(ROOM8 if size==8 else ROOM10)
    grid=np.array([list(r) for r in rows]); offset=(10-size)//2
    # Perturb actual wall geometry, retaining a connected walkable component.
    for _ in range(1+int(rng.integers(4))):
        r,c=rng.integers(1,size-1,size=2); grid[r,c]='.' if grid[r,c]=='#' else '#'
    floor={(r+offset)*10+c+offset for r in range(size) for c in range(size) if grid[r,c]!='#'}
    if not connected(floor) or len(floor)>(16 if size==6 else 33 if size==8 else 44): return None
    # Boundary goals encourage reverse-pull reachability; all selection is geometric.
    corners=[p for p in sorted(floor) if any(p+a not in floor and p+b not in floor
              for a,b in [(-10,-1),(-10,1),(10,-1),(10,1)])]
    if len(corners)<boxes: return None
    goals=sorted(rng.choice(corners,size=boxes,replace=False).tolist())
    return dict(walls=sorted(set(range(100))-floor),goals=goals,boxes=boxes,size=size)


def transform(level,k,flip):
    def convert(p):
        r,c=divmod(p,10)
        if flip: c=9-c
        for _ in range(k): r,c=9-c,r
        return r*10+c
    a=[(-1,0),(1,0),(0,-1),(0,1)]
    def action(old):
        dr,dc=a[old]
        if flip: dc=-dc
        for _ in range(k): dr,dc=-dc,dr
        return a.index((dr,dc))
    return dict(level,id=f"{level['id']}-R{k}F{int(flip)}",split='transform',
        walls=sorted(map(convert,level['walls'])),goals=sorted(map(convert,level['goals'])),
        start=[convert(level['start'][0]),sorted(map(convert,level['start'][1]))],
        reference_actions=list(map(action,level['reference_actions'])),variant_of=level['id'])


def prepare():
    out=STUDY/'tasks'; out.mkdir(parents=True,exist_ok=True)
    if (out/'lock.json').exists():
        from registry import digest
        lock=read(out/'lock.json')
        for name,value in lock['files'].items():
            if digest(out/name)!=value: raise RuntimeError('Locked tasks changed: '+name)
        return read(out/'levels.json')
    historical=read(ROOT/'archive/experiments/progressive/20261001-single-seed01/levels.json')
    used={family(l) for l in historical}; levels=[]; variants=[]; certificates={}; bank=[]
    rng=np.random.default_rng(2026100210)
    specs=[(f'D{d}-{s}',d,size,boxes,lo,hi,'train' if s.startswith('T') else 'heldout')
        for d,(size,boxes,lo,hi) in enumerate(TIERS) for s in ('T0','T1','T2','H')]
    specs += [('C0',1,*TIERS[1],'calibration'),('C1',3,*TIERS[3],'calibration')]
    for task_id,tier,size,boxes,lo,hi,split in specs:
        for attempt in range(2000):
            l=candidate_geometry(size,boxes,rng)
            if l is None or family(l) in used: continue
            try:
                ds,cert=reverse_distances(l)
                candidates=sorted(s for s,d in ds.items() if lo<=d<=hi and not set(s[1])&set(l['goals']))
                if not candidates: continue
                start=candidates[int(rng.integers(len(candidates)))]
                l.update(id=task_id,tier=tier,split=split,start=[start[0],list(start[1])],family_id=family(l))
                l.update(describe(l,ds)); cert['forward']=forward_certificate(l)
            except (RuntimeError,AssertionError) as e:
                append(out/'rejected.jsonl',dict(task=task_id,attempt=attempt,geometry=l,
                    reason=str(e),classification='solver_uncertified',brain_evaluated=False)); continue
            used.add(l['family_id']); levels.append(l); certificates[task_id]=cert
            actions,states,_=witness(l,ds,start)
            for index,s in enumerate(states):
                bank.append(dict(task=task_id,family=l['family_id'],tier=tier,index=index,
                    state=[s[0],list(s[1])],action=actions[index],remaining=ds[s],
                    push=step(restore(l),s,actions[index])[1]!=s[1],
                    role='discovery' if task_id.endswith('T0') else 'confirmation' if task_id.endswith('T1') else 'descriptive'))
            # Fixed selection predates all model evaluations.
            unseen=[s for s in candidates if s not in set(states)]
            for i in range(min(5,len(unseen))):
                s=unseen[int(i*(len(unseen)-1)/max(1,min(5,len(unseen))-1))]
                v=dict(l,id=task_id+f'-S{i}',split='new_start',start=[s[0],list(s[1])],variant_of=task_id)
                v.update(describe(v,ds)); variants.append(v)
            variants.extend(transform(l,k,flip) for k,flip in ((1,False),(0,True)))
            print(json.dumps(dict(event='task_certified',task=task_id,steps=l['optimal_steps'],
                states=cert['solvable_states'],family=l['family_id'])),flush=True)
            break
        else: raise RuntimeError('No certified task after fixed candidate limit: '+task_id)
    write(out/'levels.json',levels);write(out/'variants.json',variants)
    write(out/'certificates.json',certificates);write(out/'matched-bank.json',bank)
    write(STUDY/'protocol.json',protocol()); write(STUDY/'schedule.json',make_schedule(levels))
    from registry import digest
    write(out/'lock.json',dict(created_unix=time.time(),selection='fixed RNG + geometry + exact solver; no brain calls',
        files={p.name:digest(p) for p in out.glob('*.json') if p.name!='lock.json'},
        training_maps=24,heldout_layouts=8,calibration_maps=2))
    return levels


def separate_wall_families():
    """Resolve wall-only duplicates before any difficulty/control training.

    Goal-sensitive map identity alone is insufficient for an unseen-geometry
    claim. Preserve v1 and replace only geometrically overlapping, unused maps;
    calibration maps and all observed results remain exactly unchanged.
    """
    import shutil
    from registry import digest
    out=STUDY/'tasks';lock=read(out/'lock.json')
    if lock.get('version',1)>=2:return
    levels=read(out/'levels.json');variants=read(out/'variants.json');certs=read(out/'certificates.json')
    geometry=lambda l:canonical(dict(walls=set(l['walls']),goals=[]))
    used=set();replace=[]
    ordered=sorted(levels,key=lambda l:({'heldout':0,'calibration':1,'train':2}[l['split']],l['id']))
    for l in ordered:
        key=geometry(l)
        if key in used:replace.append(l['id'])
        else:used.add(key)
    for p in (STUDY/'runs').glob('*/config.json'):
        if read(p)['task'] in replace:raise RuntimeError('Cannot amend a task after its model has been evaluated')
    version=out/'versions/v1';version.mkdir(parents=True,exist_ok=True)
    for p in out.glob('*.json'):shutil.copy2(p,version/p.name)
    rng=np.random.default_rng(2026100211);changes=[]
    for task_id in replace:
        index=next(i for i,l in enumerate(levels) if l['id']==task_id);old=levels[index]
        size,boxes,lo,hi=TIERS[old['tier']]
        for attempt in range(2000):
            l=candidate_geometry(size,boxes,rng)
            if l is None or geometry(l) in used:continue
            ds,cert=reverse_distances(l)
            candidates=sorted(s for s,d in ds.items() if lo<=d<=hi and not set(s[1])&set(l['goals']))
            if not candidates:continue
            s=candidates[int(rng.integers(len(candidates)))];l.update(id=task_id,tier=old['tier'],split=old['split'],
                start=[s[0],list(s[1])],family_id=family(l))
            l.update(describe(l,ds));cert['forward']=forward_certificate(l)
            used.add(geometry(l));levels[index]=l;certs[task_id]=cert
            variants=[v for v in variants if v.get('variant_of')!=task_id]
            _,states,_=witness(l,ds,s);unseen=[s for s in candidates if s not in set(states)]
            for i in range(min(5,len(unseen))):
                q=unseen[int(i*(len(unseen)-1)/max(1,min(5,len(unseen))-1))]
                v=dict(l,id=task_id+f'-S{i}',split='new_start',start=[q[0],list(q[1])],variant_of=task_id)
                v.update(describe(v,ds));variants.append(v)
            variants.extend(transform(l,k,f) for k,f in ((1,False),(0,True)))
            changes.append(dict(task=task_id,old_family=old['family_id'],new_family=l['family_id'],
                reason='wall geometry duplicated another train or heldout family; no model had evaluated this task'))
            break
        else:raise RuntimeError('No geometry-independent replacement')
    bank=[]
    for l in levels:
        s=state_from(l['start']);m=restore(l)
        for i,a in enumerate(l['reference_actions']):
            bank.append(dict(task=l['id'],family=l['family_id'],tier=l['tier'],index=i,state=[s[0],list(s[1])],
                action=a,remaining=len(l['reference_actions'])-i,push=step(m,s,a)[1]!=s[1],
                role='discovery' if l['id'].endswith('T0') else 'confirmation' if l['id'].endswith('T1') else 'descriptive'))
            s=step(m,s,a)
    assert len({geometry(l) for l in levels})==34
    write(out/'levels.json',levels);write(out/'variants.json',variants);write(out/'certificates.json',certs);write(out/'matched-bank.json',bank)
    write(STUDY/'registration-amendments.json',dict(reason='geometry-only independence audit before main training',changes=changes,
        calibration_unchanged=True,original_lock='tasks/versions/v1/lock.json'))
    write(out/'lock.json',dict(lock,version=2,files={p.name:digest(p) for p in out.glob('*.json') if p.name!='lock.json'},
        independent_wall_geometries=34))
