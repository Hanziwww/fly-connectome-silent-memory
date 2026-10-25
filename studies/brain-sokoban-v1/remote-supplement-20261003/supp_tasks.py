"""Model-blind, solver-certified follow-up memory tasks on the shared canvas."""
from collections import deque
from pathlib import Path
import sys
import numpy as np

SESSION = Path(__file__).resolve().parent
STUDY = SESSION.parent
ROOT = STUDY.parents[1]
for p in (ROOT/'flybrain', ROOT/'sokoban', STUDY/'code'):
    sys.path.insert(0, str(p))
import bootstrap
from game import encode, restore, state_from, step, legal, solved, DELTAS
from progressive_tasks import reverse_distances, forward_certificate
from tasks import witness, family, connected
from registry import read, write


def state_key(s):
    return str(s[0])+':'+','.join(map(str,s[1]))


def board(case, state, cue=False):
    value=encode(restore(case['cue_level'] if cue else case['level']),state)
    if not cue and not case.get('visible',False):value[1]=0
    return value


def prefix_states(case, windows, variant=0):
    """Cue-independent legal closed walks; one window is stationary observation."""
    q=state_from(case['level']['start']);m=restore(case['level'])
    if windows==0:return []
    if windows==1:
        if case.get('legacy_prefix') and variant==0:return [(54,(34,))]
        return [q]
    assert windows%2==0
    neighbors=[n for a in range(4) if (n:=step(m,q,a)) is not None and n[1]==q[1]]
    assert len(neighbors)>=2
    n=neighbors[variant%len(neighbors)]
    return [q,n]*(windows//2)


def frames(case, delay, prefix, include_solution=True, variant=0):
    cue_state=state_from(case['cue_level']['start'])
    result=[dict(board=board(case,cue_state,True),ms=40,kind='cue',target=None)]
    if delay:result.append(dict(board=np.zeros((4,10,10),np.float32),ms=delay,kind='delay',target=None))
    for s in prefix_states(case,prefix,variant):
        result.append(dict(board=board(case,s),ms=40,kind='prefix',target=None))
    s=state_from(case['level']['start']);m=restore(case['level'])
    actions=case['level']['reference_actions'] if include_solution else [None]
    for a in actions:
        result.append(dict(board=board(case,s),ms=40,kind='decision',target=a,
                           legal=legal(m,s),state=[s[0],list(s[1])]))
        if a is not None:s=step(m,s,a)
    return result


def certify(level):
    ds,cert=reverse_distances(level)
    actions,_,_=witness(level,ds,state_from(level['start']))
    level=dict(level,reference_actions=actions,optimal_steps=len(actions),family_id=family(level))
    cert['forward']=forward_certificate(level)
    return level,ds,cert


def task_from_geometry(name,floor,start,box,goals,contexts=(0,)):
    cases=[]
    base=[]
    assert connected(floor)
    for cue,goal in enumerate(goals):
        l=dict(id=f'{name}-cue{cue}',size=10,boxes=1,walls=sorted(set(range(100))-floor),
               goals=[goal],start=[start,[box]],split='supplement_memory')
        l,_,_=certify(l);base.append(l)
    for context in contexts:
        transform=(lambda p:99-p) if context else (lambda p:p)
        for cue,cue_level in enumerate(base):
            l=dict(cue_level,id=f'{name}-context{context}-cue{cue}',
                   walls=sorted(transform(x) for x in cue_level['walls']),
                   goals=[transform(x) for x in cue_level['goals']],
                   start=[transform(start),[transform(box)]])
            l,ds,cert=certify(l)
            cases.append(dict(id=f'c{context}-cue{cue}',cue=cue,context=context,visible=False,
                 level=l,cue_level=cue_level,solvable_states=sorted(state_key(s) for s in ds),certificate=cert))
    task=dict(id=name,family_id=name,contexts=list(contexts),cases=cases,
        context_protocol='180-degree scene reorientation after cue; orientation visible only through the subsequent board' if len(contexts)>1 else 'fixed scene',
        scope='Conditional delayed-cue probe followed by autonomous Sokoban; the context reveal is an experimental manipulation, not a player move.',
        prefix_protocol='Cue-independent legal closed walks; one-window prefix is stationary.',
        acceptance='All registered training histories independently solve within 512 actions; any route accepted.',
        choice_measure='Direction of first box push; first movement is not a success gate.')
    validate_task(task)
    return task


def first_push(case):
    s=state_from(case['level']['start']);m=restore(case['level'])
    for a in case['level']['reference_actions']:
        n=step(m,s,a)
        if n[1]!=s[1]:return a
        s=n
    raise AssertionError('No push in solution')


def validate_task(task):
    for context in task['contexts']:
        pair=[c for c in task['cases'] if c['context']==context and not c.get('visible')]
        assert len(pair)==2
        q=state_from(pair[0]['level']['start'])
        assert np.array_equal(board(pair[0],q),board(pair[1],q))
        assert np.array_equal(legal(restore(pair[0]['level']),q),legal(restore(pair[1]['level']),q))
        assert first_push(pair[0])!=first_push(pair[1])
        for case in pair:
            m=restore(case['level']); safe=set(case['solvable_states'])
            expected=first_push(case); seen={q};todo=deque([q]);bad=good=0
            while todo:
                s=todo.popleft()
                for a in range(4):
                    n=step(m,s,a)
                    if n is None:continue
                    if n[1]!=s[1]:
                        if a!=expected:
                            assert state_key(n) not in safe
                            bad+=1
                        else:good+=1
                    elif n not in seen:seen.add(n);todo.append(n)
            assert bad and good
            case['certificate']['all_wrong_first_pushes_unsolvable']=True
            case['first_push_target']=expected
            for count in (0,1,2,4,8):
                for variant in (0,1):
                    states=prefix_states(case,count,variant)+[q]
                    for a,b in zip(states,states[1:]):
                        assert a==b or any(step(m,a,i)==b for i in range(4))
        for delay in (0,40,160):
            for prefix in (0,1,4):
                fs=[frames(c,delay,prefix,False) for c in pair]
                assert all(np.array_equal(a['board'],b['board']) for a,b in zip(fs[0][1:],fs[1][1:]))
    if len(task['contexts'])>1:
        for cue in (0,1):
            cs=[c for c in task['cases'] if c['cue']==cue]
            assert np.array_equal(frames(cs[0],40,0,False)[0]['board'],frames(cs[1],40,0,False)[0]['board'])
            assert cs[0]['first_push_target']!=cs[1]['first_push_target']
    return True


def prepare(rebuild=False):
    path=SESSION/'tasks.json'
    if path.exists() and not rebuild:return read(path)
    if rebuild:assert not (SESSION/'plan.json').exists(), 'Tasks already locked'
    s1=set(range(32,39))|{44,45,46,54,56,64,65,66,75,76}
    s2={14,24,34,44,54,64,74,35,36,37,47,55,56,57,48,58,68}
    tasks=[task_from_geometry('S1-ring',s1,65,35,(32,38)),
           task_from_geometry('S2-context',s2,47,44,(14,74),(0,1))]
    from memory_tasks import prepare_memory
    old=prepare_memory()[0]
    oldfloor=set(range(100))-set(old['levels'][0]['walls'])
    m0=task_from_geometry('M0-audit',oldfloor,44,34,(32,36))
    # Preserve the legacy prefix convention only for exact old-checkpoint audit.
    m0['legacy_prefix']=True
    for c in m0['cases']:c['legacy_prefix']=True
    tasks.append(m0)
    visible=dict(m0,id='M0-visibility',family_id='M0',legacy_prefix=True,cases=[])
    for visibility in (False,True):
        for c in m0['cases']:
            visible['cases'].append(dict(c,id=c['id']+('-visible' if visibility else '-hidden'),visible=visibility))
    tasks.append(visible)
    assert tasks[0]['cases'][0]['level']['family_id']!=tasks[1]['cases'][0]['level']['family_id']
    write(path,tasks)
    return tasks


if __name__=='__main__':
    import json
    ts=prepare()
    print(json.dumps([dict(id=t['id'],cases=len(t['cases']),steps=[c['level']['optimal_steps'] for c in t['cases']]) for t in ts]))
