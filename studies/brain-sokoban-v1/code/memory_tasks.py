"""Cue-dependent irreversible box pushes and matched-history memory probes."""
import numpy as np
import bootstrap
from game import encode,restore,state_from,step,legal,solved
from progressive_tasks import reverse_distances,forward_certificate
from tasks import family,witness
from registry import STUDY,read,write


def prepare_memory():
    target=STUDY/'tasks/memory.json'
    if target.exists():return read(target)
    result=[]
    base=set(range(32,37))|{43,44,45,53,54,55}
    for i,extra in enumerate((set(),{64},{52,64})):
        floor=base|extra;pair=[];certs=[]
        for cue,goal in enumerate((32,36)):
            l=dict(id=f'M{i}-cue{cue}',size=8,boxes=1,walls=sorted(set(range(100))-floor),
                goals=[goal],start=[44,[34]],split='memory',cue=cue)
            ds,cert=reverse_distances(l);actions,_,_=witness(l,ds,state_from(l['start']))
            l.update(reference_actions=actions,optimal_steps=len(actions),family_id=family(l))
            cert['forward']=forward_certificate(l)
            wrong=(34,(35,)) if cue==0 else (34,(33,))
            assert wrong not in ds, 'Wrong first box push must make the cued goal unreachable'
            cert['wrong_first_push_unsolvable']=True
            pair.append(l);certs.append(cert)
        # Current board, legal actions, cue duration and common prefix are exactly
        # paired. The simulator never receives hidden goals after cue presentation.
        assert np.array_equal(hidden_board(pair[0],(44,(34,))),hidden_board(pair[1],(44,(34,))))
        assert np.array_equal(legal(restore(pair[0]),(44,(34,))),legal(restore(pair[1]),(44,(34,))))
        result.append(dict(id=f'M{i}',levels=pair,certificates=certs,
            query_state=[44,[34]],choice_targets=[3,2],
            choice_semantics='left/right response selects approach side; full rollout must execute the irreversible push',
            common_prefix='teacher-forced common physical walk; ends at identical query state',
            resets_indistinguishability_upper_bound=.5))
    write(target,result);return result


def hidden_board(level,state):
    b=encode(restore(level),state);b[1]=0;return b


def prefix_states(windows):
    # One window: player 54 -> 44. Even windows: equal-length down/up walks.
    if windows==1:return [(54,(34,))]
    if windows%2:raise ValueError('Only predeclared prefix lengths are supported')
    return [(44 if i%2==0 else 54,(34,)) for i in range(windows)]


def frames(task,cue,delay,prefix,include_solution=True):
    level=task['levels'][cue];m=restore(level);query=state_from(task['query_state'])
    result=[dict(board=encode(m,query),ms=40,kind='cue',target=None)]
    if delay:result.append(dict(board=np.zeros((4,10,10),np.float32),ms=delay,kind='blank_delay',target=None))
    for s in prefix_states(prefix):result.append(dict(board=hidden_board(level,s),ms=40,kind='common_prefix',target=None))
    s=query
    actions=level['reference_actions'] if include_solution else [task['choice_targets'][cue]]
    for a in actions:
        result.append(dict(board=hidden_board(level,s),ms=40,kind='decision',target=a,
            legal=legal(m,s),state=[s[0],list(s[1])]))
        s=step(m,s,a)
    return result
