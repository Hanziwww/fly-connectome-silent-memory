"""Task-independent reverse-suffix lessons, without a shortest-path acceptance gate."""
import numpy as np

from bootstrap import HERE
from game import encode, legal, restore, solved, state_from
from lesson_tasks import trace
from hybrid_tasks import loss


class SuffixCurriculum:
    def __init__(self, level, snapshot=None):
        self.level, self.game = level, restore(level)
        self.witness, end = trace(self.game, state_from(level['start']), level['reference_actions'])
        if not solved(self.game, end) or len(set(s for s, _ in self.witness)) != len(self.witness):
            raise ValueError('Expected a simple, complete solution witness')
        self.learned = {}
        if snapshot:
            self.learned = {(int(s[0]), tuple(s[1])): set(a) for s, a in snapshot['learned']}

    def at_depth(self, depth):
        if not 1 <= depth <= len(self.witness):
            raise ValueError('Invalid suffix depth')
        state = self.witness[-depth][0]
        return dict(self.level, id=f"{self.level['id']}-suffix-{depth}", start=[state[0], list(state[1])])

    def levels(self, depth):
        return [self.level, self.at_depth(depth)] + [self.at_depth(d) for d in range(1, depth)]

    def data(self, depth):
        bridge, action = self.witness[-depth]
        targets = {s: set(a) for s, a in self.learned.items()}
        targets.setdefault(bridge, set()).add(action)
        states = sorted(targets)
        if len(states) > 128:
            raise RuntimeError('Bounded training-state capacity exceeded; preserve this attempt')
        clauses = [dict(kind='positive_witness', atoms=[(i, tuple(sorted(targets[s])))])
                   for i, s in enumerate(states)]
        return dict(states=states, boards=np.stack([encode(self.game, s) for s in states]).astype('uint8'),
            legal=np.stack([legal(self.game, s) for s in states]), clauses=clauses,
            bridge_clause=states.index(bridge), protected_clauses=[i for i, s in enumerate(states) if s in self.learned],
            depth=depth)

    def observe_success(self, level, record):
        if record['status'] != 'solved':
            raise ValueError('Only a demonstrated successful rollout can add anchors')
        rows, end = trace(self.game, state_from(level['start']), record['actions'])
        if not solved(self.game, end):
            raise ValueError('Recorded success did not reach a goal')
        for state, action in rows:
            self.learned.setdefault(state, set()).add(action)

    def snapshot(self):
        return dict(learned=[(s, sorted(a)) for s, a in sorted(self.learned.items())],
                    witness=[(s, a) for s, a in self.witness],
                    distances_used_for_scoring=False, inference_teacher=False)


def score(q, data):
    value, margins, correct = loss(q, data)
    return dict(loss=float(value), margins=margins.detach().cpu().tolist(),
                bridge_margin=float(margins[data['bridge_clause']]),
                correct=int(correct.sum()), total=len(margins))


def descriptor(data):
    return {k: data[k] for k in ('states', 'clauses', 'bridge_clause', 'protected_clauses', 'depth')}
