"""Positive goal-reaching lessons and task-local corrections; no optimality gate."""
import json
import numpy as np
import torch
import torch.nn.functional as F

from bootstrap import HERE
from game import encode, legal, restore, solved, state_from, step
from outcome_objective import Counterexamples, losses_and_margins
from progressive_tasks import reverse_distances

TARGET_MARGIN = .1


def trace(game, start, actions):
    state, rows = start, []
    for action in actions:
        action = int(action)
        nxt = step(game, state, action)
        if nxt is None or solved(game, state):
            raise ValueError('Invalid demonstration or recorded trajectory')
        rows.append((state, action))
        state = nxt
    return rows, state


class Lessons:
    def __init__(self, level, seed=10):
        self.level, self.game = level, restore(level)
        self.start = state_from(level['start'])
        self.witness, end = trace(self.game, self.start, level['reference_actions'])
        if not solved(self.game, end) or len({s for s, _ in self.witness}) != len(self.witness):
            raise ValueError('Teaching witness must be a valid simple solution')
        distances, self.certificate = reverse_distances(level)
        self.viable = set(distances)
        # Distances are discarded. Extend the supplied witness into an acyclic
        # goal-reaching teaching policy by attaching previously unknown states.
        self.teacher = dict(self.witness)
        known = set(self.teacher) | {s for s in self.viable if solved(self.game, s)}
        predecessors = {s: [] for s in self.viable}
        for state in sorted(self.viable):
            for action in range(4):
                nxt = step(self.game, state, action)
                if nxt in predecessors:
                    predecessors[nxt].append((state, action))
        frontier, rng = sorted(known), np.random.default_rng(seed)
        while frontier:
            index = int(rng.integers(len(frontier)))
            target = frontier.pop(index)
            for state, action in predecessors[target]:
                if state not in known:
                    known.add(state)
                    self.teacher[state] = action
                    frontier.append(state)
        assert known == self.viable
        self.depths = sorted({1, 2, 4, 8, len(self.witness)})
        # This experiment diagnoses the observed P01 decision, not an arbitrary
        # shortest-path state. The target leaves its recorded two-state cycle.
        self.critical_state, self.critical_action = (54, (44,)), 3
        assert self.critical_state in self.viable
        assert step(self.game, self.critical_state, self.critical_action) in self.viable
        self.learned, self.corrections, self.failures = {}, {}, []
        self.failure_keys, self.banks = set(), {}
        self.version = 0

    def level_for(self, stage):
        if stage < 0:
            return None
        state = self.witness[-self.depths[stage]][0]
        return dict(self.level, id=f'P01-suffix-{self.depths[stage]}',
                    start=[state[0], list(state[1])])

    def data(self, stage):
        targets = {}
        def add(state, actions, weight):
            if state not in targets:
                targets[state] = [set(actions), weight]
            else:
                targets[state][0].update(actions)
                targets[state][1] = max(targets[state][1], weight)
        if stage < 0:
            add(self.critical_state, [self.critical_action], 1.)
            failures = []
        else:
            add(self.critical_state, [self.critical_action], .5)
            for state, action in self.witness[-self.depths[stage]:]:
                add(state, [action], 1.)
            for state, actions in self.learned.items():
                add(state, actions, .5)
            for state, action in list(self.corrections.items())[-8:]:
                add(state, [action], .75)
            failures = self.failures[-8:]
        clauses = [dict(kind='positive_witness', atoms=[(state, tuple(sorted(actions)))], weight=weight)
                   for state, (actions, weight) in sorted(targets.items())]
        clauses += [dict(c, weight=.5) for c in failures]
        states = sorted({state for c in clauses for state, _ in c['atoms']})
        indices = {s: i for i, s in enumerate(states)}
        return dict(boards=np.stack([encode(self.game, s) for s in states]).astype('uint8'),
                    legal=np.stack([legal(self.game, s) for s in states]), states=states,
                    clauses=[dict(kind=c['kind'], atoms=[(indices[s], a) for s, a in c['atoms']]) for c in clauses],
                    weights=[c['weight'] for c in clauses], version=self.version,
                    critical_index=indices[self.critical_state])

    def observe(self, level, record):
        """Add successful actions or a witnessed failure plus a feasible correction."""
        start = state_from(level['start'])
        rows, end = trace(self.game, start, record['actions'])
        before = json.dumps(self.snapshot(), sort_keys=True)
        if record['status'] == 'solved':
            assert solved(self.game, end)
            for state, action in rows:
                self.learned.setdefault(state, set()).add(action)
                self.corrections.pop(state, None)
        else:
            bank = self.banks.setdefault(start, None)
            if bank is None:
                bank = self.banks[start] = Counterexamples(level)
            failure = bank.observe(record)
            if failure:
                signature = (failure['kind'], tuple(failure['atoms']))
                if signature not in self.failure_keys:
                    self.failure_keys.add(signature)
                    self.failures.append({k: failure[k] for k in ('kind', 'atoms')})
                # Correct actual failure states, including states outside the
                # supplied witness. The teacher is used only to build training data.
                for state, _ in failure['atoms']:
                    if state not in self.learned:
                        self.corrections[state] = self.teacher[state]
        changed = json.dumps(self.snapshot(), sort_keys=True) != before
        self.version += int(changed)
        return changed

    def snapshot(self):
        return dict(witness=[(s, a) for s, a in self.witness], depths=self.depths,
                    learned=[(s, sorted(a)) for s, a in sorted(self.learned.items())],
                    corrections=list(self.corrections.items()), failures=self.failures,
                    solvable_states=len(self.viable), distances_used_for_scoring=False)


def loss(q, data):
    _, margins, correct = losses_and_margins(q, data)
    penalty = .25 * F.softplus((TARGET_MARGIN - margins) / .25)
    weights = torch.as_tensor(data['weights'], dtype=q.dtype, device=q.device)
    value = .75 * (penalty * weights).sum() / weights.sum() + .25 * penalty.max()
    return value, margins, correct


def score(q, data):
    value, margins, correct = loss(q, data)
    index = data['critical_index']
    valid = torch.as_tensor(data['legal'][index], device=q.device).bool()
    competitors = valid.clone()
    competitors[3] = False
    critical_margin = q[index, 3] - q[index].masked_fill(~competitors, -1e9).max()
    return dict(loss=float(value), min_margin=float(margins.min()),
                correct=int(correct.sum()), total=len(margins), margins=margins.cpu().tolist(),
                critical_margin=float(critical_margin))


def better(candidate, incumbent):
    if candidate['solved'] or incumbent['solved']:
        return candidate['solved'] and not incumbent['solved']
    return candidate['loss'] < incumbent['loss'] - 1e-7
