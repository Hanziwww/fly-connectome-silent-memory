"""Task-local counterexamples: preserve solvability and escape actual policy cycles.

The solver supplies training feedback only. Distances and reference actions are
never used here; inference continues to use neural logits and game legality.
"""
import math
import numpy as np
import torch

from bootstrap import HERE
from game import encode, legal, restore, solved, state_from, step
from progressive_tasks import reverse_distances


class Counterexamples:
    def __init__(self, level, max_clauses=128):
        self.level, self.game = level, restore(level)
        self.start = state_from(level['start'])
        distances, self.certificate = reverse_distances(level)
        self.viable = frozenset(distances)  # Discard all shortest-distance values.
        self.clauses, self.keys, self.max_clauses = [], set(), max_clauses

    def exits(self, state, excluded=frozenset()):
        return tuple(a for a in range(4) if (nxt := step(self.game, state, a)) is not None
                     and nxt in self.viable and nxt not in excluded)

    def observe(self, record):
        """Return a witnessed failure clause, accepting detours of any length."""
        state, states, seen = self.start, [self.start], {self.start: 0}
        for move, action in enumerate(record['actions'], 1):
            nxt = step(self.game, state, int(action))
            if nxt is None or nxt not in self.viable:
                assert state in self.viable
                atoms = [(state, self.exits(state))]
                clause = dict(kind='irreversible_dead_end', atoms=atoms, move=move)
                break
            if solved(self.game, nxt):
                return None
            if nxt in seen:
                cycle = frozenset(states[seen[nxt]:])
                atoms = [(s, self.exits(s, cycle)) for s in sorted(cycle)]
                atoms = [(s, actions) for s, actions in atoms if actions]
                assert atoms, 'A cycle of solvable states must have a solvable exit'
                clause = dict(kind='closed_policy_cycle', atoms=atoms, move=move)
                break
            state = nxt
            seen[state] = len(states)
            states.append(state)
        else:
            # A finite rollout timeout is not proof of a bad action or deadlock.
            return None
        assert all(actions for _, actions in clause['atoms'])
        key = (clause['kind'], tuple(clause['atoms']))
        if key in self.keys:
            return dict(clause, added=False)
        if len(self.clauses) >= self.max_clauses:
            raise RuntimeError('Counterexample capacity reached')
        self.keys.add(key)
        self.clauses.append(clause)
        return dict(clause, added=True)

    def data(self):
        states = sorted({state for clause in self.clauses for state, _ in clause['atoms']})
        if not states:
            raise ValueError('No witnessed trainable failure')
        indices = {state: i for i, state in enumerate(states)}
        clauses = [dict(kind=c['kind'], atoms=[(indices[s], actions) for s, actions in c['atoms']])
                   for c in self.clauses]
        return dict(boards=np.stack([encode(self.game, s) for s in states]).astype('uint8'),
                    legal=np.stack([legal(self.game, s) for s in states]), clauses=clauses, states=states)


def losses_and_margins(logits, data):
    valid = torch.as_tensor(data['legal'], device=logits.device).bool()
    q = logits.masked_fill(~valid, -1e9)
    normalizer = q.logsumexp(1)
    losses, margins, correct = [], [], []
    actions = q.argmax(1)
    for clause in data['clauses']:
        no_exit, atom_margins, atom_correct = [], [], []
        for index, exits in clause['atoms']:
            mask = torch.zeros(4, device=q.device, dtype=torch.bool)
            mask[list(exits)] = True
            # A cycle is broken if at least one member chooses any viable exit.
            no_exit.append(q[index].masked_fill(mask, -1e9).logsumexp(0) - normalizer[index])
            atom_margins.append(q[index].masked_fill(~mask, -1e9).max()
                                - q[index].masked_fill(mask, -1e9).max())
            atom_correct.append(mask[actions[index]])
        probability = -torch.expm1(torch.stack(no_exit).sum().clamp(max=0))
        losses.append(-probability.clamp(min=1e-12).log())
        margins.append(torch.stack(atom_margins).max())
        correct.append(torch.stack(atom_correct).any())
    return torch.stack(losses), torch.stack(margins), torch.stack(correct)


def objective(losses):
    count = max(1, math.ceil(.2 * len(losses)))
    return .5 * losses.mean() + .5 * losses.topk(count).values.mean()


def better(candidate, incumbent):
    """Any solved trajectory wins; successful paths are never ranked by length."""
    if candidate['solved'] or incumbent['solved']:
        return candidate['solved'] and not incumbent['solved']
    if candidate['correct'] != incumbent['correct']:
        return candidate['correct'] > incumbent['correct']
    difference = candidate['min_margin'] - incumbent['min_margin']
    if abs(difference) > 1e-5:
        return difference > 0
    return candidate['focus_loss'] < incumbent['focus_loss'] - 1e-6
