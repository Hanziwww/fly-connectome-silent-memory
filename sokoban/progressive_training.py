"""Reuse the frozen D optimizer, generalizing only its success observer to many boxes."""
import repair_training as method
from game import restore, solved, state_from, step


class TeacherPolicyCheck:
    def __init__(self, level, data):
        self.map, self.start = restore(level), state_from(level['start'])
        self.index = {(int(p), tuple(sorted(map(int, b)))): i
                      for i, (p, b) in enumerate(zip(data['player'], data['boxes']))}

    def solved(self, actions):
        state, visited = self.start, set()
        while state in self.index and state not in visited:
            visited.add(state)
            state = step(self.map, state, int(actions[self.index[state]]))
            if state is None:
                return False
            if solved(self.map, state):
                return True
        return False


def fit(*args, **kwargs):
    # Workers are serial, isolated processes. Restore the module even on a stop.
    # All numeric updates, losses, acceptance, budgets and RNG use remain unchanged.
    original = method.TeacherPolicyCheck
    method.TeacherPolicyCheck = TeacherPolicyCheck
    try:
        return method.fit(*args, **kwargs)
    finally:
        method.TeacherPolicyCheck = original
