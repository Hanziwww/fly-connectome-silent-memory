"""Cached real neural features for training, and ordinary uncached verification."""
import numpy as np
import torch

from common import cpu_state, tensor_hash
from continual import evaluate
from game import encode, legal, restore, solved, step
from joint_tasks import decorate
from population_gpu import PopulationGPU, logits
from progressive_tasks import reverse_distances
from hybrid_tasks import score


class FeatureBank:
    def __init__(self, encoder, core, boards=None, features=None):
        self.encoder, self.core_hash = encoder, tensor_hash(core)
        encoder.set_core(core)
        self.cache = {}
        if boards is not None:
            self.cache.update((b.astype('uint8').tobytes(), f.copy()) for b, f in zip(boards, features))

    def encode(self, boards, check=lambda: None):
        check()
        if tensor_hash(cpu_state(self.encoder.core)) != self.core_hash:
            raise RuntimeError('Brain changed: invalidate its activity feature cache')
        keys = [b.astype('uint8').tobytes() for b in boards]
        missing = {k: b for k, b in zip(keys, boards) if k not in self.cache}
        if missing:
            features = self.encoder.encode(np.stack(list(missing.values())), check)
            self.cache.update(zip(missing, features))
        return np.stack([self.cache[k] for k in keys])


def assessment(encoder, head, data, levels, check=lambda: None):
    with torch.no_grad():
        q = logits(head, encoder.encode(data['boards'], check))
        metrics = score(q, data)
    suite = evaluate(encoder, head, levels, 'hybrid-neural-policy', check)
    result = decorate(dict(**metrics, complete=suite['complete'], records=suite['records'],
        solved=suite['records'][0]['status'] == 'solved',
        lesson_solved=suite['records'][1]['status'] == 'solved',
        rollout_status=suite['records'][0]['status'], rollout_steps=suite['records'][0]['steps']))
    return result, suite


def audit_states(level):
    distances, certificate = reverse_distances(level)
    game = restore(level)
    states = sorted(s for s in distances if not solved(game, s))
    boards = np.stack([encode(game, s) for s in states]).astype('uint8')
    return states, boards, certificate


def policy_audit(encoder, head, level, check=lambda: None):
    states, boards, certificate = audit_states(level)
    game = restore(level)
    q = logits(head, encoder.encode(boards, check))
    masks = torch.as_tensor(np.stack([legal(game, s) for s in states]), device='cuda')
    actions = q.masked_fill(~masks, -1e9).softmax(1).argmax(1).cpu().tolist()
    policy, good = dict(zip(states, actions)), []
    for start in states:
        state, seen = start, set()
        while state in policy and state not in seen and not solved(game, state):
            seen.add(state)
            state = step(game, state, policy[state])
        if state is not None and solved(game, state):
            good.append([state_value(start)[0], state_value(start)[1]])
    return dict(nonterminal_solvable_states=len(states), goal_reaching_nonterminal_states=len(good),
                states=good, actions=actions, logits=q.cpu().tolist(), complete=certificate['complete'],
                diagnostic_only=True, inference_uses_policy_lookup=False)


def state_value(state):
    return [state[0], list(state[1])]


def candidates(encoder, head, cores, boards, data, levels, check, consume):
    """Batch candidate brain features; every action is still computed by the MLP."""
    engine = PopulationGPU(encoder, head)
    engine.bind(cores)
    try:
        values = engine.encode(np.concatenate([boards] * len(cores)),
                               np.repeat(np.arange(len(cores)), len(boards)), check)
    finally:
        engine.release()
    observations = []
    for index, core in enumerate(cores):
        check()
        features = values[index * len(boards):(index + 1) * len(boards)]
        bank = FeatureBank(encoder, core, boards, features)
        obs, suite = assessment(bank, head, data, levels, check)
        observations.append(obs)
        consume(index, core, obs, suite)
    return observations
