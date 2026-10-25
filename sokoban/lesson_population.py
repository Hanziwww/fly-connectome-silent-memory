"""CUDA candidate batches with positive lessons and independent neural rollouts."""
import time
import numpy as np
import torch
from bootstrap import HERE
from curriculum import encode, legal, restore, solved, state_from, static_dead, step
from lesson_tasks import score
from population_gpu import logits


def evaluate_many(engine, cores, levels, data, check):
    """The first level is always the original task; later starts are lessons."""
    started, count, n = time.monotonic(), len(cores), len(data['boards'])
    game = restore(levels[0])
    engine.bind(cores)
    try:
        features = engine.encode(np.concatenate([data['boards']] * count),
                                 np.repeat(np.arange(count), n), check)
        q = logits(engine.head, features)
        observations = [score(q[i*n:(i+1)*n], data) for i in range(count)]
        keys = [b.tobytes() for b in data['boards']]
        caches = [dict(zip(keys, features[i*n:(i+1)*n])) for i in range(count)]
        episodes = []
        for i in range(count):
            for j, level in enumerate(levels):
                state = state_from(level['start'])
                episodes.append(dict(core=i, level=j, state=state, seen={state}, decisions=[],
                                     status='solved' if solved(game, state) else 'running'))
        for _ in range(512):
            check()
            active = []
            for episode in episodes:
                if episode['status'] != 'running':
                    continue
                state = episode['state']
                if static_dead(game, state):
                    episode['status'] = 'proven_corner_deadlock'
                elif not legal(game, state).any():
                    episode['status'] = 'blocked'
                else:
                    active.append(episode)
            if not active:
                break
            missing = {}
            for episode in active:
                board = encode(game, episode['state']).astype('uint8')
                key, i = board.tobytes(), episode['core']
                episode['key'] = key
                if key not in caches[i]:
                    missing[i, key] = board
            if missing:
                values = engine.encode(np.stack(list(missing.values())),
                                       np.array([i for i, _ in missing]), check)
                for (i, key), feature in zip(missing, values):
                    caches[i][key] = feature
            values = np.stack([caches[e['core']][e['key']] for e in active])
            q = logits(engine.head, values)
            masks = torch.as_tensor(np.stack([legal(game, e['state']) for e in active]), device='cuda')
            probabilities = q.masked_fill(~masks, -1e9).softmax(1).cpu().numpy()
            q = q.cpu().numpy()
            for row, episode in enumerate(active):
                action = int(probabilities[row].argmax())
                episode['decisions'].append(dict(action=action, final_probabilities=probabilities[row].tolist(),
                    brain_logits=q[row].tolist(), correction_logits=[0.] * 4, checkpoint_id='lesson-population'))
                state = episode['state'] = step(game, episode['state'], action)
                if solved(game, state):
                    episode['status'] = 'solved'
                elif state in episode['seen']:
                    episode['status'] = 'repeated_state'
                else:
                    episode['seen'].add(state)
            # Any original-start solution overrides every auxiliary score.
            if any(e['level'] == 0 and e['status'] == 'solved' for e in episodes):
                for episode in episodes:
                    if episode['status'] == 'running':
                        episode['status'] = 'superseded_by_solution'
                break
        for i, obs in enumerate(observations):
            records = []
            for episode in [e for e in episodes if e['core'] == i]:
                status = episode['status']
                if status == 'running':
                    status = 'step_timeout'
                level, decisions = levels[episode['level']], episode['decisions']
                records.append(dict(level_id=level['id'], split=level['split'], status=status,
                    steps=len(decisions), actions=[d['action'] for d in decisions], decisions=decisions))
            complete = all(r['status'] != 'superseded_by_solution' for r in records)
            obs.update(solved=records[0]['status'] == 'solved', complete=complete,
                       lesson_solved=len(records) > 1 and records[1]['status'] == 'solved',
                       rollout_status=records[0]['status'], rollout_steps=records[0]['steps'],
                       records=records, seconds=time.monotonic() - started)
        return observations
    finally:
        engine.release()
