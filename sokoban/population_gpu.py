"""Batch distinct candidate brains through the same CUDA connectome matrix.

Every row has its own gains, thresholds, board and reset LIF state. No feature
cache crosses candidates. Float64 synaptic accumulation is preserved.
"""
import time
import numpy as np
import torch
from bootstrap import HERE
from brain_model import SharedMatrix
from curriculum import encode, legal, restore, solved, state_from, static_dead, step
from outcome_objective import losses_and_margins, objective


class PopulationMatrix(SharedMatrix):
    def __init__(self, base):
        self.base = base
        self.pre = self.post = None

    def __matmul__(self, x):
        pre = self.pre[:, None] if self.pre.ndim == 1 else self.pre
        post = self.post[:, None] if self.post.ndim == 1 else self.post
        return ((self.base @ (x.double() * pre)) * post).float()


class PopulationGPU:
    def __init__(self, encoder, head):
        self.encoder, self.head = encoder, head
        encoder.matrix = PopulationMatrix(encoder.matrix.base)
        encoder.brain.W = encoder.matrix

    @torch.no_grad()
    def bind(self, cores):
        e = self.encoder
        e.release_graph()
        values = {key: torch.stack([c[key] for c in cores]).cuda() for key in cores[0]}
        self.pre = values['pre_log_gain'].exp()[:, e.neuron_group].double()
        self.post = values['post_log_gain'].exp()[:, e.neuron_group].double()
        self.threshold = e.base_threshold + values['threshold_mv'][:, e.neuron_group]

    @torch.no_grad()
    def encode(self, boards, indices, check):
        e, results = self.encoder, []
        for start in range(0, len(boards), e.batch):
            check()
            chunk = boards[start:start + e.batch]
            count = len(chunk)
            padded = np.zeros((e.batch, 4, 10, 10), dtype='float32')
            padded[:count] = chunk
            ids = np.zeros(e.batch, dtype='int64')
            ids[:count] = indices[start:start + count]
            ids = torch.as_tensor(ids, device='cuda')
            e.matrix.pre, e.matrix.post = self.pre[ids].T, self.post[ids].T
            e.brain.W = e.matrix
            e.params.v_th = self.threshold[ids]
            e.brain.reset()
            e.brain.t = 0.
            e.brain.step_count = e.brain.buf_pos = 0
            drive = torch.from_numpy(padded).cuda().reshape(e.batch, 400).repeat_interleave(8, 1) * 18
            e.brain.set_drive(e.input_index, drive)
            e.brain.run_ms(14)
            current = e.brain.g[:, e.observed]
            e.brain.run_ms(26)
            voltage = e.brain.v[:, e.observed] - e.params.v_rest
            features = torch.cat((current.reshape(e.batch, 256, 16).mean(2),
                                  voltage.reshape(e.batch, 256, 16).mean(2)), 1)
            results.append(features[:count].cpu().numpy())
            e.counts['forward_calls'] += 1
            e.counts['simulated_boards_including_padding'] += e.batch
        return np.concatenate(results)

    def release(self):
        self.encoder.release_graph()
        self.pre = self.post = self.threshold = None


def logits(head, features):
    """Match the established evaluator's fixed 128-row readout batches."""
    output = []
    with torch.no_grad():
        for start in range(0, len(features), 128):
            part = features[start:start + 128]
            padded = np.zeros((128, 512), dtype='float32')
            padded[:len(part)] = part
            output.append(head(torch.from_numpy(padded).cuda(), correction=False)['brain_logits'][:len(part)])
    return torch.cat(output)


def scores(q, data):
    with torch.no_grad():
        losses, margins, correct = losses_and_margins(q, data)
        return dict(correct=int(correct.sum()), total=len(losses), min_margin=float(margins.min()),
            focus_loss=float(objective(losses)), loss=float(losses.mean()),
            per_clause_loss=losses.cpu().tolist(), margins=margins.cpu().tolist(), solved=False)


def evaluate_many(engine, cores, level, data, check, early_success=True):
    """Evaluate each independent candidate on feedback states and its own rollout."""
    started = time.monotonic()
    game, count = restore(level), len(cores)
    engine.bind(cores)
    try:
        n = len(data['boards'])
        all_boards = np.concatenate([data['boards']] * count)
        features = engine.encode(all_boards, np.repeat(np.arange(count), n), check)
        q = logits(engine.head, features)
        observations = [scores(q[i*n:(i+1)*n], data) for i in range(count)]
        keys = [b.tobytes() for b in data['boards']]
        caches = [dict(zip(keys, features[i*n:(i+1)*n])) for i in range(count)]
        states = [state_from(level['start']) for _ in cores]
        seen = [{s} for s in states]
        histories = [[] for _ in cores]
        status = ['solved' if solved(game, s) else 'running' for s in states]
        for _ in range(512):
            check()
            active = []
            for i, state in enumerate(states):
                if status[i] != 'running':
                    continue
                if static_dead(game, state):
                    status[i] = 'proven_corner_deadlock'
                elif not legal(game, state).any():
                    status[i] = 'blocked'
                else:
                    active.append(i)
            if not active:
                break
            boards = [encode(game, states[i]).astype('uint8') for i in active]
            board_keys = [b.tobytes() for b in boards]
            missing = [(i, b, key) for i, b, key in zip(active, boards, board_keys) if key not in caches[i]]
            if missing:
                values = engine.encode(np.stack([x[1] for x in missing]), np.array([x[0] for x in missing]), check)
                for (i, _, key), value in zip(missing, values):
                    caches[i][key] = value
            values = np.stack([caches[i][key] for i, key in zip(active, board_keys)])
            brain_logits = logits(engine.head, values)
            masks = torch.as_tensor(np.stack([legal(game, states[i]) for i in active]), device='cuda')
            probabilities = brain_logits.masked_fill(~masks, -1e9).softmax(1).cpu().numpy()
            brain_logits = brain_logits.cpu().numpy()
            for row, i in enumerate(active):
                action = int(probabilities[row].argmax())
                histories[i].append(dict(action=action, final_probabilities=probabilities[row].tolist(),
                    brain_logits=brain_logits[row].tolist(), correction_logits=[0.] * 4, checkpoint_id='population-candidate'))
                states[i] = step(game, states[i], action)
                if solved(game, states[i]):
                    status[i] = 'solved'
                elif states[i] in seen[i]:
                    status[i] = 'repeated_state'
                else:
                    seen[i].add(states[i])
            if early_success and 'solved' in status:
                status = ['superseded_by_solution' if s == 'running' else s for s in status]
                break
        for i, obs in enumerate(observations):
            status[i] = 'step_timeout' if status[i] == 'running' else status[i]
            record = dict(level_id=level['id'], split=level['split'], status=status[i],
                steps=len(histories[i]), actions=[d['action'] for d in histories[i]], decisions=histories[i])
            complete = status[i] != 'superseded_by_solution'
            obs.update(solved=status[i] == 'solved', rollout_status=status[i], rollout_steps=len(histories[i]),
                rollout=dict(complete=complete, solved=int(status[i] == 'solved'), total=1,
                    success_rate=float(status[i] == 'solved') if complete else None,
                    records=[record], seconds=time.monotonic() - started))
        return observations
    finally:
        engine.release()
