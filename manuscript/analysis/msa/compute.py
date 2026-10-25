"""Rebuild simulator forward-call counters from the saved training logs.

The counting rules replicate studies/brain-sokoban-v1/code/training.py and behavior.py:
  * every assessment has its own feature cache: training-state boards are encoded first
    (ceil(n/16) calls), then rollout boards missing from the cache are encoded step by step
    in batches of 16;
  * every optimizer update encodes the training states without gradients (ceil(n/16)) and, when
    the core is trained, recomputes them with gradients (ceil(n/16));
  * after updates pass a stage, one unlogged re-assessment precedes the promotion;
  * the final rollout of the champion (after.json) costs one call per decision;
  * frozen cores (FrozenEncoder) share one persistent cache for the whole run;
  * a resumed process starts with one unlogged re-assessment of the last saved assessment.
One forward call simulates 16 boards x 167,106 neurons x 40 ms = 106,947,840 neuron-ms.
"""
import math

from . import io

NEURON_MS_PER_CALL = 16 * 167106 * 40


def _key(state):
    return (int(state[0]), tuple(sorted(int(x) for x in state[1])))


def _load(run_dir):
    def opt(name):
        p = run_dir / name
        return io.json_load(p) if p.exists() else None

    def jl(name):
        p = run_dir / name
        return io.jsonl_load(p) if p.exists() else []

    return dict(config=opt('config.json'), result=opt('result.json'), after=opt('after.json'),
                history=jl('history.jsonl'), exposure=jl('exposure.jsonl'), behavior=jl('behavior.jsonl'),
                interruptions=[io.json_load(p) for p in sorted(run_dir.glob('interruption-*/interruption.json'))])


class _Counter:
    def __init__(self, frozen, batch=16):
        self.frozen = frozen
        self.batch = batch
        self.persistent = set()
        self.calls = 0
        self.grad = 0

    def encode(self, keys):
        if self.frozen:
            keys = [k for k in keys if k not in self.persistent]
            self.persistent.update(keys)
        n = math.ceil(len(keys) / self.batch) if keys else 0
        self.calls += n
        return n

    def assessment(self, states, records):
        before = self.calls
        cache = set()
        ks = sorted({_key(s) for s in states})
        self.encode(ks)
        cache.update(ks)
        horizon = max((len(r['decisions']) for r in records), default=0)
        for t in range(horizon):
            new, seen = [], set()
            for r in records:
                if t < len(r['decisions']):
                    k = _key(r['decisions'][t]['state'])
                    if k not in cache and k not in seen:
                        new.append(k)
                        seen.add(k)
            if new:
                self.encode(new)
                cache.update(seen)
        return self.calls - before

    def update(self, states, n_states, core_on):
        self.encode(sorted({_key(s) for s in states}))
        if core_on:
            n = math.ceil(n_states / self.batch)
            self.calls += n
            self.grad += n

    def final_rollout(self, decisions):
        if self.frozen:
            for d in decisions:
                self.encode([_key(d['state'])])
        else:
            self.calls += len(decisions)


def batch_of(rec):
    return int(rec['config'].get('batch') or 16)


def _replay(rec, start=0, resume=False):
    cond = rec['config']['condition']
    frozen = not cond['train_core']
    counter = _Counter(frozen, batch_of(rec))
    h, e, b = rec['history'], rec['exposure'], rec['behavior']
    if resume and start > 0:
        counter.assessment(e[start - 1]['states'], b[start - 1]['records'])
    prev = h[start - 1]['updates'] if start > 0 else 0
    for i in range(start, len(h)):
        du = h[i]['updates'] - prev
        prev = h[i]['updates']
        for u in range(du):
            index = h[i]['updates'] - du + u
            core_on = cond['train_core']
            if cond.get('update_mode') == 'alternating':
                core_on = index % 2 == 0
            counter.update(e[i]['states'], h[i]['training_states'], core_on)
        cost = counter.assessment(e[i]['states'], b[i]['records'])
        if h[i]['event'] == 'update' and h[i]['stage_passed'] and not h[i]['solved'] and cond['schedule'] == 'suffix':
            if frozen:
                counter.assessment(e[i]['states'], b[i]['records'])
            else:
                counter.calls += cost
    if rec['after'] is not None:
        counter.final_rollout(rec['after']['decisions'])
    return counter


def rebuild(run_dir):
    rec = _load(run_dir)
    out = dict(run_dir=run_dir, condition=rec['config']['condition']['name'], batch=batch_of(rec),
               frozen_core=not rec['config']['condition']['train_core'], interrupted=bool(rec['interruptions']))
    tc = (rec['result'] or {}).get('training_compute') or {}
    out['saved_forward_calls'] = tc.get('forward_calls')
    out['saved_gradient_calls'] = tc.get('gradient_forward_calls')
    out['saved_neuron_ms'] = tc.get('neuron_ms')
    if not rec['history'] or rec['config']['condition'].get('model') == 'board_mlp':
        out.update(rebuilt_forward_calls=None, rebuilt_gradient_calls=None, match=None)
        return out
    full = _replay(rec)
    out['rebuilt_forward_calls'] = full.calls
    out['rebuilt_gradient_calls'] = full.grad
    if not rec['interruptions']:
        out['match'] = (None if out['saved_forward_calls'] is None else
                        (full.calls == out['saved_forward_calls'] and full.grad == out['saved_gradient_calls']))
        out['uninterrupted_equivalent_calls'] = full.calls
        out['lifetime_calls_low'] = out['lifetime_calls_high'] = full.calls
        return out
    last = rec['interruptions'][-1]['last_saved_updates']
    start = next(i for i, x in enumerate(rec['history']) if x['updates'] > last)
    seg = _replay(rec, start=start, resume=True)
    out['post_resume_rebuilt_calls'] = seg.calls
    out['post_resume_rebuilt_gradient_calls'] = seg.grad
    out['match'] = seg.calls == out['saved_forward_calls'] and seg.grad == out['saved_gradient_calls']
    probe = _Counter(False, batch_of(rec))
    resume_cost = probe.assessment(rec['exposure'][start - 1]['states'], rec['behavior'][start - 1]['records'])
    lost_updates = rec['history'][start]['updates'] - last
    lost_cost = lost_updates * 2 * math.ceil(rec['history'][start]['training_states'] / batch_of(rec))
    probe2 = _Counter(False, batch_of(rec))
    inflight = probe2.assessment(rec['exposure'][start]['states'], rec['behavior'][start]['records'])
    out['uninterrupted_equivalent_calls'] = full.calls
    out['resume_reassessment_calls'] = resume_cost
    out['lost_update_calls'] = lost_cost
    out['lifetime_calls_low'] = full.calls + resume_cost + lost_cost
    out['lifetime_calls_high'] = full.calls + resume_cost + lost_cost + inflight
    return out
