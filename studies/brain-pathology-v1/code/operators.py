"""Pathology operators on the healthy study brain (plan v2, Phases 2-3). CPU-importable; applied on the GPU host.

An operator changes one measured property of identified cells, using only mechanisms the healthy model already has
(criterion C3). It wraps a constructed StudyBrain instance (studies/brain-sokoban-v1/code/backend.py, never edited):

    threshold_offset(cells, mV)   spike threshold v_th + offset (Kv4/A-current loss: negative; lower rheobase)
    adaptation_scale(cells, k)    spike-frequency-adaptation jump x k (KCNQ/M-current loss: k < 1)
    release_scale(cells, k)       presynaptic output of the cells x k (all their synapses; release probability)
    input_scale(cells, k)         all synaptic input onto the cells x k (postsynaptic gain)
    remove(cells)                 the cells never spike (neuron loss)

Operators compose; apply() takes a list of specs. The threshold, release and input operators act after the brain's
own per-group parameters are bound at every advance, so they compose with trained parameters and keep gradients
(joint training of a lesioned brain). At zero dose (offset 0, factor 1, no cells) apply() changes nothing: the
brain stays bit-identical, which tests/test_operators.py checks on the GPU.
"""
import types

import numpy as np

KINDS = ('threshold_offset', 'adaptation_scale', 'release_scale', 'input_scale', 'remove')
IDENTITY = {'threshold_offset': 0.0, 'adaptation_scale': 1.0, 'release_scale': 1.0, 'input_scale': 1.0}


def spec(kind, cells, value=None, label=None):
    if kind not in KINDS:
        raise ValueError(f'unknown operator {kind}')
    cells = np.unique(np.asarray(cells, dtype=np.int64))
    if kind != 'remove' and value is None:
        raise ValueError(f'{kind} needs a value')
    return dict(kind=kind, cells=cells, value=None if kind == 'remove' else float(value), label=label)


def is_identity(s):
    return len(s['cells']) == 0 or (s['kind'] != 'remove' and s['value'] == IDENTITY[s['kind']])


def apply(enc, specs):
    """Apply operator specs to a StudyBrain `enc` (in place). Returns an audit dict. Call once, after the checkpoint
    is loaded and before simulation; a second call replaces the bind-time operators of the first."""
    import torch
    active = [s for s in specs if not is_identity(s)]
    n = enc.brain.n
    dev = enc.brain.device
    audit = dict(operators=[dict(kind=s['kind'], cells=int(len(s['cells'])), value=s['value'], label=s['label']) for s in specs],
                 effective=len(active))
    theta = torch.zeros(n, device=dev)
    pre = torch.ones(n, device=dev, dtype=torch.float64)
    post = torch.ones(n, device=dev, dtype=torch.float64)
    jump = None
    for s in active:
        idx = torch.as_tensor(s['cells'], device=dev)
        if s['kind'] == 'threshold_offset':
            theta[idx] += s['value']
        elif s['kind'] == 'release_scale':
            pre[idx] *= s['value']
        elif s['kind'] == 'input_scale':
            post[idx] *= s['value']
        elif s['kind'] == 'adaptation_scale':
            if jump is None:
                base = enc.brain.adapt_jump_vec
                jump = base.clone() if base is not None else torch.full((n,), float(enc.brain.p.adapt_jump), device=dev)
            jump[idx] *= s['value']
        elif s['kind'] == 'remove':
            enc.brain.active[idx] = 0.0
    if jump is not None:
        enc.brain.adapt_jump_vec = jump
    bind_ops = any(s['kind'] in ('threshold_offset', 'release_scale', 'input_scale') for s in active)
    original = type(enc).bind
    if bind_ops:
        def bind(self):
            original(self)
            if any(s['kind'] == 'threshold_offset' for s in active):
                self.params.v_th = self.params.v_th + theta
            m = self.matrix
            if any(s['kind'] == 'release_scale' for s in active):
                m.pre = m.pre * pre.to(m.pre.dtype)
            if any(s['kind'] == 'input_scale' for s in active):
                m.post = m.post * post.to(m.post.dtype)
        enc.bind = types.MethodType(bind, enc)
    else:
        enc.bind = types.MethodType(original, enc)
    audit['bind_wrapped'] = bind_ops
    audit['adaptation_vector'] = jump is not None
    audit['removed'] = int(sum(len(s['cells']) for s in active if s['kind'] == 'remove'))
    return audit


def lif_isolated(drive_mv, ms, v_rest=-52.0, v_th=-45.0, tau_m=20.0, t_ref=2.2, adapt_jump=1.5, adapt_tau=200.0, dt=1.0):
    """Spike times of one isolated LIF cell with spike-frequency adaptation under a drive time course (mV per ms),
    with the model's exact update order (flybrain.brain._step_inference): target = v_rest + drive - adapt,
    v = target + (v - target) a_m, refractory clamp, spike at v >= v_th, reset to v_rest, adapt = adapt a + jump x spike."""
    a_m, a_ad = np.exp(-dt / tau_m), np.exp(-dt / adapt_tau)
    v, adapt, refrac, spikes = v_rest, 0.0, 0.0, []
    drive = np.broadcast_to(np.asarray(drive_mv, float), (ms,))
    for t in range(ms):
        target = v_rest + drive[t] - adapt
        v = target + (v - target) * a_m
        if refrac > 0:
            v = v_rest
        refrac = max(refrac - dt, 0.0)
        s = v >= v_th
        if s:
            v, refrac = v_rest, t_ref
            spikes.append(t + 1)
        adapt = adapt * a_ad + (adapt_jump if s else 0.0)
    return np.array(spikes)


def rheobase(v_th_offset=0.0, ms=500, v_rest=-52.0, v_th=-45.0, **kw):
    """Smallest constant drive (mV, to 1e-4) that makes the isolated cell spike within `ms`."""
    lo, hi = 0.0, 40.0
    th = v_th + v_th_offset
    while hi - lo > 1e-4:
        mid = (lo + hi) / 2
        if len(lif_isolated(mid, ms, v_rest=v_rest, v_th=th, **kw)):
            hi = mid
        else:
            lo = mid
    return hi


def ramp_threshold(slope_mv_per_ms, v_th_offset=0.0, ms=2000, v_rest=-52.0, v_th=-45.0, **kw):
    """Drive at the first spike and the time integral of the drive up to it (charge analogue) under a linear ramp."""
    t = np.arange(1, ms + 1) * slope_mv_per_ms
    s = lif_isolated(t, ms, v_rest=v_rest, v_th=v_th + v_th_offset, **kw)
    if not len(s):
        return np.nan, np.nan
    k = int(s[0])
    return float(t[k - 1]), float(t[:k].sum())


def adaptation_index(drive_mv, jump_scale=1.0, ms=1000, **kw):
    """Steady-state over initial firing rate under a constant drive (1 = no adaptation): spikes in the last 500 ms
    over spikes in the first 100 ms, both as rates."""
    kw = dict(kw)
    kw['adapt_jump'] = kw.get('adapt_jump', 1.5) * jump_scale
    s = lif_isolated(drive_mv, ms, **kw)
    first = np.sum(s <= 100) / 0.1
    last = np.sum(s > ms - 500) / 0.5
    return float(last / first) if first > 0 else np.nan, float(first), float(last)
