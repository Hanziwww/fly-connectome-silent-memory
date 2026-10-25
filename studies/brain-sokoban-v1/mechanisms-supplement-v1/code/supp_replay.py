"""Frozen histories, selective continuous-state interventions and readout taps."""
from supp_base import *
import copy
import torch
from mechanism_model import Frozen, tasklib
from game import legal, restore, solved, state_from, step


def paired_indices(cases):
    return [next(j for j, d in enumerate(cases) if d['cue'] == 1-c['cue'] and
                 d['context'] == c['context'] and d.get('visible', False) == c.get('visible', False))
            for c in cases]


def matched_control(value, delta, nonnegative=False, seed=10):
    """Component-wise, per-case norm match; boundary effects are audited."""
    shape = value.shape
    x = value.reshape(len(value), -1)
    d = delta.reshape(len(value), -1)
    generator = torch.Generator(device=value.device); generator.manual_seed(seed)
    perm = torch.randperm(d.shape[1], generator=generator, device=d.device)
    direction = d[:, perm]
    norm2 = d.square().sum(1, keepdim=True)
    direction = direction - (direction*d).sum(1, keepdim=True) / norm2.clamp_min(1e-30) * d
    # A degenerate permutation is replaced by a fixed Gaussian direction.
    replacement = torch.randn(d.shape, generator=generator, device=d.device, dtype=d.dtype)
    replacement -= (replacement*d).sum(1, keepdim=True) / norm2.clamp_min(1e-30) * d
    direction = torch.where(direction.square().sum(1, keepdim=True) < 1e-20, replacement, direction)
    direction *= norm2.sqrt() / direction.norm(dim=1, keepdim=True).clamp_min(1e-30)
    if nonnegative:
        target = d.norm(dim=1, keepdim=True)
        lo = torch.zeros_like(target); hi = torch.ones_like(target)
        def change(scale):
            return (x + direction*scale).clamp_min(0) - x
        for _ in range(12):
            hi = torch.where(change(hi).norm(dim=1, keepdim=True) < target, hi*2, hi)
        for _ in range(22):
            mid = (lo+hi)/2
            less = change(mid).norm(dim=1, keepdim=True) < target
            lo = torch.where(less, mid, lo); hi = torch.where(less, hi, mid)
        direction = change((lo+hi)/2)
    return direction.reshape(shape)


def manipulate(policy, cases, spec):
    e = policy.encoder; b = e.brain
    saved = e.capture_state(detach=True)
    index = torch.as_tensor(paired_indices(cases), device=b.v.device)
    kind = spec['kind']; fraction = spec.get('fraction', .5)
    audit = []
    for field in PARTS[spec.get('part', 'continuous')]:
        value = saved['tensors'][field]
        axis = 1 if field == 'g_slow_cls' else 0
        donor = value.index_select(axis, index)
        delta = (donor-value)*fraction
        if kind in ('same_cue', 'sham'):
            applied = torch.zeros_like(value)
        elif kind == 'scrambled':
            v = value.movedim(axis, 0); d = delta.movedim(axis, 0)
            applied = matched_control(v, d, field == 'adapt', 10 + sum(map(ord, field))).movedim(0, axis)
        else:
            applied = delta
        setattr(b, field, value + applied)
        a = applied.movedim(axis, 0).reshape(len(cases), -1)
        d = delta.movedim(axis, 0).reshape(len(cases), -1)
        norm = a.norm(dim=1); requested = d.norm(dim=1)
        error = (norm-requested).abs()/requested.clamp_min(1e-20)
        cosine = (a*d).sum(1)/(norm*requested).clamp_min(1e-20)
        for i, c in enumerate(cases):
            audit.append(dict(case=c['id'], field=field, requested_l2=float(requested[i]),
                applied_l2=float(norm[i]), relative_norm_error=float(error[i]),
                cue_direction_cosine=float(cosine[i]),
                norm_matched=bool(error[i] < .01 or requested[i] < 1e-12),
                zero_boundary_fraction=float((getattr(b, field).movedim(axis, 0)[i] == 0).float().mean())
                    if field == 'adapt' else 0.))
    if kind == 'restore':
        e.restore_state(saved)
        for k, v in saved['tensors'].items():
            assert torch.equal(v, getattr(b, k)), 'State restoration was not exact'
    changed = set(PARTS[spec.get('part', 'continuous')])
    for k, v in saved['tensors'].items():
        if k not in changed:
            assert torch.equal(v, getattr(b, k)), 'Unexpected discrete/input state mutation: ' + k
    assert (b.t, b.step_count, b.buf_pos) == (saved['t'], saved['step_count'], saved['buf_pos'])
    assert torch.equal(b.gen.get_state(), saved['generator'])
    b._rate_np_key = None
    return audit


class FrozenFollowup(Frozen):
    def advance_recorded(self, boards, ms=40, route=None, record=False):
        e = self.encoder
        x = torch.as_tensor(np.asarray(boards), dtype=torch.float32, device='cuda')
        if x.ndim == 3:
            x = x[None]
        slots = None
        if route and route['channel'] in ('readout', 'both'):
            indices = torch.as_tensor(route['indices'], device='cuda')
            slots = torch.nonzero(torch.isin(e.observed, indices)).flatten()
        if route and route['channel'] in ('network', 'both'):
            gate = torch.ones(e.brain.n, device='cuda'); gate[route['indices']] = 1-route['strength']
            e.matrix.gate = gate
        currents = []; voltage_taps = []; instants = []; times = []
        def hook(brain, t):
            if slots is not None and t == 14:
                currents.append(brain.g[:, e.observed[slots]].clone())
            if slots is not None and t == ms:
                voltage_taps.append((brain.v[:, e.observed[slots]]-e.params.v_rest).clone())
            if record:
                g = brain.g[:, e.observed].reshape(e.batch, 256, 16).mean(2)
                v = (brain.v[:, e.observed]-e.params.v_rest).reshape(e.batch, 256, 16).mean(2)
                instants.append(torch.cat((g, v), 1).cpu().numpy())
                times.append(float(brain.t))
        use_hook = hook if record or slots is not None else None
        with torch.no_grad():
            f = e.advance(x, ms, hook=use_hook)
            original = f.clone()
            if slots is not None and len(slots):
                assert ms == 40 and len(currents) == 1 and len(voltage_taps) == 1
                pools = slots//16
                for offset, tapped in ((0, currents[0]), (256, voltage_taps[0])):
                    correction = (e.batch, 256)
                    diff = torch.zeros(correction, device='cuda')
                    replacements = self.head.mean[offset+pools][None]
                    diff.scatter_add_(1, pools[None].expand(e.batch, -1), (replacements-tapped)/16)
                    f[:, offset:offset+256] += route['strength']*diff
            q = self.head(f)['final_logits']
        if not torch.isfinite(f).all() or not torch.isfinite(q).all():
            raise FloatingPointError('Nonfinite policy feature or logits')
        return q.cpu().numpy(), f.cpu().numpy(), dict(
            raw_features=original.cpu().numpy(),
            instantaneous=np.stack(instants) if instants else None,
            milliseconds=np.asarray(times, dtype=np.float32))


def ordinary_frames(task, delay, prefix):
    lib = tasklib()
    return [lib.frames(c, delay, prefix, False)[:-1] for c in task['cases']]


def timing_frames(task, total_ms, prefix, input_kind):
    lib = tasklib(); cases = task['cases']
    canonical = copy.deepcopy(cases[0]); canonical['level'] = canonical['cue_level']
    canonical['visible'] = False; canonical.pop('legacy_prefix', None)
    start = state_from(canonical['level']['start'])
    if input_kind == 'blank':
        assert prefix == 0
        neutral = []
    elif input_kind == 'neutral_stationary':
        neutral = [start]*prefix
    else:
        neutral = lib.prefix_states(canonical, prefix, int(input_kind[-1]))
    result = []
    for case in cases:
        seq = [lib.frames(case, 0, 0, False)[0]]
        wait = total_ms - 40*prefix
        assert wait >= 0
        if wait:
            seq.append(dict(board=np.zeros((4, 10, 10), np.float32), ms=wait, kind='delay'))
        seq += [dict(board=lib.board(canonical, s), ms=40, kind='neutral') for s in neutral]
        assert sum(f['ms'] for f in seq) == total_ms+40
        result.append(seq)
    for i in range(1, len(result[0])):
        for seq in result[1:]:
            np.testing.assert_array_equal(seq[i]['board'], result[0][i]['board'])
    return result


def rollout(policy, task, delay=0, prefix=0, spec=None, route=None, frames=None, record_path=None):
    lib = tasklib(); cases = task['cases']; policy.reset(len(cases))
    xs = frames if frames is not None else ordinary_frames(task, delay, prefix)
    traces = []; input_blocks = []; phase_blocks = []; segment_features = []; events = []
    audit = []; elapsed = 0
    def advance(boards, ms, kind, decision=-1):
        nonlocal elapsed
        active = route if route and decision >= 0 and (route['window'] == 'all_decisions' or decision == 0) else None
        q, f, extra = policy.advance_recorded(boards, ms, active, bool(record_path) and decision <= 0)
        if record_path and decision <= 0:
            traces.append(extra['instantaneous'])
            input_blocks.append(np.repeat(np.asarray(boards).reshape(len(cases), 400)[None], ms, axis=0))
            phase_blocks.append(np.repeat({'cue': 0, 'delay': 1, 'neutral': 2, 'prefix': 2, 'decision': 3}[kind], ms))
            segment_features.append(f)
            events.append(dict(kind=kind, start_ms=elapsed, ms=ms))
        elapsed += ms
        return q, f, extra
    for i, frame in enumerate(xs[0]):
        check_stop(); boards = np.stack([x[i]['board'] for x in xs]); ms = frame['ms']; kind = frame['kind']
        if spec and spec['window'] == 'hold' and kind == 'delay':
            half = ms//2
            advance(boards, half, kind); audit.extend(manipulate(policy, cases, spec))
            advance(boards, ms-half, kind)
        else:
            advance(boards, ms, kind)
        if spec and spec['window'] == 'write' and kind == 'cue':
            audit.extend(manipulate(policy, cases, spec))
    if spec and spec['window'] == 'read':
        audit.extend(manipulate(policy, cases, spec))
    before_state = None
    if record_path:
        before_state = np.stack([getattr(policy.encoder.brain, k).cpu().numpy().copy() for k in ('v', 'g', 'adapt')])
    maps = [restore(c['level']) for c in cases]
    states = [state_from(c['level']['start']) for c in cases]
    safe = [set(c['solvable_states']) for c in cases]
    rows = [dict(case=c['id'], cue=c['cue'], context=c['context'], visible=c.get('visible', False),
        status='running', first_push_correct=None, decisions=[]) for c in cases]
    for index in range(512):
        check_stop()
        boards = np.stack([lib.board(c, s) for c, s in zip(cases, states)])
        q, f, extra = advance(boards, 40, 'decision', index)
        for r, c, m, s, safe_states, qi, fi, raw in zip(rows, cases, maps, states, safe, q, f, extra['raw_features']):
            if r['status'] != 'running':
                continue
            mask = legal(m, s)
            if not mask.any():
                r['status'] = 'blocked'; continue
            action = int(np.where(mask, qi, -np.inf).argmax()); following = step(m, s, action)
            r['decisions'].append(dict(state=[s[0], list(s[1])], action=action, legal=mask.tolist(),
                logits=qi.tolist(), features=fi.tolist(), raw_feature_change_l2=float(np.linalg.norm(fi-raw))))
            if following[1] != s[1] and r['first_push_correct'] is None:
                r['first_push_correct'] = lib.state_key(following) in safe_states; r['first_push_action'] = action
            if solved(m, following):
                r['status'] = 'solved'
            elif following[1] != s[1] and lib.state_key(following) not in safe_states:
                r['status'] = 'irreversible_deadlock'
            states[rows.index(r)] = following
        if all(r['status'] != 'running' for r in rows):
            break
    for r in rows:
        if r['status'] == 'running': r['status'] = 'action_limit'
        r['steps'] = len(r['decisions'])
    if record_path:
        save_npz(record_path, instantaneous=np.concatenate(traces).astype(np.float32),
            input_boards=np.concatenate(input_blocks).astype(np.float32), phase=np.concatenate(phase_blocks),
            segment_features=np.stack(segment_features).astype(np.float32),
            predecision_state=before_state.astype(np.float32), fields=np.asarray(['v', 'g', 'adapt']))
        write(Path(record_path).with_suffix('.json'), dict(events=events,
            cases=[{k: c.get(k, False) for k in ('id', 'cue', 'context', 'visible')} for c in cases]))
    return rows, audit


def assert_equal(rows, baseline):
    assert len(rows) == len(baseline)
    for r, b in zip(rows, baseline):
        assert r['case'] == b['case'] and r['status'] == b['status']
        assert [d['action'] for d in r['decisions']] == [d['action'] for d in b['decisions']]
        np.testing.assert_allclose([d['features'] for d in r['decisions']],
                                   [d['features'] for d in b['decisions']], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose([d['logits'] for d in r['decisions']],
                                   [d['logits'] for d in b['decisions']], rtol=1e-5, atol=1e-5)
