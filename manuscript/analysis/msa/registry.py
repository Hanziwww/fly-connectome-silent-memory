"""Model registry (brain/readout hashes) and the history-set registry that defines denominators."""
import hashlib
import json

import numpy as np

from . import io, paths, static_runs

CORE_KEYS = ('pre_log_gain', 'post_log_gain', 'threshold_mv')
HEAD_KEYS = ('mean', 'scale', 'proposal.weight', 'proposal.bias')

GRID_DELAYS = (0, 20, 40, 80, 160, 320, 640)
GRID_PREFIXES = (0, 1, 2, 4, 8)
TRAINED_DELAYS = (0, 40, 160)
TRAINED_PREFIXES = (0, 1, 4)
INTERVENTION_DELAYS = (40, 160, 640)
INTERVENTION_PREFIXES = (0, 4)

FAMILY_CASES = {
    'M0': [('c0-cue0', 0, 0, False), ('c0-cue1', 1, 0, False)],
    'S1': [('c0-cue0', 0, 0, False), ('c0-cue1', 1, 0, False)],
    'S2': [('c0-cue0', 0, 0, False), ('c0-cue1', 1, 0, False), ('c1-cue0', 0, 1, False), ('c1-cue1', 1, 1, False)],
    'M0vis': [('c0-cue0-hidden', 0, 0, False), ('c0-cue1-hidden', 1, 0, False),
              ('c0-cue0-visible', 0, 0, True), ('c0-cue1-visible', 1, 0, True)],
}


def _np(t):
    return np.ascontiguousarray(t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t), dtype=np.float32)


def digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(_np(a).tobytes())
    return h.hexdigest()


def split_checkpoint(ckpt):
    """Return (core dict or None, head dict or None, other dict) for either checkpoint layout."""
    if ckpt is None:
        return None, None, {}
    pair = ckpt.get('pair', ckpt)
    core = pair.get('core') or None
    head = pair.get('head') or None
    other = {k: v for k, v in pair.items() if k not in ('core', 'head')}
    return core, head, other


def core_hash(core):
    if not core or not all(k in core for k in CORE_KEYS):
        return None
    return digest([core[k] for k in CORE_KEYS])


def head_hash(head):
    if not head:
        return None
    keys = [k for k in HEAD_KEYS if k in head] or sorted(head)
    return digest([head[k] for k in keys])


def n_elements(d):
    return int(sum(int(np.prod(_np(v).shape)) for v in d.values())) if d else 0


def classify_kind(core, head, other):
    if 'gru' in other:
        return 'gru'
    if core and all(k in core for k in CORE_KEYS):
        return 'connectome_linear'
    if head and 'net.0.weight' in head:
        return 'board_mlp'
    if head and 'proposal.weight' in head:
        return 'fixed_trace_linear'
    return 'other'


# ---------------------------------------------------------------- history sets

_sets = {}


def split_of(delay, prefix):
    return 'train' if delay in TRAINED_DELAYS and prefix in TRAINED_PREFIXES else 'untrained'


def grid_members(family, splits=('train', 'untrained')):
    rows = []
    for case, cue, context, visible in FAMILY_CASES[family]:
        for d in GRID_DELAYS:
            for p in GRID_PREFIXES:
                s = split_of(d, p)
                if s in splits:
                    rows.append(dict(family=family, case=case, cue=cue, context=context, visible=visible,
                                     delay_ms=d, prefix_windows=p, split=s))
    return rows


def register_set(set_id, family, description, members, unit='history'):
    if set_id in _sets:
        old = _sets[set_id]
        if old['n'] != len(members):
            raise ValueError(f'history set {set_id} redefined with a different size')
        return set_id
    _sets[set_id] = dict(history_set_id=set_id, family=family, description=description, unit=unit,
                         n=len(members), members=members)
    return set_id


def save_sets(phase):
    target = paths.work(f'history_sets/{phase}.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(list(_sets.values()), default=_jd), encoding='utf-8')


def _jd(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    raise TypeError(type(v))


def all_sets():
    merged = {}
    folder = paths.work('history_sets')
    if folder.exists():
        for f in sorted(folder.glob('*.json')):
            for s in json.loads(f.read_text(encoding='utf-8')):
                merged.setdefault(s['history_set_id'], s)
    for k, s in _sets.items():
        merged.setdefault(k, s)
    return [merged[k] for k in sorted(merged)]


def register_protocol_sets():
    """Timing grids and intervention timing sets defined by the protocol."""
    for fam in ('M0', 'S1', 'S2', 'M0vis'):
        for split in ('train', 'untrained'):
            m = grid_members(fam, (split,))
            register_set(f'{fam}.grid.{split}', fam, f'{fam} timing grid, {split} delay x prefix combinations, all cases', m)
            pairs = {}
            for r in m:
                k = (r['context'], r['visible'], r['delay_ms'], r['prefix_windows'])
                pairs.setdefault(k, dict(family=fam, context=r['context'], visible=r['visible'], delay_ms=r['delay_ms'],
                                         prefix_windows=r['prefix_windows'], split=split))
            register_set(f'{fam}.pairs.{split}', fam, f'{fam} cue pairs (one per context x visibility x timing), {split}',
                         list(pairs.values()), unit='cue_pair')
        inter = []
        for case, cue, context, visible in FAMILY_CASES[fam]:
            for d in INTERVENTION_DELAYS:
                for p in INTERVENTION_PREFIXES:
                    inter.append(dict(family=fam, case=case, cue=cue, context=context, visible=visible,
                                      delay_ms=d, prefix_windows=p, split=split_of(d, p)))
        register_set(f'{fam}.interv6', fam, f'{fam} intervention timings: delays 40/160/640 ms x prefixes 0/4, all cases', inter)
