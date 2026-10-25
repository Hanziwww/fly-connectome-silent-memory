"""G3a, circuit dependency map: build the lesion conditions (local, CPU; before any run).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\phase3_dependency_prepare.py

Target populations of the retained cases (G2 shortlist): the Kenyon cells (C04 late KC loss, C16 tau in KCs) and the
GABAergic ALLNs (C13 Fragile X), removed at 50% (a seeded subset) and 100%. Null sets: 10 per target set, of the same
size, drawn from the cells that are not input, readout or target cells, matched to the target set's joint distribution
of hop class from the input cells (0, 1, 2, >=3, none; the paper's Fig. 6 classes) and degree quintile. A stratum with
too few candidates borrows from the nearest degree quintile of the same hop class (counted per set). Anchors: every
monoamine cell (an exact no-op: their synapses have sign 0) and every descending neuron (a movement positive control:
the giant fibre is one).
Cell sets follow credibility.cell_sets on the cached annotations; the GPU run recomputes them from the brain it builds
and stops if they differ. Writes tasks/phase3-dependency-conditions.json (cell indices in model order; tracked, so the
push sends it).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
CONDITIONS = STUDY / 'tasks' / 'phase3-dependency-conditions.json'
N_NULL = 10
SEED = 20261019
LEVELS = (0.5, 1.0)


def hops_from(A, sources):
    """Shortest synaptic path from any source (rows postsynaptic: A @ x moves activity downstream); -1 unreachable."""
    n = A.shape[0]
    dist = np.full(n, -1, np.int32)
    dist[sources] = 0
    frontier = np.zeros(n, np.float32)
    frontier[sources] = 1
    k = 0
    while frontier.any():
        k += 1
        new = ((A @ frontier) > 0) & (dist < 0)
        dist[new] = k
        frontier = new.astype(np.float32)
    return dist


def strata_codes(hops, degree):
    hop_class = np.where(hops < 0, 4, np.minimum(hops, 3))                     # 0, 1, 2, >=3, none
    quint = np.clip(np.searchsorted(np.quantile(degree, [0.2, 0.4, 0.6, 0.8]), degree, side='right'), 0, 4)
    return hop_class, quint


def matched_set(target, candidates, hop_class, quint, rng):
    """One random set matched to the target's (hop class, degree quintile) counts; returns (cells, borrowed)."""
    pick, borrowed = [], 0
    taken = np.zeros(len(hop_class), bool)
    need = pd.Series(list(zip(hop_class[target], quint[target]))).value_counts().sort_index()
    for (h, q), k in need.items():
        got = 0
        for dq in sorted(range(5), key=lambda x: (abs(x - q), x)):                # own quintile first, then the nearest
            pool = candidates[(hop_class[candidates] == h) & (quint[candidates] == dq) & ~taken[candidates]]
            m = min(k - got, len(pool))
            if m:
                c = rng.choice(pool, m, replace=False)
                taken[c] = True
                pick.append(c)
                got += m
                borrowed += m if dq != q else 0
            if got == k:
                break
        if got < k:
            raise RuntimeError(f'hop class {h}: not enough candidate cells')
    return np.sort(np.concatenate(pick)), borrowed


def cell_sets(n):
    """credibility.cell_sets (the subset used here), on the cached annotations."""
    t, cls, nt = n['type'].fillna('').astype(str), n['class'].fillna('').astype(str), n['nt'].fillna('').astype(str)
    return dict(kc=np.flatnonzero(t.str.startswith('KC').to_numpy()),
                alln_gaba=np.flatnonzero(((cls == 'ALLN') & (nt == 'gaba')).to_numpy()),
                monoamine=np.flatnonzero(nt.isin(['dopamine', 'serotonin', 'octopamine']).to_numpy()),
                descending=np.flatnonzero((n['superclass'].fillna('') == 'descending_neuron').to_numpy()))


def main():
    neurons = pd.read_parquet(ROOT / 'flybrain' / 'cache' / 'neurons.parquet')
    ann = pd.read_parquet(MECH / 'neurons' / 'annotations.parquet', columns=['bodyId', 'direct_input', 'observed_by_policy'])
    if not np.array_equal(neurons['bodyId'].to_numpy(np.int64), ann['bodyId'].to_numpy(np.int64)):
        raise RuntimeError('cell order of the cached neurons and the mechanism annotations differ')
    W = sp.load_npz(ROOT / 'flybrain' / 'cache' / 'W_post_pre.npz').tocsr()
    W.eliminate_zeros()
    A = (W != 0).astype(np.float32).tocsr()
    n = A.shape[0]
    inputs = np.flatnonzero(ann['direct_input'].to_numpy(bool))
    readout = np.flatnonzero(ann['observed_by_policy'].to_numpy(bool))
    hops = hops_from(A, inputs)
    degree = np.asarray((A != 0).sum(0)).ravel() + np.asarray((A != 0).sum(1)).ravel()
    hop_class, quint = strata_codes(hops, degree)
    sets = cell_sets(neurons)
    rng = np.random.default_rng(SEED)
    conds = [dict(name='healthy', population=None, level=0.0, kind='healthy', cells=[], borrowed=0)]
    for pop in ('kc', 'alln_gaba'):
        full = sets[pop]
        excluded = np.zeros(n, bool)
        excluded[inputs] = excluded[readout] = excluded[full] = True
        candidates = np.flatnonzero(~excluded)
        for level in LEVELS:
            target = full if level == 1.0 else np.sort(rng.choice(full, int(round(level * len(full))), replace=False))
            conds.append(dict(name=f'{pop}-{level:g}', population=pop, level=level, kind='target', cells=target.tolist(), borrowed=0))
            for i in range(N_NULL):
                cells, borrowed = matched_set(target, candidates, hop_class, quint, rng)
                conds.append(dict(name=f'{pop}-{level:g}-null{i}', population=pop, level=level, kind='null', cells=cells.tolist(), borrowed=int(borrowed)))
    for pop in ('monoamine', 'descending'):
        conds.append(dict(name=f'{pop}-1', population=pop, level=1.0, kind='anchor', cells=sets[pop].tolist(), borrowed=0))
    CONDITIONS.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(seed=SEED, nulls=N_NULL, levels=list(LEVELS), cells=int(n), inputs=int(len(inputs)), readout=int(len(readout)),
                set_sizes={k: int(len(v)) for k, v in sets.items()},
                target_strata={pop: {f'{h},{q}': int(c) for (h, q), c in pd.Series(list(zip(hop_class[sets[pop]], quint[sets[pop]]))).value_counts().sort_index().items()}
                               for pop in ('kc', 'alln_gaba')},
                sources=dict(neurons='flybrain/cache/neurons.parquet', annotations='mechanisms-v1/neurons/annotations.parquet',
                             connectome='flybrain/cache/W_post_pre.npz'))
    text = json.dumps(dict(meta=meta, conditions=conds), separators=(',', ':'))
    CONDITIONS.write_text(text, encoding='utf-8', newline='\n')
    print(f'{len(conds)} conditions, sha256 {hashlib.sha256(text.encode()).hexdigest()}')
    print(json.dumps(meta['set_sizes']), 'borrowed per null set:', [c['borrowed'] for c in conds if c['kind'] == 'null'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
