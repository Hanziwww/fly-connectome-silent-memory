"""X6/X7: the cue ensembles and their matched null cell and edge sets (local, CPU, before any run; reused on the host).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-supplement-v2\\code\\x6_x7_prepare.py

Ensemble: the top 1% of cells by |cue difference of v| (ties broken by index, as in Fig. 6) in the 160-ms-delay,
4-prefix state of M0 readout-only and of S1 (mechanisms-v1/interpretation/<model>-memory-cells-d160-p4.parquet).
Cell strata: cell type x hop class from the input cells (0, 1, 2, >=3, none) x degree quintile x role (readout or not);
coarser levels superclass x hop x degree x role and hop x degree x role.
- X6 null cell sets: 20 per ensemble, the same size, drawn from the cells outside the ensemble and the input cells,
  with the ensemble's counts per stratum (a stratum short of cells borrows from the next coarser level; counted).
- X7 null edge sets: 20 per ensemble, as many synapse pairs as within the ensemble, drawn from the pairs that are not
  within the ensemble and touch no input cell, with the ensemble edges' counts per (postsynaptic stratum, presynaptic
  stratum, sign) at the superclass level (then the coarsest level when short). Edge sets are regenerated on the host
  from the same seed and checked against the SHA-256 stored here (they are too large to track).
Writes tasks/x6-x7-sets.json (tracked; cell indices in model order).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
OUT = SUPP / 'tasks' / 'x6-x7-sets.json'
SOURCES = {'m0': ('remote-dynamic-M0-head_only', 0, 0), 's1': ('supp-S1-ring-brain', 0, 0)}
TOP = 0.01
N_NULL = 20
SEED = 20261019


def hops_from(A, sources):
    dist = np.full(A.shape[0], -1, np.int32)
    dist[sources] = 0
    frontier = np.zeros(A.shape[0], np.float32)
    frontier[sources] = 1
    k = 0
    while frontier.any():
        k += 1
        new = ((A @ frontier) > 0) & (dist < 0)
        dist[new] = k
        frontier = new.astype(np.float32)
    return dist


def load():
    """Connectome (rows postsynaptic, no stored zeros), cell strata codes at three levels, input and readout cells."""
    neurons = pd.read_parquet(ROOT / 'flybrain' / 'cache' / 'neurons.parquet', columns=['bodyId', 'type', 'superclass'])
    ann = pd.read_parquet(MECH / 'neurons' / 'annotations.parquet', columns=['bodyId', 'direct_input', 'observed_by_policy'])
    if not np.array_equal(neurons['bodyId'].to_numpy(np.int64), ann['bodyId'].to_numpy(np.int64)):
        raise RuntimeError('cell order differs')
    W = sp.load_npz(ROOT / 'flybrain' / 'cache' / 'W_post_pre.npz').tocsr()
    W.eliminate_zeros()
    W.sort_indices()
    A = (W != 0).astype(np.float32).tocsr()
    inputs = np.flatnonzero(ann['direct_input'].to_numpy(bool))
    readout = np.flatnonzero(ann['observed_by_policy'].to_numpy(bool))
    hop = hops_from(A, inputs)
    hop_class = np.where(hop < 0, 4, np.minimum(hop, 3))
    degree = np.asarray((A != 0).sum(0)).ravel() + np.asarray((A != 0).sum(1)).ravel()
    quint = np.clip(np.searchsorted(np.quantile(degree, [0.2, 0.4, 0.6, 0.8]), degree, side='right'), 0, 4)
    role = np.zeros(W.shape[0], np.int64)
    role[readout] = 1
    sc = neurons['superclass'].fillna('unannotated').astype(str)
    ctype = neurons['type'].astype(object).where(neurons['type'].notna(), 'untyped:' + sc).astype(str)
    hdr = pd.Series(hop_class).astype(str) + '|' + pd.Series(quint).astype(str) + '|' + pd.Series(role).astype(str)
    levels = [pd.factorize(ctype + '|' + hdr)[0], pd.factorize(sc + '|' + hdr)[0], pd.factorize(hdr)[0]]
    return W, A, inputs, readout, levels


def ensemble(n, model, ctx, vis):
    f = pd.read_parquet(MECH / 'interpretation' / f'{model}-memory-cells-d160-p4.parquet', columns=[f'c{ctx}-visible{vis}-delta-v'])
    return np.sort(np.argsort(-np.abs(f.iloc[:, 0].to_numpy(np.float64)), kind='mergesort')[:int(np.ceil(TOP * n))])


def matched_cells(target, candidates, levels, rng):
    taken = np.zeros(len(levels[0]), bool)
    pick, borrowed = [], 0
    for code in np.unique(levels[0][target]):
        members = target[levels[0][target] == code]
        need = len(members)
        for depth, codes in enumerate(levels):
            pool = candidates[(codes[candidates] == codes[members[0]]) & ~taken[candidates]]
            m = min(need, len(pool))
            if m:
                c = rng.choice(pool, m, replace=False)
                taken[c] = True
                pick.append(c)
                need -= m
                borrowed += m if depth else 0
            if need == 0:
                break
        if need:
            raise RuntimeError('not enough candidate cells')
    return np.sort(np.concatenate(pick)), borrowed


def edge_arrays(W):
    post = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
    return post, W.indices.astype(np.int64), np.sign(W.data).astype(np.int64)


def within_mask(post, pre, cells, n):
    m = np.zeros(n, bool)
    m[cells] = True
    return m[post] & m[pre]


def edge_nulls(W, ens, inputs, readout, levels, rng, n_sets=N_NULL):
    """n_sets arrays of edge positions (indices into the CSR data, sorted) matched to the ensemble's edges by
    (postsynaptic stratum, presynaptic stratum, sign), from the superclass level down to (role pair, sign) and sign."""
    n = W.shape[0]
    post, pre, sign = edge_arrays(W)
    ens_edges = within_mask(post, pre, ens, n)
    is_input = np.zeros(n, bool)
    is_input[inputs] = True
    role = np.zeros(n, np.int32)
    role[readout] = 1
    pool = np.flatnonzero(~ens_edges & ~is_input[post] & ~is_input[pre])
    ens_idx = np.flatnonzero(ens_edges)
    s1 = (sign + 1).astype(np.int32)
    keys = []
    for codes in (levels[1].astype(np.int32), levels[2].astype(np.int32), role):
        k = np.int64(codes.max() + 1)
        keys.append(((codes[post].astype(np.int64) * k + codes[pre]) * 3 + s1).astype(np.int64))
    keys.append(s1.astype(np.int64))
    del post, pre
    order = [np.argsort(key[pool], kind='mergesort') for key in keys]
    sorted_keys = [key[pool][o] for key, o in zip(keys, order)]
    fine, first, counts = np.unique(keys[0][ens_idx], return_index=True, return_counts=True)
    coarser = [key[ens_idx][first] for key in keys[1:]]
    sets, borrowed_all = [], []
    for _ in range(n_sets):
        taken = np.zeros(len(pool), bool)
        pick, borrowed = [], 0
        for i, (key, count) in enumerate(zip(fine, counts)):
            need = int(count)
            values = [key] + [c[i] for c in coarser]
            for depth, value in enumerate(values):
                lo, hi = np.searchsorted(sorted_keys[depth], value, 'left'), np.searchsorted(sorted_keys[depth], value, 'right')
                cand = order[depth][lo:hi]
                cand = cand[~taken[cand]]
                m = min(need, len(cand))
                if m:
                    c = rng.choice(cand, m, replace=False)
                    taken[c] = True
                    pick.append(pool[c])
                    need -= m
                    borrowed += m if depth else 0
                if need == 0:
                    break
            if need:
                raise RuntimeError('not enough candidate synapse pairs')
        sets.append(np.sort(np.concatenate(pick)))
        borrowed_all.append(borrowed)
    return ens_idx, sets, borrowed_all


def edge_hash(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.asarray(a, np.int64).tobytes())
    return h.hexdigest()


def build():
    W, A, inputs, readout, levels = load()
    n = W.shape[0]
    out = {}
    for key, (model, ctx, vis) in SOURCES.items():
        rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(sum(map(ord, key)),)))
        ens = ensemble(n, model, ctx, vis)
        excluded = np.zeros(n, bool)
        excluded[inputs] = excluded[ens] = True
        candidates = np.flatnonzero(~excluded)
        nulls = [matched_cells(ens, candidates, levels, rng) for _ in range(N_NULL)]
        ens_edges, edge_sets, edge_borrowed = edge_nulls(W, ens, inputs, readout, levels, rng)
        out[key] = dict(model=model, ensemble=ens, nulls=nulls, ensemble_edges=ens_edges, edge_sets=edge_sets, edge_borrowed=edge_borrowed,
                        readout_cells=int(np.isin(ens, readout).sum()))
    return W, out


def main():
    _, sets = build()
    doc = dict(meta=dict(seed=SEED, top=TOP, nulls=N_NULL, sources=SOURCES), sets={})
    for key, s in sets.items():
        doc['sets'][key] = dict(model=s['model'], cells=s['ensemble'].tolist(), readout_cells=s['readout_cells'],
                                nulls=[dict(cells=c.tolist(), borrowed=int(b)) for c, b in s['nulls']],
                                ensemble_edges=len(s['ensemble_edges']), ensemble_edges_sha256=edge_hash([s['ensemble_edges']]),
                                edge_sets=[dict(edges=len(e), borrowed=int(b)) for e, b in zip(s['edge_sets'], s['edge_borrowed'])],
                                edge_sets_sha256=edge_hash(s['edge_sets']))
        print(f"{key}: {len(s['ensemble'])} cells ({s['readout_cells']} readout); cell nulls borrowed {[int(b) for _, b in s['nulls']]}; "
              f"{len(s['ensemble_edges'])} internal edges; edge nulls borrowed {s['edge_borrowed']}")
    text = json.dumps(doc, separators=(',', ':'))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding='utf-8', newline='\n')
    print('sha256', hashlib.sha256(text.encode()).hexdigest())
    return 0


if __name__ == '__main__':
    sys.exit(main())
