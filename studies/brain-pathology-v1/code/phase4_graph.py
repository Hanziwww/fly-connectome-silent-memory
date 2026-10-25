"""Phase 4 (local, CPU): graph robustness of the task pathways (plan v2: k-hop path counts, max-flow, percolation).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\phase4_graph.py

Graph: the stored connectome (flybrain/cache/W_post_pre.npz; rows postsynaptic), sign ignored; capacity of an edge =
its synapse count. Sets from the mechanism annotations (visual input cells, readout cells) and the cached cell
annotations (ORNs, Kenyon cells, MBONs, GABAergic ALLNs; credibility.cell_sets).
1. Walks: number of directed walks of length k = 1..6 from the visual input cells to the readout cells (binary edges),
   and the readout cells reached within k synapses; the same from the ORNs to the MBONs.
2. Max-flow (scipy.sparse.csgraph.maximum_flow, Dinic) from a super-source on the input set to a super-sink on the
   target set: visual inputs -> readout cells, ORNs -> MBONs; and the olfactory flow after the retained cases'
   structural lesions (C04: the regime A sets of 17% and 34% of the KCs; G3a: 50% and 100% of the KCs).
3. Percolation: random removal of a share q of the cells that are neither input nor readout (3 seeded draws per q):
   share of readout cells still reachable from the inputs, and walks of length 3 relative to the intact graph; the
   same for degree-ordered (hub-first) removal. The share removed at which reachability falls below one half is the
   structural threshold to compare with the behavioural thresholds of the lesion runs.
Writes results/phase4_graph_paths.csv, phase4_graph_flow.csv and phase4_graph_percolation.csv.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
MECH = ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1'
SEED = 20261019
SHARES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DRAWS = 3
K_MAX = 6


def walks(A, sources, targets, k_max=K_MAX, alive=None):
    """Walk counts of length 1..k_max from sources to targets and the share of targets reached within k (A: rows post)."""
    x = np.zeros(A.shape[0])
    x[sources] = 1.0
    if alive is not None:
        x *= alive
    reached = np.zeros(A.shape[0], bool)
    out = []
    for k in range(1, k_max + 1):
        x = A @ x
        if alive is not None:
            x *= alive
        reached |= x > 0
        out.append((k, float(x[targets].sum()), float(reached[targets].mean())))
    return out


def reach_all(A, sources, targets, alive):
    """Share of targets reachable from the sources through alive cells (any length)."""
    seen = np.zeros(A.shape[0], bool)
    frontier = np.zeros(A.shape[0], np.float32)
    frontier[sources] = 1
    seen[sources] = True
    while frontier.any():
        new = ((A @ frontier) > 0) & alive & ~seen
        seen |= new
        frontier = new.astype(np.float32)
    return float(seen[targets].mean())


def max_flow(C, sources, targets, removed=None):
    """Max-flow from a super-source joined to every source to a super-sink joined from every target. C: capacities as
    (pre, post) CSR with int32 synapse counts; removed cells lose every edge."""
    n = C.shape[0]
    M = C
    if removed is not None and len(removed):
        keep = np.ones(n, bool)
        keep[removed] = False
        D = sp.diags(keep.astype(np.int32))
        M = (D @ C @ D).tocsr()
        M.eliminate_zeros()
    big = int(M.sum()) + 1
    src, snk = n, n + 1
    extra_rows = np.concatenate([np.full(len(sources), src), targets])
    extra_cols = np.concatenate([sources, np.full(len(targets), snk)])
    E = sp.csr_matrix((np.full(len(extra_rows), big, np.int64), (extra_rows, extra_cols)), shape=(n + 2, n + 2))
    G = (sp.block_diag([M.astype(np.int64), sp.csr_matrix((2, 2), dtype=np.int64)], format='csr') + E).tocsr()
    G.sum_duplicates()
    G = G.astype(np.int32) if G.data.max() < 2 ** 31 else G
    return int(maximum_flow(G, src, snk, method='dinic').flow_value)


def write(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    t0 = time.time()
    neurons = pd.read_parquet(ROOT / 'flybrain' / 'cache' / 'neurons.parquet')
    ann = pd.read_parquet(MECH / 'neurons' / 'annotations.parquet', columns=['bodyId', 'direct_input', 'observed_by_policy'])
    if not np.array_equal(neurons['bodyId'].to_numpy(np.int64), ann['bodyId'].to_numpy(np.int64)):
        raise RuntimeError('cell order differs')
    W = sp.load_npz(ROOT / 'flybrain' / 'cache' / 'W_post_pre.npz').tocsr()
    W.eliminate_zeros()
    n = W.shape[0]
    A = (W != 0).astype(np.float64).tocsr()                     # rows post: A @ x moves activity downstream
    C = abs(W).T.tocsr().astype(np.int32)                       # capacities, rows pre
    t, cls, nt = neurons['type'].fillna('').astype(str), neurons['class'].fillna('').astype(str), neurons['nt'].fillna('').astype(str)
    inputs = np.flatnonzero(ann['direct_input'].to_numpy(bool))
    readout = np.flatnonzero(ann['observed_by_policy'].to_numpy(bool))
    orn = np.flatnonzero((cls == 'olfactory').to_numpy())
    mbon = np.flatnonzero((cls == 'MBON').to_numpy())
    kc = np.flatnonzero(t.str.startswith('KC').to_numpy())
    paths = []
    for name, s_, t_ in (('visual inputs -> readout', inputs, readout), ('ORNs -> MBONs', orn, mbon)):
        for k, count, share in walks(A, s_, t_):
            paths.append(dict(pathway=name, k=k, walks=count, log10_walks=float(np.log10(count)) if count > 0 else None, targets_reached_within_k=share))
    write(STUDY / 'results' / 'phase4_graph_paths.csv', paths)
    print(f'walks done ({time.time() - t0:.0f} s)', flush=True)
    regime_a = {c['name']: c for c in json.loads((STUDY / 'runs' / 'phase3-regime-a' / 'conditions.json').read_text(encoding='utf-8'))}
    dep = {c['name']: c for c in json.loads((STUDY / 'tasks' / 'phase3-dependency-conditions.json').read_text(encoding='utf-8'))['conditions']}
    flows = []
    base_visual = max_flow(C, inputs, readout)
    flows.append(dict(pathway='visual inputs -> readout', lesion='none', cells_removed=0, max_flow=base_visual, relative=1.0))
    print(f'visual max-flow {base_visual} ({time.time() - t0:.0f} s)', flush=True)
    base_olf = max_flow(C, orn, mbon)
    flows.append(dict(pathway='ORNs -> MBONs', lesion='none', cells_removed=0, max_flow=base_olf, relative=1.0))
    lesions = [('C04 dose 1 (17% of KCs, regime A set)', regime_a['C04-dose1']['operators'][0]['cells']),
               ('C04 dose 2 (34% of KCs, regime A set)', regime_a['C04-dose2']['operators'][0]['cells']),
               ('50% of KCs (G3a set)', dep['kc-0.5']['cells']), ('all KCs', kc.tolist())]
    for name, cells in lesions:
        for pw, s_, t_, base in (('ORNs -> MBONs', orn, mbon, base_olf), ('visual inputs -> readout', inputs, readout, base_visual)):
            f = max_flow(C, s_, t_, removed=np.asarray(cells, np.int64))
            flows.append(dict(pathway=pw, lesion=name, cells_removed=len(cells), max_flow=f, relative=f / base))
        print(f'{name} done ({time.time() - t0:.0f} s)', flush=True)
    write(STUDY / 'results' / 'phase4_graph_flow.csv', flows)
    rng = np.random.default_rng(SEED)
    pool = np.setdiff1d(np.arange(n), np.concatenate([inputs, readout]))
    degree = np.asarray(A.sum(0)).ravel() + np.asarray(A.sum(1)).ravel()
    hubs = pool[np.argsort(-degree[pool], kind='mergesort')]
    base3 = walks(A, inputs, readout, 3)[-1][1]
    perc = []
    for q in SHARES:
        m = int(round(q * len(pool)))
        for draw, removed in [(d, rng.choice(pool, m, replace=False)) for d in range(DRAWS)] + [('hubs', hubs[:m])]:
            alive = np.ones(n, bool)
            alive[removed] = False
            perc.append(dict(order='random' if draw != 'hubs' else 'degree (hubs first)', draw=draw if draw != 'hubs' else None, share_removed=q,
                             cells_removed=m, readout_reachable=reach_all(A, inputs, readout, alive),
                             walks3_relative=walks(A, inputs, readout, 3, alive.astype(np.float64))[-1][1] / base3))
        print(f'percolation q={q} done ({time.time() - t0:.0f} s)', flush=True)
    write(STUDY / 'results' / 'phase4_graph_percolation.csv', perc)
    print(f'done in {time.time() - t0:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
