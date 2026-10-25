"""Unit tests for the P6b circuit helpers (panels/p6b_circuits.py) on small synthetic connectomes.

Connectome convention (as in the stored W_post_pre): rows are postsynaptic, columns presynaptic, values are signed
synapse counts (positive excitatory); explicitly stored zeros are not connections.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panels import p6b_circuits as p6b  # noqa: E402


def connectome(n, edges):
    """edges: (pre, post, weight); a weight of 0.0 is stored explicitly (not a connection)."""
    pre, post, w = (np.array(x) for x in zip(*edges)) if edges else (np.array([], int),) * 2 + (np.array([], float),)
    return sp.csr_matrix((w.astype(float), (post.astype(int), pre.astype(int))), shape=(n, n))


def annotations(n, inputs=(), readout=(), superclass=None):
    a = pd.DataFrame({'bodyId': np.arange(1000, 1000 + n), 'superclass': superclass if superclass is not None else ['x'] * n,
                      'direct_input': False, 'observed_by_policy': False})
    a.loc[list(inputs), 'direct_input'] = True
    a.loc[list(readout), 'observed_by_policy'] = True
    return a


def frames_for(dv):
    """Every P6 state set reads the same cue difference (enough for the helpers, which loop over all sets)."""
    out = {}
    for model, ctx, vis, _ in p6b.P6_SETS:
        out.setdefault(model, pd.DataFrame(index=range(len(dv))))[f'c{ctx}-visible{vis}-delta-v'] = dv
    return out


class Capture:
    def __init__(self):
        self.tables, self.tests = {}, []

    def write_table(self, df, rel, **kw):
        self.tables[rel] = df.copy()

    def record(self, result, **meta):
        self.tests.append({**meta, **result})

    def patches(self, n_null=None):
        ps = [mock.patch.object(p6b.csvout, 'write_table', self.write_table), mock.patch.object(p6b.stats, 'record', self.record)]
        if n_null is not None:
            ps.append(mock.patch.object(p6b, 'N_NULL', n_null))
        return ps


class BreadthFirst(unittest.TestCase):
    def setUp(self):
        # 0 -> 1 -> 2 -> 3, 4 isolated, 5 -> 0 (pre -> post)
        self.A = (connectome(6, [(0, 1, 1), (1, 2, -2), (2, 3, 5), (5, 0, 1)]) != 0).astype(np.float32).tocsr()

    def test_downstream_hops(self):
        np.testing.assert_array_equal(p6b._bfs(self.A, np.array([0])), [0, 1, 2, 3, -1, -1])

    def test_upstream_hops_use_the_transpose(self):
        np.testing.assert_array_equal(p6b._bfs(self.A.T.tocsr(), np.array([3])), [3, 2, 1, 0, -1, 4])

    def test_nearest_of_several_sources(self):
        np.testing.assert_array_equal(p6b._bfs(self.A, np.array([5, 2])), [1, 2, 0, 1, -1, 0])

    def test_cycle_terminates(self):
        A = (connectome(3, [(0, 1, 1), (1, 2, 1), (2, 0, 1)]) != 0).astype(np.float32).tocsr()
        np.testing.assert_array_equal(p6b._bfs(A, np.array([0])), [0, 1, 2])

    def test_stored_zero_is_not_a_connection(self):
        W = connectome(3, [(0, 1, 0.0), (1, 2, 1)])
        self.assertEqual(W.nnz, 2)
        A = (W != 0).astype(np.float32).tocsr()
        self.assertEqual(A.nnz, 1)
        np.testing.assert_array_equal(p6b._bfs(A, np.array([0])), [0, -1, -1])

    def test_hop_classes(self):
        np.testing.assert_array_equal(p6b._hop_class(np.array([-1, 0, 1, 2, 3, 9])), ['none', '0', '1', '2', '>=3', '>=3'])
        self.assertEqual(set(p6b._hop_class(np.array([-1, 0, 1, 2, 3]))), set(p6b.HOP_CLASSES))


class CellDistance(unittest.TestCase):
    def test_direction_and_table(self):
        W = connectome(5, [(0, 1, 1), (1, 2, -1), (2, 3, 1)])      # input 0, readout 3, cell 4 unconnected
        cap = Capture()
        with cap.patches()[0]:
            d_in, d_out = p6b.cell_distance(W, annotations(5, inputs=[0], readout=[3]))
        np.testing.assert_array_equal(d_in, [0, 1, 2, 3, -1])
        np.testing.assert_array_equal(d_out, [3, 2, 1, 0, -1])
        t = cap.tables[p6b.FD + 'fd_cell_distance.csv']
        self.assertEqual(list(t.columns), ['cell', 'body_id', 'hops_from_input', 'hops_to_readout'])
        np.testing.assert_array_equal(t['body_id'], np.arange(1000, 1005))


class CueByDistance(unittest.TestCase):
    def test_classes_counts_and_quantiles(self):
        dv = np.array([0.05, -2.0, 0.5, 3.0, -0.2, 1.5])
        d_in = np.array([0, 1, 1, 2, 5, -1])
        d_out = np.array([-1, -1, 0, 0, 0, 3])
        cap = Capture()
        with cap.patches()[0]:
            p6b.cue_by_distance(frames_for(dv), d_in, d_out)
        t = cap.tables[p6b.FD + 'fd_cue_by_distance.csv']
        self.assertEqual(len(t), len(p6b.P6_SETS) * 2 * len(p6b.HOP_CLASSES))
        self.assertTrue((t.groupby(['state_key', 'direction'])['cells'].sum() == len(dv)).all())
        r = t[(t.state_key == 'm0') & (t.direction == 'from_input')].set_index('hop_class')
        self.assertEqual(int(r.loc['1', 'cells']), 2)
        self.assertAlmostEqual(r.loc['1', 'median_abs_dv_mv'], 1.25)
        self.assertAlmostEqual(r.loc['1', 'fraction_over_1_mv'], 0.5)
        self.assertAlmostEqual(r.loc['>=3', 'median_abs_dv_mv'], 0.2)
        self.assertAlmostEqual(r.loc['none', 'median_abs_dv_mv'], 1.5)
        u = t[(t.state_key == 'm0') & (t.direction == 'to_readout')].set_index('hop_class')
        self.assertEqual(int(u.loc['1', 'cells']), 0)
        self.assertTrue(pd.isna(u.loc['1', 'median_abs_dv_mv']))
        self.assertAlmostEqual(u.loc['0', 'fraction_over_0p1_mv'], 1.0)


class Strata(unittest.TestCase):
    def test_codes_group_superclass_role_degree_and_location(self):
        n = 8
        W = connectome(n, [(i, (i + 1) % n, 1) for i in range(n)])      # every cell: in-degree 1, out-degree 1
        sc = ['a', 'a', 'a', 'b', 'a', 'a', None, None]
        ann = annotations(n, inputs=[5], superclass=sc)
        loc = np.array([[10, 10], [140, 20], [160, 20], [10, 10], [10, 10], [10, 10], [np.nan, np.nan], [np.nan, np.nan]], float)
        codes = p6b.strata(W, ann, loc)
        self.assertEqual(codes[0], codes[1])                 # same 150-um bin, superclass, role, degree
        self.assertEqual(codes[0], codes[4])
        self.assertNotEqual(codes[0], codes[2])              # next location bin
        self.assertNotEqual(codes[0], codes[3])              # other superclass
        self.assertNotEqual(codes[0], codes[5])              # input role
        self.assertEqual(codes[6], codes[7])                 # no soma, unannotated: one stratum
        self.assertNotEqual(codes[6], codes[0])

    def test_degree_quintiles_split_hubs(self):
        n = 10
        edges = [(9, j, 1) for j in range(9)] + [(8, j, 1) for j in range(4)]      # degrees: 2 x4, 1 x4, 5, 9
        codes = p6b.strata(connectome(n, edges), annotations(n), np.full((n, 2), 10.0))
        self.assertEqual(codes[9], codes[8])                 # both in the top degree quintile
        self.assertNotEqual(codes[9], codes[3])
        self.assertNotEqual(codes[3], codes[4])


class Ensemble(unittest.TestCase):
    n = 1200                                                  # top 1%: 12 cells

    def dv(self, top_signs):
        rng = np.random.default_rng(0)
        dv = rng.uniform(-0.5, 0.5, self.n)
        dv[:len(top_signs)] = 10.0 * np.asarray(top_signs, float)
        return dv

    def run_ensemble(self, W, dv, codes, n_null=25):
        cap = Capture()
        ps = cap.patches(n_null)
        with ps[0], ps[1], ps[2]:
            p6b.ensemble(W, frames_for(dv), codes)
        return cap

    def test_one_cell_strata_make_the_null_equal_the_observed_set(self):
        k = int(np.ceil(p6b.TOP_FRACTION * self.n))
        edges = [(i, j, 1) for i in range(k) for j in range(k) if i != j and (i + j) % 2 == 0]
        cap = self.run_ensemble(connectome(self.n, edges), self.dv([1] * k), np.arange(self.n))
        e = cap.tables[p6b.FD + 'fd_ensemble.csv']
        r = e[(e.state_key == 'm0') & (e.top_fraction == p6b.TOP_FRACTION)].iloc[0]
        self.assertEqual(int(r['observed_connections']), len(edges))
        self.assertAlmostEqual(r['null_mean'], len(edges))
        self.assertAlmostEqual(r['ratio'], 1.0)
        self.assertAlmostEqual(r['p_value'], 1.0)

    def test_planted_wiring_is_detected_and_counted_exactly(self):
        k = int(np.ceil(p6b.TOP_FRACTION * self.n))
        rng = np.random.default_rng(1)
        back = [(int(a), int(b), 1) for a, b in rng.integers(0, self.n, size=(900, 2)) if a != b and not (a < k and b < k)]
        block = [(i, j, 1) for i in range(k) for j in range(k) if i != j]
        W = connectome(self.n, back + block)
        cap = self.run_ensemble(W, self.dv([1] * k), np.zeros(self.n, int), n_null=40)
        e = cap.tables[p6b.FD + 'fd_ensemble.csv']
        r = e[(e.state_key == 'm0') & (e.top_fraction == p6b.TOP_FRACTION)].iloc[0]
        dense = (W != 0).toarray()
        self.assertEqual(int(r['observed_connections']), int(dense[:k, :k].sum()))
        self.assertGreater(r['ratio'], 5.0)
        self.assertAlmostEqual(r['p_value'], 1 / 41)
        self.assertEqual(sorted(e['top_fraction'].unique()), [0.005, 0.01, 0.02, 0.05])
        t = {x['test_id']: x for x in cap.tests}
        self.assertEqual(t['fig_circuit.ensemble.m0']['family'], 'F6g.ensemble')
        self.assertEqual(len(t), 2 * len(p6b.P6_SETS))

    def test_sign_motif_counts_and_edges(self):
        k = int(np.ceil(p6b.TOP_FRACTION * self.n))
        signs = [1] * (k // 2) + [-1] * (k - k // 2)
        same = [(i, j, 2) for i in range(k) for j in range(k) if i != j and signs[i] == signs[j]]
        opposite = [(i, j, -3) for i in range(k) for j in range(k) if signs[i] != signs[j] and (i + j) % 2]
        stored_zero = [(0, k // 2, 0.0)]                      # an opposite pair with even index sum: not an edge above
        self.assertEqual((0 + k // 2) % 2, 0)
        W = connectome(self.n, same + opposite + stored_zero)
        self.assertEqual(W.nnz, len(same) + len(opposite) + 1)
        cap = self.run_ensemble(W, self.dv(signs), np.arange(self.n), n_null=60)
        s = cap.tables[p6b.FD + 'fd_sign_motif.csv'].set_index('state_key').loc['m0']
        self.assertEqual(int(s['connections']), len(same) + len(opposite))
        self.assertEqual(int(s['same_excitatory']), len(same))
        self.assertEqual(int(s['opposite_inhibitory']), len(opposite))
        self.assertEqual(int(s['same_inhibitory']) + int(s['opposite_excitatory']), 0)
        self.assertAlmostEqual(s['difference'], 1.0)
        self.assertLess(s['p_value'], 0.05)
        edges = cap.tables[p6b.FD + 'fd_ensemble_edges.csv']
        self.assertEqual(len(edges), len(same) + len(opposite))
        exc = edges[edges['weight'] > 0]
        self.assertTrue(exc['same_preference'].all() and not edges[edges['weight'] < 0]['same_preference'].any())
        self.assertTrue(set(edges['post_cell']) | set(edges['pre_cell']) <= set(range(k)))


if __name__ == '__main__':
    unittest.main()
