"""P6b: circuit-level analyses of the cue state on the connectome (figure phase, post hoc; families F6g.* declared in
msa/testplan.py before computation). For each distinct state set (P6):

* cue difference |dv| by synaptic distance from the input cells (downstream) and to the readout cells (upstream);
* wiring among the top-1% cue-responsive cells: within-set connections against random sets matched by superclass,
  role, degree quintile and 150-um location on the unfolded display (1,000 draws);
* the sign of those connections: excitatory share between cells preferring the same cue minus between cells
  preferring opposite cues, against permutation of the preference signs (1,000 draws).

The wiring is the stored connectome (signed synapse counts, rows postsynaptic). Outputs go to
manuscript/results/figure_data/ (fd_cell_distance, fd_cue_by_distance, fd_ensemble, fd_sign_motif, fd_ensemble_edges).
"""
import numpy as np
import pandas as pd

from msa import cnsview, csvout, io, paths, stats
from panels.p6_mechanisms import STATE_SETS as P6_SETS

FD = 'figure_data/'
TOP_FRACTION = 0.01
N_NULL = 1000
N_BOOT_CI = 2000
KEYS = {('remote-dynamic-M0-head_only', 0, 0): 'm0', ('remote-dynamic-M0-joint', 0, 0): 'm0_joint',
        ('supp-M0-visibility', 0, 1): 'm0vis_visible', ('supp-S1-ring-brain', 0, 0): 's1',
        ('supp-S2-context-brain', 0, 0): 's2_o0', ('supp-S2-context-brain', 1, 0): 's2_o180'}
HOP_CLASSES = ['0', '1', '2', '>=3', 'none']


def _bfs(M, sources):
    n = M.shape[0]
    dist = np.full(n, -1, np.int32)
    dist[sources] = 0
    frontier = np.zeros(n, np.float32)
    frontier[sources] = 1
    k = 0
    while frontier.any():
        k += 1
        reached = (M @ frontier) > 0
        new = reached & (dist < 0)
        dist[new] = k
        frontier = new.astype(np.float32)
    return dist


def _hop_class(d):
    return np.where(d < 0, 'none', np.where(d >= 3, '>=3', d.astype(str)))


def load():
    W = io.sparse_load(paths.CONNECTOME / 'W_post_pre.npz').tocsr()
    ann = io.parquet_load(paths.M1 / 'neurons' / 'annotations.parquet',
                          columns=['bodyId', 'superclass', 'direct_input', 'observed_by_policy'])
    idx = io.npz_load(paths.M1 / 'neurons' / 'index.npz', keys=['body_ids'])
    if not np.array_equal(ann['bodyId'].to_numpy(np.int64), idx['body_ids']) or W.shape[0] != len(ann):
        raise RuntimeError('connectome, annotations and index disagree')
    anat = paths.ROOT / 'sokoban' / 'artifacts' / 'anatomy'
    xyz = np.frombuffer(io.read_bytes(anat / 'somata.bin'), '<f4').reshape(-1, 3).astype(np.float64)
    ids = np.frombuffer(io.read_bytes(anat / 'soma-body-ids.bin'), '<u4').astype(np.int64)
    hit = pd.Series(np.arange(len(ids)), index=ids).reindex(ann['bodyId'].to_numpy()).to_numpy()
    loc = np.full((len(ann), 2), np.nan)
    has = ~np.isnan(hit)
    loc[has] = cnsview.unfold(xyz[hit[has].astype(np.int64)])[0][:, :2]
    return W, ann, loc


def cell_distance(W, ann):
    A = (W != 0).astype(np.float32).tocsr()
    inp = np.flatnonzero(ann['direct_input'].to_numpy(bool))
    obs = np.flatnonzero(ann['observed_by_policy'].to_numpy(bool))
    d_in = _bfs(A, inp)                         # rows are postsynaptic: A @ x moves activity downstream
    d_out = _bfs(A.T.tocsr(), obs)              # upstream of the readout cells
    df = pd.DataFrame({'cell': np.arange(len(ann)), 'body_id': ann['bodyId'].to_numpy(np.int64), 'hops_from_input': d_in, 'hops_to_readout': d_out})
    csvout.write_table(df, FD + 'fd_cell_distance.csv', title=('Shortest synaptic path from any input cell and to any readout cell, per cell '
                                                              '(-1: no path; source: flybrain/cache/W_post_pre.npz)'), describe={
        'cell': 'Model cell index', 'body_id': 'MaleCNS body ID', 'hops_from_input': 'Synapses from the nearest input cell (0 = input cell; -1 = unreachable)',
        'hops_to_readout': 'Synapses to the nearest readout cell (0 = readout cell; -1 = no path)'}, key=['cell'])
    return d_in, d_out


def cue_by_distance(frames, d_in, d_out):
    rows = []
    for model, ctx, vis, label in P6_SETS:
        key = KEYS[(model, ctx, vis)]
        dv = np.abs(frames[model][f'c{ctx}-visible{vis}-delta-v'].to_numpy(np.float64))
        for direction, d in (('from_input', d_in), ('to_readout', d_out)):
            cls = _hop_class(d)
            for c in HOP_CLASSES:
                m = cls == c
                x = dv[m]
                rows.append(dict(state_set=label, state_key=key, direction=direction, hop_class=c, cells=int(m.sum()),
                                 median_abs_dv_mv=float(np.median(x)) if m.any() else None,
                                 q25_abs_dv_mv=float(np.quantile(x, 0.25)) if m.any() else None,
                                 q75_abs_dv_mv=float(np.quantile(x, 0.75)) if m.any() else None,
                                 fraction_over_0p1_mv=float((x > 0.1).mean()) if m.any() else None,
                                 fraction_over_1_mv=float((x > 1.0).mean()) if m.any() else None))
    df = pd.DataFrame(rows)
    df['analysis_status'] = 'post_hoc'
    df['source_file'] = 'mechanisms-v1/interpretation/*-memory-cells-d160-p4.parquet;flybrain/cache/W_post_pre.npz'
    csvout.write_table(df, FD + 'fd_cue_by_distance.csv', title='Cue-related voltage difference by synaptic distance from the inputs and to the readout (descriptive)',
                       describe={'state_set': 'Distinct state set', 'state_key': 'State-set key', 'direction': 'from_input (downstream) or to_readout (upstream)',
                                 'hop_class': 'Synapses: 0, 1, 2, >=3, or none (no path)', 'cells': 'Cells in the class',
                                 'median_abs_dv_mv': 'Median |cue 1 - cue 0| membrane potential (mV)', 'q25_abs_dv_mv': '25th percentile (mV)',
                                 'q75_abs_dv_mv': '75th percentile (mV)', 'fraction_over_0p1_mv': 'Share of cells over 0.1 mV',
                                 'fraction_over_1_mv': 'Share of cells over 1 mV'},
                       key=['state_key', 'direction', 'hop_class'], sort=['state_key', 'direction', 'hop_class'])


def strata(W, ann, loc):
    A = (W != 0)
    deg = np.asarray(A.sum(0)).ravel() + np.asarray(A.sum(1)).ravel()
    dbin = np.clip(np.searchsorted(np.quantile(deg, [0.2, 0.4, 0.6, 0.8]), deg, side='right'), 0, 4)
    role = np.where(ann['direct_input'].to_numpy(bool), 'input', np.where(ann['observed_by_policy'].to_numpy(bool), 'readout', 'other'))
    sc = ann['superclass'].fillna('unannotated').to_numpy()
    sx = np.floor(np.nan_to_num(loc[:, 0], nan=-1e9) / 150).astype(np.int64)
    sy = np.floor(np.nan_to_num(loc[:, 1], nan=-1e9) / 150).astype(np.int64)
    space = np.where(np.isnan(loc[:, 0]), 'nosoma', pd.Series(sx).astype(str) + ',' + pd.Series(sy).astype(str))
    codes, _ = pd.factorize(pd.Series(sc) + '|' + pd.Series(role) + '|' + pd.Series(dbin).astype(str) + '|' + pd.Series(space))
    return codes


def ensemble(W, frames, codes):
    A = (W != 0).tocsr().astype(np.int8)
    n = A.shape[0]
    k = int(np.ceil(TOP_FRACTION * n))
    pools = pd.Series(np.arange(n)).groupby(codes).apply(lambda s: s.to_numpy()).to_dict()
    ens_rows, sign_rows, edge_rows = [], [], []
    for model, ctx, vis, label in P6_SETS:
        key = KEYS[(model, ctx, vis)]
        dvs = frames[model][f'c{ctx}-visible{vis}-delta-v'].to_numpy(np.float64)
        order = np.argsort(-np.abs(dvs), kind='mergesort')
        top = np.sort(order[:k])
        mask = np.zeros(n, bool)
        mask[top] = True
        observed = int(A[mask][:, mask].nnz)
        tid = f'fig_circuit.ensemble.{key}'
        rng = stats.rng_for(tid)
        cnt = np.bincount(codes[mask], minlength=codes.max() + 1)
        null = np.empty(N_NULL)
        for i in range(N_NULL):
            pick = np.concatenate([rng.choice(pools[c], cnt[c], replace=False) for c in np.flatnonzero(cnt)])
            mm = np.zeros(n, bool)
            mm[pick] = True
            null[i] = A[mm][:, mm].nnz
        p = stats.perm_p(observed, null, two_sided=False)
        boot = observed / null[stats.rng_for(f'{tid}.ci').integers(0, N_NULL, size=(N_BOOT_CI, N_NULL))].mean(1)
        res = dict(test='stratified_random_sets', statistic_name='within_set_connections', statistic=float(observed), p_value=p,
                   p_method='permutation', n=int(k), n_used=int(k), n_resamples=N_NULL, mc_se=stats.mc_se(p, N_NULL),
                   effect_name='observed_over_null_mean', effect=float(observed / null.mean()), effect_ci_low=float(np.quantile(boot, 0.025)),
                   effect_ci_high=float(np.quantile(boot, 0.975)),
                   effect_ci_method=f'bootstrap over the null draws (Monte Carlo error of the null mean; {N_BOOT_CI:,} resamples)',
                   null_mean=float(null.mean()), null_q025=float(np.quantile(null, 0.025)), null_q975=float(np.quantile(null, 0.975)),
                   informative=True)
        stats.record(res, test_id=tid, panel='fig_circuit', family='F6g.ensemble', contrast=f'{label}: connections among the top-1% cue-responsive cells vs matched random sets',
                     unit='cell (within one state set)', level='B', analysis_status='post_hoc', source_table='figure_data/fd_ensemble.csv')
        ens_rows.append(dict(state_set=label, state_key=key, top_fraction=TOP_FRACTION, cells=int(k), observed_connections=observed,
                             null_mean=float(null.mean()), null_q025=float(np.quantile(null, 0.025)), null_q975=float(np.quantile(null, 0.975)),
                             ratio=float(observed / null.mean()), p_value=p, test_id=tid))
        # sensitivity: other top fractions (descriptive, 200 draws)
        for frac in (0.005, 0.02, 0.05):
            kk = int(np.ceil(frac * n))
            tt = np.sort(order[:kk])
            mk = np.zeros(n, bool)
            mk[tt] = True
            ob = int(A[mk][:, mk].nnz)
            cc = np.bincount(codes[mk], minlength=codes.max() + 1)
            r2 = np.random.default_rng(int(frac * 1e4) + len(key))
            nl = np.empty(200)
            for i in range(200):
                pick = np.concatenate([r2.choice(pools[c], cc[c], replace=False) for c in np.flatnonzero(cc)])
                mm = np.zeros(n, bool)
                mm[pick] = True
                nl[i] = A[mm][:, mm].nnz
            ens_rows.append(dict(state_set=label, state_key=key, top_fraction=frac, cells=int(kk), observed_connections=ob,
                                 null_mean=float(nl.mean()), null_q025=float(np.quantile(nl, 0.025)), null_q975=float(np.quantile(nl, 0.975)),
                                 ratio=float(ob / nl.mean()), p_value=None, test_id=None))
        # sign motif inside the 1% ensemble
        sub = W[top][:, top].tocsr()
        sub.eliminate_zeros()                      # stored zeros are not connections
        sub = sub.tocoo()
        pref = np.sign(dvs[top])
        same = pref[sub.row] == pref[sub.col]
        exc = sub.data > 0
        diff = float(exc[same].mean() - exc[~same].mean())
        tid2 = f'fig_circuit.sign.{key}'
        r3 = stats.rng_for(tid2)
        nulls = np.empty(N_NULL)
        for i in range(N_NULL):
            pp = r3.permutation(pref)
            sm = pp[sub.row] == pp[sub.col]
            nulls[i] = exc[sm].mean() - exc[~sm].mean()
        p2 = stats.perm_p(diff, nulls, two_sided=True, center=0.0)
        n_same, n_opp = int(same.sum()), int((~same).sum())
        if n_same and n_opp:                       # binomial resampling = bootstrap of binary connection signs
            rb = stats.rng_for(f'{tid2}.ci')
            boot2 = rb.binomial(n_same, exc[same].mean(), N_BOOT_CI) / n_same - rb.binomial(n_opp, exc[~same].mean(), N_BOOT_CI) / n_opp
            ci2 = (float(np.quantile(boot2, 0.025)), float(np.quantile(boot2, 0.975)))
        else:
            ci2 = (float('nan'), float('nan'))
        res2 = dict(test='preference_sign_permutation', statistic_name='excitatory_share_same_minus_opposite', statistic=diff, p_value=p2,
                    p_method='permutation', n=int(len(exc)), n_used=int(len(exc)), n_resamples=N_NULL, mc_se=stats.mc_se(p2, N_NULL),
                    effect_name='excitatory_share_difference', effect=diff, effect_ci_low=ci2[0], effect_ci_high=ci2[1],
                    effect_ci_method=f'bootstrap over connections (binomial resampling within same- and opposite-preference connections; '
                                     f'{N_BOOT_CI:,} resamples)',
                    null_mean=float(nulls.mean()), null_q025=float(np.quantile(nulls, 0.025)), null_q975=float(np.quantile(nulls, 0.975)),
                    informative=True)
        stats.record(res2, test_id=tid2, panel='fig_circuit', family='F6g.sign', contrast=f'{label}: excitatory share, same-preference minus opposite-preference connections',
                     unit='connection (within one state set)', level='B', analysis_status='post_hoc', source_table='figure_data/fd_sign_motif.csv')
        sign_rows.append(dict(state_set=label, state_key=key, connections=int(len(exc)), same_excitatory=int((same & exc).sum()),
                              same_inhibitory=int((same & ~exc).sum()), opposite_excitatory=int((~same & exc).sum()),
                              opposite_inhibitory=int((~same & ~exc).sum()), excitatory_share_same=float(exc[same].mean()),
                              excitatory_share_opposite=float(exc[~same].mean()), difference=diff, null_q025=float(np.quantile(nulls, 0.025)),
                              null_q975=float(np.quantile(nulls, 0.975)), p_value=p2, test_id=tid2))
        if key == 'm0':
            for r, c_, w in zip(sub.row, sub.col, sub.data):
                edge_rows.append(dict(post_cell=int(top[r]), pre_cell=int(top[c_]), weight=float(w), same_preference=bool(pref[r] == pref[c_])))
    e = pd.DataFrame(ens_rows)
    e['analysis_status'] = 'post_hoc'
    csvout.write_table(e, FD + 'fd_ensemble.csv', title=('Wiring among the most cue-responsive cells vs random sets matched by superclass, role, degree '
                                                         'and location (1% tested; 0.5, 2 and 5% descriptive)'), describe={
        'state_set': 'Distinct state set', 'state_key': 'State-set key', 'top_fraction': 'Share of all cells taken by |dv| rank',
        'cells': 'Cells in the set', 'observed_connections': 'Connections (nonzero synapse counts) within the set',
        'null_mean': 'Mean within-set connections over matched random sets', 'null_q025': '2.5th percentile of the null', 'null_q975': '97.5th percentile of the null',
        'ratio': 'Observed over null mean', 'p_value': 'One-sided permutation p (1% sets only; Holm in ST2)'}, key=['state_key', 'top_fraction'])
    s_ = pd.DataFrame(sign_rows)
    s_['analysis_status'] = 'post_hoc'
    csvout.write_table(s_, FD + 'fd_sign_motif.csv', title='Sign of the connections within the 1% cue ensemble by cue preference', describe={
        'state_set': 'Distinct state set', 'state_key': 'State-set key', 'connections': 'Connections within the ensemble',
        'same_excitatory': 'Excitatory connections between cells preferring the same cue', 'same_inhibitory': 'Inhibitory, same preference',
        'opposite_excitatory': 'Excitatory, opposite preference', 'opposite_inhibitory': 'Inhibitory, opposite preference',
        'excitatory_share_same': 'Excitatory share among same-preference connections', 'excitatory_share_opposite': 'Excitatory share among opposite-preference connections',
        'difference': 'Same minus opposite', 'null_q025': '2.5th percentile under preference permutation', 'null_q975': '97.5th percentile',
        'p_value': 'Two-sided permutation p (Holm in ST2)'}, key=['state_key'])
    ed = pd.DataFrame(edge_rows)
    csvout.write_table(ed, FD + 'fd_ensemble_edges.csv', title='Connections within the 1% cue ensemble of the M0 readout-only state (for drawing)', describe={
        'post_cell': 'Postsynaptic model cell', 'pre_cell': 'Presynaptic model cell', 'weight': 'Signed synapse count (positive excitatory)',
        'same_preference': 'Both cells more depolarised after the same cue'}, key=['post_cell', 'pre_cell'])


def main():
    W, ann, loc = load()
    d_in, d_out = cell_distance(W, ann)
    frames = {m: io.parquet_load(paths.M1 / 'interpretation' / f'{m}-memory-cells-d160-p4.parquet') for m in sorted({s[0] for s in P6_SETS})}
    cue_by_distance(frames, d_in, d_out)
    ensemble(W, frames, strata(W, ann, loc))
