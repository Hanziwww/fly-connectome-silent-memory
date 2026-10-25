"""P11: follow-up analyses on saved data (plan v2, Phase 1B; after freeze-v1.1). CPU only, no new simulation.

Families F11.* are declared in msa/testplan.py before these tests run. They are post hoc: the questions were raised by
the evidence audit of the frozen paper.

1. Timing lock (Fig. 7). Why does memory work only at the trained timings? At the first decision, the readout margin
   between the two cue-specific pushes splits exactly into a cue-differential part D (half the margin difference between
   cues) and a cue-independent part C (the mean margin). Both cues are pushed correctly iff D > |C|.
   - If D stays positive at untrained timings while |C| grows, the cue signal is still in the readout direction and a
     shift of the readout's operating point (a timing-dependent bias) causes the failures.
   - If D shrinks or flips, the code itself has moved.
   A leave-one-timing-out shrinkage-LDA decoder then asks whether any fixed linear readout separates the cues at every
   timing.
   Data: mechanisms-v1/runs/record-<model>/free-after-d<delay>-p<prefix>-behavior.json (first-decision features and
   logits for each case) and parameters-after.npz (the readout head). Recomputed logits are checked against the
   recorded ones.
2. Swap specificity (Fig. 4c/d). Opposite-cue vs same-cue whole-state swaps on the same histories. The write, hold and
   read windows feed the swapped state into the same post-cue input stream, so they repeat one experiment and are
   collapsed to one record per history. The two histories of a cue pair are swapped into each other, so a test
   clustered by timing is reported beside the per-history test.
3. Claims of absence (Fig. 7, ED5), with a 10-percentage-point margin and Tango's score test for paired proportions.
   "No net memory use at untrained timings" is one-sided (net use below +10 pp, a non-superiority test); "cue decoding
   at chance" and "neutral input acts like blank waiting" are two-sided equivalence (TOST). A failed test is reported as
   inconclusive, with the margin the data support.
4. State dynamics (Figs 6-7). From the 1-ms records (delay 160 and 640 ms, prefix 4): the cue difference of each state
   variable through the delay, its effective decay time against the intrinsic one, and an exact split of the voltage
   difference of non-spiking cells into its initial value, the synaptic part and the adaptation part. At all 35 timings:
   the cue-difference energy at the first decision.
5. Ensemble wiring under a stricter null (Fig. 6g): random sets matched by cell type, synaptic distance from the inputs
   and degree quintile; and the ensemble and its wiring at each of the 35 timings.
6. Brain x readout pairing in the memory task (M0 joint): the readouts from before, mid and after training applied to
   the brain states from before, mid and after training on identical inputs.
8. Sign motif vs signed propagation (Fig. 6e). A cell's cue preference is inherited from its inputs with the sign of the
   synapse, so same-preference connections are excitatory more often under any propagation, and the preference
   permutation of Fig. 6e is the wrong null for a specific motif. Null: the same statistic in the top-1% sets of the
   linear response of the signed connectome, each row divided by its total absolute input (each cell passes on a gain
   times the signed, input-weighted mean of its inputs), to the state's own adaptation-difference pattern with random
   signs. Descriptive: the statistic by |dv| rank band, and for the unflipped pattern.
9. Drive split at every timing (Fig. 7; extends 4). At the first decision window of all 35 timings, the cue difference
   of the drive (synaptic minus adaptation) splits exactly into its synaptic and adaptation parts; and the adaptation
   difference's effective decay time across delays, against the intrinsic 200 ms.
"""
import io as _io
import re
import zipfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import optimize
from scipy import stats as st

from msa import csvout, io, paths, stats

FU = 'followup/'
MODELS = [('remote-dynamic-M0-head_only', 'M0 readout-only'), ('remote-dynamic-M0-joint', 'M0 joint'),
          ('supp-M0-visibility', 'M0 shared readout'), ('supp-S1-ring-brain', 'S1 readout-only'),
          ('supp-S2-context-brain', 'S2 readout-only')]
TRAINED_DELAYS, TRAINED_PREFIXES = (0, 40, 160), (0, 1, 4)
N_PERM, N_DECODER_NULL = 10000, 1000
CASE = re.compile(r'^c(?P<context>\d+)-cue(?P<cue>[01])(?:-(?P<visibility>hidden|visible))?$')
FILE = re.compile(r'free-after-d(?P<d>\d+)-p(?P<p>\d+)-behavior\.json$')


def _head(model):
    z = io.npz_load(paths.M1 / 'runs' / f'record-{model}' / 'parameters-after.npz',
                    keys=['head_proposal.weight', 'head_proposal.bias', 'head_mean', 'head_scale'])
    return (z['head_proposal.weight'].astype(np.float64), z['head_proposal.bias'].astype(np.float64),
            z['head_mean'].astype(np.float64), z['head_scale'].astype(np.float64))


def _z(features, mean, scale):
    return np.clip((np.asarray(features, np.float64) - mean) / scale, -10.0, 10.0)


def _timings(model):
    folder = paths.M1 / 'runs' / f'record-{model}'
    out = []
    for f in sorted(folder.glob('free-after-d*-p*-behavior.json'), key=lambda p: (int(FILE.search(p.name)['d']), int(FILE.search(p.name)['p']))):
        m = FILE.search(f.name)
        out.append((int(m['d']), int(m['p']), io.json_load(f)))
    if len(out) != 35:
        raise RuntimeError(f'{model}: expected 35 timings, found {len(out)}')
    return out


def _lw_solve(X, v):
    """Solve (Ledoit-Wolf shrinkage covariance of the class-centred rows X) w = v, exactly, through the n x n Gram matrix:
    with S = X'X/n, mu = tr(S)/p and shrinkage a, Sigma = a mu I + (1 - a) S, and Woodbury gives
    w = (v - k X'(c I + k G)^-1 X v) / c with c = a mu, k = (1 - a)/n, G = X X'."""
    n, p = X.shape
    G = X @ X.T
    s2 = np.sum(G ** 2) / n ** 2                      # ||S||_F^2
    mu = np.trace(G) / (n * p)
    d2 = (s2 - p * mu ** 2) / p
    b2 = (np.sum(np.einsum('ij,ij->i', X, X) ** 2) - n * s2) / n ** 2 / p
    a = min(b2, d2) / d2 if d2 > 0 else 1.0
    c, k = a * mu, (1 - a) / n
    if c <= 0:
        raise RuntimeError('degenerate covariance')
    return (v - k * X.T @ np.linalg.solve(c * np.eye(n) + k * G, X @ v)) / c


def _loto_accuracy(Z, y, fold):
    """Leave-one-timing-out shrinkage-LDA cue decoding; returns the share of held-out states decoded correctly."""
    correct = 0
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        m0, m1 = Z[tr & (y == 0)].mean(0), Z[tr & (y == 1)].mean(0)
        Xc = np.vstack([Z[tr & (y == 0)] - m0, Z[tr & (y == 1)] - m1])
        w = _lw_solve(Xc, m1 - m0)
        score = (Z[te] - (m0 + m1) / 2) @ w
        correct += int(np.sum((score > 0).astype(int) == y[te]))
    return correct / len(y)


def _pairs(cases):
    """Group case ids into cue pairs keyed by (context, visibility)."""
    pairs = {}
    for c in cases:
        m = CASE.match(c['case'])
        if not m:
            raise RuntimeError(f'unexpected case id {c["case"]}')
        pairs.setdefault((int(m['context']), m['visibility'] or 'hidden'), {})[int(m['cue'])] = c['case']
    return pairs


def _solving_action(timings, case):
    acts = [c['decisions'][0]['action'] for _, _, cases in timings for c in cases if c['case'] == case and c['status'] == 'solved' and c['decisions']]
    vals, counts = np.unique(acts, return_counts=True)
    return int(vals[np.argmax(counts)]), int(counts.max()), len(acts)


def timing_lock():
    rows, summary = [], []
    for model, label in MODELS:
        W, b, mean, scale = _head(model)
        timings = _timings(model)
        for (ctx, vis), cue_case in sorted(_pairs(timings[0][2]).items()):
            a0, n0, t0 = _solving_action(timings, cue_case[0])
            a1, n1, t1 = _solving_action(timings, cue_case[1])
            if a0 == a1:
                raise RuntimeError(f'{model} c{ctx} {vis}: both cues solved by the same first push')
            w_d, b_d = W[a1] - W[a0], b[a1] - b[a0]
            unit = f'{model}|c{ctx}|{vis}'
            recs, Zs, ys, folds = [], [], [], []
            for k, (d, p, cases) in enumerate(timings):
                by_case = {c['case']: c for c in cases}
                z, status, act = {}, {}, {}
                for cue in (0, 1):
                    c = by_case[cue_case[cue]]
                    dec = c['decisions'][0]
                    z[cue] = _z(dec['features'], mean, scale)
                    logits = W @ z[cue] + b
                    if np.max(np.abs(logits - np.asarray(dec['logits'], np.float64))) > 1e-3:
                        raise RuntimeError(f'{unit} d{d} p{p} cue{cue}: recomputed logits differ from the record')
                    status[cue], act[cue] = c['status'], int(dec['action'])
                    Zs.append(z[cue])
                    ys.append(cue)
                    folds.append(k)
                m0, m1 = float(w_d @ z[0] + b_d), float(w_d @ z[1] + b_d)
                dz = z[1] - z[0]
                recs.append(dict(model=model, model_label=label, context=ctx, visibility=vis, delay_ms=d, prefix_windows=p,
                                 trained=bool(d in TRAINED_DELAYS and p in TRAINED_PREFIXES), cue0_action=a0, cue1_action=a1,
                                 margin_cue0=m0, margin_cue1=m1, differential=(m1 - m0) / 2, common=(m1 + m0) / 2,
                                 cue_difference_norm=float(np.linalg.norm(dz)),
                                 cosine_with_readout=float(w_d @ dz / (np.linalg.norm(w_d) * np.linalg.norm(dz))) if np.linalg.norm(dz) > 0 else np.nan,
                                 both_solved=bool(status[0] == 'solved' and status[1] == 'solved'), same_first_push=bool(act[0] == act[1]),
                                 _dz=dz))
            ref = np.mean([r['_dz'] for r in recs if r['trained']], axis=0)
            for r in recs:
                dz = r.pop('_dz')
                r['cosine_with_trained_difference'] = float(dz @ ref / (np.linalg.norm(dz) * np.linalg.norm(ref)))
            rows.extend(recs)
            df = pd.DataFrame(recs)
            tr = df['trained'].to_numpy()
            # permutation tests over the timing labels (9 trained of 35)
            for fam, col, transform, name in (('F11.timing.differential', 'differential', lambda v: v, 'mean differential margin'),
                                              ('F11.timing.common', 'common', np.abs, 'mean |common margin|')):
                v = transform(df[col].to_numpy(float))
                obs = float(v[~tr].mean() - v[tr].mean())
                tid = f'fu1.{col}.{unit}'
                rng = stats.rng_for(tid)
                null = np.array([v[perm[9:]].mean() - v[perm[:9]].mean() for perm in (rng.permutation(35) for _ in range(N_PERM))])
                pv = stats.perm_p(obs, null)
                stats.record(dict(test='timing_label_permutation', statistic_name=f'{name}: untrained minus trained', statistic=obs, p_value=pv,
                                  p_method='permutation', n=35, n_used=35, n_resamples=N_PERM, mc_se=stats.mc_se(pv, N_PERM),
                                  effect_name='untrained_minus_trained', effect=obs, null_mean=float(null.mean()),
                                  null_q025=float(np.quantile(null, 0.025)), null_q975=float(np.quantile(null, 0.975)), informative=True),
                             test_id=tid, panel='followup/fu1', family=fam, contrast=f'{label}, context {ctx}, {vis}: {name} at 26 untrained vs 9 trained timings',
                             unit='timing (within one model)', level='B', analysis_status='post_hoc', source_table='followup/fu1_timing_lock.csv')
            Z, y, fold = np.vstack(Zs), np.array(ys), np.array(folds)
            acc = _loto_accuracy(Z, y, fold)
            tid = f'fu1.decoder.{unit}'
            rng = stats.rng_for(tid)
            null = np.empty(N_DECODER_NULL)
            for i in range(N_DECODER_NULL):
                flip = rng.random(35) < 0.5            # swap the two cue labels within a random subset of timings
                null[i] = _loto_accuracy(Z, np.where(flip[fold], 1 - y, y), fold)
            pv = stats.perm_p(acc, null, two_sided=False)
            stats.record(dict(test='leave_one_timing_out_lda', statistic_name='held-out cue decoding accuracy', statistic=acc, p_value=pv,
                              p_method='permutation', n=70, n_used=70, n_resamples=N_DECODER_NULL, mc_se=stats.mc_se(pv, N_DECODER_NULL),
                              effect_name='accuracy', effect=acc, null_mean=float(null.mean()), null_q025=float(np.quantile(null, 0.025)),
                              null_q975=float(np.quantile(null, 0.975)), informative=True),
                         test_id=tid, panel='followup/fu1', family='F11.timing.decoder',
                         contrast=f'{label}, context {ctx}, {vis}: one fixed linear cue readout across all 35 timings (leave one timing out)',
                         unit='state (within one model)', level='B', analysis_status='post_hoc', source_table='followup/fu1_timing_summary.csv')
            un = df[~df['trained']]
            summary.append(dict(model=model, model_label=label, context=ctx, visibility=vis, cue0_action=a0, cue1_action=a1,
                                cue0_action_solved_share=n0 / t0, cue1_action_solved_share=n1 / t1,
                                trained_both_solved=int(df[df['trained']]['both_solved'].sum()), untrained_both_solved=int(un['both_solved'].sum()),
                                untrained_differential_positive=int((un['differential'] > 0).sum()),
                                untrained_common_exceeds_differential=int((un['common'].abs() >= un['differential']).sum()),
                                trained_mean_differential=float(df[df['trained']]['differential'].mean()), untrained_mean_differential=float(un['differential'].mean()),
                                trained_mean_abs_common=float(df[df['trained']]['common'].abs().mean()), untrained_mean_abs_common=float(un['common'].abs().mean()),
                                untrained_mean_cosine_with_trained=float(un['cosine_with_trained_difference'].mean()),
                                loto_decoder_accuracy=acc, loto_decoder_null_mean=float(null.mean()), loto_decoder_p=pv))
    common_desc = {'model': 'Model', 'model_label': 'Model (label)', 'context': 'S2 orientation context (0 elsewhere)',
                   'visibility': 'Goal visibility of the case (hidden unless the shared-readout visible condition)',
                   'cue0_action': 'First push that solves cue 0 (most common among solved histories)', 'cue1_action': 'First push that solves cue 1'}
    csvout.write_table(pd.DataFrame(rows), FU + 'fu1_timing_lock.csv',
                       title='Timing lock: readout margin at the first decision split into a cue-differential and a cue-independent part, per timing',
                       describe={**common_desc, 'delay_ms': 'Blank delay (ms)', 'prefix_windows': 'Hidden-goal prefix windows',
                                 'trained': 'Trained timing (delay 0/40/160 ms x prefix 0/1/4)',
                                 'margin_cue0': 'Logit(cue-1 push) minus logit(cue-0 push) after cue 0 (negative = correct)',
                                 'margin_cue1': 'The same margin after cue 1 (positive = correct)',
                                 'differential': 'Half the margin difference between cues (cue signal along the readout)',
                                 'common': 'Mean margin over cues (cue-independent drive along the readout)',
                                 'cue_difference_norm': 'Norm of the z-scored feature difference between cues',
                                 'cosine_with_readout': 'Cosine between the feature difference and the readout contrast',
                                 'cosine_with_trained_difference': 'Cosine between this feature difference and the mean over trained timings',
                                 'both_solved': 'Both cues solved', 'same_first_push': 'Both cues gave the same first push'},
                       key=['model', 'context', 'visibility', 'delay_ms', 'prefix_windows'])
    csvout.write_table(pd.DataFrame(summary), FU + 'fu1_timing_summary.csv', title='Timing lock: per cue pair (model, context, visibility)',
                       describe={**common_desc, 'cue0_action_solved_share': 'Share of solved cue-0 histories using that push',
                                 'cue1_action_solved_share': 'Share of solved cue-1 histories using that push',
                                 'trained_both_solved': 'Trained timings (of 9) with both cues solved',
                                 'untrained_both_solved': 'Untrained timings (of 26) with both cues solved',
                                 'untrained_differential_positive': 'Untrained timings whose differential margin has the correct sign',
                                 'untrained_common_exceeds_differential': 'Untrained timings with |common| >= differential (the readout cannot separate the cues)',
                                 'trained_mean_differential': 'Mean differential margin, trained timings', 'untrained_mean_differential': 'Mean differential margin, untrained timings',
                                 'trained_mean_abs_common': 'Mean |common| margin, trained timings', 'untrained_mean_abs_common': 'Mean |common| margin, untrained timings',
                                 'untrained_mean_cosine_with_trained': 'Mean cosine of the untrained cue difference with the trained one',
                                 'loto_decoder_accuracy': 'Leave-one-timing-out shrinkage-LDA cue accuracy (70 states)',
                                 'loto_decoder_null_mean': 'Mean accuracy under within-timing label swaps', 'loto_decoder_p': 'One-sided permutation p (unadjusted)'},
                       key=['model', 'context', 'visibility'])


# ---------------------------------------------------------------- 2. swap specificity

SWAP_UNITS = ('M0', 'M0vis.hidden', 'M0vis.visible', 'S1', 'S2')
HIDDEN_SWAP_UNITS = ('M0', 'M0vis.hidden', 'S1', 'S2')
SWAP_SRC = 'followup/fu2_swap_histories.csv'


def _bool(s):
    return s.astype(str).str.lower().isin(['true', '1', '1.0'])


def _swap_records():
    """One record per history: solved after the opposite-cue and after the same-cue whole-state swap (majority over the
    write, hold and read windows, which repeat the same swap) and the unperturbed baseline."""
    st5 = pd.read_csv(paths.out('supplementary/st5_intervention_records.csv'), low_memory=False)
    x = st5[(st5['source'] == 'mechanisms_v1') & st5['intervention'].isin(['swap_all_state', 'same_cue_swap'])].copy()
    x['solved'] = _bool(x['solved'])
    x['baseline_solved'] = _bool(x['baseline_solved'])
    hist = ['unit', 'case', 'context', 'cue', 'delay_ms', 'prefix_windows']
    if set(x['window']) != {'write', 'hold', 'read'} or (x.groupby(hist + ['intervention'])['window'].nunique() != 3).any():
        raise RuntimeError('swap records: expected the write, hold and read windows for every history and swap')
    if (x.groupby(hist)['baseline_solved'].nunique() != 1).any():
        raise RuntimeError('swap records: the baseline differs between windows')
    agg = x.groupby(hist + ['intervention']).agg(windows_solved=('solved', 'sum'), baseline_solved=('baseline_solved', 'first')).reset_index()
    w = agg.pivot_table(index=hist, columns='intervention', values='windows_solved', aggfunc='first')
    H = agg.groupby(hist)['baseline_solved'].first().to_frame()
    H['same_cue_windows_solved'] = w['same_cue_swap'].astype(int)
    H['opposite_cue_windows_solved'] = w['swap_all_state'].astype(int)
    H = H.reset_index()
    H['same_cue_solved'] = H['same_cue_windows_solved'] >= 2
    H['opposite_cue_solved'] = H['opposite_cue_windows_solved'] >= 2
    H['windows_disagree'] = H['same_cue_windows_solved'].isin([1, 2]) | H['opposite_cue_windows_solved'].isin([1, 2])
    return H.sort_values(['unit', 'context', 'delay_ms', 'prefix_windows', 'cue']).reset_index(drop=True)


def swap_specificity():
    H = _swap_records()
    rows, strata = [], []
    for u in SWAP_UNITS:
        g = H[H.unit == u]
        if g.empty:
            raise RuntimeError(f'swap records: no histories for {u}')
        r = stats.mcnemar(g['same_cue_solved'], g['opposite_cue_solved'])
        stats.record(r, test_id=f'fu2.swap_history.{u}', panel='followup/fu2', family='F11.swap.history',
                     contrast=f'{u}: histories solved after a same-cue vs an opposite-cue whole-state swap (windows collapsed)',
                     unit='history', level='B', analysis_status='post_hoc', source_table=SWAP_SRC)
        t = g.groupby(['context', 'delay_ms', 'prefix_windows'])[['same_cue_solved', 'opposite_cue_solved']].sum()
        diff = (t['same_cue_solved'] - t['opposite_cue_solved']).to_numpy(float)
        rt = stats.sign_test(diff)
        stats.record(rt, test_id=f'fu2.swap_timing.{u}', panel='followup/fu2', family='F11.swap.timing',
                     contrast=f'{u}: histories solved per timing after same-cue minus opposite-cue swaps (clustered by cue pair)',
                     unit='timing (cue pair)', level='B', analysis_status='post_hoc', source_table=SWAP_SRC)
        if u in HIDDEN_SWAP_UNITS:
            strata.append((r['a_only'], r['b_only']))
        rows.append(dict(unit=u, histories=len(g), timings=len(t), baseline_solved=int(g['baseline_solved'].sum()),
                         same_cue_solved=int(g['same_cue_solved'].sum()), opposite_cue_solved=int(g['opposite_cue_solved'].sum()),
                         solved_only_after_same_cue=int(r['a_only']), solved_only_after_opposite_cue=int(r['b_only']),
                         histories_windows_disagree=int(g['windows_disagree'].sum()),
                         timings_same_cue_better=int((diff > 0).sum()), timings_opposite_cue_better=int((diff < 0).sum()),
                         p_history=r['p_value'], p_timing=rt['p_value']))
    stats.record(stats.stratified_mcnemar(strata), test_id='fu2.swap_combined.hidden', panel='followup/fu2', family='F11.swap.combined',
                 contrast='hidden-goal units (M0, M0 shared readout, S1, S2): solved only after the same-cue vs only after the opposite-cue swap',
                 unit='history', level='B', analysis_status='post_hoc', source_table=SWAP_SRC)
    hv = H[H.unit.isin(['M0vis.hidden', 'M0vis.visible'])].copy()
    hv['disruption'] = hv['same_cue_solved'].astype(int) - hv['opposite_cue_solved'].astype(int)
    m = hv.pivot_table(index=['cue', 'delay_ms', 'prefix_windows'], columns='unit', values='disruption', aggfunc='first').dropna()
    rv = stats.sign_test((m['M0vis.hidden'] - m['M0vis.visible']).to_numpy(float))
    stats.record(rv, test_id='fu2.swap_visibility.shared_readout', panel='followup/fu2', family='F11.swap.visibility',
                 contrast='shared readout: disruption by the opposite- vs the same-cue swap, hidden minus visible goal (matched histories)',
                 unit='history pair (hidden, visible)', level='B', analysis_status='post_hoc', source_table=SWAP_SRC)
    rows.append(dict(unit='M0vis hidden minus visible', histories=len(m), timings=None, baseline_solved=None, same_cue_solved=None,
                     opposite_cue_solved=None, solved_only_after_same_cue=int(rv['n_positive']), solved_only_after_opposite_cue=int(rv['n_negative']),
                     histories_windows_disagree=None, timings_same_cue_better=None, timings_opposite_cue_better=None, p_history=rv['p_value'], p_timing=None))
    H['analysis_status'] = 'post_hoc'
    csvout.write_table(H, SWAP_SRC, title='Opposite-cue vs same-cue whole-state swaps, one record per history (windows collapsed)', describe={
        'unit': 'Model unit', 'case': 'Case', 'context': 'Context', 'cue': 'Cue', 'delay_ms': 'Delay (ms)', 'prefix_windows': 'Prefix windows',
        'baseline_solved': 'Solved without intervention', 'same_cue_windows_solved': 'Windows (of write, hold, read) solved after the same-cue swap',
        'opposite_cue_windows_solved': 'Windows solved after the opposite-cue swap', 'same_cue_solved': 'Solved after the same-cue swap (majority of windows)',
        'opposite_cue_solved': 'Solved after the opposite-cue swap (majority of windows)', 'windows_disagree': 'The three windows disagree for either swap'},
        key=['unit', 'case', 'delay_ms', 'prefix_windows'])
    csvout.write_table(pd.DataFrame(rows), 'followup/fu2_swap_summary.csv', title='Swap specificity per unit (windows collapsed)', describe={
        'unit': 'Model unit (last row: shared readout, hidden minus visible disruption)', 'histories': 'Histories (or matched hidden-visible pairs)',
        'timings': 'Timings (cue pairs)', 'baseline_solved': 'Solved without intervention', 'same_cue_solved': 'Solved after the same-cue swap',
        'opposite_cue_solved': 'Solved after the opposite-cue swap', 'solved_only_after_same_cue': 'Discordant: solved only after the same-cue swap '
        '(last row: pairs with more disruption when hidden)', 'solved_only_after_opposite_cue': 'Discordant: solved only after the opposite-cue swap '
        '(last row: pairs with more disruption when visible)', 'histories_windows_disagree': 'Histories whose three windows disagree',
        'timings_same_cue_better': 'Timings with more histories solved after the same-cue swap', 'timings_opposite_cue_better': 'Timings with more solved after the opposite-cue swap',
        'p_history': 'Exact McNemar (or sign test), unadjusted', 'p_timing': 'Exact sign test over timings, unadjusted'}, key=['unit'])


# ---------------------------------------------------------------- 3. equivalence

EQ_MARGIN = 0.10
MEMORY_LEARNERS = ('connectome_readout_only', 'connectome_joint', 'connectome_shared_visibility_readout')
Z_TOST = float(st.norm.ppf(0.95))


def _tango_z(b, c, n, d0):
    """Score statistic for H0: p_b - p_c = d0 from paired multinomial counts (b and c the two informative cells of n),
    with the restricted maximum-likelihood estimate of p_c (Tango 1998, Stat Med 17:891-908)."""
    A = 2.0 * n
    B = -b - c + (2.0 * n - b + c) * d0
    C = -c * d0 * (1.0 - d0)
    q = (np.sqrt(max(B * B - 4.0 * A * C, 0.0)) - B) / (2.0 * A)
    var = n * (2.0 * q + d0 * (1.0 - d0))
    num = b - c - n * d0
    if var <= 0:
        return float(np.copysign(np.inf, num)) if num else 0.0
    return float(num / np.sqrt(var))


def tost_paired(b, c, n, margin=EQ_MARGIN):
    """Equivalence of the paired difference (b - c) / n to zero within +-margin: two one-sided Tango score tests at 5%,
    the 90% score interval that matches them, and the smallest symmetric margin the data support."""
    d = (b - c) / n
    p_low = float(st.norm.sf(_tango_z(b, c, n, -margin)))     # H0: difference <= -margin
    p_up = float(st.norm.cdf(_tango_z(b, c, n, margin)))      # H0: difference >= +margin
    eps = 1e-9
    f_lo = lambda x: _tango_z(b, c, n, x) - Z_TOST           # noqa: E731 (the statistic decreases in x)
    f_hi = lambda x: _tango_z(b, c, n, x) + Z_TOST           # noqa: E731
    lo = optimize.brentq(f_lo, -1 + eps, d, xtol=1e-10) if d > -1 + eps and f_lo(-1 + eps) > 0 else -1.0
    hi = optimize.brentq(f_hi, d, 1 - eps, xtol=1e-10) if d < 1 - eps and f_hi(1 - eps) < 0 else 1.0
    return dict(estimate=d, ci90_low=lo, ci90_high=hi, p_lower=p_low, p_upper=p_up, p_tost=max(p_low, p_up),
                equivalent=bool(max(p_low, p_up) < 0.05), margin_supported=max(abs(lo), abs(hi)))


def _tost_record(rows, tid, family, contrast, unit, b, c, n, source, b_label, c_label, claim, one_sided=False):
    t = tost_paired(b, c, n)
    if one_sided:                                  # non-superiority: H0 difference >= +margin
        p, shown, supported = t['p_upper'], t['p_upper'] < 0.05, max(t['ci90_high'], 0.0)
        name, method = 'non_superiority_tango_score', 'asymptotic score (one-sided, H0: difference >= +0.10)'
        verdict = 'below +10 pp' if shown else f'inconclusive (data support < +{100 * supported:.0f} pp)'
    else:
        p, shown, supported = t['p_tost'], t['equivalent'], t['margin_supported']
        name, method = 'tost_tango_score', 'asymptotic score (two one-sided tests, margin +-0.10)'
        verdict = 'equivalent within +-10 pp' if shown else f'inconclusive (data support +-{100 * supported:.0f} pp)'
    res = dict(test=name, statistic_name=f'paired difference: share {b_label} minus share {c_label}', statistic=t['estimate'],
               p_value=p, p_method=method, n=int(n), n_used=int(n),
               effect_name='paired_difference', effect=t['estimate'], effect_ci_low=t['ci90_low'], effect_ci_high=t['ci90_high'],
               effect_ci_method='Tango score interval, 90% (each bound is a one-sided 95% bound)', a_only=int(b), b_only=int(c), informative=True)
    stats.record(res, test_id=tid, panel='followup/fu3', family=family, contrast=contrast, unit=unit, level='B', analysis_status='post_hoc',
                 source_table='followup/fu3_equivalence.csv')
    rows.append(dict(test_id=tid, claim=claim, contrast=contrast, n=int(n), count_a=int(b), label_a=b_label, count_b=int(c), label_b=c_label,
                     difference=t['estimate'], ci90_low=t['ci90_low'], ci90_high=t['ci90_high'], p_lower=t['p_lower'], p_upper=t['p_upper'],
                     p_tost=t['p_tost'], test=name, p_test=p, margin=EQ_MARGIN, shown=bool(shown), margin_supported=supported, verdict=verdict,
                     source_table=source))


def equivalence():
    rows = []
    # (a) no net memory use at untrained timings: cue pairs with both cues solved vs neither (ED5c)
    c5 = pd.read_csv(paths.out('extended/ed05/ed05c_cue_pair_classes.csv'), low_memory=False)
    c5['visible'] = _bool(c5['visible'])
    # the connectome learners that use the memory at the trained timings (hidden goal)
    sel = c5[c5['learner'].isin(MEMORY_LEARNERS) & (c5['split'] == 'untrained') & ~c5['visible']]
    if len(sel) != 5:
        raise RuntimeError(f'equivalence: expected 5 connectome memory models at untrained timings in ED5c, found {len(sel)}')
    for _, r in sel.sort_values('model').iterrows():
        _tost_record(rows, f'fu3.memory_use.{r["model"]}', 'F11.equivalence.memory_use',
                     f'{r["model"]} at untrained timings: share of cue pairs with both cues solved minus share with neither is below +10 pp',
                     'cue_pair', int(r['both_solved']), int(r['none_solved']), int(r['cue_pairs']), 'extended/ed05/ed05c_cue_pair_classes.csv',
                     'both solved', 'neither solved', 'no net memory use at untrained timings', one_sided=True)
    # (b) transfer at chance: M0-fitted cue decoder on S1/S2, and M0 behaviour on S1/S2 without updates (Fig. 5d)
    f5 = pd.read_csv(paths.out('main/fig5/fig5d_cross_layout_transfer.csv'), low_memory=False)
    dec = f5[f5['record_type'] == 'cross_family_cue_decoding'].copy()
    dec['decoded_correct'] = _bool(dec['decoded_correct'])
    for (fam, split), g in sorted(dec.groupby(['family', 'split'])):
        if fam == 'M0':
            continue
        k = g.groupby(['context', 'delay_ms', 'prefix_windows'])['decoded_correct'].sum()
        _tost_record(rows, f'fu3.transfer_decoding.{fam}.{split}', 'F11.equivalence.transfer',
                     f'M0-fitted cue decoder on {fam} ({split}): share of cue pairs decoded both-correct minus both-wrong, within +-10 pp',
                     'cue_pair', int((k == 2).sum()), int((k == 0).sum()), len(k), 'main/fig5/fig5d_cross_layout_transfer.csv',
                     'both correct', 'both wrong', 'cue code does not transfer across layouts')
    d5 = pd.read_csv(paths.out('extended/ed05/ed05d_dense_delay_scan.csv'), low_memory=False)
    d5['train_time'] = _bool(d5['train_time'])
    d5['decoded_correct'] = _bool(d5['decoded_correct'])
    k = d5[~d5['train_time']].groupby(['delay_ms', 'prefix_windows'])['decoded_correct'].sum()
    _tost_record(rows, 'fu3.transfer_decoding.dense_untrained', 'F11.equivalence.transfer',
                 'dense delay scan, untrained timings: share of cue pairs decoded both-correct minus both-wrong by the fixed linear decoder, within +-10 pp',
                 'cue_pair', int((k == 2).sum()), int((k == 0).sum()), len(k), 'extended/ed05/ed05d_dense_delay_scan.csv',
                 'both correct', 'both wrong', 'a fixed cue decoder is at chance at untrained timings')
    # (c) two neutral input windows vs duration-matched blank waiting (Fig. 5c)
    c = pd.read_csv(paths.out('main/fig5/fig5c_matched_duration.csv'), low_memory=False)
    c = c[(c['input_kind'] != 'blank') & (c['prefix_windows'] == 2)].copy()
    c['solved'] = _bool(c['solved'])
    c['baseline_solved'] = _bool(c['baseline_solved'])
    for (u, kind), x in sorted(c.groupby(['unit', 'input_kind'])):
        b = int((x['solved'] & ~x['baseline_solved']).sum())
        cc = int((~x['solved'] & x['baseline_solved']).sum())
        _tost_record(rows, f'fu3.duration.{u}.{kind}', 'F11.equivalence.duration',
                     f'{u}: share solved after 2 {kind} input windows minus after duration-matched blank waiting, within +-10 pp',
                     'history', b, cc, len(x), 'main/fig5/fig5c_matched_duration.csv', f'solved only after {kind} input',
                     'solved only after blank waiting', 'neutral input acts like blank waiting of the same duration')
    df = pd.DataFrame(rows)
    df['analysis_status'] = 'post_hoc'
    csvout.write_table(df, 'followup/fu3_equivalence.csv', title='Equivalence tests (TOST, margin +-10 percentage points) for the claims of absence', describe={
        'test_id': 'Test ID (ST2)', 'claim': 'Claim tested', 'contrast': 'Contrast', 'n': 'Units (cue pairs or histories)',
        'count_a': 'Units in the first category', 'label_a': 'First category', 'count_b': 'Units in the second category', 'label_b': 'Second category',
        'difference': '(count_a - count_b) / n', 'ci90_low': 'Tango 90% score interval, lower', 'ci90_high': 'Tango 90% score interval, upper',
        'p_lower': 'One-sided p, H0: difference <= -0.10', 'p_upper': 'One-sided p, H0: difference >= +0.10', 'p_tost': 'TOST p (the larger)',
        'test': 'non_superiority (one-sided, memory use) or tost (two-sided equivalence)', 'p_test': 'p of that test (unadjusted; Holm in ST2)',
        'margin': 'Margin', 'shown': 'Claim shown at 5% (unadjusted)',
        'margin_supported': 'Smallest margin the data support (upper 95% bound for one-sided; largest |bound| of the 90% interval for TOST)',
        'verdict': 'Verdict', 'source_table': 'Source table'}, key=['test_id'])


# ---------------------------------------------------------------- 4. state dynamics

FIELDS = ('spikes', 'g', 'v', 'adapt', 'res')
DENSE_TIMINGS = ((160, 4), (640, 4))
SKIP_MS = 20          # the first ms after cue offset are left out of the decay fits (the cue response is still settling)
STATE_SRC = 'followup/fu4_state_dynamics.csv'


def _record_meta(model, d, p, phase='free-after'):
    folder = paths.M1 / 'runs' / f'record-{model}'
    return folder, io.json_load(folder / f'{phase}-d{d}-p{p}.json')


def _first_decision(meta):
    for i, e in enumerate(meta['events']):
        if e['kind'] == 'decision' and e['decision_index'] == 0:
            return i
    raise RuntimeError('record without a first decision')


def _record_pairs(meta):
    """{(context, visibility): (case index of cue 0, case index of cue 1)} from a record's case list."""
    pairs = {}
    for i, c in enumerate(meta['cases']):
        pairs.setdefault((int(c['context']), 'visible' if c['visible'] else 'hidden'), {})[int(c['cue'])] = i
    return {k: (v[0], v[1]) for k, v in sorted(pairs.items())}


def _npz_rows(data, key, chunk):
    """Rows of one array of an .npz held in memory, inflated chunk by chunk (a dense delay record is up to 2.1 GB per case)."""
    with zipfile.ZipFile(_io.BytesIO(data)) as zf, zf.open(f'{key}.npy') as f:
        version = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(f)
        if fortran:
            raise RuntimeError('Fortran-ordered array in a dense record')
        row = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        for start in range(0, shape[0], chunk):
            m = min(chunk, shape[0] - start)
            buf = bytearray()
            while len(buf) < m * row:
                got = f.read(m * row - len(buf))
                if not got:
                    raise RuntimeError('truncated array in a dense record')
                buf.extend(got)
            yield start, np.frombuffer(buf, dtype).reshape((m,) + tuple(shape[1:]))


def _lif(model):
    lif = io.json_load(paths.M1 / 'runs' / f'record-{model}' / 'brain.json')['lif']
    if lif.get('slow_gain_by_class') or lif.get('dt') != 1.0:
        raise RuntimeError(f'{model}: unexpected LIF settings')
    return lif


class VoltageSplit:
    """Exact split of the voltage cue difference of non-spiking, non-refractory cells after cue offset. The LIF update is
    v(t) = target + (v(t-1) - target) a with target = v_rest + g(t) + drive(t) - adapt(t-1) and a = exp(-dt / tau_m); the
    drive is the same for both cues after cue offset, so
        dv(t) = a^(t - t0) dv(t0) + sum_s a^(t - s) (1 - a) dg(s) - sum_s a^(t - s) (1 - a) dadapt(s - 1).
    Spike resets and refractory clamping make the residual."""

    def __init__(self, dv0, dadapt0, a):
        self.a = a
        self.initial = dv0.astype(np.float64).copy()
        self.synaptic = np.zeros_like(self.initial)
        self.adaptation = np.zeros_like(self.initial)
        self.prev_adapt = dadapt0.astype(np.float64).copy()

    def step(self, dg, dadapt):
        a = self.a
        self.initial *= a
        self.synaptic *= a
        self.synaptic += (1 - a) * dg
        self.adaptation *= a
        self.adaptation -= (1 - a) * self.prev_adapt
        self.prev_adapt = dadapt.astype(np.float64).copy()
        return self.initial, self.synaptic, self.adaptation


def _shares(dv, parts):
    """Additive split of |dv|^2: <part, dv> / |dv|^2 for each part (the shares and the residual's sum to 1)."""
    e = float(dv @ dv)
    if e <= 0:
        return [np.nan] * (len(parts) + 1)
    s = [float(p @ dv) / e for p in parts]
    return s + [1.0 - sum(s)]


def _dense_pair(model, folder, meta, i0, i1, pools, a_m, split_until):
    """Per-ms cue-difference energies of the five fields and the voltage split, cue onset to the end of the first decision.
    Steps are stamped 1, 2, ... from cue onset. The split needs identical input for both cues, so it runs from the first
    step after cue offset to step split_until."""
    first = meta['events'][_first_decision(meta)]
    end = first['start_ms'] + first['ms']
    files = {}
    for e in meta['dense_files']:
        if e['start_ms'] < end:
            files.setdefault(e['case'], []).append(e)
    s0 = sorted(files.get(i0, []), key=lambda e: e['start_ms'])
    s1 = sorted(files.get(i1, []), key=lambda e: e['start_ms'])
    if [(e['start_ms'], e['ms']) for e in s0] != [(e['start_ms'], e['ms']) for e in s1] or not s0 or s0[-1]['start_ms'] + s0[-1]['ms'] != end:
        raise RuntimeError(f'{model}: dense segments of the cue pair do not cover cue onset to the first decision')
    cue_end = next(e['start_ms'] + e['ms'] for e in meta['events'] if e['kind'] == 'cue')
    rows, split, last = [], None, None
    for e0, e1 in zip(s0, s1):
        d0, d1 = io.read_bytes(folder / e0['file']), io.read_bytes(folder / e1['file'])
        for dd in (d0, d1):
            with np.load(_io.BytesIO(dd), allow_pickle=False) as z:
                if tuple(z['fields'].astype(str)) != FIELDS:
                    raise RuntimeError(f'{model}: unexpected dense fields')
                ms = z['milliseconds'].astype(np.int64)
            if ms[0] != e0['start_ms'] + 1 or len(ms) != e0['ms']:     # stamps count steps: a segment from s covers s+1..s+ms
                raise RuntimeError(f'{model}: dense milliseconds disagree with the segment list')
        for (start, a), (_, b) in zip(_npz_rows(d0, 'values', 20), _npz_rows(d1, 'values', 20)):
            D = b.astype(np.float64) - a.astype(np.float64)              # (ms, field, cell): cue 1 minus cue 0
            for k in range(D.shape[0]):
                t = int(ms[start + k])
                dsp, dg, dv, dad, dres = D[k]
                row = dict(ms=t, segment=e0['kind'], **{f'energy_{f}': float(D[k, j] @ D[k, j]) for j, f in enumerate(FIELDS)})
                pdg, pdv = dg[pools].mean(1), dv[pools].mean(1)
                row.update(energy_readout_g=float(pdg @ pdg), energy_readout_v=float(pdv @ pdv))
                if t == cue_end + 1:                                      # first step after cue offset: start the split
                    split = VoltageSplit(last[0], last[1], a_m)
                if split is not None and t <= split_until:
                    parts = split.step(dg, dad)
                    sh = _shares(dv, parts)
                    shr = _shares(pdv, [p[pools].mean(1) for p in parts])
                else:
                    sh = shr = [np.nan] * 4
                row.update(zip(['share_v_initial', 'share_v_synaptic', 'share_v_adaptation', 'share_v_residual'], sh))
                row.update(zip(['share_readout_v_initial', 'share_readout_v_synaptic', 'share_readout_v_adaptation', 'share_readout_v_residual'], shr))
                rows.append(row)
                last = (dv, dad)
    return rows


def _decay_tau(ms, energy, start, stop):
    """Effective time constant of a cue difference from its energy: |d(t)|^2 ~ exp(-2 t / tau), least squares on log energy."""
    m = (ms >= start) & (ms < stop) & (energy > 0)
    if m.sum() < 10:
        return np.nan, np.nan
    x, y = ms[m].astype(float), np.log(energy[m])
    slope, icept = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (icept + slope * x)) ** 2) / np.sum((y - y.mean()) ** 2)
    return (float(-2.0 / slope) if slope < 0 else np.inf), float(r2)


def state_dynamics(pools):
    rows, summary = [], []
    for model, label in MODELS:
        lif = _lif(model)
        a_m = float(np.exp(-lif['dt'] / lif['tau_m']))
        intrinsic = {'g': lif['tau_syn'], 'v': lif['tau_m'], 'adapt': lif['adapt_tau'], 'res': lif['std_tau']}
        for d, p in DENSE_TIMINGS:
            folder, meta = _record_meta(model, d, p)
            delay = next(e for e in meta['events'] if e['kind'] == 'delay')
            first = meta['events'][_first_decision(meta)]
            for (ctx, vis), (i0, i1) in _record_pairs(meta).items():
                # a visible goal changes the input from the first prefix window on; a hidden goal only at the first push
                until = delay['start_ms'] + delay['ms'] if vis == 'visible' else first['start_ms'] + first['ms']   # last step stamp
                rr = _dense_pair(model, folder, meta, i0, i1, pools, a_m, until)
                for r in rr:
                    r.update(model=model, model_label=label, context=ctx, visibility=vis, delay_ms=d, prefix_windows=p)
                rows.extend(rr)
                df = pd.DataFrame(rr)
                ms = df['ms'].to_numpy()
                lo, hi = delay['start_ms'] + 1 + SKIP_MS, delay['start_ms'] + delay['ms'] + 1      # step stamps lo..hi-1
                s = dict(model=model, model_label=label, context=ctx, visibility=vis, delay_ms=d, prefix_windows=p, fit_start_ms=lo, fit_end_ms=hi)
                for f in FIELDS:
                    tau, r2 = _decay_tau(ms, df[f'energy_{f}'].to_numpy(float), lo, hi)
                    s[f'tau_eff_{f}_ms'], s[f'r2_{f}'] = tau, r2
                    s[f'intrinsic_tau_{f}_ms'] = intrinsic.get(f)
                end_delay = df[df['ms'] == hi - 1].iloc[0]
                dec = df[df['segment'] == 'decision']
                for part in ('initial', 'synaptic', 'adaptation', 'residual'):
                    s[f'share_v_{part}_end_of_delay'] = float(end_delay[f'share_v_{part}'])
                    s[f'share_v_{part}_first_decision'] = float(dec[f'share_v_{part}'].mean())
                    s[f'share_readout_v_{part}_first_decision'] = float(dec[f'share_readout_v_{part}'].mean())
                s['spike_differences_last_half_of_delay'] = float(df[(df['ms'] >= (lo + hi) // 2) & (df['ms'] < hi)]['energy_spikes'].mean())
                summary.append(s)
    df = pd.DataFrame(rows)
    df['analysis_status'] = 'post_hoc'
    lead = ['model', 'model_label', 'context', 'visibility', 'delay_ms', 'prefix_windows']
    df = df[lead + [c for c in df.columns if c not in lead]]
    share_desc = {f'share_{w}v_{p}': f'Share of |cue difference in {"readout-pooled " if w else ""}voltage|^2 from the {p} part'
                  + (' (exact split; residual = spike resets)' if p == 'residual' else '') for w in ('', 'readout_') for p in ('initial', 'synaptic', 'adaptation', 'residual')}
    csvout.write_table(df, STATE_SRC, title='Cue difference of each state variable per ms, from cue onset to the first decision (dense records)', describe={
        'model': 'Model', 'model_label': 'Model (label)', 'context': 'Context', 'visibility': 'Goal visibility', 'delay_ms': 'Delay (ms)',
        'prefix_windows': 'Prefix windows', 'ms': 'Simulation step from cue onset (1 ms each; 1 = first step of the cue)', 'segment': 'Segment (cue, delay, prefix, decision)',
        **{f'energy_{f}': f'Sum over cells of (cue 1 - cue 0)^2 for {f}' for f in FIELDS},
        'energy_readout_g': 'Same for the readout-pooled synaptic current (256 pools of observed cells)',
        'energy_readout_v': 'Same for the readout-pooled voltage', **share_desc},
        key=['model', 'context', 'visibility', 'delay_ms', 'prefix_windows', 'ms'])
    sm = pd.DataFrame(summary)
    sm['analysis_status'] = 'post_hoc'
    csvout.write_table(sm, 'followup/fu4_state_summary.csv', title='Effective decay times of the cue difference vs intrinsic time constants, and the voltage split', describe={
        'model': 'Model', 'model_label': 'Model (label)', 'context': 'Context', 'visibility': 'Goal visibility', 'delay_ms': 'Delay (ms)',
        'prefix_windows': 'Prefix windows', 'fit_start_ms': 'First step of the fit window (21 steps after cue offset)', 'fit_end_ms': 'Step after the last step of the fit window (delay end + 1)',
        **{f'tau_eff_{f}_ms': f'Effective decay time of the {f} cue difference (ms; inf = not decaying)' for f in FIELDS},
        **{f'r2_{f}': f'R^2 of the log-linear fit ({f})' for f in FIELDS},
        **{f'intrinsic_tau_{f}_ms': f'Intrinsic time constant of {f} (ms; brain.json)' for f in FIELDS},
        **{f'share_v_{p}_end_of_delay': f'Share of the voltage cue difference from the {p} part, last ms of the delay' for p in ('initial', 'synaptic', 'adaptation', 'residual')},
        **{f'share_v_{p}_first_decision': f'Same, mean over the first decision window' for p in ('initial', 'synaptic', 'adaptation', 'residual')},
        **{f'share_readout_v_{p}_first_decision': f'Same for the readout-pooled voltage, first decision window' for p in ('initial', 'synaptic', 'adaptation', 'residual')},
        'spike_differences_last_half_of_delay': 'Mean number of cells whose spike state differs between cues per ms, second half of the delay'},
        key=['model', 'context', 'visibility', 'delay_ms', 'prefix_windows'])


def _drive_split(dg, dadapt):
    """Exact split of the cue difference of the drive. The LIF target is v_rest + g + drive - adapt and the external drive
    is the same for both cues after cue offset, so du = dg - dadapt and |du|^2 = <dg, du> + <-dadapt, du>; returns
    (|du|^2, synaptic share, adaptation share), the shares summing to 1."""
    du = dg - dadapt
    e = float(du @ du)
    if e <= 0:
        return e, np.nan, np.nan
    s = float(dg @ du) / e
    return e, s, 1.0 - s


def _trace_decay(delays, energy):
    """Effective decay time of a cue difference across delays from its energies, |d|^2 ~ exp(-2 delay / tau): least
    squares on log energy; (tau in ms, R^2, delays used). tau = inf if not decaying."""
    x, e = np.asarray(delays, float), np.asarray(energy, float)
    ok = e > 0
    if ok.sum() < 3:
        return np.nan, np.nan, int(ok.sum())
    y = np.log(e[ok])
    slope, icept = np.polyfit(x[ok], y, 1)
    r2 = 1 - np.sum((y - (icept + slope * x[ok])) ** 2) / np.sum((y - y.mean()) ** 2)
    return (float(-2.0 / slope) if slope < 0 else np.inf), float(r2), int(ok.sum())


def decision_states(pools):
    """Cue difference at the first decision window (segment means) at all 35 timings: per-field energies, the drive split
    (fu9) and the voltage difference vector for the ensemble analysis."""
    rows, parts, dv = [], [], {}
    for model, label in MODELS:
        for d, p, _ in _timings(model):
            folder, meta = _record_meta(model, d, p)
            z = io.npz_load(folder / f'free-after-d{d}-p{p}.npz', keys=['mean'])['mean']
            if z.shape[0] != len(meta['events']) or z.shape[2] != len(FIELDS):
                raise RuntimeError(f'{model} d{d} p{p}: record shape disagrees with its events')
            s = _first_decision(meta)
            for (ctx, vis), (i0, i1) in _record_pairs(meta).items():
                D = z[s, i1].astype(np.float64) - z[s, i0].astype(np.float64)
                r = dict(model=model, model_label=label, context=ctx, visibility=vis, delay_ms=d, prefix_windows=p,
                         trained=bool(d in TRAINED_DELAYS and p in TRAINED_PREFIXES), elapsed_after_cue_ms=d + 40 * p)
                r.update({f'energy_{f}': float(D[j] @ D[j]) for j, f in enumerate(FIELDS)})
                pg, pv = D[1][pools].mean(1), D[2][pools].mean(1)
                r.update(energy_readout_g=float(pg @ pg), energy_readout_v=float(pv @ pv))
                rows.append(r)
                dv[(model, ctx, vis, d, p)] = D[2].astype(np.float32)
                q = {k: r[k] for k in ('model', 'model_label', 'context', 'visibility', 'delay_ms', 'prefix_windows', 'trained', 'elapsed_after_cue_ms')}
                q['energy_drive'], q['share_drive_synaptic'], q['share_drive_adaptation'] = _drive_split(D[1], D[3])
                q['energy_readout_drive'], q['share_readout_drive_synaptic'], q['share_readout_drive_adaptation'] = _drive_split(D[1][pools].mean(1), D[3][pools].mean(1))
                parts.append(q)
    drive_split(pd.DataFrame(parts), pd.DataFrame(rows))
    df = pd.DataFrame(rows)
    df['analysis_status'] = 'post_hoc'
    csvout.write_table(df, 'followup/fu4_decision_energy.csv', title='Cue-difference energy per state variable at the first decision window, all 35 timings', describe={
        'model': 'Model', 'model_label': 'Model (label)', 'context': 'Context', 'visibility': 'Goal visibility', 'delay_ms': 'Delay (ms)',
        'prefix_windows': 'Prefix windows', 'trained': 'Trained timing', 'elapsed_after_cue_ms': 'Delay + 40 ms per prefix window',
        **{f'energy_{f}': f'Sum over cells of the squared cue difference of {f} (first decision window mean)' for f in FIELDS},
        'energy_readout_g': 'Same for the readout-pooled synaptic current', 'energy_readout_v': 'Same for the readout-pooled voltage'},
        key=['model', 'context', 'visibility', 'delay_ms', 'prefix_windows'])
    return dv


def drive_split(parts, energies):
    """fu9: the drive split at every timing, and the adaptation difference's decay across delays per cue pair and prefix."""
    parts = parts.copy()
    parts['analysis_status'] = 'post_hoc'
    lead = ['model', 'model_label', 'context', 'visibility', 'delay_ms', 'prefix_windows']
    csvout.write_table(parts, 'followup/fu9_drive_split.csv', title='Synaptic and adaptation parts of the cue difference of the drive at the first decision window, all 35 timings', describe={
        'model': 'Model', 'model_label': 'Model (label)', 'context': 'Context', 'visibility': 'Goal visibility', 'delay_ms': 'Delay (ms)',
        'prefix_windows': 'Prefix windows', 'trained': 'Trained timing', 'elapsed_after_cue_ms': 'Delay + 40 ms per prefix window',
        'energy_drive': 'Sum over cells of the squared cue difference of the drive (synaptic current minus adaptation; window means)',
        'share_drive_synaptic': 'Share of that energy from the synaptic current (exact split; the two shares sum to 1)',
        'share_drive_adaptation': 'Share from adaptation', 'energy_readout_drive': 'Same for the readout-pooled cells',
        'share_readout_drive_synaptic': 'Synaptic share, readout-pooled', 'share_readout_drive_adaptation': 'Adaptation share, readout-pooled'},
        key=lead)
    rows = []
    for (model, label, ctx, vis, p), g in energies.groupby(['model', 'model_label', 'context', 'visibility', 'prefix_windows'], sort=True):
        g = g.sort_values('delay_ms')
        tau, r2, used = _trace_decay(g['delay_ms'], g['energy_adapt'])
        sh = parts[(parts['model'] == model) & (parts['context'] == ctx) & (parts['visibility'] == vis) & (parts['prefix_windows'] == p)]
        rows.append(dict(model=model, model_label=label, context=ctx, visibility=vis, prefix_windows=p, delays_used=used,
                         tau_eff_adapt_ms=tau, r2_adapt=r2, intrinsic_tau_adapt_ms=float(_lif(model)['adapt_tau']),
                         share_drive_adaptation_min=float(sh['share_drive_adaptation'].min()),
                         share_drive_adaptation_max=float(sh['share_drive_adaptation'].max())))
    df = pd.DataFrame(rows)
    df['analysis_status'] = 'post_hoc'
    csvout.write_table(df, 'followup/fu9_trace_decay.csv', title='Decay of the adaptation cue difference at the first decision across the 7 delays, per prefix', describe={
        'model': 'Model', 'model_label': 'Model (label)', 'context': 'Context', 'visibility': 'Goal visibility', 'prefix_windows': 'Prefix windows',
        'delays_used': 'Delays with a nonzero adaptation difference (of 7)',
        'tau_eff_adapt_ms': 'Effective decay time across delays (ms): least squares of log energy on delay, energy ~ exp(-2 delay / tau)',
        'r2_adapt': 'R^2 of that fit', 'intrinsic_tau_adapt_ms': 'Intrinsic adaptation time constant (brain.json)',
        'share_drive_adaptation_min': 'Smallest adaptation share of the drive difference over the 7 delays',
        'share_drive_adaptation_max': 'Largest adaptation share over the 7 delays'},
        key=['model', 'context', 'visibility', 'prefix_windows'])


# ---------------------------------------------------------------- 5. ensemble wiring: stricter null, all timings

N_STRICT = 1000
N_TIMING_NULL = 100


def _pools(codes):
    """{stratum code: cell indices in ascending order}."""
    order = np.argsort(codes, kind='mergesort')
    return {int(codes[g[0]]): g for g in np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)}


def _within(A, idx, mask):
    """Connections among the cells idx (rows postsynaptic), counted from the row slice."""
    mask[idx] = True
    k = int(mask[A[idx].indices].sum())
    mask[idx] = False
    return k


def _null_counts(A, codes, pools, idx, rng, draws, mask):
    cnt = np.bincount(codes[idx], minlength=codes.max() + 1)
    nz = np.flatnonzero(cnt)
    out = np.empty(draws)
    for i in range(draws):
        pick = np.concatenate([rng.choice(pools[c], cnt[c], replace=False) for c in nz])
        out[i] = _within(A, np.sort(pick), mask)
    return out, cnt


def ensemble_followup(dv_timings):
    from panels import p6b_circuits as p6b
    from panels.p6_mechanisms import STATE_SETS
    W, ann, loc = p6b.load()
    A = (W != 0).tocsr()
    n = A.shape[0]
    k = int(np.ceil(p6b.TOP_FRACTION * n))
    typ = io.parquet_load(paths.M1 / 'neurons' / 'annotations.parquet', columns=['bodyId', 'type'])
    if not np.array_equal(typ['bodyId'].to_numpy(), ann['bodyId'].to_numpy()):
        raise RuntimeError('annotations disagree')
    d_in = p6b._bfs(A.astype(np.float32).tocsr(), np.flatnonzero(ann['direct_input'].to_numpy(bool)))
    deg = np.asarray(A.sum(0)).ravel() + np.asarray(A.sum(1)).ravel()
    dbin = np.clip(np.searchsorted(np.quantile(deg, [0.2, 0.4, 0.6, 0.8]), deg, side='right'), 0, 4)
    cell_type = typ['type'].astype(object).where(typ['type'].notna(), 'untyped:' + ann['superclass'].fillna('unannotated').astype(str))
    strict, _ = pd.factorize(cell_type.astype(str) + '|' + pd.Series(p6b._hop_class(d_in)) + '|' + pd.Series(dbin).astype(str))
    loose = p6b.strata(W, ann, loc)
    pools = {'strict': _pools(strict), 'loose': _pools(loose)}
    codes = {'strict': strict, 'loose': loose}
    mask = np.zeros(n, bool)
    frames = {m: io.parquet_load(paths.M1 / 'interpretation' / f'{m}-memory-cells-d160-p4.parquet') for m in sorted({s[0] for s in STATE_SETS})}
    rows, trows = [], []
    for model, ctx, vis, label in STATE_SETS:
        key = p6b.KEYS[(model, ctx, vis)]
        dvs = frames[model][f'c{ctx}-visible{vis}-delta-v'].to_numpy(np.float64)
        top = np.sort(np.argsort(-np.abs(dvs), kind='mergesort')[:k])
        observed = _within(A, top, mask)
        tid = f'fu5.ensemble_strict.{key}'
        null, cnt = _null_counts(A, strict, pools['strict'], top, stats.rng_for(tid), N_STRICT, mask)
        forced = int(sum(cnt[c] for c in np.flatnonzero(cnt) if len(pools['strict'][c]) == cnt[c]))
        pv = stats.perm_p(observed, null, two_sided=False)
        boot = observed / null[stats.rng_for(f'{tid}.ci').integers(0, N_STRICT, size=(2000, N_STRICT))].mean(1)
        stats.record(dict(test='stratified_random_sets', statistic_name='within_set_connections', statistic=float(observed), p_value=pv,
                          p_method='permutation', n=k, n_used=k, n_resamples=N_STRICT, mc_se=stats.mc_se(pv, N_STRICT),
                          effect_name='observed_over_null_mean', effect=float(observed / null.mean()), effect_ci_low=float(np.quantile(boot, 0.025)),
                          effect_ci_high=float(np.quantile(boot, 0.975)), effect_ci_method='bootstrap over the null draws (2,000 resamples)',
                          null_mean=float(null.mean()), null_q025=float(np.quantile(null, 0.025)), null_q975=float(np.quantile(null, 0.975)), informative=True),
                     test_id=tid, panel='followup/fu5', family='F11.ensemble.strict',
                     contrast=f'{label}: connections among the top-1% cue-responsive cells vs sets matched by cell type, hops from the inputs and degree',
                     unit='cell (within one state set)', level='B', analysis_status='post_hoc', source_table='followup/fu5_ensemble_strict.csv')
        rows.append(dict(state_set=label, state_key=key, cells=k, cells_in_forced_strata=forced, strata_used=int((cnt > 0).sum()),
                         observed_connections=observed, null_mean=float(null.mean()), null_q025=float(np.quantile(null, 0.025)),
                         null_q975=float(np.quantile(null, 0.975)), ratio=float(observed / null.mean()), p_value=pv, test_id=tid))
        # the same ensemble definition at every timing (first decision window), against the original matched null
        vis_name = 'visible' if vis else 'hidden'
        ref = set(top.tolist())
        per = []
        for (m_, c_, v_, d, p), x in sorted(dv_timings.items()):
            if (m_, c_, v_) != (model, ctx, vis_name):
                continue
            tt = np.sort(np.argsort(-np.abs(x.astype(np.float64)), kind='mergesort')[:k])
            ob = _within(A, tt, mask)
            nl, _ = _null_counts(A, loose, pools['loose'], tt, stats.rng_for(f'fu5.timing.{key}.d{d}.p{p}'), N_TIMING_NULL, mask)
            per.append(ob - nl.mean())
            trows.append(dict(state_set=label, state_key=key, delay_ms=d, prefix_windows=p, trained=bool(d in TRAINED_DELAYS and p in TRAINED_PREFIXES),
                              observed_connections=ob, null_mean=float(nl.mean()), null_q975=float(np.quantile(nl, 0.975)), ratio=float(ob / nl.mean()),
                              above_null_q975=bool(ob > np.quantile(nl, 0.975)), overlap_with_reference=len(ref.intersection(tt.tolist())) / k))
        if len(per) != 35:
            raise RuntimeError(f'{key}: expected 35 timings, found {len(per)}')
        stats.record(stats.sign_test(np.array(per)), test_id=f'fu5.ensemble_timings.{key}', panel='followup/fu5', family='F11.ensemble.timings',
                     contrast=f'{label}: top-1% ensemble at the first decision more interconnected than matched sets, over the 35 timings',
                     unit='timing', level='B', analysis_status='post_hoc', source_table='followup/fu5_ensemble_timings.csv')
    e = pd.DataFrame(rows)
    e['analysis_status'] = 'post_hoc'
    csvout.write_table(e, 'followup/fu5_ensemble_strict.csv', title='Ensemble wiring vs random sets matched by cell type, hops from the inputs and degree quintile', describe={
        'state_set': 'Distinct state set', 'state_key': 'State-set key', 'cells': 'Cells in the ensemble (top 1% by |dv|)',
        'cells_in_forced_strata': 'Ensemble cells whose stratum has no other cell (drawn as themselves)', 'strata_used': 'Strata containing ensemble cells',
        'observed_connections': 'Connections within the ensemble', 'null_mean': 'Mean over 1,000 matched sets', 'null_q025': '2.5th percentile', 'null_q975': '97.5th percentile',
        'ratio': 'Observed over null mean', 'p_value': 'One-sided permutation p (Holm in ST2)', 'test_id': 'Test ID'}, key=['state_key'])
    t = pd.DataFrame(trows)
    t['analysis_status'] = 'post_hoc'
    csvout.write_table(t, 'followup/fu5_ensemble_timings.csv', title='The top-1% ensemble at the first decision window at every timing, and its wiring', describe={
        'state_set': 'Distinct state set', 'state_key': 'State-set key', 'delay_ms': 'Delay (ms)', 'prefix_windows': 'Prefix windows', 'trained': 'Trained timing',
        'observed_connections': 'Connections within the ensemble', 'null_mean': 'Mean over 100 sets matched by superclass, role, degree and location',
        'null_q975': '97.5th percentile of the null', 'ratio': 'Observed over null mean', 'above_null_q975': 'Above the 97.5th percentile',
        'overlap_with_reference': 'Share of cells shared with the Fig. 6 ensemble (160 ms delay, 4 prefixes, last recorded window)'},
        key=['state_key', 'delay_ms', 'prefix_windows'])


# ---------------------------------------------------------------- 6. brain x readout pairing (M0 joint)

PAIR_MODEL = 'remote-dynamic-M0-joint'
STAGES = ('before', 'mid', 'after')


def pairing():
    folder = paths.M1 / 'runs' / f'record-{PAIR_MODEL}'
    index = io.json_load(folder / 'matched-index.json')
    heads = {}
    for s in STAGES:
        z = io.npz_load(folder / f'parameters-{s}.npz', keys=['head_proposal.weight', 'head_proposal.bias', 'head_mean', 'head_scale'])
        heads[s] = tuple(z[k].astype(np.float64) for k in ('head_proposal.weight', 'head_proposal.bias', 'head_mean', 'head_scale'))
    timings = _timings(PAIR_MODEL)
    pairs = _pairs(timings[0][2])
    if list(pairs) != [(0, 'hidden')]:
        raise RuntimeError('pairing: expected one hidden cue pair')
    a0, _, _ = _solving_action(timings, pairs[(0, 'hidden')][0])
    a1, _, _ = _solving_action(timings, pairs[(0, 'hidden')][1])
    legal = {}                                       # legal moves at the first decision (same board for both cues)
    for d, p, cases in timings:
        masks = {tuple(c['decisions'][0]['legal']) for c in cases}
        if len(masks) != 1:
            raise RuntimeError(f'pairing: legal moves differ between cues at d{d} p{p}')
        legal[(d, p)] = np.array(masks.pop(), bool)
    rows = []
    for e in index:
        f = io.npz_load(folder / e['readout'], keys=['features', 'logits'])
        meta = io.json_load(folder / f'matched-before-d{e["delay"]}-p{e["prefix"]}.json')
        s0 = _first_decision(meta)
        if f['features'].shape[0] != len(meta['events']):
            raise RuntimeError('pairing: matched record shape disagrees with its events')
        (i0, i1), = _record_pairs(meta).values()
        for s in STAGES:
            Wh, bh, mean, scale = heads[s]
            lg = {cue: Wh @ _z(f['features'][s0, i], mean, scale) + bh for cue, i in ((0, i0), (1, i1))}
            if s == e['phase']:
                for cue, i in ((0, i0), (1, i1)):
                    if np.max(np.abs(lg[cue] - f['logits'][s0, i].astype(np.float64))) > 1e-3:
                        raise RuntimeError(f'pairing: recomputed logits differ (brain {e["phase"]}, d{e["delay"]} p{e["prefix"]})')
            m0, m1 = lg[0][a1] - lg[0][a0], lg[1][a1] - lg[1][a0]
            ok = legal[(e['delay'], e['prefix'])]
            push = {cue: int(np.argmax(np.where(ok, lg[cue], -np.inf))) for cue in (0, 1)}
            rows.append(dict(brain_stage=e['phase'], readout_stage=s, delay_ms=e['delay'], prefix_windows=e['prefix'],
                             trained=bool(e['delay'] in TRAINED_DELAYS and e['prefix'] in TRAINED_PREFIXES), margin_cue0=m0, margin_cue1=m1,
                             differential=(m1 - m0) / 2, common=(m1 + m0) / 2, separated=bool(m1 > 0 > m0),
                             first_push_correct_cue0=bool(push[0] == a0), first_push_correct_cue1=bool(push[1] == a1)))
    df = pd.DataFrame(rows)
    if len(df) != 3 * 3 * 35:
        raise RuntimeError('pairing: expected 3 brains x 3 readouts x 35 timings')
    piv = df.pivot_table(index=['delay_ms', 'prefix_windows', 'trained'], columns=['brain_stage', 'readout_stage'], values='differential').reset_index()
    tr = piv[piv['trained']]
    for name, a, b, text in (('trained_brain_for_trained_readout', ('after', 'after'), ('before', 'after'),
                              'trained readout on the trained vs the initial brain'),
                             ('trained_readout_for_trained_brain', ('after', 'after'), ('after', 'before'),
                              'trained brain read by the trained vs the initial readout'),
                             ('brain_training_alone', ('after', 'before'), ('before', 'before'),
                              'initial readout on the trained vs the initial brain')):
        diff = (tr[a] - tr[b]).to_numpy(float)
        stats.record(stats.sign_test(diff), test_id=f'fu6.pairing.{name}', panel='followup/fu6', family='F11.pairing',
                     contrast=f'M0 joint, 9 trained timings: differential margin, {text}', unit='timing', level='B', analysis_status='post_hoc',
                     source_table='followup/fu6_pairing.csv')
    df['analysis_status'] = 'post_hoc'
    csvout.write_table(df, 'followup/fu6_pairing.csv', title='M0 joint: readouts from before, mid and after training applied to brain states from each stage (identical inputs)',
                       describe={'brain_stage': 'Stage of the brain parameters that produced the state', 'readout_stage': 'Stage of the readout applied',
                                 'delay_ms': 'Delay (ms)', 'prefix_windows': 'Prefix windows', 'trained': 'Trained timing',
                                 'margin_cue0': 'Logit(cue-1 push) minus logit(cue-0 push) after cue 0 (negative = correct)',
                                 'margin_cue1': 'The same after cue 1 (positive = correct)', 'differential': 'Half the margin difference between cues',
                                 'common': 'Mean margin over cues', 'separated': 'Both margins on the correct side',
                                 'first_push_correct_cue0': 'Highest-logit legal move is the cue-0 solving first move',
                                 'first_push_correct_cue1': 'Highest-logit legal move is the cue-1 solving first move'},
                       key=['brain_stage', 'readout_stage', 'delay_ms', 'prefix_windows'])


# ---------------------------------------------------------------- 7. brain change vs its contribution (Fig. 3)

def _rank_residual(a, b):
    """Residual of rank(a) after a least-squares line on rank(b)."""
    ra, rb = st.rankdata(a), st.rankdata(b)
    B = np.c_[np.ones_like(rb), rb]
    return ra - B @ np.linalg.lstsq(B, ra, rcond=None)[0]


def partial_spearman(x, y, z, test_id):
    """Spearman correlation of x and y controlling for z (Pearson correlation of the rank residuals on rank(z)); Freedman-Lane
    permutation of the x residuals, and a bootstrap interval over units with the ranks recomputed in each resample."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    ex, ey = _rank_residual(x, z), _rank_residual(y, z)
    norm = np.sqrt((ex @ ex) * (ey @ ey))
    r = float(ex @ ey / norm)
    rng = stats.rng_for(test_id)
    null = np.array([ex[rng.permutation(len(x))] @ ey / norm for _ in range(N_PERM)])
    p = stats.perm_p(r, null, center=0.0)
    rb = stats.rng_for(f'{test_id}.ci')
    boot = []
    for _ in range(2000):
        i = rb.integers(0, len(x), len(x))
        bx, by = _rank_residual(x[i], z[i]), _rank_residual(y[i], z[i])
        d = np.sqrt((bx @ bx) * (by @ by))
        if d > 0:
            boot.append(bx @ by / d)
    return dict(test='partial_spearman_permutation', statistic_name='partial rho', statistic=r, p_value=p, p_method='permutation (Freedman-Lane)',
                n=len(x), n_used=len(x), n_resamples=N_PERM, mc_se=stats.mc_se(p, N_PERM), effect_name='partial_spearman_rho', effect=r,
                effect_ci_low=float(np.quantile(boot, 0.025)), effect_ci_high=float(np.quantile(boot, 0.975)),
                effect_ci_method='percentile bootstrap over maps (2,000 resamples)', informative=True)


def brain_change():
    st4 = pd.read_csv(paths.out('supplementary/st4_parameter_change.csv'), usecols=['run_id', 'normalized_abs_change_sum'], low_memory=False)
    d3 = pd.read_csv(paths.out('main/fig3/fig3d_margin_decomposition.csv'), low_memory=False)
    d3 = d3[d3['set'] == 'difficulty'].set_index('model')
    c2 = pd.read_csv(paths.out('main/fig2/fig2c_cost_vs_difficulty.csv'), low_memory=False).set_index('run_id')
    df = pd.DataFrame({'map_id': d3['map_id'], 'brain_change': st4.groupby('run_id')['normalized_abs_change_sum'].sum().reindex(d3.index),
                       'updates': c2['updates'].reindex(d3.index), 'brain_term': d3['brain_term'], 'readout_term': d3['readout_term'],
                       'interaction_term': d3['interaction_term'], 'total_margin_change': d3['total']})
    if len(df) != 24 or df.isna().any().any():
        raise RuntimeError('brain change: expected 24 difficulty maps with change, updates and decomposition')
    df.index.name = 'run_id'
    logu = np.log(df['updates'].to_numpy(float))
    for name, y, text in (('brain_term', df['brain_term'], 'brain term of the margin decomposition'),
                          ('total_margin_change', df['total_margin_change'], 'total margin change')):
        tid = f'fu7.partial.{name}'
        stats.record(partial_spearman(df['brain_change'], y, logu, tid), test_id=tid, panel='followup/fu7', family='F11.brain_change',
                     contrast=f'24 maps: total brain parameter change vs the {text}, controlling for log(updates)', unit='map', level='B',
                     analysis_status='post_hoc', source_table='followup/fu7_brain_change.csv')
    stats.record(stats.spearman(df['brain_change'], logu, 'fu7.spearman.updates'), test_id='fu7.spearman.updates', panel='followup/fu7',
                 family='F11.brain_change', contrast='24 maps: total brain parameter change vs log(updates)', unit='map', level='B',
                 analysis_status='post_hoc', source_table='followup/fu7_brain_change.csv')
    out = df.reset_index()
    out['analysis_status'] = 'post_hoc'
    csvout.write_table(out, 'followup/fu7_brain_change.csv', title='Brain parameter change, training length and the brain contribution to the decision margin (24 maps)',
                       describe={'run_id': 'Run', 'map_id': 'Map', 'brain_change': 'Sum over type groups of the normalised absolute parameter change (ST4)',
                                 'updates': 'Updates to acquisition (Fig. 2c)', 'brain_term': 'Brain term of the margin decomposition (Fig. 3d)',
                                 'readout_term': 'Readout term', 'interaction_term': 'Interaction term', 'total_margin_change': 'Total margin change'},
                       key=['run_id'])


# ---------------------------------------------------------------- 8. sign motif vs signed propagation (Fig. 6e)

PROP_GAINS = (0.5, 0.8)
PROP_DRAWS = 200
PROP_BLOCK = 50
PROP_TOL = 1e-8                 # the propagation series is cut after K steps with gain^K <= PROP_TOL
BANDS = ((0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.10))


def _row_normalised(W):
    """Signed connectome (rows postsynaptic) with each row divided by its total absolute input, so |Wn| is row-stochastic;
    rows without input stay zero."""
    W = W.tocsr().astype(np.float64)
    W.eliminate_zeros()
    total = np.asarray(abs(W).sum(1)).ravel()
    scale = np.zeros_like(total)
    scale[total > 0] = 1.0 / total[total > 0]
    return (sp.diags(scale) @ W).tocsr()


def _propagate(Wn, U, gain):
    """Linear response sum_{k>=1} (gain Wn)^k U, cut after K steps with gain^K <= PROP_TOL (|Wn| row-stochastic, so the
    remainder is below gain^(K+1) / (1 - gain) max|U|)."""
    steps = int(np.ceil(np.log(PROP_TOL) / np.log(gain)))
    X = np.zeros_like(U)
    for _ in range(steps):
        X = gain * (Wn @ (X + U))
    return X


def _top(x, k):
    return np.sort(np.argsort(-np.abs(x), kind='mergesort')[:k])


def _motif(W, idx, x):
    """The Fig. 6e statistic among the cells idx (sorted): excitatory share of the connections between cells with the same
    sign of x minus that between cells of opposite sign; returns (statistic, same-sign mask, excitatory mask)."""
    sub = W[idx][:, idx].tocoo()
    pref = np.sign(x[idx])
    same = pref[sub.row] == pref[sub.col]
    exc = sub.data > 0
    if not same.any() or same.all():
        return np.nan, same, exc
    return float(exc[same].mean() - exc[~same].mean()), same, exc


def sign_propagation():
    from panels import p6b_circuits as p6b
    from panels.p6_mechanisms import STATE_SETS
    W, _, _ = p6b.load()
    W = W.tocsr()
    W.eliminate_zeros()                            # stored zeros are not connections
    Wn = _row_normalised(W)
    n = W.shape[0]
    k = int(np.ceil(p6b.TOP_FRACTION * n))
    fig = io.csv_load(paths.out(p6b.FD + 'fd_sign_motif.csv')).set_index('state_key')['difference']
    frames = {m: io.parquet_load(paths.M1 / 'interpretation' / f'{m}-memory-cells-d160-p4.parquet') for m in sorted({s[0] for s in STATE_SETS})}
    rows, brows = [], []
    for model, ctx, vis, label in STATE_SETS:
        key = p6b.KEYS[(model, ctx, vis)]
        f = frames[model]
        dv = f[f'c{ctx}-visible{vis}-delta-v'].to_numpy(np.float64)
        source = -f[f'c{ctx}-visible{vis}-delta-adapt'].to_numpy(np.float64)     # adaptation lowers the drive
        order = np.argsort(-np.abs(dv), kind='mergesort')
        top = np.sort(order[:k])
        observed, same, exc = _motif(W, top, dv)
        if abs(observed - float(fig[key])) > 1e-12:
            raise RuntimeError(f'{key}: sign motif disagrees with figure_data/fd_sign_motif.csv')
        for lo, hi in BANDS:
            idx = np.sort(order[int(np.ceil(lo * n)):int(np.ceil(hi * n))])
            m, s_, e_ = _motif(W, idx, dv)
            brows.append(dict(state_set=label, state_key=key, band_from=lo, band_to=hi, cells=len(idx), connections=len(e_),
                              same_sign_connections=int(s_.sum()), statistic=m))
        support = np.flatnonzero(source != 0)
        signs = stats.rng_for(f'fu8.sign_propagation.{key}').choice([-1.0, 1.0], size=(PROP_DRAWS, len(support)))   # same draws for each gain
        n_same, n_opp = int(same.sum()), int((~same).sum())
        for gain in PROP_GAINS:
            tid = f'fu8.sign_propagation.{key}.g{gain:g}'
            x0 = _propagate(Wn, source[:, None], gain)[:, 0]
            t0 = _top(x0, k)
            m0, _, _ = _motif(W, t0, x0)
            null, conns = np.full(PROP_DRAWS, np.nan), np.zeros(PROP_DRAWS)
            for b in range(0, PROP_DRAWS, PROP_BLOCK):
                m_ = min(PROP_BLOCK, PROP_DRAWS - b)
                U = np.zeros((n, m_))
                U[support] = (signs[b:b + m_] * source[support]).T
                X = _propagate(Wn, U, gain)
                for j in range(m_):
                    null[b + j], _, e_ = _motif(W, _top(X[:, j], k), X[:, j])
                    conns[b + j] = len(e_)
            valid = null[~np.isnan(null)]
            p = stats.perm_p(observed, valid, two_sided=True)
            rb = stats.rng_for(f'{tid}.ci')                # binomial resampling of the observed connection signs, as in Fig. 6e
            boot = rb.binomial(n_same, exc[same].mean(), 2000) / n_same - rb.binomial(n_opp, exc[~same].mean(), 2000) / n_opp - valid.mean()
            stats.record(dict(test='signed_propagation_null', statistic_name='excitatory_share_same_minus_opposite', statistic=observed, p_value=p,
                              p_method='Monte Carlo (signed-propagation surrogates)', n=int(len(exc)), n_used=int(len(exc)), n_resamples=int(len(valid)),
                              mc_se=stats.mc_se(p, len(valid)), effect_name='observed_minus_null_mean', effect=float(observed - valid.mean()),
                              effect_ci_low=float(np.quantile(boot, 0.025)), effect_ci_high=float(np.quantile(boot, 0.975)),
                              effect_ci_method='binomial resampling of the observed connection signs (2,000 resamples) minus the null mean',
                              null_mean=float(valid.mean()), null_q025=float(np.quantile(valid, 0.025)), null_q975=float(np.quantile(valid, 0.975)),
                              informative=True),
                         test_id=tid, panel='followup/fu8', family='F11.sign.propagation',
                         contrast=f'{label}: sign motif of the 1% ensemble vs signed propagation (gain {gain:g}) of the adaptation pattern with random signs',
                         unit='connection (within one state set)', level='B', analysis_status='post_hoc', source_table='followup/fu8_sign_propagation.csv')
            rows.append(dict(state_set=label, state_key=key, gain=gain, connections=int(len(exc)), statistic=observed,
                             source_cells=int(len(support)), null_draws=int(len(valid)), null_mean=float(valid.mean()),
                             null_q025=float(np.quantile(valid, 0.025)), null_q975=float(np.quantile(valid, 0.975)),
                             null_connections_median=float(np.median(conns)), p_value=p,
                             unflipped_statistic=m0, unflipped_overlap_with_ensemble=len(np.intersect1d(t0, top)) / k,
                             unflipped_sign_agreement=float((np.sign(x0[top]) == np.sign(dv[top])).mean()), test_id=tid))
    df = pd.DataFrame(rows)
    df['analysis_status'] = 'post_hoc'
    csvout.write_table(df, 'followup/fu8_sign_propagation.csv', title='Sign motif of the 1% cue ensemble vs a signed-propagation null', describe={
        'state_set': 'Distinct state set', 'state_key': 'State-set key', 'gain': 'Propagation gain (each cell passes on gain x the signed, input-weighted mean of its inputs)',
        'connections': 'Connections within the ensemble', 'statistic': 'Excitatory share, same-preference minus opposite-preference connections (Fig. 6e)',
        'source_cells': 'Cells with an adaptation difference (the propagated pattern)', 'null_draws': 'Surrogates with both same- and opposite-sign connections (of 200)',
        'null_mean': 'Mean statistic in the top-1% sets of the surrogates (pattern signs drawn at random)', 'null_q025': '2.5th percentile', 'null_q975': '97.5th percentile',
        'null_connections_median': 'Median connections within the surrogate sets', 'p_value': 'Two-sided Monte Carlo p (Holm in ST2)',
        'unflipped_statistic': 'Statistic in the top-1% set of the propagated pattern itself (signs as observed)',
        'unflipped_overlap_with_ensemble': 'Share of that set in the observed ensemble',
        'unflipped_sign_agreement': 'Share of ensemble cells whose propagated sign matches the observed preference', 'test_id': 'Test ID'},
        key=['state_key', 'gain'])
    b = pd.DataFrame(brows)
    b['analysis_status'] = 'post_hoc'
    csvout.write_table(b, 'followup/fu8_sign_bands.csv', title='The Fig. 6e statistic among the cells of each |dv| rank band (descriptive)', describe={
        'state_set': 'Distinct state set', 'state_key': 'State-set key', 'band_from': 'Band start (share of cells by |dv| rank)', 'band_to': 'Band end',
        'cells': 'Cells in the band', 'connections': 'Connections within the band', 'same_sign_connections': 'Connections between same-preference cells',
        'statistic': 'Excitatory share, same-preference minus opposite-preference connections'},
        key=['state_key', 'band_from'])


def _readout_pools():
    z = io.npz_load(paths.M1 / 'neurons' / 'index.npz', keys=['populations'])
    pools = z['populations'].astype(np.int64)
    if pools.shape != (256, 16):
        raise RuntimeError('unexpected readout populations')
    return pools


def main():
    timing_lock()
    swap_specificity()
    equivalence()
    pools = _readout_pools()
    state_dynamics(pools)
    ensemble_followup(decision_states(pools))
    pairing()
    brain_change()
    sign_propagation()
