"""P8: statistics table (ST2) with family-wise corrections, claims ledger and coverage sweep, discrepancies, gaps."""
import json
import re

import numpy as np
import pandas as pd

from msa import csvout, io, paths, stats, testplan

DOCS = {
    'remote_results': paths.STUDY / 'remote-results.en.md',
    'remote_final': paths.R0_ROOT / 'final.en.md',
    'supplement_analysis': paths.RS_ROOT / 'analysis.en.md',
    'mechanisms_v1': paths.M1 / 'interpretation.en.md',
    'mechanisms_supplement': paths.MS / 'interpretation.en.md',
    'static_analysis_report': paths.STUDY / 'analysis' / 'analysis-report.en.md',
    'readme': paths.ROOT / 'README.md',
    'method': paths.DOCS / 'method.md',
}

ST2_COLUMNS = ['test_id', 'panel', 'family', 'family_correction', 'family_primary', 'family_posthoc', 'level', 'analysis_status', 'contrast', 'unit', 'history_set_id',
               'test', 'statistic_name', 'statistic', 'n', 'n_used', 'p_value', 'p_method', 'n_resamples', 'mc_se', 'p_adjusted', 'significant_adjusted',
               'informative', 'min_attainable_p', 'effect_name', 'effect', 'effect_ci_low', 'effect_ci_high', 'effect_ci_method', 'a_only', 'b_only', 'both',
               'neither', 'n_positive', 'n_negative', 'n_zero', 'ties', 'rank_biserial', 'mean', 'mean_ci_low', 'mean_ci_high', 'null_mean', 'null_q025', 'null_q975', 'direction',
               'kendall_ci_low', 'kendall_ci_high', 'intercept', 'source_table']


def st2():
    rows = stats.load_all()
    fams = {f['family']: f for f in testplan.FAMILIES}
    undeclared = sorted({r['family'] for r in rows} - set(fams))
    if undeclared:
        raise RuntimeError(f'tests in undeclared families: {undeclared}')
    df = pd.DataFrame(rows)
    if df['test_id'].duplicated().any():
        raise RuntimeError('duplicate test ids')
    df['family_correction'] = df['family'].map(lambda f: fams[f]['correction'])
    df['family_primary'] = df['family'].map(lambda f: fams[f]['primary'])
    df['family_posthoc'] = df['family'].map(lambda f: fams[f]['posthoc'])
    wrong = sorted(set(df.loc[df['family_posthoc'] & (df['analysis_status'] != 'post_hoc'), 'family']))
    if wrong:
        raise RuntimeError(f'families declared post hoc hold registered tests: {wrong}')
    df['p_adjusted'] = np.nan
    for fam, idx in df.groupby('family').groups.items():
        sub = df.loc[idx]
        ok = sub['p_value'].notna()
        if not ok.any():
            continue
        corr = fams[fam]['correction']
        p = sub.loc[ok, 'p_value'].astype(float).to_numpy()
        adj = stats.holm(p) if corr == 'holm' else (stats.bh(p) if corr == 'bh' else p)
        df.loc[sub.index[ok.to_numpy()], 'p_adjusted'] = adj
    df['significant_adjusted'] = df['p_adjusted'].map(lambda v: None if pd.isna(v) else bool(v < stats.ALPHA))
    for c in ST2_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[ST2_COLUMNS]
    desc = {
        'panel': 'Panel or table', 'family': 'Test family (qa/test_plan.csv)', 'family_correction': 'Correction within the family',
        'family_primary': 'Primary family', 'family_posthoc': 'Every test of the family was added after the registered analysis', 'level': 'A maps; B histories/cue pairs/timings within one model', 'contrast': 'What is compared',
        'test': 'Test', 'statistic_name': 'Statistic', 'statistic': 'Statistic value', 'n_used': 'Units contributing (e.g. discordant pairs, non-zero differences)',
        'p_value': 'Unadjusted two-sided p (exact unless p_method says otherwise)', 'p_method': 'exact, permutation or as returned',
        'n_resamples': 'Permutation/bootstrap draws', 'mc_se': 'Monte Carlo standard error of p', 'p_adjusted': 'Adjusted p within the family',
        'significant_adjusted': 'p_adjusted < 0.05', 'informative': 'At least one informative unit (e.g. a discordant pair)',
        'min_attainable_p': 'Smallest p attainable with the informative units', 'effect_name': 'Effect measure', 'effect': 'Effect estimate',
        'effect_ci_low': 'Lower limit of the effect interval', 'effect_ci_high': 'Upper limit of the effect interval', 'effect_ci_method': 'Interval method',
        'a_only': 'McNemar: units positive under the first condition only', 'b_only': 'McNemar: units positive under the second condition only',
        'both': 'McNemar: positive under both', 'neither': 'McNemar: positive under neither', 'n_positive': 'Sign test: positive differences',
        'n_negative': 'Sign test: negative differences', 'n_zero': 'Zero differences (dropped)', 'ties': 'Tied absolute differences present',
        'rank_biserial': 'Matched-pairs rank-biserial correlation', 'mean': 'Mean of the paired values', 'mean_ci_low': 'BCa lower limit of the mean',
        'mean_ci_high': 'BCa upper limit of the mean', 'null_mean': 'Mean of the permutation null statistic',
        'null_q025': '2.5th percentile of the permutation null statistic', 'null_q975': '97.5th percentile of the permutation null statistic',
        'direction': 'Direction of a trend',
        'kendall_ci_low': 'Theil-Sen Kendall interval, lower', 'kendall_ci_high': 'Theil-Sen Kendall interval, upper', 'intercept': 'Theil-Sen intercept',
        'source_table': 'Deliverable holding the underlying data'}
    for rel in ('supplementary/st2_statistical_tests.csv', 'stats/statistical_tests.csv'):
        csvout.write_table(df, rel, title='Statistical tests (every test, with family-wise adjusted p)', describe=desc, key=['test_id'],
                           sort=['panel', 'family', 'test_id'])
    counts = df.groupby('family').size().rename('tests').reset_index()
    plan = pd.DataFrame(testplan.FAMILIES)[['family', 'panel', 'correction', 'primary', 'posthoc']]
    counts = plan.merge(counts, on='family', how='left').fillna({'tests': 0})
    counts['tests'] = counts['tests'].astype(int)
    csvout.write_table(counts, 'qa/test_family_counts.csv', title='Tests realized per declared family', describe={
        'family': 'Family', 'panel': 'Panel', 'correction': 'Correction', 'primary': 'Primary family', 'posthoc': 'Post-hoc family',
        'tests': 'Tests recorded'}, key=['family'])
    return df


# ---------------------------------------------------------------- claims ledger

def R(rel):
    return pd.read_csv(paths.out(rel), low_memory=False)


def fr(k, n):
    return f'{int(k)}/{int(n)}'


def ci(x):
    return f'{int(round(x)):,}'


def claims():
    L = []

    def add(doc, statement, reported, computed, source):
        L.append(dict(document=doc, statement=statement, reported=str(reported), rederived=str(computed), match=str(reported) == str(computed),
                      rederived_from=source))

    a2 = R('main/fig2/fig2a_acquisition_outcomes.csv')
    c2 = R('main/fig2/fig2c_cost_vs_difficulty.csv')
    add('static_analysis_report', 'maps solved after independent reload', '24/24', fr((a2.reload_status == 'solved').sum(), len(a2)), 'fig2a')
    med_min = c2.groupby('tier')['training_seconds'].median() / 60
    med_upd = c2.groupby('tier')['updates'].median()
    rng = c2.groupby('tier')['training_seconds'].agg(['min', 'max']) / 60
    rep_min = ['1.49', '1.39', '3.88', '7.19', '7.78', '12.26', '28.50', '51.59']
    rep_upd = ['100', '92', '232', '360', '392', '612', '1,044', '1,556']
    rep_rng = ['1.45–1.54', '1.36–1.42', '3.03–14.45', '7.12–7.30', '7.46–9.73', '8.95–13.50', '20.95–38.26', '47.37–76.58']
    for t in range(8):
        add('mechanisms_v1', f'D{t} median effective training minutes', rep_min[t], f'{med_min[t]:.2f}', 'fig2c')
        add('mechanisms_v1', f'D{t} median updates', rep_upd[t], ci(med_upd[t]), 'fig2c')
        add('mechanisms_v1', f'D{t} minimum-maximum training minutes', rep_rng[t], f'{rng.loc[t, "min"]:.2f}–{rng.loc[t, "max"]:.2f}', 'fig2c')
    rs = a2.set_index('map_id')['reload_steps']
    add('remote_results', 'D1/D3/D5/D7 representatives reload solution moves', '13/25/37/60',
        '/'.join(str(int(rs[m])) for m in ('D1-T0', 'D3-T0', 'D5-T0', 'D7-T0')), 'fig2a')
    e2 = R('extended/ed02/ed02a_precision_calibration.csv').set_index('id')
    add('static_analysis_report', 'FP64 reference total training seconds', '912.42', f'{e2.loc["fp64-b16", "training_seconds"]:.2f}', 'ed02a')
    add('static_analysis_report', 'mixed-precision total training seconds', '494.24', f'{e2.loc["mixed-b16", "training_seconds"]:.2f}', 'ed02a')
    add('static_analysis_report', 'speed ratio', '1.85', f'{e2.loc["fp64-b16", "training_seconds"] / e2.loc["mixed-b16", "training_seconds"]:.2f}', 'ed02a')
    st1 = R('supplementary/st1_run_registry.csv')
    add('static_analysis_report', 'local runs recorded', '68', str((st1.layer == 'local_static').sum()), 'st1')
    add('static_analysis_report', 'scheduled training items unfinished', '31', str((st1.layer == 'registered_not_run').sum()), 'st1')
    f3 = R('main/fig3/fig3a_training_controls.csv').set_index(['condition', 'map_id'])
    for cond, vals in (('joint', ['84', '360', '440', '1,924']), ('head_only', ['60', '240', '356', '868']),
                       ('demonstration', ['24', '72', '88', '504'])):
        for m, v in zip(('D1-T0', 'D3-T0', 'D5-T0', 'D7-T0'), vals):
            add('mechanisms_v1', f'{cond} updates on {m}', v, ci(f3.loc[(cond, m), 'updates']), 'fig3a')
    for m, v in zip(('D1-T0', 'D3-T0', 'D5-T0'), ['380', '7,020', '6,292']):
        add('mechanisms_v1', f'brain-only updates on {m}', v, ci(f3.loc[('brain_only', m), 'updates']), 'fig3a')
    for m, v in zip(('D1-T0', 'D3-T0', 'D5-T0'), ['4,344', '4,604', '4,464']):
        add('mechanisms_v1', f'no-recovery updates at stagnation on {m}', v, ci(f3.loc[('no_recovery', m), 'updates']), 'fig3a')
    for m, v, j in zip(('D1-T0', 'D3-T0', 'D5-T0'), ['4.68', '22.91', '25.19'], ['83.56', '437.74', '536.94']):
        add('mechanisms_v1', f'readout-only training seconds on {m}', v, f'{f3.loc[("head_only", m), "training_seconds"]:.2f}', 'fig3a')
        add('mechanisms_v1', f'joint training seconds on {m}', j, f'{f3.loc[("joint", m), "training_seconds"]:.2f}', 'fig3a')
    add('remote_final', 'D7 readout-only training seconds', '61.77', f'{f3.loc[("head_only", "D7-T0"), "training_seconds"]:.2f}', 'fig3a')
    add('remote_final', 'D7 demonstration training seconds', '778.25', f'{f3.loc[("demonstration", "D7-T0"), "training_seconds"]:.2f}', 'fig3a')
    e = R('main/fig2/fig2e_zero_update_transfer.csv')
    for _, r in e[e.row_type == 'summary_heldout'].iterrows():
        add('remote_results', f'{r.model} zero-update held-out layouts solved', '0/8', fr(r.numerator, r.denominator), 'fig2e')
    fc = R('main/fig3/fig3c_four_combination.csv')
    fc = fc[fc.set == 'difficulty']
    for comb, rep in (('00', '0/24'), ('10', '0/24'), ('01', '1/24'), ('11', '24/24')):
        x = fc[fc.combination.astype(str).str.zfill(2) == comb]
        add('mechanisms_v1', f'four-combination {comb} complete solutions', rep, fr(x.solved.sum(), len(x)), 'fig3c')
    add('mechanisms_v1', 'the only old-brain + new-readout solution', 'D0-T1', ';'.join(fc[(fc.combination.astype(str).str.zfill(2) == '01') & fc.solved].map_id), 'fig3c')
    d3 = R('main/fig3/fig3d_margin_decomposition.csv')
    dd = d3[d3.set == 'difficulty']
    add('mechanisms_v1', 'equal-map mean brain term', '+0.0275', f'{dd.brain_term.mean():+.4f}', 'fig3d')
    add('mechanisms_v1', 'equal-map mean readout term', '+0.7575', f'{dd.readout_term.mean():+.4f}', 'fig3d')
    add('mechanisms_v1', 'equal-map mean interaction term', '+0.0555', f'{dd.interaction_term.mean():+.4f}', 'fig3d')
    add('mechanisms_v1', 'maps with positive readout term', '24/24', fr((dd.readout_term > 0).sum(), len(dd)), 'fig3d')
    add('mechanisms_v1', 'brain term positive/negative maps', '13/11', f'{(dd.brain_term > 0).sum()}/{(dd.brain_term < 0).sum()}', 'fig3d')
    add('mechanisms_v1', 'interaction term positive/negative maps', '16/8', f'{(dd.interaction_term > 0).sum()}/{(dd.interaction_term < 0).sum()}', 'fig3d')
    add('mechanisms_v1', 'maximum reconstruction error across 34 static models', '9.54e-07', f'{d3.reconstruction_error.max():.2e}', 'fig3d')
    add('mechanisms_v1', 'D1-T1 total reference-margin change', '-0.0484', f'{float(dd.set_index("map_id").loc["D1-T1", "total"]):.4f}', 'fig3d')
    a6 = R('main/fig6/fig6a_activity_sparsity.csv')
    af = a6[a6.phase == 'after']
    allp = af[af.population == 'all']
    add('mechanisms_v1', 'mean cells with spikes per map', '4,923', ci(allp.active_cells.mean()), 'fig6a')
    add('mechanisms_v1', 'fraction of cells with spikes', '2.95%', f'{100 * allp.active_cells.mean() / 167106:.2f}%', 'fig6a')
    add('mechanisms_v1', 'per-map range of active cells', '4,402–5,342', f'{ci(allp.active_cells.min())}–{ci(allp.active_cells.max())}', 'fig6a')
    add('mechanisms_v1', 'mean per-cell spike probability per ms', '0.000779', f'{allp.spike_probability_per_ms.mean():.6f}', 'fig6a')
    for pop, rep_cells, rep_dv in (('input', '797', '0.0106'), ('observed', '941', '0.0524'), ('other', '3,185', '0.00857')):
        x = af[af.population == pop]
        add('mechanisms_v1', f'mean cells with spikes per map ({pop})', rep_cells, ci(x.active_cells.mean()), 'fig6a')
        add('mechanisms_v1', f'mean |voltage change| ({pop}, mV)', rep_dv, f'{x.mean_abs_voltage_change_mv.mean():.{len(rep_dv.split(".")[1])}f}', 'fig6a')
    add('mechanisms_v1', 'overall mean |voltage change| (mV)', '0.00968', f'{allp.mean_abs_voltage_change_mv.mean():.5f}', 'fig6a')
    b6 = R('main/fig6/fig6b_cue_difference_coverage.csv')
    h = b6[(b6.model == 'remote-dynamic-M0-head_only') & (b6.population == 'all')].set_index('variable')
    add('mechanisms_v1', 'cells with voltage cue difference > 1e-6 mV', '119,503', ci(h.loc['v', 'cells_above_threshold']), 'fig6b')
    add('mechanisms_v1', 'cells with adaptation cue difference', '6,379', ci(h.loc['adapt', 'cells_above_threshold']), 'fig6b')
    add('mechanisms_v1', 'cells with resource cue difference', '0', ci(h.loc['res', 'cells_above_threshold']), 'fig6b')
    add('mechanisms_v1', 'mean |voltage cue difference| (mV)', '0.423', f'{h.loc["v", "mean_abs_difference"]:.3f}', 'fig6b')
    s1a = R('main/fig1/fig1a_model_specification.csv').set_index(['section', 'quantity'])['value']
    for q, rep in (('cells_missing_type', '2,605'), ('cells_missing_superclass', '516'), ('cells_missing_soma_neuromere', '145,304'),
                   ('cells_transmitter_unknown', '2,361')):
        add('mechanisms_v1', q.replace('_', ' '), rep, ci(float(s1a[('annotation', q)])), 'fig1a')
    for q, rep, doc in (('neurons', '167,106', 'method'), ('stored_connectome_entries', '25,578,600', 'method'), ('shared_parameter_groups', '11,770', 'method'),
                        ('input_cells', '3,200', 'mechanisms_v1'), ('observed_cells', '4,096', 'mechanisms_v1'), ('feature_dim', '512', 'method')):
        sec = 'readout' if q == 'feature_dim' else 'network'
        add(doc, q.replace('_', ' '), rep, ci(float(s1a[(sec, q)])), 'fig1a')
    b1 = R('main/fig1/fig1b_parameter_budget.csv')
    tot = b1[(b1.model_kind == 'connectome_linear') & (b1.component != 'total')]
    add('method', 'brain trainable parameters', '35,310', ci(tot[tot.component == 'brain'].n_elements.sum()), 'fig1b')
    add('method', 'readout trainable parameters', '2,052', ci(tot[(tot.component == 'readout') & tot.trainable_in_joint].n_elements.sum()), 'fig1b')
    mlp = b1[(b1.model_kind == 'board_mlp') & (b1.component != 'total')].n_elements.sum()
    add('readme', 'board MLP trainable parameters', '37,264', ci(mlp), 'fig1b')
    # Memory outcomes.
    a4 = R('main/fig4/fig4a_memory_training_outcomes.csv')
    c5 = R('extended/ed05/ed05c_cue_pair_classes.csv')

    def outcome(model, visible=None):
        x = c5[c5.model == model]
        if visible is not None:
            x = x[x.visible == visible]
        t, u = x[x.split == 'train'], x[x.split == 'untrained']
        return fr(t.histories_solved.sum(), t.histories.sum()), fr(u.histories_solved.sum(), u.histories.sum())

    for model, doc, rt, ru, vis in [
            ('remote-dynamic-M0-head_only', 'remote_results', '18/18', '26/52', None), ('remote-dynamic-M0-joint', 'mechanisms_v1', '15/18', '16/52', None),
            ('supp-M0-visibility', 'mechanisms_v1', '18/18', '27/52', False), ('supp-M0-visibility', 'mechanisms_v1', '18/18', '28/52', True),
            ('supp-M0-visibility', 'remote_results', '36/36', '55/104', None), ('supp-S1-ring-brain', 'remote_results', '18/18', '24/52', None),
            ('supp-origin-S1-ring-p05-fresh', 'remote_results', '18/18', '32/52', None), ('supp-S2-context-brain', 'remote_results', '29/36', '57/104', None),
            ('remote-dynamic-M0-carry_untrained', 'remote_final', '10/18', '24/52', None), ('remote-dynamic-M0-reset', 'remote_final', '9/18', '26/52', None),
            ('remote-dynamic-M0-gru', 'remote_final', '18/18', '42/52', None), ('supp-S1-ring-gru', 'supplement_analysis', '18/18', '42/52', None),
            ('supp-S2-context-gru', 'supplement_analysis', '36/36', '92/104', None), ('supp-origin-S1-ring-leaky', 'remote_results', '18/18', '43/52', None),
            ('supp-origin-S2-context-leaky', 'remote_results', '35/36', '78/104', None),
            ('supp-origin-S1-ring-neutral-real', 'supplement_analysis', '17/18', '26/52', None),
            ('supp-origin-S1-ring-neutral-rewired', 'supplement_analysis', '16/18', '27/52', None),
            ('supp-origin-S2-context-p05-fresh', 'supplement_analysis', '22/36', '51/104', None),
            ('supp-origin-S2-context-neutral-real', 'supplement_analysis', '19/36', '54/104', None),
            ('supp-origin-S2-context-neutral-rewired', 'supplement_analysis', '26/36', '54/104', None)]:
        t, u = outcome(model, vis)
        tag = '' if vis is None else (' (visible goal)' if vis else ' (hidden goal)')
        add(doc, f'{model}{tag}: trained histories solved', rt, t, 'ed05c')
        add(doc, f'{model}{tag}: untrained timings solved', ru, u, 'ed05c')
    add('remote_results', 'zero-update M0 -> S1 transfer', '36/70', '/'.join(map(str, [c5[c5.model == 'supp-transfer-S1-ring'].histories_solved.sum(),
                                                                                       c5[c5.model == 'supp-transfer-S1-ring'].histories.sum()])), 'ed05c')
    add('remote_results', 'zero-update M0 -> S2 transfer', '0/140', '/'.join(map(str, [c5[c5.model == 'supp-transfer-S2-context'].histories_solved.sum(),
                                                                                        c5[c5.model == 'supp-transfer-S2-context'].histories.sum()])), 'ed05c')
    a4i = a4.set_index(['model', 'visible'])
    for model, upd, mins in [('supp-S1-ring-brain', '280', '51.58'), ('supp-S2-context-brain', '196', '62.03'), ('supp-M0-visibility', '264', '10.78'),
                             ('supp-origin-S1-ring-p05-fresh', '276', '17.56'), ('supp-origin-S1-ring-neutral-real', '328', '20.17'),
                             ('supp-origin-S1-ring-neutral-rewired', '376', '20.20'), ('supp-origin-S2-context-p05-fresh', '28', '23.31'),
                             ('supp-origin-S2-context-neutral-real', '32', '21.97'), ('supp-origin-S2-context-neutral-rewired', '328', '66.05'),
                             ('remote-dynamic-M0-reset', '6304', '140.19'), ('remote-dynamic-M0-head_only', '272', '7.14'), ('remote-dynamic-M0-joint', '380', '20.28')]:
        r = a4i.loc[(model, False)]
        add('supplement_analysis' if model.startswith('supp') else 'remote_final', f'{model}: updates', upd, str(int(r['updates'])), 'fig4a')
        add('supplement_analysis' if model.startswith('supp') else 'remote_final', f'{model}: training-loop minutes', mins, f'{r["training_minutes"]:.2f}', 'fig4a')
    for model, rep, lost in [('supp-S2-context-brain', '+5.377', 2), ('supp-origin-S1-ring-neutral-real', '+1.982', 1),
                             ('supp-origin-S1-ring-neutral-rewired', '+2.792', 1), ('supp-origin-S2-context-p05-fresh', '+1.313', 3),
                             ('supp-origin-S2-context-neutral-real', '+0.829', 4), ('supp-origin-S2-context-neutral-rewired', '+3.988', 2)]:
        r = a4i.loc[(model, False)]
        add('supplement_analysis', f'{model}: entry-margin change at the last segment', rep, f'{r["entry_margin_change_at_last_segment"]:+.3f}', 'fig4a')
        add('supplement_analysis', f'{model}: required histories lost', str(lost), str(int(r['required_histories_lost'])), 'fig4a')
    ed5d = R('extended/ed05/ed05d_dense_delay_scan.csv')
    add('supplement_analysis', 'dense-scan decoder, training times', '18/18', fr(ed5d[ed5d.train_time].decoded_correct.sum(), ed5d.train_time.sum()), 'ed05d')
    add('supplement_analysis', 'dense-scan decoder, untrained times', '158/312', fr(ed5d[~ed5d.train_time].decoded_correct.sum(), (~ed5d.train_time).sum()), 'ed05d')
    ea = R('extended/ed05/ed05a_memory_behaviour_all.csv')
    lk = ea[(ea.model == 'supp-origin-S2-context-leaky') & (ea.split == 'train') & ~ea.solved]
    add('remote_results', 'leaky S2 remaining trained error (context, delay, prefix)', '1/160/0',
        ';'.join(f'{int(r.context)}/{int(r.delay_ms)}/{int(r.prefix_windows)}' for _, r in lk.iterrows()), 'ed05a')
    # Interventions.
    c4 = R('main/fig4/fig4c_state_interventions.csv')
    rsc = c4[(c4.source == 'remote_supplement_causal') & (c4.job == 'supp-M0-causal') & (c4.window == 'decision')]
    base = rsc[rsc.intervention == 'sham']  # sham at decision equals baseline in this job; baseline solved recorded per row
    bpairs = rsc.groupby(['delay_ms'])
    sub = rsc[rsc.groupby('delay_ms')['baseline_solved'].transform(lambda s: s.astype(bool).all())]
    for kind, rep in (('sham', '6/6'), ('restore', '6/6'), ('neural_clear', '3/6'), ('opposite_cue_swap', '0/6')):
        x = sub[sub.intervention == kind]
        add('remote_results', f'M0 causal probe, both-baselines subset: {kind}', rep, fr(x.solved.sum(), len(x)), 'fig4c')
    dv = R('main/fig4/fig4d_visibility_contrast.csv')
    m = dv[(dv.source == 'remote_supplement_causal') & (dv.window == 'decision') & (dv.intervention == 'neural_clear') & dv.timing_strict_subset]
    add('remote_results', 'visibility-matched clear: hidden goal', '3/6', fr(m.hidden_solved.sum(), len(m)), 'fig4d')
    add('remote_results', 'visibility-matched clear: visible goal', '6/6', fr(m.visible_solved.sum(), len(m)), 'fig4d')
    for kind, rep in (('clear_current', '3'), ('neural_clear', '5'), ('reset_each_window', '0'), ('restore', '7'), ('same_cue_swap', '8'), ('sham', '7'),
                      ('opposite_cue_swap', '3'), ('clear_voltage', '4')):
        x = rsc[rsc.intervention == kind]
        add('supplement_analysis', f'supp-M0-causal decision window {kind}: solved of 10 (baseline 7)', rep, str(int(x.solved.sum())), 'fig4c')
    e6 = R('extended/ed06/ed06a_intervention_factorial.csv')
    m1 = e6[(e6.source == 'mechanisms_v1') & (e6.window == 'read') & (e6.intervention != 'group_suppression')].set_index(['unit', 'intervention'])
    table = {'clear_voltage': ['6/12', '6/12', '7/12', '6/12', '12/24'], 'clear_current': ['7/12', '6/12', '9/12', '8/12', '12/24'],
             'clear_adaptation': ['8/12', '8/12', '7/12', '4/12', '10/24'], 'clear_resources': ['9/12', '9/12', '12/12', '11/12', '15/24'],
             'clear_delay_buffer': ['9/12', '9/12', '10/12', '8/12', '12/24'], 'clear_all_state': ['6/12', '6/12', '12/12', '6/12', '12/24'],
             'swap_all_state': ['3/12', '3/12', '1/12', '1/12', '9/24']}
    units = ['M0', 'M0vis.hidden', 'M0vis.visible', 'S1', 'S2']
    base_rep = ['9/12', '9/12', '12/12', '11/12', '15/24']
    for u, rep in zip(units, base_rep):
        r = m1.loc[(u, 'clear_voltage')]
        add('mechanisms_v1', f'{u}: baseline solved (intervention timings)', rep, fr(r.baseline_solved, r.histories), 'ed06a')
    for kind, reps in table.items():
        for u, rep in zip(units, reps):
            r = m1.loc[(u, kind)]
            add('mechanisms_v1', f'{u}: {kind} before reading', rep, fr(r.solved, r.histories), 'ed06a')
    allw = e6[(e6.source == 'mechanisms_v1') & (e6.intervention == 'clear_adaptation')].set_index(['unit', 'window'])
    for u, reps in (('M0', ['7', '7', '8']), ('S1', ['7', '4', '4'])):
        for w, rep in zip(('write', 'hold', 'read'), reps):
            add('mechanisms_v1', f'{u}: clear adaptation at {w}: solved of 12', rep, str(int(allw.loc[(u, w), 'solved'])), 'ed06a')
    s2 = m1.loc[('S2', 'clear_all_state')]
    add('mechanisms_v1', 'S2 clear all state before reading: lost/gained', '8/5', f'{int(s2.lost)}/{int(s2.gained)}', 'ed06a')
    s2 = m1.loc[('S2', 'clear_adaptation')]
    add('mechanisms_v1', 'S2 clear adaptation before reading: lost/gained', '10/5', f'{int(s2.lost)}/{int(s2.gained)}', 'ed06a')
    st5 = R('supplementary/st5_intervention_records.csv')
    add('mechanisms_v1', 'paired memory interventions (mechanisms v1)', '7,272', ci((st5.source == 'mechanisms_v1').sum()), 'st5')
    groups = R('extended/ed07/ed07a_group_selection.csv')
    gid = groups.set_index('type_labels')['group']
    e6g = R('main/fig6/fig6e_candidate_groups.csv')
    in17 = gid.get('IN17A052')
    for u, rep_c, rep_m in (('M0', '6/12', '9/12'), ('S1', '10/12', '10/12'), ('S2', '10/24', '15/24')):
        x = e6g[(e6g.unit == u) & (e6g.group == in17) & (e6g.window == 'hold') & (e6g.strength == 1.0) & e6g.set.str.startswith('memory')]
        cnd, mtc = x[x.side == 'candidate'].iloc[0], x[x.side == 'matched'].iloc[0]
        add('mechanisms_v1', f'IN17A052 full holding suppression on {u}: candidate', rep_c, fr(cnd.solved, cnd.histories), 'fig6e')
        add('mechanisms_v1', f'IN17A052 full holding suppression on {u}: matched', rep_m, fr(mtc.solved, mtc.histories), 'fig6e')
    gm = R('extended/ed07/ed07b_group_matching.csv').drop_duplicates('group').set_index('group')
    for label, rep in (('IN06B056', '0.0038'), ('IN19A016', '0.0142'), ('AVLP297', '0.0098'), ('MNad18,MNad27', '0.3077'), ('SLP324', '0'),
                       ('IN17A052', '0.0012')):
        g = gid[label]
        v = gm.loc[g, 'match_mean_distance']
        add('mechanisms_v1', f'{label} ({g}) mean standardized matching distance', rep, '0' if abs(v) < 5e-5 and rep == '0' else f'{v:.4f}', 'ed07b')
    f6 = R('main/fig6/fig6f_decoding_dynamics.csv')
    dyn = f6[f6.record_type == 'm1_dynamics_heldout_r2']
    for hz, rep_a in ((1, '0.954–0.957'), (20, '0.380–0.455'), (40, '-0.068 to -0.047'), (80, '-0.365 to -0.264')):
        x = dyn[(dyn.horizon_ms == hz) & (dyn.method == 'affine_autonomous')].value
        txt = f'{x.min():.3f}–{x.max():.3f}' if hz in (1, 20) else f'{x.min():.3f} to {x.max():.3f}'
        add('mechanisms_v1', f'affine model held-out R2 at {hz} ms (range)', rep_a, txt, 'fig6f')
    x = dyn[(dyn.horizon_ms == 1) & (dyn.method == 'persistence')].value
    add('mechanisms_v1', 'persistence baseline R2 at 1 ms (range)', '0.942–0.947', f'{x.min():.3f}–{x.max():.3f}', 'fig6f')
    td = f6[(f6.record_type == 'm1_time_decoder') & (f6.method == 'same_time_mean_accuracy')].value
    add('mechanisms_v1', 'same-time held-history decoding accuracy (range)', '50.4%–54.2%', f'{100 * td.min():.1f}%–{100 * td.max():.1f}%', 'fig6f')
    d5 = R('main/fig5/fig5d_cross_layout_transfer.csv')
    dec = d5[d5.record_type == 'cross_family_cue_decoding']
    for fam, rt, ru in (('M0', '18/18', '26/52'), ('S1', '8/18', '27/52'), ('S2', '19/36', '53/104')):
        t = dec[(dec.family == fam) & (dec.split == 'train')]
        u = dec[(dec.family == fam) & (dec.split == 'untrained')]
        add('mechanisms_v1', f'M0-fitted cue decoder on {fam}: trained timings', rt, fr(t.decoded_correct.sum(), len(t)), 'fig5d')
        add('mechanisms_v1', f'M0-fitted cue decoder on {fam}: untrained timings', ru, fr(u.decoded_correct.sum(), len(u)), 'fig5d')
    g6 = R('main/fig6/fig6c_readout_geometry.csv')
    gs = g6[~g6.visible].groupby(['model', 'phase']).agg(frac=('readout_energy_fraction', 'mean'), con=('centered_logit_contrast', 'mean'))
    add('mechanisms_v1', 'S2 readout fraction before training', '0.770%', f'{100 * gs.loc[("supp-S2-context-brain", "before"), "frac"]:.3f}%', 'fig6c')
    add('mechanisms_v1', 'S2 readout fraction after training', '0.460%', f'{100 * gs.loc[("supp-S2-context-brain", "after"), "frac"]:.3f}%', 'fig6c')
    add('mechanisms_v1', 'S2 centred-logit cue contrast before', '3.923', f'{gs.loc[("supp-S2-context-brain", "before"), "con"]:.3f}', 'fig6c')
    add('mechanisms_v1', 'S2 centred-logit cue contrast after', '2.655', f'{gs.loc[("supp-S2-context-brain", "after"), "con"]:.3f}', 'fig6c')
    add('mechanisms_v1', 'isotropic reference', '0.586%', f'{100 * 3 / 512:.3f}%', 'fig6c')
    t5 = R('main/fig5/fig5a_timing_profiles.csv')
    s1 = t5[(t5.model == 'supp-S1-ring-brain') & (t5.split == 'untrained')]
    for d, rep in ((320, '1/10'), (640, '8/10')):
        x = s1[s1.delay_ms == d]
        add('mechanisms_v1', f'S1 untrained timings solved at {d} ms', rep, fr(x.numerator.sum(), x.denominator.sum()), 'fig5a')
    # Supplement: state table, direction, dose, components, norm.
    d6 = R('main/fig6/fig6d_component_interventions.csv').set_index(['unit', 'window', 'component', 'fraction'])
    ms_units = ['M0', 'M0joint', 'M0vis.hidden', 'M0vis.visible', 'S1', 'S2']
    state = {'M0': ('9/12', '7/12', '9/12', '4/12', '8/12'), 'M0joint': ('8/12', '7/12', '8/12', '5/12', '9/12'),
             'M0vis.hidden': ('9/12', '7/12', '10/12', '4/12', '10/12'), 'M0vis.visible': ('12/12', '8/12', '8/12', '2/12', '11/12'),
             'S1': ('11/12', '7/12', '5/12', '2/12', '9/12'), 'S2': ('15/24', '11/24', '12/24', '9/24', '13/24')}
    for u in ms_units:
        mid = d6.loc[(u, 'read', 'continuous_combined', 0.5)]
        sw = d6.loc[(u, 'read', 'continuous_combined', 1.0)]
        got = (fr(mid.baseline_solved, mid.histories), fr(mid.cue_solved, mid.histories), fr(mid.scrambled_solved, mid.histories),
               fr(sw.cue_solved, sw.histories), fr(sw.scrambled_solved, sw.histories))
        for name, rep, g in zip(('baseline', 'cue midpoint', 'midpoint matched control', 'cue swap', 'swap matched control'), state[u], got):
            add('mechanisms_supplement', f'{u}: {name} (read window, combined state)', rep, g, 'fig6d')
    e4 = R('main/fig4/fig4e_directional_transfer.csv')
    direction = {'M0': ('6/8', '0/8'), 'M0joint': ('5/6', '0/6'), 'M0vis.hidden': ('6/8', '0/8'), 'M0vis.visible': ('10/12', '1/12'),
                 'S1': ('9/10', '2/10'), 'S2': ('8/10', '2/10')}
    for u, (rc, rs_) in direction.items():
        x = e4[(e4.unit == u) & e4.eligible_both_baselines_solved]
        c, s = x[x.kind == 'cue'], x[x.kind == 'scrambled']
        add('mechanisms_supplement', f'{u}: follows donor after cue swap (eligible)', rc, fr(c.follows_donor.sum(), len(c)), 'fig4e')
        add('mechanisms_supplement', f'{u}: follows donor after scrambling (eligible)', rs_, fr(s.follows_donor.sum(), len(s)), 'fig4e')
    for frac, rep in ((0.25, '5/12'), (0.5, '7/12'), (1.0, '4/12')):
        r = d6.loc[('M0', 'read', 'continuous_combined', frac)]
        add('mechanisms_supplement', f'M0 readout-only dose {frac}: solved', rep, fr(r.cue_solved, r.histories), 'fig6d')
    ms_e = e6[(e6.source == 'mechanisms_supplement') & (e6.intervention == 'interpolate_cue') & (e6.window == 'read') &
              (e6.component == 'continuous_combined')].set_index(['unit', 'fraction'])
    r = ms_e.loc[('S2', 0.25)]
    add('mechanisms_supplement', 'S2 dose 0.25: solved', '15/24', fr(r.solved, r.histories), 'ed06a')
    add('mechanisms_supplement', 'S2 dose 0.25: lost/gained', '4/4', f'{int(r.lost)}/{int(r.gained)}', 'ed06a')
    comps = [('M0', 'read', 'voltage', '6/12', '9/12'), ('M0', 'read', 'current', '6/12', '9/12'), ('M0', 'read', 'adaptation', '8/12', '8/12'),
             ('M0joint', 'read', 'adaptation', '6/12', '10/12'), ('M0joint', 'write', 'adaptation', '3/12', '8/12'), ('S1', 'read', 'adaptation', '9/12', '7/12'),
             ('S1', 'read', 'continuous_combined', '2/12', '9/12'), ('S2', 'read', 'continuous_combined', '9/24', '13/24'),
             ('S2', 'hold', 'continuous_combined', '13/24', '12/24')]
    for u, w, comp, rc, rs_ in comps:
        r = d6.loc[(u, w, comp, 1.0)]
        add('mechanisms_supplement', f'{u}: {comp} swap at {w}', rc, fr(r.cue_solved, r.histories), 'fig6d')
        add('mechanisms_supplement', f'{u}: {comp} matched control at {w}', rs_, fr(r.scrambled_solved, r.histories), 'fig6d')
    nb = R('extended/ed06/ed06b_norm_audit.csv')
    add('mechanisms_supplement', 'maximum relative norm error', '0.5294%', f'{100 * nb.max_relative_norm_error.max():.4f}%', 'ed06b')
    add('mechanisms_supplement', 'maximum absolute direction cosine', '0.01815', f'{nb.max_absolute_direction_cosine.max():.5f}', 'ed06b')
    rt = st5[st5.intervention.astype(str).str.startswith('route_')]
    nonzero = rt[rt.strength > 0]
    add('mechanisms_supplement', 'channel-intervention histories without any solution loss or gain', '3,024',
        ci(((nonzero.solved.astype(str).str.lower() == nonzero.baseline_solved.astype(str).str.lower())).sum()), 'st5')
    hd = f6[(f6.record_type == 'held_history_cue_decoding') & (f6.split == 'held_time')]
    for (model, vis), rep in ((('remote-dynamic-M0-head_only', False), '26/62'), (('remote-dynamic-M0-joint', False), '35/62'),
                              (('supp-M0-visibility', False), '26/62'), (('supp-M0-visibility', True), '43/62'), (('supp-S1-ring-brain', False), '35/62'),
                              (('supp-S2-context-brain', False), '65/124')):
        x = hd[(hd.model == model) & (hd.visible == vis)]
        add('mechanisms_supplement', f'{model}{" visible" if vis else ""}: held-out-duration cue decoding', rep, fr(x.correct.sum(), len(x)), 'fig6f')
    icd = f6[(f6.record_type == 'input_conditioned_dynamics_r2') & (f6.horizon_ms == 40) & (f6.split == 'held_time')]
    dyn_rep = {'remote-dynamic-M0-head_only': ('-0.836', '-0.082', '0.375'), 'remote-dynamic-M0-joint': ('-0.860', '0.012', '0.437'),
               'supp-S1-ring-brain': ('-0.822', '-0.001', '0.429'), 'supp-S2-context-brain': ('-0.864', '-0.014', '0.269')}
    for model, reps in dyn_rep.items():
        for meth, rep in zip(('persistence', 'affine', 'input_phase'), reps):
            x = icd[(icd.model == model) & (icd.method == meth)].value
            add('mechanisms_supplement', f'{model}: 40-ms R2 ({meth})', rep, f'{x.iloc[0]:.3f}' if len(x) else 'missing', 'fig6f')
    lostgain = {0.5: {'M0': '4/2', 'M0joint': '3/2', 'M0vis.hidden': '4/2', 'M0vis.visible': '4/0', 'S1': '4/0', 'S2': '6/2'},
                1.0: {'M0': '6/1', 'M0joint': '6/3', 'M0vis.hidden': '6/1', 'M0vis.visible': '10/0', 'S1': '9/0', 'S2': '10/4'}}
    for frac, reps in lostgain.items():
        for u, rep in reps.items():
            r = ms_e.loc[(u, frac)]
            add('mechanisms_supplement', f'{u}: {"cue midpoint" if frac == 0.5 else "cue swap"} lost/gained vs baseline', rep, f'{int(r.lost)}/{int(r.gained)}', 'ed06a')
    tm = R('main/fig5/fig5c_matched_duration.csv')

    def tcount(unit, ms, kind, windows):
        x = tm[(tm.unit == unit) & (tm.total_postcue_ms == ms) & (tm.input_kind == kind) & (tm.prefix_windows == windows)]
        return fr(x.solved.sum(), len(x))

    for unit, ms, kind, w, rep in [('M0', 240, 'blank', 0, '2/2'), ('M0', 240, 'neutral_stationary', 2, '0/2'), ('M0', 240, 'neutral_walk_0', 2, '0/2'),
                                   ('M0', 240, 'neutral_walk_1', 2, '0/2'), ('M0', 480, 'blank', 0, '0/2'), ('M0', 480, 'neutral_stationary', 2, '2/2'),
                                   ('S2', 80, 'blank', 0, '2/4'), ('S2', 80, 'neutral_stationary', 2, '4/4'), ('S2', 160, 'blank', 0, '4/4'),
                                   ('S2', 160, 'neutral_stationary', 2, '2/4')]:
        add('mechanisms_supplement', f'{unit} {ms} ms {kind} ({w} windows)', rep, tcount(unit, ms, kind, w), 'fig5c')
    seq = [tcount('M0joint', ms, 'blank', 0) for ms in (80, 120, 160, 240, 320, 480, 640)]
    add('mechanisms_supplement', 'M0 joint blank-only outcomes over increasing durations', '1/2, 2/2, 2/2, 0/2, 1/2, 1/2, 0/2', ', '.join(seq), 'fig5c')
    for t in range(8):
        x = a2[a2.tier == t]
        add('mechanisms_v1', f'D{t} maps solved after reload', '3/3', fr((x.reload_status == 'solved').sum(), len(x)), 'fig2a')
    add('mechanisms_v1', 'mean cells with spikes per window', '4,923/167,106', f'{ci(allp.active_cells.mean())}/{ci(167106)}', 'fig6a')
    for layer, doc, rep, statuses in (('mechanisms_v1', 'mechanisms_v1', '55/55', ('complete',)), ('mechanisms_supplement', 'mechanisms_supplement', '20/20', ('complete',)),
                                      ('remote_batch1', 'remote_final', '16/16', None), ('remote_supplement', 'remote_results', '19/19', None)):
        x = st1[st1.layer == layer]
        ended = x if statuses is None else x[x.status.isin(statuses)]
        add(doc, f'{layer}: queue entries or jobs ended', rep, fr(len(ended), len(x)), 'st1')
    rec = st1[(st1.layer == 'mechanisms_v1') & st1.run_id.str.startswith('record-')]
    add('mechanisms_v1', 'existing-model record jobs completed', '39/39', fr((rec.status == 'complete').sum(), len(rec)), 'st1')
    tr_rows = st1[(st1.layer == 'remote_supplement') & (st1.module == 'train')]
    add('supplement_analysis', 'supplementary training conditions accepted after independent reload', '6/13',
        fr(tr_rows.accepted.astype(str).str.lower().eq('true').sum(), len(tr_rows)), 'st1')
    l1 = R('extended/ed01/ed01b_formal_lineage.csv')
    lv = l1.set_index(['model', 'item'])['value']
    for name, upd, sec, mv in (('20261002-p03-01', '216', '410.156', '22'), ('20261002-p05-01', '384', '771.891', '34')):
        add('readme', f'{name}: joint updates', upd, str(json.loads(lv[(name, 'training.joint_updates')])), 'ed01b')
        add('readme', f'{name}: training seconds', sec, f'{json.loads(lv[(name, "training.training_seconds")]):.3f}', 'ed01b')
        add('readme', f'{name}: solution moves', mv, str(json.loads(lv[(name, 'after.original_steps')])), 'ed01b')
    df = pd.DataFrame(L)
    df.insert(0, 'claim_id', [f'C{i + 1:03d}' for i in range(len(df))])
    df['document_path'] = df['document'].map(lambda d: paths.rel(DOCS[d]))
    return df


def _tokens(values):
    out = set()
    for v in values:
        for piece in [str(v)] + [p.strip() for p in str(v).split(',') if '/' in p]:
            parts = piece.split('/')
            out.add(piece)
            for i in range(len(parts) - 1):
                out.add(f'{parts[i]}/{parts[i + 1]}')
    return out


NOT_CLAIMS = {
    ('mechanisms_v1', '14/40'): 'feature snapshot times (ms), not a count',
    ('mechanisms_v1', '40/160'): 'delay values (ms), not a count',
    ('mechanisms_v1', '40/80'): 'prediction horizons (ms), not a count',
    ('mechanisms_supplement', '40/160'): 'delay values (ms), not a count',
    ('mechanisms_supplement', '0/4'): 'prefix values (0 and 4 windows), not a count',
    ('mechanisms_supplement', '120/240'): 'held-out durations (ms), not a count',
    ('mechanisms_supplement', '480/640'): 'held-out durations (ms), not a count',
    ('mechanisms_supplement', '18/36'): 'reference to the training-history denominators quoted above',
    ('static_analysis_report', '0/4'): 'snapshot status written before the mechanism studies (superseded)',
    ('static_analysis_report', '0/15'): 'snapshot status written before the remote batches (superseded)',
}
NUM = r'(?:\d{1,3}(?:,\d{3})+|\d+)'


def coverage(ledger):
    rows = []
    every = _tokens(ledger['reported'])
    for doc, p in DOCS.items():
        text = io.text(p)
        found = {}
        for m in re.finditer(rf'(?<![\w.,/]){NUM}/{NUM}(?![\w./]|,\d)', text):
            found[m.group(0)] = found.get(m.group(0), 0) + 1
        same = _tokens(ledger.loc[ledger.document == doc, 'reported'])
        for frac, n in sorted(found.items()):
            rows.append(dict(document=doc, fraction=frac, occurrences=n, in_ledger_same_document=frac in same,
                             in_ledger_any_document=frac in every, not_a_claim=NOT_CLAIMS.get((doc, frac)), document_path=paths.rel(p)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- discrepancies and gaps

DISCREPANCIES = [
    ('stale_record', 'remote-20261003/memory-analysis.json', 'The reset run shows a stale progress block (training, 4,176 updates) next to its final status (stagnation, 6,304 updates).',
     'Final values taken from result.json and verification.json.', 'fig4a, fig4b, m4'),
    ('report_inconsistency', 'remote-20261003/final.en.md vs progress.en.md', 'carry_untrained training time is reported as 0 minutes in one report and 1.01 minutes in the other.',
     'result.json training_seconds (60.46 s) is used; the evaluation-only nature is stated.', 'fig4a, m4'),
    ('vocabulary', 'remote batches', 'Irreversible failures are labelled wrong_irreversible_push (batch 1) and irreversible_deadlock (supplement).',
     'Harmonized to irreversible_failure; raw labels kept (raw_status).', 'ed05a, st5'),
    ('field_semantics', 'remote-supplement-20261003 training rollouts', 'The cue field holds the case index; the true cue is cue_label.', 'cue_label used.', 'ed05a, fig5b'),
    ('missing_field', 'remote-20261003 verification rollouts', 'No first-push field is recorded.', 'First push derived from decisions (DELTAS rule).', 'ed05a, fig4b'),
    ('field_semantics', 'remote-20261003 interventions', 'The "all" intervention is a full reset including clock and RNG, unlike the supplement neural_clear.',
     'Mapped to full_reset in the taxonomy.', 'fig4c, st5'),
    ('stale_record', 'remote-supplement-20261003 probe jobs', 'progress.json of completed probe jobs still reads "evaluating".', 'Completion taken from result.json and observations.', 'st1'),
    ('hand_edited_report', 'remote-supplement-20261003/analysis.en.md', 'The report was hand-edited after generation.',
     'All its numbers were re-derived from raw records (claims ledger).', 'qa/claims_ledger.csv'),
    ('path_semantics', 'remote JSON records', 'Embedded paths refer to the remote machine (e.g. /root/autodl-tmp/...).', 'Never resolved; local paths used.', 'all'),
    ('documented_output_missing', 'studies/brain-sokoban-v1/analysis/analysis-report.en.md', 'generalization.csv and memory-delays.csv are cited but were never produced; the study README cites activity/, which does not exist.',
     'Equivalent data re-derived from remote-20261003/results/generalization-* and the memory records.', 'fig2e, fig5a'),
    ('location', 'runs/difficulty-D1-T0/generalization.json', 'The generalization data referenced locally live under remote-20261003/results/generalization-D*.', 'Remote copies used.', 'fig2e'),
    ('interruption', 'runs/difficulty-D6-T1, runs/difficulty-D7-T0', 'Compute counters and wall_seconds cover only the resumed process; gradients.jsonl repeats update numbers around the resume.',
     'Forward calls rebuilt from logs (gate passed); duplicates removed (last kept); wall_seconds dropped; training_seconds flagged.', 'fig2c, ed02b, meta/compute_reconstruction.csv'),
    ('code_version', 'runs/difficulty-D0-T0..D1-T1', 'Five early runs used four older code versions with the same protocol hash (the only training change adds an option used by memory runs).',
     'Listed in meta/code_versions.csv; runs retained.', 'fig2, meta/code_versions.csv'),
    ('benchmark_mixture', 'runs/benchmark-*', 'Eight of ten benchmark runs used the v1 task lock; the FP64 reference total combines a v2 (C0) and a v3 (C1) run.', 'Reported as recorded.', 'ed02a'),
    ('sentinel', 'runs/*/history.jsonl', 'Margins contain a 1e9 sentinel when no competing legal action exists (D3-T0, D5-T0, D7-*).', 'Excluded from margin statistics and counted.', 'fig1c, fig2b, ed03a'),
    ('checkpoint_metadata', 'runs/*/model.pt', 'Checkpoint-embedded config.code_version differs from run config.json (English-edition fingerprint renewal).', 'config.json used for code versions.', 'meta/code_versions.csv'),
    ('benchmark_failure', 'runs/benchmark-v2-fp32-b16-C0', 'The FP32 candidate exceeded the recomputation tolerance (the recorded reason it was rejected).', 'Reported in ED2b; not gated.', 'ed02b'),
]

GAPS = [
    ('never_run', 'D7 training controls', '10 registered D7 controls; 2 equivalents (readout-only, demonstration) ran remotely under other IDs; D7 neutral stopped', 'fig3a, fig3b'),
    ('never_run', 'curriculum', '7 registered continual-curriculum runs', 'no claim on continual learning'),
    ('never_run', 'transfer retraining', '8 registered transfer-by-retraining runs', 'no claim on retraining transfer'),
    ('never_run', 'parameter memory (A->B->A)', '6 registered runs', 'no claim on forgetting or relearning'),
    ('never_run', 'memory tasks M1/M2', 'defined in tasks/memory.json, never trained or evaluated', 'memory claims limited to M0/S1/S2'),
    ('needs_gpu', 'mid-checkpoint brain/readout swaps', 'behaviour for pairings not replayed', 'fig3c limited to before/after pairings'),
    ('needs_gpu', 'S2 cross-orientation swaps', 'donor and recipient always share orientation', 'fig4e scope'),
    ('needs_gpu', 'additional doses, groups or scrambled draws', 'one scrambled draw per condition', 'fig4e, fig4f, fig6d sampling uncertainty of the control not captured'),
    ('needs_gpu', 'current-flow recordings and effective-matrix pathway sums', 'one recorded event; effective matrix not saved', 'ed07c scope'),
    ('needs_gpu', 'behaviour at stage-aligned checkpoints', 'only parameters are compared', 'ed04a stage rows are parameter-only'),
    ('not_recorded', 'pre-interruption counters and wall clock', 'D6-T1, D7-T0', 'forward calls rebuilt; wall clock dropped'),
    ('not_applicable', 'neuron-ms for board MLP, GRU and fixed trace', 'no connectome simulation', 'fig3b, m4'),
    ('single_seed', 'all results', 'seed 10 only; no between-seed variance', 'all inferences are within this lineage'),
]


def main():
    st2()
    ledger = claims()
    csvout.write_table(ledger, 'qa/claims_ledger.csv', title='Claims ledger: every re-derived quantitative statement of the existing reports', describe={
        'claim_id': 'Claim identifier', 'document': 'Source report', 'statement': 'Statement (paraphrased)', 'reported': 'Value as reported',
        'rederived': 'Value re-derived from the raw records through this analysis', 'match': 'Re-derived value equals the reported value',
        'rederived_from': 'Deliverable used', 'document_path': 'Report path'}, key=['claim_id'])
    cov = coverage(ledger)
    csvout.write_table(cov, 'qa/claims_coverage.csv', title='Regex sweep of k/n fractions in the reports and their ledger coverage', describe={
        'document': 'Report', 'fraction': 'Fraction string found', 'occurrences': 'Occurrences in the report',
        'in_ledger_same_document': 'Covered by a ledger claim from the same report', 'in_ledger_any_document': 'Covered by any ledger claim',
        'not_a_claim': 'Reason the string is not a quantitative claim (durations, superseded snapshot statements)',
        'document_path': 'Report path'}, key=['document', 'fraction'])
    disc = pd.DataFrame(DISCREPANCIES, columns=['category', 'location', 'description', 'handling', 'affects'])
    mism = ledger[~ledger.match]
    for _, r in mism.iterrows():
        disc.loc[len(disc)] = ['claim_mismatch', r.document_path, f'{r.claim_id} "{r.statement}": reported {r.reported}, re-derived {r.rederived}',
                               'Re-derived value used in the deliverables; source report not changed.', r.rederived_from]
    disc.insert(0, 'discrepancy_id', [f'D{i + 1:02d}' for i in range(len(disc))])
    csvout.write_table(disc, 'qa/discrepancies.csv', title='Discrepancies found in the inputs and how they were handled', describe={
        'discrepancy_id': 'Identifier', 'category': 'Category', 'location': 'Where', 'description': 'What', 'handling': 'How this analysis handles it',
        'affects': 'Affected deliverables'}, key=['discrepancy_id'])
    gaps = pd.DataFrame(GAPS, columns=['reason', 'item', 'detail', 'consequence'])
    gaps.insert(0, 'gap_id', [f'G{i + 1:02d}' for i in range(len(gaps))])
    csvout.write_table(gaps, 'qa/gaps.csv', title='Evidence gaps (reported, never imputed)', describe={
        'gap_id': 'Identifier', 'reason': 'needs_gpu, never_run, not_recorded, not_applicable or single_seed', 'item': 'Missing evidence',
        'detail': 'Detail', 'consequence': 'Consequence for the claims'}, key=['gap_id'])
