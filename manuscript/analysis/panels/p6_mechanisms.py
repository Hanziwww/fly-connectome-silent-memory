"""P6: whole-brain atlas (6a), cue-difference coverage (6b), readout geometry (6c), decoding and dynamics (6f), ED8, ED9.
Brain-level statistics are computed once per distinct state set (the shared-readout hidden-goal table duplicates
the M0 readout-only table and is dropped)."""
import json

import numpy as np
import pandas as pd

from msa import csvout, io, paths, stats

ISOTROPIC = 3 / 512
THRESH = 1e-6
STATE_SETS = [  # model, context, visibility, label
    ('remote-dynamic-M0-head_only', 0, 0, 'M0 readout-only (hidden)'),
    ('remote-dynamic-M0-joint', 0, 0, 'M0 joint (hidden)'),
    ('supp-M0-visibility', 0, 1, 'M0 shared readout (visible)'),
    ('supp-S1-ring-brain', 0, 0, 'S1 readout-only (hidden)'),
    ('supp-S2-context-brain', 0, 0, 'S2 readout-only, orientation 0 (hidden)'),
    ('supp-S2-context-brain', 1, 0, 'S2 readout-only, orientation 180 (hidden)'),
]
DUPLICATES = [('supp-M0-visibility', 0, 0, 'remote-dynamic-M0-head_only')]


def _cellsets(df):
    inp = df['direct_input'].astype(bool).to_numpy()
    obs = df['observed_by_policy'].astype(bool).to_numpy()
    return {'all': np.ones(len(df), bool), 'input': inp, 'observed': obs, 'other': ~inp & ~obs}


def fig6a():
    maps = sorted(p.stem.replace('-before', '') for p in (paths.M1 / 'atlas').glob('difficulty-*-before.parquet'))
    saved_p = paths.M1 / 'interpretation' / 'neuron-weighted-atlas.csv'
    saved = io.csv_load(saved_p).set_index(['model', 'phase', 'population'])
    rows = []
    cols = ['direct_input', 'observed_by_policy', 'spikes_mean', 'v_mean']
    for m in maps:
        b = io.parquet_load(paths.M1 / 'atlas' / f'{m}-before.parquet', columns=cols)
        a = io.parquet_load(paths.M1 / 'atlas' / f'{m}-after.parquet', columns=cols)
        dv = np.abs(a['v_mean'].to_numpy(np.float64) - b['v_mean'].to_numpy(np.float64))
        for phase, df in (('before', b), ('after', a)):
            for pop, mask in _cellsets(df).items():
                spikes = df['spikes_mean'].to_numpy(np.float64)[mask]
                ch = dv[mask] if phase == 'after' else np.zeros(mask.sum())
                s = saved.loc[(m, phase, pop)] if (m, phase, pop) in saved.index else None
                rows.append(dict(map_id=m.replace('difficulty-', ''), model=m, phase=phase, population=pop, cells=int(mask.sum()),
                                 active_cells=int((spikes > 0).sum()), spike_probability_per_ms=float(spikes.mean()),
                                 mean_abs_voltage_change_mv=float(ch.mean()), voltage_change_p50_mv=float(np.median(ch)),
                                 voltage_change_p95_mv=float(np.quantile(ch, 0.95)),
                                 matches_saved_atlas=(None if s is None else bool(int(s['active_cells']) == int((spikes > 0).sum()) and
                                                                                   abs(float(s['mean_abs_voltage_change']) - float(ch.mean())) < 1e-6)),
                                 analysis_status='re_derived', source_file=f'{paths.rel(paths.M1 / "atlas")}/{m}-{{before,after}}.parquet'))
    df = pd.DataFrame(rows)
    if df['matches_saved_atlas'].eq(False).any():
        raise RuntimeError('fig6a: re-derived atlas disagrees with neuron-weighted-atlas.csv')
    csvout.write_table(df, 'main/fig6/fig6a_activity_sparsity.csv', title='Spiking sparsity and matched-state voltage change by cell set', describe={
        'phase': 'before (P05 start) or after (final) parameters at the same matched reference states', 'population': 'all, input, observed or other cells',
        'cells': 'Cells in the set', 'active_cells': 'Cells with any recorded spike in the 40-ms windows',
        'spike_probability_per_ms': 'Mean per-cell spike probability per ms', 'mean_abs_voltage_change_mv': 'Mean |v_after - v_before| (mV; 0 for before rows)',
        'voltage_change_p50_mv': 'Median |voltage change| (mV)', 'voltage_change_p95_mv': '95th percentile |voltage change| (mV)',
        'matches_saved_atlas': 'Active cells and mean change equal interpretation/neuron-weighted-atlas.csv'}, key=['model', 'phase', 'population'])
    aft = df[df.phase == 'after'].pivot(index='model', columns='population', values='mean_abs_voltage_change_mv')
    for pop in ('observed', 'input'):
        x = (aft[pop] - aft['other']).to_numpy()
        stats.record(stats.signed_rank(x, f'fig6a.wilcoxon.{pop}_vs_other'), test_id=f'fig6a.wilcoxon.{pop}_vs_other', panel='fig6a', family='F6a',
                     contrast=f'mean |voltage change|: {pop} minus other cells', unit='map', level='A', analysis_status='new_registered',
                     source_table='main/fig6/fig6a_activity_sparsity.csv')


def memory_cells(model):
    return io.parquet_load(paths.M1 / 'interpretation' / f'{model}-memory-cells-d160-p4.parquet')


def fig6b():
    saved = io.csv_load(paths.M1 / 'interpretation' / 'memory-cell-summary.csv')
    rows = []
    cache = {}
    for model, ctx, vis, label in STATE_SETS:
        df = cache.setdefault(model, memory_cells(model))
        sets = _cellsets(df)
        for var in ('v', 'g', 'adapt', 'res'):
            col = f'c{ctx}-visible{vis}-delta-{var}'
            d = np.abs(df[col].to_numpy(np.float64))
            for pop, mask in sets.items():
                x = d[mask]
                rows.append(dict(model=model, context=ctx, visible=bool(vis), state_set=label, variable=var, population=pop, cells=int(mask.sum()),
                                 cells_above_threshold=int((x > THRESH).sum()), threshold=THRESH, mean_abs_difference=float(x.mean()),
                                 median_abs_difference=float(np.median(x)), p95_abs_difference=float(np.quantile(x, 0.95)), duplicate_of=None,
                                 analysis_status='re_derived', source_file=paths.rel(paths.M1 / 'interpretation' / f'{model}-memory-cells-d160-p4.parquet')))
    for model, ctx, vis, dup in DUPLICATES:
        a = memory_cells(model)
        b = cache[dup]
        same = all(np.array_equal(a[f'c{ctx}-visible{vis}-delta-{v}'].to_numpy(), b[f'c0-visible0-delta-{v}'].to_numpy()) for v in ('v', 'g', 'adapt', 'res'))
        rows.append(dict(model=model, context=ctx, visible=bool(vis), state_set='M0 shared readout (hidden)', variable='all', population='all',
                         cells=len(a), cells_above_threshold=None, threshold=THRESH, mean_abs_difference=None, median_abs_difference=None,
                         p95_abs_difference=None, duplicate_of=(dup if same else None), analysis_status='re_derived',
                         source_file=paths.rel(paths.M1 / 'interpretation' / f'{model}-memory-cells-d160-p4.parquet')))
    df = pd.DataFrame(rows)
    df['cells_above_threshold'] = df['cells_above_threshold'].astype('Int64')
    chk = df[(df.population == 'all') & (df.variable == 'v')].merge(saved, on=['model', 'context', 'visible'], how='left')
    if (chk['cells_above_threshold'] != chk['nonzero_voltage_cue']).any():
        raise RuntimeError('fig6b: coverage disagrees with memory-cell-summary.csv')
    csvout.write_table(df, 'main/fig6/fig6b_cue_difference_coverage.csv', title='Coverage of cue-related state differences by variable (distinct state sets)', describe={
        'state_set': 'Distinct state set (brain x task x visibility x orientation)', 'variable': 'v voltage, g current, adapt adaptation, res synaptic resources',
        'population': 'Cell set', 'cells': 'Cells', 'cells_above_threshold': 'Cells with |cue1 - cue0| > threshold in the last recorded window (160 ms delay, 4 prefixes)',
        'threshold': 'Numerical threshold', 'mean_abs_difference': 'Mean |difference|', 'median_abs_difference': 'Median |difference|',
        'p95_abs_difference': '95th percentile |difference|', 'duplicate_of': 'State set identical to this model (counted once)'},
        key=['model', 'context', 'visible', 'variable', 'population'])
    return cache


def fig6c():
    p = paths.M1 / 'tables' / 'memory-geometry.csv'
    g = io.csv_load(p)
    g['visible'] = g['visible'].astype(str).str.lower() == 'true'
    g['split'] = g['split'].map({'train': 'train', 'heldout': 'untrained'})
    g['isotropic_reference'] = ISOTROPIC
    g['analysis_status'] = 're_derived'
    g['source_file'] = paths.rel(p)
    csvout.write_table(g, 'main/fig6/fig6c_readout_geometry.csv', title='Readout row-space fraction of cue-difference energy', describe={
        'phase': 'before, mid or after (readout parameters)', 'delay': 'Delay (ms)', 'prefix': 'Prefix windows',
        'cue_distance': 'Euclidean distance between the two cue feature vectors (normalized, clipped features)',
        'readout_energy_fraction': 'Fraction of squared cue distance inside the centred readout row space',
        'centered_logit_contrast': 'Norm of the centred logit difference between cues', 'isotropic_reference': 'Expected fraction for an isotropic difference (3/512)'},
        key=['model', 'phase', 'delay', 'prefix', 'context', 'visible'])
    for (model, vis), x in g[g.phase == 'after'].groupby(['model', 'visible']):
        sfx = '.visible' if vis else ''
        stats.record(stats.sign_test((x['readout_energy_fraction'] - ISOTROPIC).to_numpy()), test_id=f'fig6c.sign.{model}{sfx}', panel='fig6c',
                     family='F6c', contrast=f'{model}{sfx}: readout fraction minus isotropic 3/512 over timings (final readout)', unit='timing', level='B',
                     analysis_status='new_registered', source_table='main/fig6/fig6c_readout_geometry.csv')


def fig6f():
    rows = []
    dp = paths.MS / 'analysis' / 'held-history-cue-decoding.csv'
    d = io.csv_load(dp)
    d['visible'] = d['visible'].astype(str).str.lower() == 'true'
    for _, r in d.iterrows():
        rows.append(dict(record_type='held_history_cue_decoding', model=r['model'], visible=bool(r['visible']), split=r['split'], case=r['case'],
                         condition=r['condition'], horizon_ms=None, method='ridge_cue_decoder', value=float(r['decoder_score']),
                         correct=bool(r['correct']), solved=bool(r['solved']), total_postcue_ms=r['total_postcue_ms'], source_file=paths.rel(dp)))
    for (m, vis, split), x in d.groupby(['model', 'visible', 'split']):
        pairs = x.groupby(['context', 'condition'])['correct'].sum()
        both, none = int((pairs == 2).sum()), int((pairs == 0).sum())
        sfx = '.visible' if vis else ''
        stats.record(stats.sign_test(np.r_[np.ones(both), -np.ones(none)]), test_id=f'fig6f.decoder_pairs.{m}{sfx}.{split}', panel='fig6f',
                     family='F6f.decoding', contrast=f'{m}{sfx} ({split}): cue pairs decoded both-correct vs both-wrong', unit='cue_pair', level='B',
                     analysis_status='post_hoc', source_table='main/fig6/fig6f_decoding_dynamics.csv')
    ip = paths.MS / 'analysis' / 'input-conditioned-dynamics.csv'
    for _, r in io.csv_load(ip).iterrows():
        rows.append(dict(record_type='input_conditioned_dynamics_r2', model=r['model'], visible=None, split=r['split'], case=None, condition=None,
                         horizon_ms=int(r['horizon_ms']), method=r['method'], value=float(r['r2']), correct=None, solved=None, total_postcue_ms=None,
                         source_file=paths.rel(ip)))
    pp = paths.M1 / 'interpretation' / 'dynamics-persistence-check.csv'
    for _, r in io.csv_load(pp).iterrows():
        for method, col in (('affine_autonomous', 'affine_r2'), ('persistence', 'persistence_r2')):
            rows.append(dict(record_type='m1_dynamics_heldout_r2', model=r['model'], visible=None, split='heldout_histories', case=None, condition=None,
                             horizon_ms=int(r['horizon_ms']), method=method, value=float(r[col]), correct=None, solved=None, total_postcue_ms=None,
                             source_file=paths.rel(pp)))
    tp = paths.M1 / 'tables' / 'time-decoders.json'
    for r in io.json_load(tp):
        for key in ('same_time_mean_accuracy', 'all_time_mean_accuracy'):
            rows.append(dict(record_type='m1_time_decoder', model=r['model'], visible=None, split='heldout_histories', case=None, condition=None,
                             horizon_ms=None, method=key, value=float(r[key]), correct=None, solved=None, total_postcue_ms=None, source_file=paths.rel(tp)))
    df = pd.DataFrame(rows)
    df.insert(0, 'row', range(len(df)))
    df['analysis_status'] = 're_derived'
    csvout.write_table(df, 'main/fig6/fig6f_decoding_dynamics.csv', title='Cue decoding (scored by cue pairs) and input-driven low-dimensional dynamics', describe={
        'row': 'Row index', 'record_type': 'held_history_cue_decoding, input_conditioned_dynamics_r2, m1_dynamics_heldout_r2 or m1_time_decoder',
        'condition': 'Supplement condition (duration and input)', 'horizon_ms': 'Prediction horizon (ms)',
        'method': 'Decoder or dynamics model (persistence, affine/autonomous, input_phase)', 'value': 'Decoder score, R2 or accuracy',
        'correct': 'Decoded cue correct', 'total_postcue_ms': 'Cue offset to decision (ms)'}, key=['row'])


def ed08(cache):
    tp = paths.M1 / 'tables' / 'type-functional-atlas.parquet'
    t = io.parquet_load(tp)
    t = t[t['model'].str.startswith('difficulty-')]
    agg = t.groupby(['phase', 'type']).agg(cells=('cells', 'first'), maps=('model', 'nunique'), spike_probability=('spike_probability', 'mean'),
                                           voltage_mv=('voltage', 'mean'), adaptation=('adaptation', 'mean'), resource=('resource', 'mean')).reset_index()
    agg.insert(0, 'record', 'static_activity_24_maps')
    ann = io.parquet_load(paths.M1 / 'neurons' / 'annotations.parquet', columns=['bodyId', 'type', 'superclass', 'nt'])
    ann['type'] = ann['type'].fillna('unannotated')
    ann['superclass'] = ann['superclass'].fillna('unannotated')
    mem = []
    enr = []
    for model, ctx, vis, label in STATE_SETS:
        df = cache[model]
        dv = np.abs(df[f'c{ctx}-visible{vis}-delta-v'].to_numpy(np.float64))
        a = ann.assign(dv=dv)
        g = a.groupby('type').agg(cells=('dv', 'size'), mean_abs_voltage_cue_difference=('dv', 'mean')).reset_index()
        g.insert(0, 'state_set', label)
        mem.append(g)
        k = int(np.ceil(0.01 * len(dv)))
        top = np.zeros(len(dv), bool)
        top[np.argsort(-dv, kind='mergesort')[:k]] = True
        for var in ('superclass', 'nt'):
            for cat in sorted(a[var].unique()):
                inc = (a[var] == cat).to_numpy()
                table = [[int((top & inc).sum()), int((top & ~inc).sum())], [int((~top & inc).sum()), int((~top & ~inc).sum())]]
                r = stats.fisher(table)
                tid = f'ed08b.enrichment.{model}.c{ctx}v{vis}.{var}.{cat}'
                stats.record(r, test_id=tid, panel='ed08b', family='F8.enrichment', contrast=f'{label}: {var} = {cat} among the top-1% cue-responsive cells',
                             unit='cell (descriptive, within one model)', level='B', analysis_status='new_registered',
                             source_table='extended/ed08/ed08b_atlas_enrichment.csv')
                enr.append(dict(state_set=label, model=model, variable=var, category=str(cat), cells_in_category=int(inc.sum()), top_cells=k,
                                top_in_category=int((top & inc).sum()), expected_in_top=float(k * inc.mean()), odds_ratio=r['effect'],
                                ci_low=r['effect_ci_low'], ci_high=r['effect_ci_high'], p_value=r['p_value'], test_id=tid, analysis_status='new_registered'))
    m = pd.concat(mem, ignore_index=True)
    out = pd.concat([agg.rename(columns={'phase': 'phase_or_state_set'}),
                     m.rename(columns={'state_set': 'phase_or_state_set'}).assign(record='cue_difference_by_type')], ignore_index=True)
    out['source_file'] = np.where(out['record'] == 'static_activity_24_maps', paths.rel(tp), 'mechanisms-v1/interpretation/*-memory-cells-d160-p4.parquet')
    csvout.write_table(out, 'extended/ed08/ed08a_type_atlas.csv', title='Atlas by cell type: static activity (24 maps) and cue differences (memory state sets)', describe={
        'record': 'static_activity_24_maps or cue_difference_by_type', 'phase_or_state_set': 'Phase (static) or state set (memory)', 'type': 'Cell type (or unannotated)',
        'cells': 'Cells of the type', 'maps': 'Maps averaged', 'spike_probability': 'Mean spike probability per ms', 'voltage_mv': 'Mean voltage (mV)',
        'adaptation': 'Mean adaptation', 'resource': 'Mean synaptic resource', 'mean_abs_voltage_cue_difference': 'Mean |cue1 - cue0| voltage (mV)'},
        key=['record', 'phase_or_state_set', 'type'])
    csvout.write_table(pd.DataFrame(enr), 'extended/ed08/ed08b_atlas_enrichment.csv', title='Enrichment among the top-1% cue-responsive cells (descriptive, BH in ST2)', describe={
        'state_set': 'Distinct state set', 'variable': 'superclass or transmitter (nt)', 'category': 'Category', 'cells_in_category': 'Cells in the category',
        'top_cells': 'Top-1% cells by |voltage cue difference|', 'top_in_category': 'Top cells in the category', 'expected_in_top': 'Expected count',
        'odds_ratio': 'Conditional odds ratio', 'p_value': 'Fisher exact p (BH within F8.enrichment in ST2)'}, key=['state_set', 'variable', 'category'])


def ed09():
    rows = []

    def add(record_type, model, method, value, horizon=None, split=None, extra=None, src=''):
        rows.append(dict(record_type=record_type, model=model, method=method, horizon_ms=horizon, split=split, value=value,
                         details=json.dumps(extra, sort_keys=True) if extra else None, source_file=src))

    lp = paths.M1 / 'tables' / 'cross-family-learning-decoding.csv'
    for _, r in io.csv_load(lp).iterrows():
        add('static_cross_family_decoding', r['model'], f"{r['phase']}:{r['representation']}:{r['target']}", float(r['correct']) / float(r['states']),
            extra=dict(correct=int(r['correct']), states=int(r['states']), brier=float(r['brier']), majority_baseline=float(r['majority_baseline_accuracy'])),
            src=paths.rel(lp))
    mp = paths.M1 / 'tables' / 'cross-family-memory-decoding.csv'
    md = io.csv_load(mp)
    for (dm, em, fam, held), g in md.groupby(['discovery_model', 'evaluation_model', 'family', 'heldout_history']):
        add('memory_cross_family_decoding', em, f'decoder fitted on {dm}', float(g['decoded_correct'].mean()), split='untrained' if held else 'train',
            extra=dict(correct=int(g['decoded_correct'].sum()), histories=len(g), family=fam), src=paths.rel(mp))
    for f, rt in (('time-decoders.json', 'time_decoder'), ('demixed-reconstruction.json', 'demixed_reconstruction'), ('dynamics.json', 'affine_dynamics')):
        p = paths.M1 / 'tables' / f
        for r in io.json_load(p):
            if rt == 'time_decoder':
                add(rt, r['model'], 'same_time_mean_accuracy', r['same_time_mean_accuracy'], src=paths.rel(p))
                add(rt, r['model'], 'all_time_mean_accuracy', r['all_time_mean_accuracy'], src=paths.rel(p))
            elif rt == 'demixed_reconstruction':
                add(rt, r['model'], f"factor:{r['factor']}", r['heldout_reconstruction_r2'], extra=dict(components=r['components']), src=paths.rel(p))
            else:
                add(rt, r['model'], 'heldout_r2_1ms', r['heldout_r2'], extra={k: v for k, v in r.items() if k not in ('model', 'heldout_r2')}, src=paths.rel(p))
    sp = paths.MS / 'analysis' / 'final-decoder-summary.csv'
    for _, r in io.csv_load(sp).iterrows():
        add('supplement_decoder_summary', r['model'], f"visible={str(r['visible']).lower()}", float(r['cue_correct']) / float(r['histories']), split=r['split'],
            extra=dict(cue_correct=int(r['cue_correct']), histories=int(r['histories']), solved=int(r['solved'])), src=paths.rel(sp))
    cp = paths.MS / 'analysis' / 'final-decoder-behavior-cross.csv'
    cx = io.csv_load(cp)
    for i, r in cx.iterrows():
        add('decoder_behaviour_cross', str(r.get('model')), 'cross_table_row', None, extra={k: (v.item() if hasattr(v, 'item') else v) for k, v in r.items()}, src=paths.rel(cp))
    a = pd.DataFrame(rows)
    a.insert(0, 'row', range(len(a)))
    csvout.write_table(a, 'extended/ed09/ed09a_decoding_all.csv', title='Decoding results at every horizon and family, including failures', describe={
        'row': 'Row index', 'record_type': 'Decoder family', 'method': 'Decoder, target or factor', 'horizon_ms': 'Horizon (ms)',
        'value': 'Accuracy or R2', 'details': 'Counts and baselines (JSON)'}, key=['row'])
    rows = []
    ip = paths.MS / 'analysis' / 'input-conditioned-dynamics.csv'
    for _, r in io.csv_load(ip).iterrows():
        add('input_conditioned_dynamics', r['model'], r['method'], float(r['r2']), horizon=int(r['horizon_ms']), split=r['split'],
            extra=dict(windows=int(r['windows']), histories=int(r['histories'])), src=paths.rel(ip))
    pp = paths.M1 / 'interpretation' / 'dynamics-persistence-check.csv'
    for _, r in io.csv_load(pp).iterrows():
        add('m1_dynamics', r['model'], 'affine', float(r['affine_r2']), horizon=int(r['horizon_ms']), split='heldout_histories',
            extra=dict(histories=int(r['histories']), state_pairs=int(r['state_pairs'])), src=paths.rel(pp))
        add('m1_dynamics', r['model'], 'persistence', float(r['persistence_r2']), horizon=int(r['horizon_ms']), split='heldout_histories', src=paths.rel(pp))
    b = pd.DataFrame(rows)
    b.insert(0, 'row', range(len(b)))
    csvout.write_table(b, 'extended/ed09/ed09b_dynamics_all.csv', title='Low-dimensional dynamics at every horizon (all models and methods)', describe={
        'row': 'Row index', 'record_type': 'Dynamics analysis', 'method': 'persistence, affine (autonomous) or input_phase', 'horizon_ms': 'Prediction horizon (ms)',
        'value': 'Held-out R2', 'details': 'Counts (JSON)'}, key=['row'])


def main():
    fig6a()
    cache = fig6b()
    fig6c()
    fig6f()
    ed08(cache)
    ed09()
