"""P5: interventions in one taxonomy (gates: M1 memory-intervention-summary and MS final-state-factorial reproduced
exactly), state interventions (4c), visibility (4d), directional transfer (4e), dose (4f), matched duration (5c),
component swaps vs norm-matched controls (6d), candidate groups (6e), ED6, ED7, ST5."""
import numpy as np
import pandas as pd

from msa import csvout, io, memory, paths, stats, taxonomy

M1_FAMILY = {'remote-dynamic-M0-head_only': 'M0', 'supp-M0-visibility': 'M0vis', 'supp-S1-ring-brain': 'S1', 'supp-S2-context-brain': 'S2',
             'remote-dynamic-M0-joint': 'M0'}
COMPONENT = {'all': 'all_state', 'delay': 'delay_buffer', 'resource': 'resources', 'voltage': 'voltage', 'current': 'current',
             'adaptation': 'adaptation', 'continuous': 'continuous_combined'}
WINDOW_DESC = 'write = just after cue offset, hold = mid-delay, read = just before the first decision (M1/MS); decision = at the decision (remote probes)'


def unit_of(model, visible):
    base = {'remote-dynamic-M0-head_only': 'M0', 'remote-dynamic-M0-joint': 'M0joint', 'supp-S1-ring-brain': 'S1', 'supp-S2-context-brain': 'S2'}
    if model == 'supp-M0-visibility':
        return 'M0vis.visible' if bool(visible) else 'M0vis.hidden'
    return base[model]


def _b(s):
    return s.astype(str).str.lower().isin(['true', '1', '1.0'])


# ---------------------------------------------------------------- sources

def m1_pairs():
    p = paths.M1 / 'interpretation' / 'memory-intervention-pairs.csv'
    d = io.csv_load(p)
    for c in ('visible', 'solved', 'baseline_solved', 'first_push_correct', 'baseline_push', 'initial_action_changed'):
        d[c] = _b(d[c])
    d['source'] = 'mechanisms_v1'
    d['source_file'] = paths.rel(p)
    return d


def m1_canonical(d):
    group_qa = (d['kind'] == 'group') & d['group'].isin(['restore', 'sham'])
    qa = d['component'].isin(['restore', 'same_cue', 'sham']) | group_qa
    inter = np.where(group_qa, d['group'].astype(str),
                     np.where(d['kind'] == 'group', 'group_suppression',
                              np.where(qa, d['component'].map({'restore': 'restore', 'same_cue': 'same_cue_swap', 'sham': 'sham'}),
                                       d['intervention'].astype(str) + '_' + d['component'].map(COMPONENT).fillna(''))))
    out = pd.DataFrame(dict(source='mechanisms_v1', job=d['job'], model=d['model'], family=d['model'].map(M1_FAMILY), visible=d['visible'],
                            unit=[unit_of(m, v) for m, v in zip(d['model'], d['visible'])], case=d['case'], cue=d['cue'].astype(int),
                            context=d['context'].astype(int), delay_ms=d['delay'].astype(int), prefix_windows=d['prefix'].astype(int), split=d['split'],
                            condition_raw=d['condition'], intervention=inter, component=np.where(d['kind'] == 'group', None, d['component'].map(COMPONENT)),
                            operation=np.where(qa | (d['kind'] == 'group'), None, d['intervention']),
                            window=np.where(group_qa, 'group_job_check', d['window']), fraction=np.nan,
                            group=np.where(group_qa, None, d['group']), side=d['side'].map({'indices': 'candidate', 'controls': 'matched'}),
                            strength=np.where(group_qa, np.nan, d['strength']),
                            channel=None, input_kind=None, total_postcue_ms=np.nan, qa_check=qa.to_numpy(), solved=d['solved'],
                            baseline_solved=d['baseline_solved'], status=d['status'].map(taxonomy.behavior), first_push_correct=d['first_push_correct'],
                            baseline_first_push_correct=d['baseline_push'], margin_change=d['first_move_margin_delta'],
                            source_file=d['source_file']))
    return out


def gate_m1(d):
    s_p = paths.M1 / 'interpretation' / 'memory-intervention-summary.csv'
    lg_p = paths.M1 / 'interpretation' / 'lost-and-gained.csv'
    saved = io.csv_load(s_p)
    saved['visible'] = _b(saved['visible'])
    g = d.groupby(['model', 'kind', 'condition', 'visible'])
    mine = g.agg(histories=('solved', 'size'), solved=('solved', 'sum'), baseline_solved=('baseline_solved', 'sum'),
                 first_push=('first_push_correct', 'sum'), baseline_push=('baseline_push', 'sum'),
                 mean_first_move_margin_change=('first_move_margin_delta', 'mean'), changed_initial_action=('initial_action_changed', 'sum')).reset_index()
    mine['net_change'] = mine['solved'] - mine['baseline_solved']
    j = saved.merge(mine, on=['model', 'kind', 'condition', 'visible'], suffixes=('_saved', ''))
    ok = len(j) == len(saved) == len(mine)
    for c in ('histories', 'solved', 'baseline_solved', 'net_change', 'first_push', 'baseline_push', 'changed_initial_action'):
        ok &= bool((j[c + '_saved'] == j[c]).all())
    ok &= bool(np.allclose(j['mean_first_move_margin_change_saved'], j['mean_first_move_margin_change'], atol=1e-9, rtol=0))
    lg = io.csv_load(lg_p)
    lg['visible'] = _b(lg['visible'])
    lost = d.groupby(['model', 'condition', 'visible']).apply(lambda x: pd.Series(dict(lost=int((x.baseline_solved & ~x.solved).sum()),
                                                                                      gained=int((~x.baseline_solved & x.solved).sum()))),
                                                              include_groups=False).reset_index()
    jl = lg.merge(lost, on=['model', 'condition', 'visible'], suffixes=('_saved', ''))
    ok &= len(jl) == len(lg) and bool((jl.lost_saved == jl.lost).all() and (jl.gained_saved == jl.gained).all())
    return ok, len(saved), len(lg)


def ms_outcomes():
    p = paths.MS / 'analysis' / 'condition-outcomes.csv'
    d = io.csv_load(p)
    for c in ('visible', 'norm_valid'):
        d[c] = _b(d[c])
    d['solved'] = d['solved'].astype(bool)
    d['source_file'] = paths.rel(p)
    return d


def ms_canonical(d):
    rows = d[d['module'] != 'timing'].copy()
    rows = rows[rows['kind'].fillna('') != 'baseline']
    kind = rows['kind'].fillna('')
    inter = np.select([kind == 'cue', kind == 'scrambled', kind == 'restore', kind == 'same_cue', kind == 'sham', rows['module'] == 'routes'],
                      ['interpolate_cue', 'norm_matched_scramble', 'restore', 'same_cue_swap', 'sham',
                       'route_' + rows['channel'].fillna('none').astype(str)], 'other')
    bs = rows['baseline_solved']
    out = pd.DataFrame(dict(source='mechanisms_supplement', job=rows['source'].str.split('/').str[1], model=rows['model'],
                            family=rows['model'].map(M1_FAMILY), visible=rows['visible'], unit=[unit_of(m, v) for m, v in zip(rows['model'], rows['visible'])],
                            case=rows['case'], cue=rows['cue'].astype(int), context=rows['context'].astype(int), delay_ms=rows['delay'],
                            prefix_windows=rows['prefix'], split=rows['split'], condition_raw=rows['condition'], intervention=inter,
                            component=rows['part'].map(COMPONENT), operation=None, window=rows['window'], fraction=rows['fraction'],
                            group=rows['group'], side=rows['side'].map({'indices': 'candidate', 'controls': 'matched'}), strength=rows['strength'],
                            channel=rows['channel'], input_kind=None, total_postcue_ms=np.nan,
                            qa_check=kind.isin(['restore', 'same_cue', 'sham']).to_numpy(), solved=rows['solved'],
                            baseline_solved=bs.astype(float).map(lambda v: None if pd.isna(v) else bool(v)), status=rows['status'].map(taxonomy.behavior),
                            first_push_correct=rows['first_push_correct'], baseline_first_push_correct=None, margin_change=rows['margin_effect'],
                            source_file=rows['source_file']))
    return out


def gate_ms(d):
    f_p = paths.MS / 'analysis' / 'final-state-factorial.csv'
    saved = io.csv_load(f_p)
    saved['visible'] = _b(saved['visible'])
    sel = d[(d['module'] == 'selective') & d['kind'].isin(['cue', 'scrambled'])].copy()
    sel['bs'] = sel['baseline_solved'].astype(bool)
    g = sel.groupby(['model', 'visible', 'window', 'part', 'fraction', 'kind'])
    mine = g.apply(lambda x: pd.Series(dict(histories=len(x), baseline_solved=float(x.bs.sum()), solved=int(x.solved.sum()),
                                            first_push_correct=int(x.first_push_correct.astype(bool).sum()),
                                            lost=int((x.bs & ~x.solved).sum()), gained=int((~x.bs & x.solved).sum()),
                                            margin_mean_change=float(x.margin_effect.mean()))), include_groups=False).reset_index()
    j = saved.merge(mine, on=['model', 'visible', 'window', 'part', 'fraction', 'kind'], suffixes=('_saved', ''))
    ok = len(j) == len(saved) == len(mine)
    for c in ('histories', 'baseline_solved', 'solved', 'first_push_correct', 'lost', 'gained'):
        ok &= bool((j[c + '_saved'] == j[c]).all())
    ok &= bool(np.allclose(j['margin_mean_change_saved'], j['margin_mean_change'], atol=1e-9, rtol=0))
    return ok, len(saved)


def r0_interventions():
    rows = []
    for m in ('remote-dynamic-M0-head_only', 'remote-dynamic-M0-joint', 'remote-dynamic-M0-reset', 'remote-dynamic-M0-carry_untrained'):
        v = memory.verification(m)
        src = paths.rel(memory.root(m) / m / 'verification.json')
        base = {(int(r['cue']), int(r['delay_ms']), int(r['prefix_windows'])): r for r in v['results']['rollouts']}
        base_probe = {}
        for pr in v['results']['probes']:
            for rec in pr['records']:
                base_probe[(int(rec['cue']), int(pr['delay_ms']), int(pr['prefix_windows']))] = rec
        probe = {}
        for pr in v.get('interventions', []):
            for rec in pr['records']:
                probe[(pr['intervention'], int(rec['cue']), int(pr['delay_ms']), int(pr['prefix_windows']))] = rec
        for ro in v.get('intervention_rollouts', []):
            cue, d, p = int(ro['cue']), int(ro['delay_ms']), int(ro['prefix_windows'])
            b = base[(cue, d, p)]
            rec = probe.get((ro['intervention'], cue, d, p))
            brec = base_probe.get((cue, d, p))
            tgt, oth = memory.M0_CHOICE_TARGET[cue], memory.M0_CHOICE_TARGET[1 - cue]
            margin = (rec['logits'][tgt] - rec['logits'][oth]) if rec else None
            bmargin = (brec['logits'][tgt] - brec['logits'][oth]) if brec else None
            canon = taxonomy.R0_KIND[ro['intervention']]
            rows.append(dict(source='remote_batch1', job=m, model=m, family='M0', visible=False, unit=m.replace('remote-dynamic-', ''),
                             case=f'c0-cue{cue}', cue=cue, context=0, delay_ms=d, prefix_windows=p, split=memory.split_label(b['history_split']),
                             condition_raw=ro['intervention'], intervention=canon, component=None, operation=None, window='decision', fraction=np.nan,
                             group=None, side=None, strength=np.nan, channel=None, input_kind=None, total_postcue_ms=np.nan,
                             qa_check=canon in taxonomy.QA_CHECKS, solved=ro['status'] == 'solved', baseline_solved=b['status'] == 'solved',
                             status=taxonomy.behavior(ro['status']), first_push_correct=None, baseline_first_push_correct=None,
                             margin_change=(margin - bmargin) if (margin is not None and bmargin is not None) else None,
                             choice_follows_donor=(rec['choice'] == oth) if (rec and canon == 'opposite_cue_swap') else None, source_file=src))
    return pd.DataFrame(rows)


def rs_causal():
    rows = []
    for job in ('supp-M0-causal', 'supp-M0-visibility-causal'):
        src = paths.RS / job / 'observations.jsonl'
        lines = io.jsonl_load(src)
        base = {}
        for ln in lines:
            if ln.get('intervention') in (None, 'sham'):
                for rec in ln['records']:
                    base[(ln.get('intervention'), ln.get('window'), int(ln['delay_ms']), int(ln['prefix_windows']), rec['case'])] = rec
        for ln in lines:
            kind = ln.get('intervention')
            if kind is None:
                continue
            win = ln.get('window')
            d, p = int(ln['delay_ms']), int(ln['prefix_windows'])
            for rec in ln['records']:
                key = (None, win, d, p, rec['case']) if win == 'decision' else ('sham', win, d, p, rec['case'])
                b = base.get(key)
                if b is None or (win != 'decision' and kind == 'sham'):
                    b = base.get((None, 'decision', d, p, rec['case'])) if win != 'decision' and kind == 'sham' else b
                canon = taxonomy.RS_KIND.get(kind, kind)
                model = 'remote-dynamic-M0-head_only' if job == 'supp-M0-causal' else 'supp-M0-visibility'
                rows.append(dict(source='remote_supplement_causal', job=job, model=model, family='M0' if job == 'supp-M0-causal' else 'M0vis',
                                 visible=bool(rec['visible']), unit=('M0' if job == 'supp-M0-causal' else ('M0vis.visible' if rec['visible'] else 'M0vis.hidden')),
                                 case=rec['case'], cue=int(rec['cue']), context=int(rec['context']), delay_ms=d, prefix_windows=p,
                                 split='train' if d in (0, 40, 160) and p in (0, 1, 4) else 'untrained', condition_raw=kind, intervention=canon,
                                 component=None, operation=None, window=win, fraction=np.nan, group=None, side=None, strength=np.nan, channel=None,
                                 input_kind=None, total_postcue_ms=np.nan, qa_check=canon in taxonomy.QA_CHECKS, solved=bool(rec['solved']),
                                 baseline_solved=(bool(b['solved']) if b is not None else None), status=taxonomy.behavior(rec['status']),
                                 first_push_correct=rec.get('first_push_correct'),
                                 baseline_first_push_correct=(b.get('first_push_correct') if b is not None else None), margin_change=None,
                                 choice_follows_donor=None, source_file=paths.rel(src)))
    return pd.DataFrame(rows)


def ms_timing():
    p = paths.MS / 'analysis' / 'final-timing-pairs.csv'
    d = io.csv_load(p)
    d['visible'] = _b(d['visible'])
    out = pd.DataFrame(dict(source='mechanisms_supplement', job=d['source'].str.split('/').str[1], model=d['model'], family=d['model'].map(M1_FAMILY),
                            visible=d['visible'], unit=[unit_of(m, v) for m, v in zip(d['model'], d['visible'])], case=d['case'], cue=d['cue'].astype(int),
                            context=d['context'].astype(int), delay_ms=np.nan, prefix_windows=d['prefix'], split=d['split'], condition_raw=d['condition'],
                            intervention='timing_' + d['input_kind'].astype(str), component=None, operation=None, window=None, fraction=np.nan, group=None,
                            side=None, strength=np.nan, channel=None, input_kind=d['input_kind'], total_postcue_ms=d['total_postcue_ms'],
                            qa_check=False, solved=d['solved'].astype(bool), baseline_solved=d['matched_blank_solved'].astype(bool),
                            status=d['status'].map(taxonomy.behavior), first_push_correct=d['first_push_correct'], baseline_first_push_correct=None,
                            margin_change=None, choice_follows_donor=None, source_file=paths.rel(p)))
    return out, d


# ---------------------------------------------------------------- tests helpers

def _rec(r, tid, panel, fam, contrast, unit='history', status='new_registered', source='', hs=None):
    stats.record(r, test_id=tid, panel=panel, family=fam, contrast=contrast, unit=unit, level='B', analysis_status=status,
                 history_set_id=hs, source_table=source)


def _paired(df, cond_mask, keys=('case', 'delay_ms', 'prefix_windows')):
    x = df[cond_mask]
    return x.sort_values(list(keys))


# ---------------------------------------------------------------- panels

def fig4c(M1c, R0, RSc):
    keep = M1c[(M1c.component == 'all_state') | M1c.qa_check]
    out = pd.concat([keep, R0, RSc], ignore_index=True)
    out['history_set_id'] = np.where(out.source == 'mechanisms_v1', out.family + '.interv6', None)
    out['analysis_status'] = 're_derived'
    csvout.write_table(out, 'main/fig4/fig4c_state_interventions.csv', title='Whole-state interventions per history (with QA checks)', describe=ST5_DESC,
                       key=['source', 'model', 'unit', 'intervention', 'window', 'case', 'delay_ms', 'prefix_windows'],
                       sort=['source', 'unit', 'intervention', 'window', 'delay_ms', 'prefix_windows', 'case'])
    for u, g in M1c.groupby('unit'):
        for win in ('write', 'hold', 'read'):
            for comp in ('all_state', 'voltage', 'current', 'adaptation', 'resources', 'delay_buffer'):
                for op in ('clear', 'swap'):
                    x = g[(g.window == win) & (g.component == comp) & (g.operation == op)].sort_values(['case', 'delay_ms', 'prefix_windows'])
                    if x.empty:
                        continue
                    fam = f'F4c.primary.{u}' if comp == 'all_state' else f'F4c.components.{u}'
                    tid = f'fig4c.mcnemar.{u}.{win}.{op}_{comp}'
                    _rec(stats.mcnemar(x.solved, x.baseline_solved.astype(bool)), tid, 'fig4c' if comp == 'all_state' else 'ed06a', fam,
                         f'{u}: {op} {comp} at {win} vs baseline (solved)', source='main/fig4/fig4c_state_interventions.csv',
                         hs=f'{g.family.iloc[0]}.interv6')
                    if comp == 'all_state':
                        _rec(stats.signed_rank(x.margin_change.to_numpy(float), f'fig4c.margin.{u}.{win}.{op}'), f'fig4c.wilcoxon_margin.{u}.{win}.{op}_all_state',
                             'fig4c', f'F4c.margin.{u}', f'{u}: first-move margin change after {op} of the whole state at {win}',
                             source='main/fig4/fig4c_state_interventions.csv', hs=f'{g.family.iloc[0]}.interv6')
    h = R0[R0.model == 'remote-dynamic-M0-head_only']
    for kind in ('clear_voltage', 'clear_current', 'clear_adaptation', 'clear_delay_buffer', 'full_reset', 'opposite_cue_swap'):
        x = h[h.intervention == kind].sort_values(['cue', 'delay_ms'])
        _rec(stats.mcnemar(x.solved, x.baseline_solved), f'fig4c.mcnemar.original.{kind}', 'fig4c', 'F4c.original',
             f'remote batch 1 M0 readout-only, decision moment (prefix 4, 7 delays): {kind} vs baseline', source='main/fig4/fig4c_state_interventions.csv')
    s = RSc[RSc.job == 'supp-M0-causal']
    for (win, kind), x in s.groupby(['window', 'intervention']):
        if kind in taxonomy.QA_CHECKS or x.baseline_solved.isna().any():
            continue
        x = x.sort_values(['case', 'delay_ms'])
        _rec(stats.mcnemar(x.solved, x.baseline_solved.astype(bool)), f'fig4c.mcnemar.supplement.{win}.{kind}', 'fig4c', 'F4c.supplement',
             f'supplement causal probe, M0 readout-only: {kind} at {win} vs {"no intervention" if win == "decision" else "same-window sham"}',
             source='main/fig4/fig4c_state_interventions.csv')


def fig4d(M1c, RSc):
    v = M1c[M1c.model == 'supp-M0-visibility']
    rows = []
    for (win, comp, op), g in v.groupby(['window', 'component', 'operation']):
        hid = g[~g.visible].set_index(['cue', 'delay_ms', 'prefix_windows'])
        vis = g[g.visible].set_index(['cue', 'delay_ms', 'prefix_windows'])
        j = hid.join(vis, lsuffix='_hidden', rsuffix='_visible', how='inner')
        eh = j.solved_hidden.astype(int) - j.baseline_solved_hidden.astype(int)
        ev = j.solved_visible.astype(int) - j.baseline_solved_visible.astype(int)
        both = j.baseline_solved_hidden.astype(bool) & j.baseline_solved_visible.astype(bool)
        for idx in j.index:
            rows.append(dict(source='mechanisms_v1', window=win, intervention=f'{op}_{comp}', cue=idx[0], delay_ms=idx[1], prefix_windows=idx[2],
                             hidden_baseline_solved=bool(j.loc[idx, 'baseline_solved_hidden']), hidden_solved=bool(j.loc[idx, 'solved_hidden']),
                             visible_baseline_solved=bool(j.loc[idx, 'baseline_solved_visible']), visible_solved=bool(j.loc[idx, 'solved_visible']),
                             effect_hidden=int(eh.loc[idx]), effect_visible=int(ev.loc[idx]), both_baselines_solved=bool(both.loc[idx]),
                             source_file=g.source_file.iloc[0]))
        if comp == 'all_state':
            _rec(stats.sign_test((eh - ev).to_numpy(float)), f'fig4d.sign.{win}.{op}_all_state', 'fig4d', 'F4d',
                 f'shared readout: hidden-minus-visible effect of {op} whole state at {win}', source='main/fig4/fig4d_visibility_contrast.csv',
                 hs='M0vis.interv6')
    r = RSc[RSc.job == 'supp-M0-visibility-causal']
    for (win, kind), g in r.groupby(['window', 'intervention']):
        hid = g[~g.visible].set_index(['cue', 'delay_ms', 'prefix_windows'])
        vis = g[g.visible].set_index(['cue', 'delay_ms', 'prefix_windows'])
        j = hid.join(vis, lsuffix='_hidden', rsuffix='_visible', how='inner')
        if j.baseline_solved_hidden.isna().any() or j.baseline_solved_visible.isna().any():
            continue
        eh = j.solved_hidden.astype(int) - j.baseline_solved_hidden.astype(int)
        ev = j.solved_visible.astype(int) - j.baseline_solved_visible.astype(int)
        both = j.baseline_solved_hidden.astype(bool) & j.baseline_solved_visible.astype(bool)
        for idx in j.index:
            rows.append(dict(source='remote_supplement_causal', window=win, intervention=kind, cue=idx[0], delay_ms=idx[1], prefix_windows=idx[2],
                             hidden_baseline_solved=bool(j.loc[idx, 'baseline_solved_hidden']), hidden_solved=bool(j.loc[idx, 'solved_hidden']),
                             visible_baseline_solved=bool(j.loc[idx, 'baseline_solved_visible']), visible_solved=bool(j.loc[idx, 'solved_visible']),
                             effect_hidden=int(eh.loc[idx]), effect_visible=int(ev.loc[idx]), both_baselines_solved=bool(both.loc[idx]),
                             source_file=g.source_file.iloc[0]))
        if kind not in taxonomy.QA_CHECKS and win == 'decision':
            _rec(stats.sign_test((eh - ev).to_numpy(float)), f'fig4d.sign.supplement.{kind}', 'fig4d', 'F4d.supplement',
                 f'supplement visibility probe: hidden-minus-visible effect of {kind} at the decision', source='main/fig4/fig4d_visibility_contrast.csv')
    df = pd.DataFrame(rows)
    both = df.groupby(['source', 'window', 'intervention', 'delay_ms', 'prefix_windows'])['both_baselines_solved'].transform('all')
    df['timing_strict_subset'] = both.astype(bool)
    df['analysis_status'] = 're_derived'
    csvout.write_table(df, 'main/fig4/fig4d_visibility_contrast.csv', title='Hidden-goal vs visible-goal intervention effects on matched histories', describe={
        'timing_strict_subset': 'Both cues at this timing solve under both visibility baselines (the strictly matched subset of the reports)',
        'source': ST5_DESC['source'], 'window': WINDOW_DESC, 'intervention': 'Canonical intervention', 'cue': 'Cue', 'delay_ms': 'Delay (ms)', 'prefix_windows': 'Prefix windows',
        'hidden_baseline_solved': 'Hidden goal, baseline solved', 'hidden_solved': 'Hidden goal, solved after intervention',
        'visible_baseline_solved': 'Visible goal, baseline solved', 'visible_solved': 'Visible goal, solved after intervention',
        'effect_hidden': 'Hidden effect: solved after minus baseline (-1, 0, 1)', 'effect_visible': 'Visible effect',
        'both_baselines_solved': 'Both visibility baselines solved (strictly matched subset)'},
        key=['source', 'window', 'intervention', 'cue', 'delay_ms', 'prefix_windows'])


def fig4e():
    p = paths.MS / 'analysis' / 'directional-swap-audit.csv'
    a = io.csv_load(p)
    for c in ('visible', 'eligible_both_baselines_solved', 'solved', 'follows_donor', 'follows_recipient'):
        a[c] = _b(a[c])
    a['unit'] = [unit_of(m, v) for m, v in zip(a['model'], a['visible'])]
    a['source_file'] = paths.rel(p)
    a['analysis_status'] = 'post_hoc'
    csvout.write_table(a, 'main/fig4/fig4e_directional_transfer.csv', title='Directional transfer: first push after a cue swap vs after norm-matched scrambling',
                       describe={'delay': 'Delay (ms)', 'prefix': 'Prefix windows', 'kind': 'cue (opposite-cue continuous-state swap) or scrambled (norm-matched control)',
                                 'case': 'Case', 'visible': 'Goal visible', 'context': 'Context', 'eligible_both_baselines_solved': 'Both opposite-cue baselines solved at this timing',
                                 'own_push': 'Recipient baseline first push', 'donor_push': 'Donor baseline first push', 'actual_push': 'First push after the intervention',
                                 'solved': 'Solved after the intervention', 'follows_donor': 'First push equals the donor push',
                                 'follows_recipient': 'First push equals the recipient push', 'unit': 'Model / visibility unit'},
                       key=['unit', 'kind', 'case', 'delay', 'prefix'])
    strata = []
    for u, g in a.groupby('unit'):
        w = g.pivot_table(index=['case', 'delay', 'prefix'], columns='kind', values='follows_donor', aggfunc='first')
        el = g[g.kind == 'cue'].set_index(['case', 'delay', 'prefix'])['eligible_both_baselines_solved']
        we = w[el.reindex(w.index).astype(bool)]
        r = stats.mcnemar(we['cue'].astype(bool), we['scrambled'].astype(bool))
        fam = 'F4e.visible' if u == 'M0vis.visible' else 'F4e.hidden'
        _rec(r, f'fig4e.mcnemar.{u}', 'fig4e', fam, f'{u}: donor-following after cue swap vs scrambling (eligible histories)', status='post_hoc',
             source='main/fig4/fig4e_directional_transfer.csv')
        if u in ('M0', 'M0joint', 'S1', 'S2'):
            strata.append((r['a_only'], r['b_only']))
        defined = g[g.kind == 'cue'].set_index(['case', 'delay', 'prefix'])
        ok = defined['own_push'].notna() & defined['donor_push'].notna() & (defined['own_push'] != defined['donor_push'])
        ws = w[ok.reindex(w.index).fillna(False).astype(bool)]
        _rec(stats.mcnemar(ws['cue'].astype(bool), ws['scrambled'].astype(bool)), f'fig4e.sensitivity.{u}', 'fig4e', 'F4e.sensitivity',
             f'{u}: donor-following after cue swap vs scrambling, all histories with distinct defined pushes', status='post_hoc',
             source='main/fig4/fig4e_directional_transfer.csv')
        t = w.reset_index().groupby(['delay', 'prefix'])[['cue', 'scrambled']].sum()
        _rec(stats.sign_test((t['cue'] - t['scrambled']).to_numpy(float)), f'fig4e.clustered.{u}', 'fig4e', 'F4e.clustered',
             f'{u}: per-timing difference in donor-following (cue swap minus scrambling)', unit='timing', status='post_hoc',
             source='main/fig4/fig4e_directional_transfer.csv')
    _rec(stats.stratified_mcnemar(strata), 'fig4e.stratified.distinct_states', 'fig4e', 'F4e.combined',
         'combined over hidden-goal models with distinct brain states (M0 readout-only, M0 joint, S1, S2); secondary', status='post_hoc',
         source='main/fig4/fig4e_directional_transfer.csv')


def fig4f(ms):
    base = ms[(ms.module == 'selective') & (ms.kind == 'baseline')][['model', 'visible', 'case', 'delay', 'prefix', 'margin', 'solved']]
    base = base.rename(columns={'margin': 'm0', 'solved': 's0'})
    sel = ms[(ms.module == 'selective') & ms.kind.isin(['cue', 'scrambled']) & (ms.window == 'read') & (ms.part == 'continuous')].copy()
    rows = []
    for _, r in base.iterrows():
        rows.append(dict(model=r.model, visible=bool(r.visible), unit=unit_of(r.model, r.visible), case=r.case, delay_ms=int(r.delay), prefix_windows=int(r.prefix),
                         kind='baseline', fraction=0.0, solved=bool(r.s0), margin=float(r.m0)))
    for _, r in sel.iterrows():
        rows.append(dict(model=r.model, visible=bool(r.visible), unit=unit_of(r.model, r.visible), case=r.case, delay_ms=int(r.delay), prefix_windows=int(r.prefix),
                         kind=r.kind, fraction=float(r.fraction), solved=bool(r.solved), margin=float(r.margin)))
    df = pd.DataFrame(rows)
    df['source_file'] = paths.rel(paths.MS / 'analysis' / 'condition-outcomes.csv')
    df['analysis_status'] = 're_derived'
    csvout.write_table(df, 'main/fig4/fig4f_dose_response.csv', title='Interpolation dose response (read window, combined continuous state)', describe={
        'unit': 'Model / visibility unit', 'case': 'Case', 'delay_ms': 'Delay (ms)', 'prefix_windows': 'Prefix windows',
        'kind': 'baseline, cue (interpolation toward the opposite-cue state) or scrambled (norm-matched control)',
        'fraction': 'Interpolation dose a (0 = baseline)', 'solved': 'Solved', 'margin': 'First-decision margin (fixed baseline action vs strongest legal competitor)'},
        key=['unit', 'case', 'delay_ms', 'prefix_windows', 'kind', 'fraction'])
    for u, g in df.groupby('unit'):
        cue = g[g.kind.isin(['baseline', 'cue'])].pivot_table(index=['case', 'delay_ms', 'prefix_windows'], columns='fraction', values='margin', aggfunc='first')
        cue = cue[[0.0, 0.25, 0.5, 1.0]].dropna()
        _rec(stats.page(cue.to_numpy()), f'fig4f.page.{u}', 'fig4f', 'F4f', f'{u}: margin trend over dose 0/0.25/0.5/1 (cue interpolation)',
             source='main/fig4/fig4f_dose_response.csv')
        for frac in (0.25, 0.5, 1.0):
            w = g[g.fraction == frac].pivot_table(index=['case', 'delay_ms', 'prefix_windows'], columns='kind', values='margin', aggfunc='first').dropna()
            _rec(stats.signed_rank((w['cue'] - w['scrambled']).to_numpy(float), f'fig4f.wilcoxon.{u}.{frac}'), f'fig4f.wilcoxon.{u}.{frac}', 'fig4f',
                 'F4f.companion', f'{u}: margin after cue interpolation minus after scrambling at dose {frac}', source='main/fig4/fig4f_dose_response.csv')


def fig5c(T, raw):
    df = T.copy()
    df['main_panel'] = df['prefix_windows'].isin([0, 2])
    df['analysis_status'] = 're_derived'
    csvout.write_table(df, 'main/fig5/fig5c_matched_duration.csv', title='Matched-duration neutral input vs blank waiting', describe={
        **ST5_DESC, 'main_panel': 'Row used in the main panel (blank or 2 neutral windows); 4/8 windows are Extended Data'},
        key=['unit', 'condition_raw', 'case'], sort=['unit', 'total_postcue_ms', 'input_kind', 'prefix_windows', 'case'])
    for u, g in df[df.input_kind != 'blank'].groupby('unit'):
        for kind, x in g[g.prefix_windows == 2].groupby('input_kind'):
            _rec(stats.mcnemar(x.solved, x.baseline_solved), f'fig5c.mcnemar.{u}.{kind}', 'fig5c', f'F5c.{u}',
                 f'{u}: {kind} (2 windows) vs duration-matched blank, pooled over durations', source='main/fig5/fig5c_matched_duration.csv')


def fig6d(ms):
    sel = ms[(ms.module == 'selective') & ms.kind.isin(['cue', 'scrambled'])].copy()
    sel['unit'] = [unit_of(m, v) for m, v in zip(sel['model'], sel['visible'])]
    rows = []
    for (u, win, part, frac), g in sel.groupby(['unit', 'window', 'part', 'fraction']):
        w = g.pivot_table(index=['case', 'delay', 'prefix'], columns='kind', values='solved', aggfunc='first')
        mg = g.pivot_table(index=['case', 'delay', 'prefix'], columns='kind', values='margin_effect', aggfunc='first')
        r = stats.mcnemar(w['cue'].astype(bool), w['scrambled'].astype(bool))
        tid = None
        if frac in (0.5, 1.0):
            fam = f'F6d.{u}' if frac == 1.0 else f'F6d.midpoint.{u}'
            tid = f'fig6d.mcnemar.{u}.{win}.{part}.{frac}'
            _rec(r, tid, 'fig6d' if frac == 1.0 else 'ed06a', fam, f'{u}: {part} at {win}, dose {frac}: cue vs norm-matched scrambled (solved)',
                 source='main/fig6/fig6d_component_interventions.csv')
        base = g[g.kind == 'cue']
        rows.append(dict(unit=u, model=g.model.iloc[0], window=win, component=COMPONENT.get(part, part), fraction=float(frac), histories=len(w),
                         baseline_solved=int(base.baseline_solved.astype(bool).sum()), cue_solved=int(w['cue'].sum()), scrambled_solved=int(w['scrambled'].sum()),
                         cue_only_solved=r['a_only'], scrambled_only_solved=r['b_only'], cue_margin_mean_change=float(mg['cue'].mean()),
                         scrambled_margin_mean_change=float(mg['scrambled'].mean()), p_value=r['p_value'], test_id=tid, analysis_status='re_derived',
                         source_file=paths.rel(paths.MS / 'analysis' / 'condition-outcomes.csv')))
    df = pd.DataFrame(rows)
    csvout.write_table(df, 'main/fig6/fig6d_component_interventions.csv', title='Component x window continuous-state swaps vs norm-matched controls', describe={
        'unit': 'Model / visibility unit', 'window': WINDOW_DESC, 'component': 'State component interpolated', 'fraction': 'Dose a (0.5 midpoint, 1 full swap)',
        'histories': 'Histories', 'baseline_solved': 'Baseline solved', 'cue_solved': 'Solved after the cue intervention',
        'scrambled_solved': 'Solved after the norm-matched scrambled control', 'cue_only_solved': 'Histories solved only after the cue intervention',
        'scrambled_only_solved': 'Histories solved only after scrambling', 'cue_margin_mean_change': 'Mean margin change after the cue intervention',
        'scrambled_margin_mean_change': 'Mean margin change after scrambling', 'p_value': 'Exact McNemar p (unadjusted; see ST2)'},
        key=['unit', 'window', 'component', 'fraction'])


def fig6e(M1c, ms):
    g = M1c[M1c.intervention == 'group_suppression']
    rows = []
    for (u, grp, side, strength, win), x in g.groupby(['unit', 'group', 'side', 'strength', 'window']):
        stage = 'confirmation_S1' if u == 'S1' else ('discovery_M0' if u in ('M0', 'M0vis.hidden', 'M0vis.visible') else 'scope_S2')
        rows.append(dict(set='memory_' + stage, unit=u, group=grp, side=side, strength=float(strength), window=win, channel='outgoing_propagation', histories=len(x),
                         baseline_solved=int(x.baseline_solved.astype(bool).sum()), solved=int(x.solved.sum()),
                         lost=int((x.baseline_solved.astype(bool) & ~x.solved).sum()), gained=int((~x.baseline_solved.astype(bool) & x.solved).sum()),
                         margin_mean_change=float(x.margin_change.mean()), source_file=x.source_file.iloc[0]))
    st_p = paths.M1 / 'interpretation' / 'static-intervention-summary.csv'
    st_ = io.csv_load(st_p)
    for _, r in st_.iterrows():
        rows.append(dict(set='learning_confirmation_T1', unit=r['model'], group=r['group'], side={'indices': 'candidate', 'controls': 'matched'}[r['side']],
                         strength=float(r['strength']), window=r['window'], channel='outgoing_propagation', histories=1,
                         baseline_solved=int(_b(pd.Series([r['baseline_solved']]))[0]), solved=int(_b(pd.Series([r['solved']]))[0]),
                         lost=int(r['solved_effect'] < 0), gained=int(r['solved_effect'] > 0), margin_mean_change=None, source_file=paths.rel(st_p)))
    rt = ms[ms.module == 'routes'].copy()
    rt['unit'] = [unit_of(m, v) for m, v in zip(rt['model'], rt['visible'])]
    for (u, side, ch, win, strength), x in rt.groupby(['unit', 'side', 'channel', 'window', 'strength']):
        rows.append(dict(set='route_' + ch, unit=u, group='memory-g8561', side={'indices': 'candidate', 'controls': 'matched'}[side], strength=float(strength),
                         window=win, channel=ch, histories=len(x), baseline_solved=int(x.baseline_solved.astype(bool).sum()), solved=int(x.solved.sum()),
                         lost=int((x.baseline_solved.astype(bool) & ~x.solved).sum()), gained=int((~x.baseline_solved.astype(bool) & x.solved).sum()),
                         margin_mean_change=float(x.margin_effect.abs().mean()), source_file=paths.rel(paths.MS / 'analysis' / 'condition-outcomes.csv')))
    df = pd.DataFrame(rows)
    df['analysis_status'] = 're_derived'
    csvout.write_table(df, 'main/fig6/fig6e_candidate_groups.csv', title='Candidate vs matched cell groups: propagation suppression and direct-readout replacement', describe={
        'set': 'memory_discovery_M0, memory_confirmation_S1, memory_scope_S2, learning_confirmation_T1 (one map per row) or route_<channel>',
        'unit': 'Model / visibility unit, or map for learning groups', 'group': 'Candidate group identifier', 'side': 'candidate or matched control group',
        'strength': 'Suppression or replacement strength', 'window': 'Intervention window', 'channel': 'outgoing_propagation, network, readout (direct) or both',
        'histories': 'Histories (or 1 map)', 'baseline_solved': 'Baseline solved', 'solved': 'Solved after the intervention', 'lost': 'Solutions lost',
        'gained': 'Solutions gained', 'margin_mean_change': 'Mean first-move margin change (routes: mean |change|)'},
        key=['set', 'unit', 'group', 'side', 'strength', 'window', 'channel'])
    s1 = g[g.unit == 'S1']
    for (grp, strength, win), x in s1.groupby(['group', 'strength', 'window']):
        w = x.pivot_table(index=['case', 'delay_ms', 'prefix_windows'], columns='side', values='solved', aggfunc='first')
        _rec(stats.mcnemar(w['candidate'].astype(bool), w['matched'].astype(bool)), f'fig6e.mcnemar.S1.{grp}.{strength}.{win}', 'fig6e', 'F6e.memory',
             f'S1 confirmation: {grp} vs matched group, strength {strength}, {win} window', source='main/fig6/fig6e_candidate_groups.csv', hs='S1.interv6')
    sel = rt[(rt.channel == 'readout') & (rt.strength == 1.0) & (rt.window == 'all_decisions')]
    for u, x in sel.groupby('unit'):
        w = x.pivot_table(index=['case', 'delay', 'prefix'], columns='side', values='margin_effect', aggfunc='first')
        d = (w['indices'].abs() - w['controls'].abs()).to_numpy(float)
        _rec(stats.signed_rank(d, f'fig6e.routes.{u}'), f'fig6e.wilcoxon_routes.{u}', 'fig6e', 'F6e.routes',
             f'{u}: |margin change| under direct-readout replacement (strength 1, all decisions), candidate minus matched',
             source='main/fig6/fig6e_candidate_groups.csv')


def ed06(all_rows):
    keys = ['source', 'unit', 'model', 'intervention', 'component', 'operation', 'window', 'fraction', 'group', 'side', 'strength', 'channel', 'input_kind']
    a = all_rows.copy()
    for k in keys:
        a[k] = a[k].astype(object).where(a[k].notna(), '')
    a['bs'] = a['baseline_solved'].map(lambda v: bool(v) if v is not None and not pd.isna(v) else False)
    a['ok_b'] = a['baseline_solved'].map(lambda v: v is not None and not pd.isna(v))
    g = a.groupby(keys)
    s = g.apply(lambda x: pd.Series(dict(histories=len(x), baseline_solved=int(x.bs.sum()), solved=int(x.solved.astype(bool).sum()),
                                         lost=int((x.bs & ~x.solved.astype(bool)).sum()), gained=int((~x.bs & x.solved.astype(bool) & x.ok_b).sum()),
                                         first_push_correct=int(np.nansum([bool(v) for v in x.first_push_correct if v is not None and not pd.isna(v)])),
                                         margin_mean_change=float(pd.to_numeric(x.margin_change, errors='coerce').mean()),
                                         qa_check=bool(x.qa_check.astype(bool).any()))), include_groups=False).reset_index()
    s['net_change'] = s['solved'] - s['baseline_solved']
    s['analysis_status'] = 're_derived'
    csvout.write_table(s, 'extended/ed06/ed06a_intervention_factorial.csv', title='Full intervention factorial (all sources, one taxonomy)', describe={
        **ST5_DESC, 'histories': 'Histories', 'baseline_solved': 'Baseline solved', 'solved': 'Solved after the intervention', 'lost': 'Solutions lost',
        'gained': 'Solutions gained', 'first_push_correct': 'Correct first pushes after the intervention', 'margin_mean_change': 'Mean margin change',
        'net_change': 'Solved minus baseline solved'}, key=keys)
    na_p = paths.MS / 'analysis' / 'final-norm-audit.csv'
    ns_p = paths.MS / 'analysis' / 'final-norm-summary.csv'
    ns = io.csv_load(ns_p)
    io.csv_load(na_p)
    ns['source_file'] = paths.rel(ns_p)
    csvout.write_table(ns, 'extended/ed06/ed06b_norm_audit.csv', title='Norm and direction audit of the norm-matched controls', describe={
        'field': 'State field', 'entries': 'Audited operations', 'invalid': 'Operations outside the 1% norm tolerance',
        'max_relative_norm_error': 'Maximum relative norm error', 'max_absolute_direction_cosine': 'Maximum |cosine| between control and cue directions',
        'max_zero_boundary_fraction': 'Maximum fraction at the zero boundary (includes initially zero cells)'}, key=['model', 'field'])


def ed07():
    rows, match = [], []
    for line in ('memory', 'learning'):
        p = paths.M1 / 'runs' / f'discover-{line}' / 'groups.json'
        g = io.json_load(p)
        for grp in g['groups']:
            rows.append(dict(line=line, group=grp['id'], parameter_group=grp['parameter_group'], type_labels=';'.join(grp['type_labels']), score=grp['score'],
                             cells=grp['count'], noninput=grp['noninput'], observed_cells=grp['observed_count'], selection_data=g['selection_data'],
                             validation_data=g['validation_data'], source_file=paths.rel(p)))
            for i, cov in enumerate(g['matching_covariates']):
                match.append(dict(line=line, group=grp['id'], covariate=cov, candidate_mean=grp['target_covariate_mean'][i],
                                  matched_mean=grp['control_covariate_mean'][i], match_mean_distance=grp['match_mean_distance'],
                                  observed_membership_exact=g['observed_membership_exact'], source_file=paths.rel(p)))
    csvout.write_table(pd.DataFrame(rows), 'extended/ed07/ed07a_group_selection.csv', title='Candidate-group selection (discovery data only)', describe={
        'line': 'learning or memory', 'group': 'Group identifier', 'parameter_group': 'Shared parameter group', 'type_labels': 'Cell-type labels',
        'score': 'Discovery score', 'cells': 'Cells', 'noninput': 'No direct-input cells', 'observed_cells': 'Readout-observed members',
        'selection_data': 'Data used for selection', 'validation_data': 'Confirmation data'}, key=['line', 'group'])
    csvout.write_table(pd.DataFrame(match), 'extended/ed07/ed07b_group_matching.csv', title='Matching quality of the control groups', describe={
        'line': 'learning or memory', 'group': 'Group identifier', 'covariate': 'Matching covariate', 'candidate_mean': 'Candidate mean',
        'matched_mean': 'Matched-group mean', 'match_mean_distance': 'Mean standardized matching distance',
        'observed_membership_exact': 'Observed-cell membership matched exactly'}, key=['line', 'group', 'covariate'])
    sp = paths.M1 / 'tables' / 'signed-pathways.csv'
    fp = paths.M1 / 'interpretation' / 'observed-direct-flux.csv'
    a = io.csv_load(sp)
    a.insert(0, 'record', 'signed_structural_paths')
    b = io.csv_load(fp)
    b.insert(0, 'record', 'observed_direct_flux')
    df = pd.concat([a, b], ignore_index=True)
    df['source_file'] = [paths.rel(sp)] * len(a) + [paths.rel(fp)] * len(b)
    df.insert(1, 'row', range(len(df)))
    desc = {c: f'As recorded in the source table ({c})' for c in df.columns if c not in ('record', 'row', 'source_file')}
    csvout.write_table(df, 'extended/ed07/ed07c_signed_pathways.csv', title='Signed structural pathways and recorded direct flux to observed cells',
                       describe={**desc, 'record': 'signed_structural_paths or observed_direct_flux', 'row': 'Row index'}, key=['row'])


ST5_DESC = {'source': 'mechanisms_v1, mechanisms_supplement, remote_batch1 or remote_supplement_causal', 'job': 'Job or run', 'family': 'Task family',
            'visible': 'Goal visible', 'unit': 'Model / visibility unit', 'case': 'Case', 'cue': 'Cue', 'context': 'Context', 'delay_ms': 'Delay (ms)',
            'prefix_windows': 'Prefix windows (timing rows: neutral windows)', 'split': 'Timing split as recorded', 'condition_raw': 'Condition label in the source',
            'intervention': 'Canonical intervention (see meta/intervention_taxonomy.csv)', 'component': 'State component', 'operation': 'clear or swap',
            'window': WINDOW_DESC, 'fraction': 'Interpolation dose', 'group': 'Cell group', 'side': 'candidate or matched', 'strength': 'Strength',
            'channel': 'Route channel', 'input_kind': 'Matched-duration input kind', 'total_postcue_ms': 'Cue offset to first decision (ms)',
            'qa_check': 'Sham, restoration or same-cue check (reported outside test families)', 'solved': 'Solved after the intervention',
            'baseline_solved': 'Baseline solved (decision window: no intervention; other windows: same-window sham; timing: matched blank)',
            'status': 'End state', 'first_push_correct': 'Correct first push after the intervention', 'baseline_first_push_correct': 'Correct first push at baseline',
            'margin_change': 'First-decision margin change vs baseline', 'choice_follows_donor': 'Remote probes: first choice followed the donor cue'}


def taxonomy_table(all_rows):
    t = all_rows.groupby(['source', 'condition_raw', 'intervention'], dropna=False).agg(
        component=('component', 'first'), operation=('operation', 'first'), window=('window', 'first'), qa_check=('qa_check', 'first'),
        records=('solved', 'size')).reset_index()
    for c in ('component', 'operation', 'window'):
        t[c] = t[c].astype(object).where(t[c].notna(), None)
    t['window'] = t['window'].astype(object)
    csvout.write_table(t, 'meta/intervention_taxonomy.csv', title='Mapping of source condition labels to the canonical intervention taxonomy', describe={
        **ST5_DESC, 'records': 'Per-history records with this label'}, key=['source', 'condition_raw'])


def main():
    m1 = m1_pairs()
    ok1, n1, n2 = gate_m1(m1)
    ms = ms_outcomes()
    ok2, n3 = gate_ms(ms)
    gates = pd.DataFrame([dict(gate='m1_memory_intervention_summary', reproduced=bool(ok1), rows=n1),
                          dict(gate='m1_lost_and_gained', reproduced=bool(ok1), rows=n2),
                          dict(gate='ms_final_state_factorial', reproduced=bool(ok2), rows=n3)])
    csvout.write_table(gates, 'qa/taxonomy_gates.csv', title='Exact reproduction of the published intervention summaries from per-history records',
                       describe={'gate': 'Published table', 'reproduced': 'Every count and mean reproduced exactly', 'rows': 'Rows in the published table'},
                       key=['gate'])
    if not (ok1 and ok2):
        raise RuntimeError('taxonomy gate failed')
    M1c = m1_canonical(m1)
    MSc = ms_canonical(ms)
    R0 = r0_interventions()
    RSc = rs_causal()
    T, raw = ms_timing()
    for df in (M1c, MSc, T):
        df['choice_follows_donor'] = None
    all_rows = pd.concat([M1c, MSc, R0, RSc, T], ignore_index=True)
    all_rows['analysis_status'] = 're_derived'
    csvout.write_table(all_rows, 'supplementary/st5_intervention_records.csv', title='Per-history intervention records in one taxonomy', describe=ST5_DESC,
                       key=None, sort=['source', 'unit', 'intervention', 'window', 'fraction', 'group', 'side', 'strength', 'channel', 'input_kind',
                                       'total_postcue_ms', 'case', 'delay_ms', 'prefix_windows'])
    taxonomy_table(all_rows)
    fig4c(M1c, R0, RSc)
    fig4d(M1c, RSc)
    fig4e()
    fig4f(ms)
    fig5c(T, raw)
    fig6d(ms)
    fig6e(M1c, ms)
    ed06(all_rows)
    ed07()
