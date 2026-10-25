"""P4: memory behaviour — per-history outcomes, cue-pair scoring, training outcomes (Fig. 4a), carried vs reset
state (Fig. 4b), timing profiles (Fig. 5a), learner comparison (Fig. 5b), transfer (Fig. 5d), S2 context (Fig. 5e), ED5."""
import numpy as np
import pandas as pd

from msa import csvout, io, memory, paths, registry, stats, taxonomy

HIST_KEYS = ['family', 'case', 'cue', 'context', 'visible', 'delay_ms', 'prefix_windows']
PAIR_KEYS = ['family', 'context', 'visible', 'delay_ms', 'prefix_windows']
DELAY_RANK = {d: i for i, d in enumerate(registry.GRID_DELAYS)}
REFERENCE = {'M0': 'remote-dynamic-M0-head_only', 'S1': 'supp-S1-ring-brain', 'S2': 'supp-S2-context-brain'}
HIST_DESC = {'family': 'Task family (M0, S1, S2, M0vis)', 'learner': 'Learner / training condition', 'kind': 'connectome, gru or trace',
             'reference': 'Reference model of the family (connectome readout-only)', 'case': 'Case (context-cue[-visibility])', 'cue': 'Cue (true cue label)',
             'context': 'Scene orientation (S2)', 'visible': 'Goal visible at decision', 'delay_ms': 'Blank delay after the cue (ms)',
             'prefix_windows': 'Common-prefix windows', 'split': 'train (training timing) or untrained', 'raw_status': 'End state as recorded',
             'status': 'Harmonized end state', 'solved': 'Full hidden-goal solution', 'steps': 'Moves', 'first_push_action': 'Action of the first box push (empty if none)',
             'first_push_correct': 'First irreversible push toward the cued goal (empty if the box never moved)',
             'first_decision_margin': 'First-decision margin (original batch: logit[target] - logit[other] of the first-step choice; supplement: entry margin)',
             'first_decision_correct': 'First-decision probe correct', 'margin_kind': 'Definition of first_decision_margin',
             'pair_class': 'Cue-pair class at this timing: both, one or none of the two cues solved'}


def histories():
    rows = []
    for m in memory.MODEL_INFO:
        rows.extend(memory.baseline_histories(m))
    H = pd.DataFrame(rows)
    H['first_push_action'] = H['first_push_action'].astype('Int64')
    return H


def cue_pairs(H):
    g = H.groupby(['model'] + PAIR_KEYS, dropna=False)
    P = g.agg(learner=('learner', 'first'), kind=('kind', 'first'), split=('split', 'first'), n_cues=('cue', 'nunique'),
              n_solved=('solved', 'sum'), n_push_correct=('first_push_correct', lambda s: int(np.nansum([bool(x) for x in s if x is not None and not pd.isna(x)]))),
              same_first_push=('first_push_action', lambda s: bool(s.notna().all() and s.nunique() == 1))).reset_index()
    if (P['n_cues'] != 2).any():
        raise RuntimeError('cue pairs must contain both cues')
    P['pair_class'] = P['n_solved'].map({2: 'both', 1: 'one', 0: 'none'})
    P['push_class'] = P['n_push_correct'].map({2: 'both', 1: 'one', 0: 'none'})
    return P


def crosschecks(H):
    rows = []
    mb_p = paths.M1 / 'tables' / 'memory-behavior.csv'
    mb = io.csv_load(mb_p)
    mb['visible'] = mb['visible'].astype(str).str.lower() == 'true'
    for model, g in mb.groupby('model'):
        mine = H[H.model == model]
        j = g.merge(mine, left_on=['delay', 'prefix', 'context', 'cue', 'visible'],
                    right_on=['delay_ms', 'prefix_windows', 'context', 'cue', 'visible'], how='left', suffixes=('_m1', ''))
        agree = (j['status_m1'].map(taxonomy.behavior) == j['status']).sum()
        rows.append(dict(check='m1_memory_behavior_vs_remote_records', model=model, n_compared=len(g), n_matched_keys=int(j['status'].notna().sum()),
                         n_status_agree=int(agree), n_steps_agree=int((j['steps_m1'] == j['steps']).sum()), source_file=paths.rel(mb_p)))
    dense = pd.DataFrame(memory.dense_delay_histories())
    ref = H[H.model == 'remote-dynamic-M0-head_only']
    j = ref.merge(dense, on=['delay_ms', 'prefix_windows', 'cue'], suffixes=('', '_dense'))
    rows.append(dict(check='remote_batch1_head_only_vs_supp_M0_delay', model='remote-dynamic-M0-head_only', n_compared=len(ref),
                     n_matched_keys=len(j), n_status_agree=int((j['status'] == j['status_dense']).sum()),
                     n_steps_agree=int((j['steps'] == j['steps_dense']).sum()), source_file=paths.rel(paths.RS / 'supp-M0-delay' / 'observations.jsonl')))
    df = pd.DataFrame(rows)
    csvout.write_table(df, 'qa/memory_behavior_crosscheck.csv', title='Cross-checks of per-history memory outcomes between independent records', describe={
        'check': 'Check', 'n_compared': 'Histories compared', 'n_matched_keys': 'Histories found in both records',
        'n_status_agree': 'Histories with identical end state', 'n_steps_agree': 'Histories with identical move count'}, key=['check', 'model'])
    return dense


def _rec(result, test_id, panel, family, contrast, unit, hs=None, status='new_registered', source='', level='B'):
    stats.record(result, test_id=test_id, panel=panel, family=family, contrast=contrast, unit=unit, level=level,
                 analysis_status=status, history_set_id=hs, source_table=source)


def _hs(family, split, unit='grid'):
    fam = 'M0vis' if family == 'M0vis' else family
    return f'{fam}.{"grid" if unit == "grid" else "pairs"}.{split}'


def main():
    H = histories()
    P = cue_pairs(H)
    H = H.merge(P[['model'] + PAIR_KEYS + ['pair_class']], on=['model'] + PAIR_KEYS, how='left')
    dense = crosschecks(H)
    H['analysis_status'] = 're_derived'
    csvout.write_table(H, 'extended/ed05/ed05a_memory_behaviour_all.csv', title='Every memory history for every model and control', describe=HIST_DESC,
                       key=['model', 'case', 'delay_ms', 'prefix_windows'], sort=['family', 'model', 'case', 'delay_ms', 'prefix_windows'])
    ed05b()
    ed05c(P)
    ed05d(dense)
    fig4a(H, P)
    fig4b(H, P)
    fig5a(H, P)
    fig5b(H, P)
    fig5d(H)
    fig5e(H, P)


def ed05b():
    rows = []
    for m in memory.MODEL_INFO:
        src = memory.root(m) / m / 'history.jsonl'
        if not src.exists():
            continue
        for i, x in enumerate(io.jsonl_load(src)):
            rows.append(dict(model=m, family=memory.MODEL_INFO[m]['family'], learner=memory.MODEL_INFO[m]['learner'], assessment=i + 1,
                             updates=int(x['updates']), elapsed_s=float(x['elapsed']), solved=int(x['solved']), total=int(x['total']),
                             probe_correct=x.get('probe_correct'), probe_total=x.get('probe_total'), loss=x.get('loss'), source_file=paths.rel(src)))
    csvout.write_table(pd.DataFrame(rows), 'extended/ed05/ed05b_memory_training_runs.csv', title='Learning histories of the memory training runs', describe={
        **HIST_DESC, 'assessment': 'Assessment index', 'updates': 'Cumulative updates', 'elapsed_s': 'Training-loop seconds',
        'solved': 'Training histories solved at the assessment', 'total': 'Training histories', 'probe_correct': 'Correct first choices',
        'probe_total': 'First-choice probes', 'loss': 'Training loss'}, key=['model', 'assessment'])


def ed05c(P):
    rows = []
    for (model, split, visible), g in P.groupby(['model', 'split', 'visible']):
        fam = g['family'].iloc[0]
        n = len(g)
        both, one, none = (int((g.pair_class == c).sum()) for c in ('both', 'one', 'none'))
        same = int(g['same_first_push'].sum())
        r = stats.sign_test(np.r_[np.ones(both), -np.ones(none)])
        tid = f'ed05c.memory_use.{model}.{split}' + ('.visible' if visible else '')
        _rec(r, tid, 'ed05c', f'F5.memory_use.{fam}', f'{model} ({split}{", visible goal" if visible else ""}): cue pairs with both cues solved vs neither',
             'cue_pair', hs=_hs(fam, split, 'pairs'), status='post_hoc', source='extended/ed05/ed05c_cue_pair_classes.csv')
        rows.append(dict(model=model, family=fam, learner=g['learner'].iloc[0], kind=g['kind'].iloc[0], split=split, visible=bool(visible),
                         cue_pairs=n, both_solved=both, one_solved=one, none_solved=none, same_first_push_both_cues=same,
                         histories_solved=int(g['n_solved'].sum()), histories=2 * n, p_value=r['p_value'], test_id=tid,
                         history_set_id=_hs(fam, split, 'pairs'), analysis_status='post_hoc'))
    csvout.write_table(pd.DataFrame(rows), 'extended/ed05/ed05c_cue_pair_classes.csv', title='Cue-pair classes for every memory model',
                       describe={**HIST_DESC, 'cue_pairs': 'Timings (cue pairs) evaluated', 'both_solved': 'Pairs with both cues solved (memory-consistent)',
                                 'one_solved': 'Pairs with exactly one cue solved (what a memoryless policy produces)', 'none_solved': 'Pairs with neither cue solved',
                                 'same_first_push_both_cues': 'Pairs whose two cues produced the same first push', 'histories_solved': 'Histories solved',
                                 'histories': 'Histories evaluated', 'p_value': 'Exact sign test, both vs none'},
                       key=['model', 'split', 'visible'], sort=['family', 'model', 'visible', 'split'])


def ed05d(dense):
    A = io.json_load(paths.RS_ROOT / 'analysis.json')
    run = next(r for r in A['runs'] if r['id'] == 'supp-M0-delay')
    dec = pd.DataFrame(run['cue_decoder']['records'])
    d = dense.merge(dec[['delay_ms', 'prefix_windows', 'cue', 'train_time', 'decoded_correct', 'solved']],
                    on=['delay_ms', 'prefix_windows', 'cue'], how='left', suffixes=('', '_decoder_record'))
    if (d['solved'] != d['solved_decoder_record']).any():
        raise RuntimeError('dense delay scan: decoder records disagree with behaviour')
    d = d.drop(columns=['solved_decoder_record'])
    d['elapsed_after_cue_ms'] = d['delay_ms'] + 40 * d['prefix_windows']
    d['analysis_status'] = 're_derived'
    d['source_file'] = d['source_file'] + ';' + paths.rel(paths.RS_ROOT / 'analysis.json')
    csvout.write_table(d, 'extended/ed05/ed05d_dense_delay_scan.csv', title='Dense delay scan (M0 readout-only checkpoint) with the fixed linear cue decoder', describe={
        **HIST_DESC, 'train_time': 'Delay and prefix both used in decoder fitting (training timing)', 'decoded_correct': 'Ridge cue decoder correct (descriptive, fitted on 18 training-timing rows)',
        'elapsed_after_cue_ms': 'Delay + 40 ms per prefix window'}, key=['delay_ms', 'prefix_windows', 'case'])
    pairs = d.groupby(['delay_ms', 'prefix_windows']).agg(train=('train_time', 'first'), dec=('decoded_correct', 'sum'), sol=('solved', 'sum')).reset_index()
    for split, g in (('train', pairs[pairs.train]), ('untrained', pairs[~pairs.train])):
        both, none = int((g.dec == 2).sum()), int((g.dec == 0).sum())
        _rec(stats.sign_test(np.r_[np.ones(both), -np.ones(none)]), f'ed05d.decoder_pairs.{split}', 'ed05d', 'F5d.decoding',
             f'dense scan ({split} timings): cue pairs decoded both-correct vs both-wrong', 'cue_pair', status='post_hoc',
             source='extended/ed05/ed05d_dense_delay_scan.csv')
        bs, ns = int((g.sol == 2).sum()), int((g.sol == 0).sum())
        _rec(stats.sign_test(np.r_[np.ones(bs), -np.ones(ns)]), f'ed05d.behaviour_pairs.{split}', 'ed05d', 'F5.memory_use.M0',
             f'dense scan ({split} timings): cue pairs with both cues solved vs neither (M0 readout-only)', 'cue_pair', status='post_hoc',
             source='extended/ed05/ed05d_dense_delay_scan.csv')


def _extension_summary(model):
    src = memory.root(model) / model / 'extensions.jsonl'
    if not src.exists():
        return {}
    e = io.jsonl_load(src)
    last = e[-1]
    lost = sorted(set(last.get('required_histories') or []) - set(last.get('solved_histories') or []))
    return dict(final_segment=int(last['phase']), extension_granted=bool(last['progress']),
                entry_margin_change=(last['min_choice_margin'] - last['phase_margin']) if last.get('phase_margin') is not None else None,
                required_histories_lost=len(lost))


def fig4a(H, P):
    res_rows = []
    for m, info in memory.MODEL_INFO.items():
        if info['kind'] != 'connectome' or info['learner'] == 'zero_update_transfer_from_M0':
            continue
        res_p = memory.root(m) / m / 'result.json'
        res = io.json_load(res_p)
        pc = (res.get('parameter_changes') or {}).get('core') or {}
        core_changed = int(sum((v or {}).get('changed', 0) for v in pc.values())) if pc else 0
        ext = _extension_summary(m)
        for visible in sorted(H[H.model == m]['visible'].unique()):
            h = H[(H.model == m) & (H.visible == visible)]
            p = P[(P.model == m) & (P.visible == visible)]
            row = dict(model=m, family=info['family'], learner=info['learner'], visible=bool(visible), status=res.get('status'),
                       status_group=taxonomy.training_group(res.get('status')), updates=res.get('updates'),
                       training_minutes=(res.get('training_seconds') or 0) / 60, core_parameters_changed=core_changed,
                       final_segment=ext.get('final_segment'), extension_granted=ext.get('extension_granted'),
                       entry_margin_change_at_last_segment=ext.get('entry_margin_change'), required_histories_lost=ext.get('required_histories_lost'))
            for split in ('train', 'untrained'):
                hh = h[h.split == split]
                k, n = int(hh.solved.sum()), len(hh)
                lo, hi = stats.cp_interval(k, n)
                pp = p[p.split == split]
                row.update({f'{split}_solved': k, f'{split}_histories': n, f'{split}_ci_low': lo, f'{split}_ci_high': hi,
                            f'{split}_pairs_both': int((pp.pair_class == 'both').sum()), f'{split}_pairs_one': int((pp.pair_class == 'one').sum()),
                            f'{split}_pairs_none': int((pp.pair_class == 'none').sum())})
            row['source_file'] = paths.rel(res_p)
            res_rows.append(row)
    df = pd.DataFrame(res_rows)
    df['updates'] = df['updates'].astype('Int64')
    desc = {**HIST_DESC, 'status': 'Training end status', 'status_group': 'Grouped status', 'updates': 'Updates', 'training_minutes': 'Training-loop minutes',
            'core_parameters_changed': 'Core parameter elements changed during memory training (0 = frozen brain)',
            'final_segment': 'Last time segment reached', 'extension_granted': 'Extension granted at the last segment boundary',
            'entry_margin_change_at_last_segment': 'Minimum choice margin minus the segment-start margin at the last boundary',
            'required_histories_lost': 'Required (previously solved) training histories lost at the last boundary'}
    for split in ('train', 'untrained'):
        desc.update({f'{split}_solved': f'{split} histories solved', f'{split}_histories': f'{split} histories evaluated',
                     f'{split}_ci_low': 'Clopper-Pearson 95% lower limit', f'{split}_ci_high': 'Clopper-Pearson 95% upper limit',
                     f'{split}_pairs_both': f'{split} cue pairs with both cues solved', f'{split}_pairs_one': f'{split} cue pairs with one cue solved',
                     f'{split}_pairs_none': f'{split} cue pairs with neither cue solved'})
    csvout.write_table(df, 'main/fig4/fig4a_memory_training_outcomes.csv', title='Memory training outcomes for connectome conditions (level C)',
                       describe=desc, key=['model', 'visible'], sort=['family', 'model', 'visible'])


def fig4b(H, P):
    models = ['remote-dynamic-M0-head_only', 'remote-dynamic-M0-reset', 'remote-dynamic-M0-carry_untrained', 'remote-dynamic-M0-joint']
    h = H[H.model.isin(models)].copy()
    out = h[['model', 'learner', 'case', 'cue', 'delay_ms', 'prefix_windows', 'split', 'status', 'solved', 'steps', 'first_push_action',
             'first_push_correct', 'first_decision_margin', 'pair_class', 'source_file']].copy()
    out['history_set_id'] = out['split'].map(lambda s: f'M0.grid.{s}')
    out['analysis_status'] = 're_derived'
    csvout.write_table(out, 'main/fig4/fig4b_carry_vs_reset.csv', title='Carried vs reset state on M0, paired by history and cue pair', describe=HIST_DESC,
                       key=['model', 'case', 'delay_ms', 'prefix_windows'], sort=['model', 'split', 'delay_ms', 'prefix_windows', 'case'])
    W = h.pivot_table(index=['case', 'delay_ms', 'prefix_windows', 'split'], columns='model', values='solved', aggfunc='first').reset_index()
    Pw = P[P.model.isin(models)].pivot_table(index=['delay_ms', 'prefix_windows', 'split'], columns='model', values='pair_class', aggfunc='first').reset_index()
    for split in ('train', 'untrained'):
        fam = 'F4b.trained' if split == 'train' else 'F4b.untrained'
        w = W[W.split == split]
        pw = Pw[Pw.split == split]
        for other in ('remote-dynamic-M0-reset', 'remote-dynamic-M0-carry_untrained'):
            short = other.replace('remote-dynamic-M0-', '')
            _rec(stats.mcnemar(w['remote-dynamic-M0-head_only'].astype(bool), w[other].astype(bool)), f'fig4b.mcnemar.head_only_vs_{short}.{split}', 'fig4b',
                 fam, f'histories solved: carried-state readout-only vs {short} ({split})', 'history', hs=f'M0.grid.{split}',
                 source='main/fig4/fig4b_carry_vs_reset.csv', status='new_registered')
            _rec(stats.mcnemar(pw['remote-dynamic-M0-head_only'] == 'both', pw[other] == 'both'), f'fig4b.mcnemar_pairs.head_only_vs_{short}.{split}',
                 'fig4b', fam, f'cue pairs with both cues solved: readout-only vs {short} ({split})', 'cue_pair', hs=f'M0.pairs.{split}',
                 source='main/fig4/fig4b_carry_vs_reset.csv', status='post_hoc')


def fig5a(H, P):
    connectome = [m for m, i in memory.MODEL_INFO.items() if i['kind'] == 'connectome' and i['learner'] != 'zero_update_transfer_from_M0']
    rows = []
    for (m, vis, ctx, d, pf), g in H[H.model.isin(connectome)].groupby(['model', 'visible', 'context', 'delay_ms', 'prefix_windows']):
        pc = g['pair_class'].iloc[0]
        rows.append(dict(model=m, family=g['family'].iloc[0], learner=g['learner'].iloc[0], visible=bool(vis), context=int(ctx), delay_ms=int(d),
                         prefix_windows=int(pf), split=g['split'].iloc[0], numerator=int(g.solved.sum()), denominator=len(g), pair_class=pc,
                         first_push_correct=int(np.nansum([bool(x) for x in g.first_push_correct if x is not None and not pd.isna(x)])),
                         unit='cue_pair', history_set_id=_hs(g['family'].iloc[0], g['split'].iloc[0], 'pairs'), analysis_status='re_derived',
                         source_file=g['source_file'].iloc[0]))
    df = pd.DataFrame(rows)
    csvout.write_table(df, 'main/fig5/fig5a_timing_profiles.csv', title='Delay x prefix profiles for connectome memory models', describe={
        **HIST_DESC, 'numerator': 'Cues solved at this timing (of 2)', 'denominator': 'Cues at this timing',
        'first_push_correct': 'Cues with a correct first push'}, key=['model', 'visible', 'context', 'delay_ms', 'prefix_windows'])
    for m in connectome:
        for vis in sorted(H[H.model == m]['visible'].unique()):
            h = H[(H.model == m) & (H.visible == vis) & (H.split == 'untrained')]
            p = P[(P.model == m) & (P.visible == vis) & (P.split == 'untrained')]
            sfx = '.visible' if vis else ''
            fam = h['family'].iloc[0]
            _rec(stats.cochran_armitage(h['delay_ms'].map(DELAY_RANK), h['solved']), f'fig5a.ca_histories.{m}{sfx}', 'fig5a', 'F5a.histories',
                 f'{m}{sfx}: trend of per-history solution over delay rank (untrained timings)', 'history', hs=_hs(fam, 'untrained'),
                 source='main/fig5/fig5a_timing_profiles.csv')
            _rec(stats.cochran_armitage(p['delay_ms'].map(DELAY_RANK), p['pair_class'] == 'both'), f'fig5a.ca_pairs.{m}{sfx}', 'fig5a', 'F5a.pairs',
                 f'{m}{sfx}: trend of both-cue solution over delay rank (untrained cue pairs)', 'cue_pair', hs=_hs(fam, 'untrained', 'pairs'),
                 source='main/fig5/fig5a_timing_profiles.csv', status='post_hoc')


def fig5b(H, P):
    fams = ['M0', 'S1', 'S2']
    h = H[H.family.isin(fams)].copy()
    out = h[['family', 'model', 'learner', 'kind', 'reference', 'case', 'cue', 'context', 'delay_ms', 'prefix_windows', 'split', 'status', 'solved',
             'first_push_correct', 'pair_class', 'source_file']].copy()
    out['history_set_id'] = [f'{f}.grid.{s}' for f, s in zip(out.family, out.split)]
    out['analysis_status'] = 're_derived'
    csvout.write_table(out, 'main/fig5/fig5b_learner_comparison.csv', title='Learner comparison on identical histories (conditional on these trained instances)',
                       describe=HIST_DESC, key=['model', 'case', 'delay_ms', 'prefix_windows'], sort=['family', 'model', 'split', 'case', 'delay_ms', 'prefix_windows'])
    for fam in fams:
        ref = REFERENCE[fam]
        others = [m for m, i in memory.MODEL_INFO.items() if i['family'] == fam and m != ref]
        for split in ('train', 'untrained'):
            hf = h[(h.family == fam) & (h.split == split)]
            W = hf.pivot_table(index=['case', 'delay_ms', 'prefix_windows'], columns='model', values='solved', aggfunc='first')
            pf = P[(P.family == fam) & (P.split == split)]
            Pw = pf.pivot_table(index=['context', 'delay_ms', 'prefix_windows'], columns='model', values='pair_class', aggfunc='first')
            for o in others:
                if W[o].isna().any() or W[ref].isna().any():
                    raise RuntimeError(f'history keys differ between {ref} and {o}')
                _rec(stats.mcnemar(W[ref].astype(bool), W[o].astype(bool)), f'fig5b.mcnemar.{fam}.{split}.{o}', 'fig5b', f'F5b.{fam}.{split}',
                     f'{fam} {split}: reference connectome readout-only vs {memory.MODEL_INFO[o]["learner"]} ({o})', 'history',
                     hs=f'{fam}.grid.{split}', source='main/fig5/fig5b_learner_comparison.csv')
                _rec(stats.mcnemar(Pw[ref] == 'both', Pw[o] == 'both'), f'fig5b.mcnemar_pairs.{fam}.{split}.{o}', 'fig5b', f'F5b.pairs.{fam}.{split}',
                     f'{fam} {split}: both-cue solution, reference vs {memory.MODEL_INFO[o]["learner"]} ({o})', 'cue_pair',
                     hs=f'{fam}.pairs.{split}', source='main/fig5/fig5b_learner_comparison.csv', status='post_hoc')


def fig5d(H):
    t = H[H.learner == 'zero_update_transfer_from_M0'][['model', 'family', 'case', 'cue', 'context', 'delay_ms', 'prefix_windows', 'split', 'status',
                                                        'solved', 'steps', 'first_push_action', 'first_push_correct', 'pair_class', 'source_file']].copy()
    t.insert(0, 'record_type', 'zero_update_behaviour')
    dp = paths.M1 / 'tables' / 'cross-family-memory-decoding.csv'
    d = io.csv_load(dp)
    dec = pd.DataFrame(dict(record_type='cross_family_cue_decoding', model=d['evaluation_model'],
                            family=d['family'].map({'M0-audit': 'M0', 'S1-ring': 'S1', 'S2-context': 'S2'}),
                            case=[f'c{c}-cue{q}' for c, q in zip(d['context'], d['cue'])], cue=d['cue'], context=d['context'], delay_ms=d['delay'],
                            prefix_windows=d['prefix'], split=np.where(d['heldout_history'], 'untrained', 'train'), status=None,
                            solved=None, steps=None, first_push_action=None, first_push_correct=None, pair_class=None, source_file=paths.rel(dp)))
    dec['decoded_correct'] = d['decoded_correct'].astype(bool).to_numpy()
    dec['decoder_fitted_on_family'] = d['fit_on_this_family'].astype(bool).to_numpy()
    t['decoded_correct'] = None
    t['decoder_fitted_on_family'] = None
    df = pd.concat([t, dec], ignore_index=True)
    df['first_push_action'] = df['first_push_action'].astype('Int64')
    df['analysis_status'] = 're_derived'
    csvout.write_table(df, 'main/fig5/fig5d_cross_layout_transfer.csv', title='Cross-layout transfer: zero-update behaviour and cross-family cue decoding', describe={
        **HIST_DESC, 'record_type': 'zero_update_behaviour (M0 model evaluated on S1/S2) or cross_family_cue_decoding (decoder fitted on M0 training histories)',
        'decoded_correct': 'Decoder predicted the cue', 'decoder_fitted_on_family': 'Decoder fitted on this family'},
        key=['record_type', 'model', 'case', 'delay_ms', 'prefix_windows'])
    for (model, fam, split), g in dec.groupby(['model', 'family', 'split']):
        pairs = g.groupby(['context', 'delay_ms', 'prefix_windows'])['decoded_correct'].sum()
        both, none = int((pairs == 2).sum()), int((pairs == 0).sum())
        _rec(stats.sign_test(np.r_[np.ones(both), -np.ones(none)]), f'fig5d.decoder_pairs.{fam}.{split}', 'fig5d', 'F5d.decoding',
             f'M0-fitted cue decoder on {fam} ({split}): cue pairs both-correct vs both-wrong', 'cue_pair', hs=f'{fam}.pairs.{split}',
             source='main/fig5/fig5d_cross_layout_transfer.csv', status='post_hoc')


def fig5e(H, P):
    h = H[H.family == 'S2']
    rows = []
    for (m, ctx, cue, split), g in h.groupby(['model', 'context', 'cue', 'split']):
        k, n = int(g.solved.sum()), len(g)
        lo, hi = stats.cp_interval(k, n)
        rows.append(dict(model=m, learner=g['learner'].iloc[0], context=int(ctx), cue=int(cue), split=split, numerator=k, denominator=n, estimate=k / n,
                         ci_low=lo, ci_high=hi, ci_method='clopper_pearson_exact', unit='history', history_set_id=f'S2.grid.{split}',
                         analysis_status='re_derived', source_file=g['source_file'].iloc[0]))
    csvout.write_table(pd.DataFrame(rows), 'main/fig5/fig5e_context_s2.csv', title='S2 conditional use: orientation x cue outcomes', describe=HIST_DESC,
                       key=['model', 'context', 'cue', 'split'], sort=['model', 'split', 'context', 'cue'])
    s2 = h[h.model == REFERENCE['S2']]
    for split in ('train', 'untrained'):
        w = s2[s2.split == split].pivot_table(index=['cue', 'delay_ms', 'prefix_windows'], columns='context', values='solved', aggfunc='first')
        _rec(stats.mcnemar(w[0].astype(bool), w[1].astype(bool)), f'fig5e.mcnemar.orientation.{split}', 'fig5e', 'F5e',
             f'S2 readout-only, {split}: orientation 0 vs 180 degrees, paired by cue, delay and prefix', 'history', hs=f'S2.grid.{split}',
             source='main/fig5/fig5e_context_s2.csv')
