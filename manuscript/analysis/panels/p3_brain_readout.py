"""P3: training controls, control costs, four-combination swaps, margin decomposition (gate),
parameter-change anatomy with stratified nulls, readout geometry across maps, ST4."""
import math

import numpy as np
import pandas as pd

from msa import anatomy, csvout, io, paths, registry, static_runs, stats, taxonomy

CONDITIONS = ['joint', 'head_only', 'brain_only', 'frozen', 'demonstration', 'no_recovery', 'entry_ce', 'alternating',
              'neutral', 'rewired', 'weight_shuffle', 'board_mlp']
MAPS = ['D1-T0', 'D3-T0', 'D5-T0', 'D7-T0']
LOG_RANGE = math.log(1.25) - math.log(0.8)
NORM = {'pre_log_gain': LOG_RANGE, 'post_log_gain': LOG_RANGE, 'threshold_mv': 2.0}
TOP_FRACTION = 0.01
N_NULL = stats.N_RESAMPLES


def _comp():
    # Our own P2 output: read directly so it is not listed as a study input.
    return pd.read_csv(paths.out('meta/compute_reconstruction.csv')).set_index('run_id')


# ---------------------------------------------------------------- Fig. 3a / 3b

def control_runs():
    comp = _comp()
    sched = {e['id']: e for e in io.json_load(paths.STUDY / 'schedule.json')}
    rows = []
    for m in MAPS:
        k = m.split('-')[0]
        for cond in CONDITIONS:
            if cond == 'joint':
                rid, loc, layer = f'difficulty-{m}', paths.RUNS / f'difficulty-{m}', 'local_static'
            elif k == 'D7' and cond in ('head_only', 'demonstration'):
                rid, loc, layer = f'remote-D7-{cond}', paths.R0 / f'remote-D7-{cond}', 'remote_static'
            else:
                rid, loc, layer = f'control-{k}-{cond}', paths.RUNS / f'control-{k}-{cond}', 'local_static'
            if not loc.exists():
                e = sched.get(rid)
                c = (e or {}).get('condition') or {}
                rows.append(dict(condition=cond, map_id=m, run_id=rid, layer='registered_not_run' if e else 'not_registered',
                                 train_core=c.get('train_core'), train_head=c.get('train_head'), loss=c.get('loss'), schedule=c.get('schedule'),
                                 initialization=c.get('initialization'), structure=c.get('structure'), model_kind=c.get('model'),
                                 status='pending' if e else None, status_group='never_run', final_solved=None, reload_verified=None,
                                 final_steps=None, updates=None, training_seconds=None, forward_calls=None, neuron_ms=None,
                                 core_pre_changed=None, core_post_changed=None, core_threshold_changed=None, core_l2=None,
                                 readout_weight_changed=None, readout_l2=None, hardware=None,
                                 source_file=paths.rel(paths.STUDY / 'schedule.json')))
                continue
            cfg = io.json_load(loc / 'config.json')
            c = cfg['condition']
            res = io.json_load(loc / 'result.json') if (loc / 'result.json').exists() else {}
            ver = io.json_load(loc / 'verification.json') if (loc / 'verification.json').exists() else {}
            env = io.json_load(loc / 'environment.json') if (loc / 'environment.json').exists() else {}
            pc = res.get('parameter_changes') or {}
            core = pc.get('core') or {}
            head = pc.get('head') or {}
            status = res.get('status') or (io.json_load(loc / 'progress.json').get('status') if (loc / 'progress.json').exists() else None)
            calls = comp.loc[rid, 'uninterrupted_equivalent_calls'] if rid in comp.index else None
            rows.append(dict(condition=cond, map_id=m, run_id=rid, layer=layer, train_core=c['train_core'], train_head=c['train_head'],
                             loss=c['loss'], schedule=c['schedule'], initialization=c['initialization'], structure=c['structure'],
                             model_kind=c['model'], status=status, status_group=taxonomy.training_group(status),
                             final_solved=res.get('final_solved'), reload_verified=(ver.get('status') == 'verified') if ver else None,
                             final_steps=res.get('final_steps'), updates=res.get('updates'), training_seconds=res.get('training_seconds'),
                             forward_calls=None if calls is None or pd.isna(calls) else int(calls),
                             neuron_ms=None if calls is None or pd.isna(calls) else int(calls) * 106947840,
                             core_pre_changed=(core.get('pre_log_gain') or {}).get('changed'),
                             core_post_changed=(core.get('post_log_gain') or {}).get('changed'),
                             core_threshold_changed=(core.get('threshold_mv') or {}).get('changed'),
                             core_l2=math.sqrt(sum((v or {}).get('l2', 0) ** 2 for v in core.values())) if core else None,
                             readout_weight_changed=(head.get('proposal.weight') or {}).get('changed'),
                             readout_l2=math.sqrt(sum((head.get(k) or {}).get('l2', 0) ** 2 for k in ('proposal.weight', 'proposal.bias'))) if head else None,
                             hardware=env.get('gpu'), source_file=paths.rel(loc)))
    return pd.DataFrame(rows)


def fig3ab():
    df = control_runs()
    for c in ('updates', 'forward_calls', 'neuron_ms', 'final_steps', 'core_pre_changed', 'core_post_changed', 'core_threshold_changed',
              'readout_weight_changed'):
        df[c] = df[c].astype('Int64')
    desc = {'condition': 'Training condition', 'layer': 'local_static, remote_static or registered_not_run',
            'train_core': 'Brain parameters trained', 'train_head': 'Readout trained', 'loss': 'Loss variant', 'schedule': 'suffix or demonstration',
            'initialization': 'p05 (inherited) or neutral', 'structure': 'real, rewired or weight_shuffle graph', 'model_kind': 'brain_linear or board_mlp',
            'status': 'End status', 'status_group': 'Grouped status', 'final_solved': 'Final rollout solved', 'reload_verified': 'Independent reload verified',
            'final_steps': 'Moves in the final rollout', 'updates': 'Optimizer updates', 'training_seconds': 'Training-loop seconds',
            'forward_calls': 'Simulator forward calls (rebuilt)', 'neuron_ms': 'Simulated neuron-ms', 'core_pre_changed': 'Pre-gain groups changed',
            'core_post_changed': 'Post-gain groups changed', 'core_threshold_changed': 'Threshold groups changed',
            'core_l2': 'L2 norm of all core parameter changes', 'readout_weight_changed': 'Readout weights changed',
            'readout_l2': 'L2 norm of readout weight and bias changes', 'hardware': 'GPU'}
    csvout.write_table(df, 'main/fig3/fig3a_training_controls.csv', title='Training-control matrix (descriptive, level C)', describe=desc,
                       key=['condition', 'map_id'], sort=['map_id', 'condition'])
    rows = []
    for m in MAPS:
        j = df[(df.map_id == m) & (df.condition == 'joint')].iloc[0]
        for _, r in df[(df.map_id == m) & (df.condition != 'joint') & (df.status_group != 'never_run')].iterrows():
            same_hw = (r['hardware'] == j['hardware']) and r['hardware'] is not None
            rows.append(dict(condition=r['condition'], map_id=m, run_id=r['run_id'], joint_run_id=j['run_id'], joint_solved=bool(j['final_solved']),
                             control_solved=bool(r['final_solved']) if r['final_solved'] is not None and not pd.isna(r['final_solved']) else False,
                             control_status=r['status'], joint_updates=int(j['updates']),
                             control_updates=None if pd.isna(r['updates']) else int(r['updates']),
                             updates_ratio=None if pd.isna(r['updates']) else int(r['updates']) / int(j['updates']),
                             neuron_ms_ratio=(None if pd.isna(r['neuron_ms']) or pd.isna(j['neuron_ms']) else int(r['neuron_ms']) / int(j['neuron_ms'])),
                             hardware_matched=bool(same_hw),
                             training_seconds_ratio=(r['training_seconds'] / j['training_seconds']) if same_hw and pd.notna(r['training_seconds']) else None,
                             source_file=r['source_file']))
    b = pd.DataFrame(rows)
    b['control_updates'] = b['control_updates'].astype('Int64')
    summ = []
    for cond, g in b.groupby('condition'):
        g3 = g[g.map_id != 'D7-T0']
        summ.append(dict(condition=cond, maps_compared=len(g3), control_solved_maps=int(g3.control_solved.sum()),
                         fewer_updates_than_joint=int((g3.updates_ratio < 1).sum()),
                         all_solved=bool(len(g3) and g3.control_solved.all()), none_solved=bool(len(g3) and not g3.control_solved.any()),
                         d7_available=bool((g.map_id == 'D7-T0').any())))
    s = pd.DataFrame(summ)
    b = b.merge(s, on='condition', how='left')
    csvout.write_table(b, 'main/fig3/fig3b_control_costs.csv', title='Control cost relative to joint training, paired by map (level C, no tests)', describe={
        'condition': 'Control condition', 'joint_run_id': 'Joint-training run on the same map', 'joint_solved': 'Joint run solved',
        'control_solved': 'Control run solved', 'control_status': 'Control end status', 'joint_updates': 'Updates of the joint run',
        'control_updates': 'Updates of the control run', 'updates_ratio': 'Control updates / joint updates',
        'neuron_ms_ratio': 'Control neuron-ms / joint neuron-ms (not applicable to the board MLP)',
        'hardware_matched': 'Control and joint ran on the same GPU type', 'training_seconds_ratio': 'Training-seconds ratio (same hardware only)',
        'maps_compared': 'D1/D3/D5 maps with both runs', 'control_solved_maps': 'D1/D3/D5 maps solved by the control',
        'fewer_updates_than_joint': 'D1/D3/D5 maps where the control used fewer updates',
        'all_solved': 'Control solved all D1/D3/D5 maps', 'none_solved': 'Control solved none of D1/D3/D5', 'd7_available': 'A D7 run exists'},
        key=['condition', 'map_id'], sort=['condition', 'map_id'])


# ---------------------------------------------------------------- Fig. 3c

def fig3c():
    p = paths.M1 / 'interpretation' / 'four-combination-behavior.csv'
    fc = io.csv_load(p, dtype={'combination': str})
    fc['combination'] = fc['combination'].str.zfill(2)
    fc['brain'] = fc['combination'].str[0].map({'0': 'old', '1': 'new'})
    fc['readout'] = fc['combination'].str[1].map({'0': 'old', '1': 'new'})
    fc['solved'] = fc['solved'].astype(str).str.lower() == 'true'
    fc['status'] = fc['status'].map(taxonomy.behavior)
    fc['set'] = np.where(fc['model'].str.startswith('difficulty-'), 'difficulty', 'other')
    fc['map_id'] = fc['model'].str.replace('difficulty-', '', regex=False).where(fc['set'] == 'difficulty')
    fc['source_file'] = paths.rel(p)
    fc['analysis_status'] = 're_derived'
    out = fc[['set', 'model', 'map_id', 'combination', 'brain', 'readout', 'status', 'steps', 'solved', 'analysis_status', 'source_file']]
    csvout.write_table(out, 'main/fig3/fig3c_four_combination.csv', title='Four-combination behaviour: old/new brain x old/new readout', describe={
        'set': 'difficulty (24 maps, tested) or other static model', 'combination': 'Brain digit then readout digit; 0 = before training, 1 = after',
        'brain': 'Brain parameters (old = before, new = after)', 'readout': 'Readout parameters', 'status': 'Rollout end state',
        'steps': 'Moves', 'solved': 'Solved'}, key=['model', 'combination'], sort=['set', 'model', 'combination'])
    d = fc[fc.set == 'difficulty'].pivot(index='model', columns='combination', values='solved')
    for other in ('00', '10', '01'):
        r = stats.mcnemar(d['11'].to_numpy(), d[other].to_numpy())
        stats.record(r, test_id=f'fig3c.mcnemar.11_vs_{other}', panel='fig3c', family='F3c', contrast=f'new brain + new readout vs pairing {other}',
                     unit='map', level='A', analysis_status='new_registered', source_table='main/fig3/fig3c_four_combination.csv')


# ---------------------------------------------------------------- Fig. 3d (gate)

def fig3d():
    saved_p = paths.M1 / 'interpretation' / 'decomposition-by-model.csv'
    saved = io.csv_load(saved_p).set_index('model')
    rows = []
    for d in sorted((paths.M1 / 'runs').glob('record-*')):
        z_p = d / 'decomposition.npz'
        if not z_p.exists():
            continue
        model = d.name.replace('record-', '')
        z = io.npz_load(z_p)
        bank = io.json_load(d / 'bank.json')
        targets = np.array([x['action'] for x in bank['labels']])
        legal = np.array([x['legal'] for x in bank['labels']], dtype=bool)
        legal[np.arange(len(targets)), targets] = False
        alt = legal.any(1)
        comp = np.where(legal, z['q11'], -np.inf).argmax(1)
        terms = {}
        for name in ('brain', 'head', 'interaction'):
            m = z[name][np.arange(len(targets)), targets] - z[name][np.arange(len(targets)), comp]
            terms[name] = float(m[alt].mean()) if alt.any() else float('nan')
        # Independent check: recompute the four logit sets from matched features and saved readouts.
        pb = io.npz_load(d / 'parameters-before.npz')
        pa = io.npz_load(d / 'parameters-after.npz')
        fb = io.npz_load(d / 'matched-before.npz', keys=('features',))['features'].astype(np.float64)
        fa = io.npz_load(d / 'matched-after.npz', keys=('features',))['features'].astype(np.float64)

        def logits(f, p):
            return np.clip((f - p['head_mean']) / p['head_scale'], -10, 10) @ p['head_proposal.weight'].T.astype(np.float64) + p['head_proposal.bias']

        q = {'q00': logits(fb, pb), 'q10': logits(fa, pb), 'q01': logits(fb, pa), 'q11': logits(fa, pa)}
        qdiff = max(float(np.max(np.abs(q[k] - z[k]))) for k in q)
        total = terms['brain'] + terms['head'] + terms['interaction']
        srow = saved.loc[model] if model in saved.index else None
        dev = max(abs(terms[k] - float(srow[k])) for k in ('brain', 'head', 'interaction')) if srow is not None else None
        rows.append(dict(model=model, set='difficulty' if model.startswith('difficulty-') else 'other',
                         map_id=model.replace('difficulty-', '') if model.startswith('difficulty-') else None,
                         brain_term=terms['brain'], readout_term=terms['head'], interaction_term=terms['interaction'], total=total,
                         compared_states=int(alt.sum()), single_legal_action_states=int((~alt).sum()),
                         reconstruction_error=float(z['reconstruction_error']), max_abs_logit_recompute_diff=qdiff,
                         max_abs_deviation_from_saved_table=dev, analysis_status='re_derived',
                         source_file=';'.join([paths.rel(z_p), paths.rel(d / 'bank.json'), paths.rel(saved_p)])))
    df = pd.DataFrame(rows)
    bad = df[(df.max_abs_deviation_from_saved_table > 1e-6) | (df.max_abs_logit_recompute_diff > 1e-3)]
    if len(bad) or (df.set == 'difficulty').sum() != 24:
        raise RuntimeError(f'fig3d gate failed for {bad.model.tolist()}')
    csvout.write_table(df, 'main/fig3/fig3d_margin_decomposition.csv', title='Margin decomposition into brain, readout and interaction terms', describe={
        'set': 'difficulty (24 maps, tested) or other static model', 'brain_term': 'Mean change of the fixed reference-action margin from the brain (q10 - q00)',
        'readout_term': 'Mean change from the readout (q01 - q00)', 'interaction_term': 'Mean interaction (q11 - q10 - q01 + q00)',
        'total': 'Sum of the three terms (= q11 - q00)', 'compared_states': 'Reference states with at least one legal alternative',
        'single_legal_action_states': 'Reference states without a legal alternative (excluded)',
        'reconstruction_error': 'Saved maximum reconstruction error of the decomposition',
        'max_abs_logit_recompute_diff': 'Maximum absolute difference between logits recomputed from matched features and saved logits',
        'max_abs_deviation_from_saved_table': 'Maximum deviation of the three terms from decomposition-by-model.csv (gate <= 1e-6)'},
        key=['model'], sort=['set', 'model'])
    dd = df[df.set == 'difficulty']
    for col, label in (('brain_term', 'brain'), ('readout_term', 'readout'), ('interaction_term', 'interaction')):
        x = dd[col].to_numpy()
        r = stats.signed_rank(x, f'fig3d.wilcoxon.{label}')
        lo, hi, meth = stats.bootstrap_ci((x,), lambda a, axis=-1: np.mean(a, axis=axis), f'fig3d.mean.{label}')
        r.update(mean=float(np.mean(x)), mean_ci_low=lo, mean_ci_high=hi, mean_ci_method=meth)
        stats.record(r, test_id=f'fig3d.wilcoxon.{label}', panel='fig3d', family='F3d', contrast=f'{label} margin term vs 0', unit='map',
                     level='A', analysis_status='new_registered', source_table='main/fig3/fig3d_margin_decomposition.csv')
        stats.record(stats.sign_test(x), test_id=f'fig3d.sign.{label}', panel='fig3d', family='F3d.companion', contrast=f'sign of the {label} term',
                     unit='map', level='A', analysis_status='new_registered', source_table='main/fig3/fig3d_margin_decomposition.csv')


# ---------------------------------------------------------------- parameter anatomy

def _core(path):
    if path.suffix == '.npz':
        z = io.npz_load(path)
        return {k: z[f'core_{k}'].astype(np.float64) for k in registry.CORE_KEYS}
    c, _, _ = registry.split_checkpoint(io.torch_load(path))
    return {k: c[k].detach().cpu().numpy().astype(np.float64) for k in registry.CORE_KEYS}


def anatomy_inputs():
    groups, ann = anatomy.group_table()
    hops = anatomy.hop_distances(ann)
    cell = ann[['bodyId', 'parameter_group']].merge(hops, on='bodyId')
    gh = cell.groupby('parameter_group').agg(hops_from_input=('hops_from_input', lambda s: s[s >= 0].min() if (s >= 0).any() else -1),
                                             hops_to_observed=('hops_to_observed', lambda s: s[s >= 0].min() if (s >= 0).any() else -1))
    groups = groups.merge(gh, left_on='group', right_index=True, how='left')
    maps = [f'difficulty-{l}' for l in sorted(r.replace('difficulty-', '') for r in static_runs.run_ids('difficulty-'))]
    acts = []
    for m in maps:
        a = io.parquet_load(paths.M1 / 'atlas' / f'{m}-before.parquet', columns=['parameter_group', 'spikes_mean'])
        acts.append(a.groupby('parameter_group')['spikes_mean'].mean().reindex(groups['group']).to_numpy())
    groups['baseline_spike_probability'] = np.nanmean(np.vstack(acts), axis=0)
    return groups, maps


def deltas(maps):
    out = {}
    for m in maps:
        d = paths.M1 / 'runs' / f'record-{m}'
        b = _core(d / 'parameters-before.npz')
        a = _core(d / 'parameters-after.npz')
        out[m] = {k: a[k] - b[k] for k in registry.CORE_KEYS}
    return out


def stage_deltas(maps, fractions=(0.25, 0.5, 0.75)):
    lv = {l['id']: l for l in io.json_load(paths.TASKS / 'levels.json')}
    out = {}
    for m in maps:
        L = int(lv[m.replace('difficulty-', '')]['optimal_steps'])
        before = _core(paths.M1 / 'runs' / f'record-{m}' / 'parameters-before.npz')
        for f in fractions:
            k = max(1, int(round(f * L)))
            p = paths.RUNS / m / f'suffix-{k}.pt'
            while not p.exists() and k > 1:
                k -= 1
                p = paths.RUNS / m / f'suffix-{k}.pt'
            c = _core(p)
            out[(m, f)] = ({q: c[q] - before[q] for q in registry.CORE_KEYS}, k)
    return out


def strata_codes(groups):
    act = groups['baseline_spike_probability'].fillna(0).to_numpy()
    q = np.quantile(act, [0.2, 0.4, 0.6, 0.8])
    act_bin = np.searchsorted(q, act, side='right')
    mem = groups['membership'].map({'neither': 0, 'input': 1, 'observed': 2, 'input_and_observed': 3}).to_numpy()
    hin = groups['hops_from_input'].to_numpy()
    hin_bin = np.where(hin < 0, 4, np.minimum(hin, 3))
    codes = act_bin * 100 + mem * 10 + hin_bin
    _, codes = np.unique(codes, return_inverse=True)
    return codes


def _standardized_ranks(X):
    from scipy.stats import rankdata
    R = rankdata(X, axis=1)
    Z = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
    return R, Z


N_BOOT_CI = 2000


def _map_bootstrap(Z, T, kk, observed, key):
    """95% percentile CIs of the mean pairwise Spearman and mean top-1% overlap, resampling maps with replacement
    (pairs of a map with itself are excluded). Uses the pairwise matrices, whose off-diagonal means equal the observed
    statistics exactly."""
    M, n = Z.shape
    C = (Z @ Z.T) / n
    O = (T @ T.T) / kk.mean()
    off = ~np.eye(M, dtype=bool)
    if abs(C[off].mean() - observed[0]) > 1e-9 or abs(O[off].mean() - observed[1]) > 1e-9:
        raise RuntimeError('pairwise matrices do not reproduce the observed consistency')
    idx = stats.rng_for(key).integers(0, M, size=(N_BOOT_CI, M))
    keep = idx[:, :, None] != idx[:, None, :]
    rho = (C[idx[:, :, None], idx[:, None, :]] * keep).sum((1, 2)) / keep.sum((1, 2))
    ov = (O[idx[:, :, None], idx[:, None, :]] * keep).sum((1, 2)) / keep.sum((1, 2))
    return np.quantile(rho, [0.025, 0.975]), np.quantile(ov, [0.025, 0.975])


def consistency(Xs, codes, test_id):
    """Mean pairwise Spearman and mean top-1% overlap across maps vs a within-strata permutation null.

    Xs maps parameter -> (maps x groups) matrix. The same within-strata permutations serve every
    parameter. With standardized rank vectors z_m, mean pairwise rho = (|sum z|^2 - M n) / (M (M-1) n),
    which depends only on the summed vector, so each permutation draw costs O(M n).
    """
    out = {}
    prepared = {}
    for p, X in Xs.items():
        M, n = X.shape
        k = int(math.ceil(TOP_FRACTION * n))
        R, Z = _standardized_ranks(X)
        top = (R > n - k).astype(np.float64)
        prepared[p] = (Z, top, top.sum(1), k)

    def summary(Zs, Ts, kk):
        M, n = Zs.shape
        rho = (np.sum(Zs.sum(0) ** 2) - M * n) / (M * (M - 1) * n)
        S = Ts.sum(0)
        ov = (np.sum(S ** 2) - kk.sum()) / (M * (M - 1) * kk.mean())
        return rho, ov

    nulls = {p: (np.empty(N_NULL), np.empty(N_NULL)) for p in Xs}
    obs = {p: summary(Z, T, kk) for p, (Z, T, kk, _) in prepared.items()}
    cis = {p: _map_bootstrap(Z, T, kk, obs[p], f'{test_id}.{p}.ci') for p, (Z, T, kk, _) in prepared.items()}
    rng = stats.rng_for(test_id)
    M, n = next(iter(Xs.values())).shape
    keys_base = codes.astype(np.float64)[None, :] * 2.0
    rows = np.arange(M)[:, None]
    for i in range(N_NULL):
        order = np.argsort(keys_base + rng.random((M, n)), axis=1)
        for p, (Z, T, kk, _) in prepared.items():
            nulls[p][0][i], nulls[p][1][i] = summary(Z[rows, order], T[rows, order], kk)
    for p in Xs:
        out[p] = (obs[p][0], obs[p][1], nulls[p][0], nulls[p][1], prepared[p][3], cis[p][0], cis[p][1])
    return out


def fig3e_ed4(groups, maps):
    D = deltas(maps)
    codes = strata_codes(groups)
    groups = groups.copy()
    groups['stratum'] = codes
    n = len(groups)
    # Fig. 3e summaries (equal-map means of normalized absolute change).
    norm = {p: np.vstack([np.abs(D[m][p]) / NORM[p] for m in maps]) for p in registry.CORE_KEYS}
    groups['size_class'] = pd.cut(groups['n_cells'], [0, 1, 3, 10, 100, 10 ** 6], labels=['1', '2-3', '4-10', '11-100', '>100']).astype(str)
    act = groups['baseline_spike_probability'].fillna(0).to_numpy()
    groups['activity_quintile'] = np.searchsorted(np.quantile(act, [0.2, 0.4, 0.6, 0.8]), act, side='right') + 1
    groups['hops_from_input_class'] = groups['hops_from_input'].map(lambda h: 'unreachable' if h < 0 else ('>=3' if h >= 3 else str(int(h))))
    groups['hops_to_observed_class'] = groups['hops_to_observed'].map(lambda h: 'unreachable' if h < 0 else ('>=3' if h >= 3 else str(int(h))))
    rows = []
    for stype in ('superclass', 'transmitter', 'membership', 'size_class', 'activity_quintile', 'hops_from_input_class', 'hops_to_observed_class'):
        for sval, idx in groups.groupby(stype).groups.items():
            idx = np.asarray(list(idx))
            for p in registry.CORE_KEYS:
                per_map = norm[p][:, idx]
                rows.append(dict(stratum_type=stype, stratum=str(sval), parameter=p, n_groups=len(idx), n_cells=int(groups.loc[idx, 'n_cells'].sum()),
                                 mean_normalized_abs_change=float(per_map.mean(1).mean()),
                                 median_group_normalized_abs_change=float(np.median(per_map.mean(0))),
                                 fraction_groups_changed=float((per_map > 0).mean(1).mean()), n_maps=len(maps),
                                 analysis_status='new_registered', source_file='mechanisms-v1/runs/record-difficulty-*/parameters-{before,after}.npz'))
    s3e = pd.DataFrame(rows)
    # ED4a consistency vs stratified null, final and stage-aligned.
    SD = stage_deltas(maps)
    cons_rows = []
    cons = consistency(norm, codes, 'ed04a.consistency.final')
    for p in registry.CORE_KEYS:
        obs_rho, obs_ov, nr, no, k, ci_rho, ci_ov = cons[p]
        for metric, obs, null, ci in (('mean_pairwise_spearman', obs_rho, nr, ci_rho), ('mean_top1pct_overlap', obs_ov, no, ci_ov)):
            pval = stats.perm_p(obs, null)
            res = dict(test='within_strata_permutation', statistic_name=metric, statistic=float(obs), p_value=pval, p_method='permutation',
                       n=len(maps), n_used=len(maps), n_resamples=N_NULL, mc_se=stats.mc_se(pval, N_NULL), effect_name='observed_minus_null_mean',
                       effect=float(obs - null.mean()), effect_ci_low=float(ci[0] - null.mean()), effect_ci_high=float(ci[1] - null.mean()),
                       effect_ci_method=f'bootstrap over maps ({N_BOOT_CI:,} resamples; self-pairs excluded), minus the null mean',
                       null_mean=float(null.mean()), null_q025=float(np.quantile(null, 0.025)), null_q975=float(np.quantile(null, 0.975)),
                       informative=True)
            stats.record(res, test_id=f'ed04a.{metric}.{p}', panel='fig3e/ed04a', family='F3e.consistency',
                         contrast=f'cross-map consistency of |{p} change| vs within-strata permutation', unit='map', level='A',
                         analysis_status='new_registered', source_table='extended/ed04/ed04a_parameter_change_consistency.csv')
            cons_rows.append(dict(parameter=p, checkpoint='final', depth_fraction=1.0, metric=metric, observed=float(obs), null_mean=float(null.mean()),
                                  null_q025=float(np.quantile(null, 0.025)), null_q975=float(np.quantile(null, 0.975)), p_value=pval,
                                  top_k=k, n_maps=len(maps), test_id=f'ed04a.{metric}.{p}', analysis_status='new_registered'))
        for f in (0.25, 0.5, 0.75):
            Xf = np.vstack([np.abs(SD[(m, f)][0][p]) / NORM[p] for m in maps])
            keep = Xf.std(1) > 0          # maps with no change yet at this checkpoint are excluded
            R, Z = _standardized_ranks(Xf[keep])
            M = int(keep.sum())
            rho = (np.sum(Z.sum(0) ** 2) - M * n) / (M * (M - 1) * n) if M > 1 else float('nan')
            # rank agreement between this stage and the final change, averaged over maps
            Rf, _ = _standardized_ranks(norm[p][keep])
            stage_final = float(np.mean([np.corrcoef(R[i], Rf[i])[0, 1] for i in range(M)])) if M else float('nan')
            cons_rows.append(dict(parameter=p, checkpoint=f'suffix_at_{f}', depth_fraction=f, metric='mean_pairwise_spearman', observed=float(rho),
                                  null_mean=None, null_q025=None, null_q975=None, p_value=None, top_k=None, n_maps=M, test_id=None,
                                  analysis_status='post_hoc'))
            cons_rows.append(dict(parameter=p, checkpoint=f'suffix_at_{f}', depth_fraction=f, metric='mean_within_map_spearman_with_final', observed=stage_final,
                                  null_mean=None, null_q025=None, null_q975=None, p_value=None, top_k=None, n_maps=M, test_id=None,
                                  analysis_status='post_hoc'))
    # Magnitude vs difficulty (map level).
    lv = {l['id']: l for l in io.json_load(paths.TASKS / 'levels.json')}
    opt = np.array([lv[m.replace('difficulty-', '')]['optimal_steps'] for m in maps], float)
    for p in registry.CORE_KEYS:
        l2 = np.array([np.linalg.norm(D[m][p]) for m in maps])
        tid = f'ed04a.spearman.l2_{p}~optimum'
        stats.record(stats.spearman(opt, l2, tid), test_id=tid, panel='ed04a', family='F3e.magnitude',
                     contrast=f'L2 norm of {p} change vs certified optimal moves', unit='map', level='A', analysis_status='new_registered',
                     source_table='extended/ed04/ed04a_parameter_change_consistency.csv')
        for i, m in enumerate(maps):
            cons_rows.append(dict(parameter=p, checkpoint='final', depth_fraction=1.0, metric=f'l2_change:{m}', observed=float(l2[i]), null_mean=None,
                                  null_q025=None, null_q975=None, p_value=None, top_k=None, n_maps=1, test_id=tid, analysis_status='new_registered'))
    csvout.write_table(pd.DataFrame(cons_rows), 'extended/ed04/ed04a_parameter_change_consistency.csv',
                       title='Cross-map consistency of parameter change (final and stage-aligned) and change magnitude per map', describe={
        'parameter': 'Core parameter type', 'checkpoint': 'final, or the suffix checkpoint nearest the depth fraction',
        'depth_fraction': 'Curriculum depth fraction of the checkpoint', 'metric': 'Statistic (mean pairwise Spearman, mean top-1% overlap, within-map agreement with final, or per-map L2 change)',
        'observed': 'Observed value', 'null_mean': 'Mean under the within-strata permutation null', 'null_q025': '2.5th percentile of the null',
        'null_q975': '97.5th percentile of the null', 'p_value': 'Two-sided permutation p-value', 'top_k': 'Top-k groups (1%)', 'n_maps': 'Maps'},
        key=['parameter', 'checkpoint', 'metric'], sort=['parameter', 'checkpoint', 'metric'])
    # ED4c enrichment among consistently most-changed groups (combined normalized change).
    comb = sum(norm[p] for p in registry.CORE_KEYS)
    from scipy.stats import rankdata
    pct = rankdata(comb, axis=1) / n
    score = pct.mean(0)
    k = int(math.ceil(TOP_FRACTION * n))
    top_idx = np.argsort(-score, kind='mergesort')[:k]
    rng = stats.rng_for('ed04c.enrichment')
    null_tops = []
    keys_base = codes.astype(np.float64) * 2.0
    for _ in range(N_NULL):
        order = np.argsort(keys_base + rng.random(n))
        base = np.argsort(codes, kind='stable')
        perm = np.empty(n)
        perm[base] = score[order]
        null_tops.append(np.argsort(-perm, kind='mergesort')[:k])
    null_tops = np.vstack(null_tops)
    enr = []
    for var in ('superclass', 'transmitter'):
        vals = groups[var].to_numpy()
        for cat in sorted(set(vals)):
            is_cat = vals == cat
            obs = int(is_cat[top_idx].sum())
            null = is_cat[null_tops].sum(1)
            pval = stats.perm_p(obs, null)
            tid = f'ed04c.enrichment.{var}.{cat}'
            res = dict(test='within_strata_permutation', statistic_name='count_in_top', statistic=float(obs), p_value=pval, p_method='permutation',
                       n=n, n_used=k, n_resamples=N_NULL, mc_se=stats.mc_se(pval, N_NULL), effect_name='observed_over_expected',
                       effect=float(obs / null.mean()) if null.mean() > 0 else float('nan'), null_mean=float(null.mean()), informative=bool(is_cat.sum() > 0))
            stats.record(res, test_id=tid, panel='fig3e/ed04c', family='F3e.enrichment', contrast=f'{var} = {cat} among top-1% consistently changed groups',
                         unit='group', level='A', analysis_status='new_registered', source_table='extended/ed04/ed04c_parameter_change_enrichment.csv')
            enr.append(dict(variable=var, category=str(cat), groups_in_category=int(is_cat.sum()), top_k=k, observed_in_top=obs,
                            expected_in_top=float(null.mean()), observed_over_expected=res['effect'], p_value=pval, test_id=tid,
                            analysis_status='new_registered'))
    csvout.write_table(pd.DataFrame(enr), 'extended/ed04/ed04c_parameter_change_enrichment.csv',
                       title='Enrichment among the consistently most-changed parameter groups (vs within-strata permutation)', describe={
        'variable': 'Annotation variable', 'category': 'Category', 'groups_in_category': 'Groups in the category', 'top_k': 'Top-1% groups by mean cross-map percentile of combined change',
        'observed_in_top': 'Category groups among the top groups', 'expected_in_top': 'Expected count under the within-strata null',
        'observed_over_expected': 'Observed / expected', 'p_value': 'Two-sided permutation p-value (BH within family in ST2)'},
        key=['variable', 'category'])
    # Pre-registered placement rule for Fig. 3e.
    reg = [r for r in stats.registry() if r['family'] in ('F3e.consistency', 'F3e.enrichment')]
    q = {}
    for fam in ('F3e.consistency', 'F3e.enrichment'):
        rr = [r for r in reg if r['family'] == fam]
        q[fam] = dict(zip([r['test_id'] for r in rr], stats.bh([r['p_value'] for r in rr])))
    cons_ok = any(v < 0.05 for v in q['F3e.consistency'].values())
    enr_ok = any(v < 0.05 for v in q['F3e.enrichment'].values())
    placement = dict(rule='Fig. 3e shows parameter-change anatomy only if cross-map consistency and enrichment both beat the within-strata null '
                          'at BH q < 0.05; otherwise readout geometry across maps becomes Fig. 3e and anatomy moves to ED4b',
                     consistency_min_q=min(q['F3e.consistency'].values()), enrichment_min_q=min(q['F3e.enrichment'].values()),
                     consistency_passed=cons_ok, enrichment_passed=enr_ok, anatomy_in_main=cons_ok and enr_ok)
    return groups, D, s3e, placement


ANATOMY_DESC = {
    'stratum_type': 'Grouping variable', 'stratum': 'Group category', 'parameter': 'Core parameter type',
    'n_groups': 'Parameter groups in the category', 'n_cells': 'Cells in those groups',
    'mean_normalized_abs_change': 'Equal-map mean of |change| / allowed range (log-gain range ln1.25-ln0.8; threshold 2 mV)',
    'median_group_normalized_abs_change': 'Median over groups of the map-averaged normalized |change|',
    'fraction_groups_changed': 'Equal-map mean fraction of groups with any change', 'n_maps': 'Maps averaged'}


def ed04b(maps, relpath, title):
    rows = []
    heads = {}
    for d in sorted((paths.M1 / 'runs').glob('record-*')):
        if not (d / 'parameters-after.npz').exists():
            continue
        model = d.name.replace('record-', '')
        pb = io.npz_load(d / 'parameters-before.npz')
        pa = io.npz_load(d / 'parameters-after.npz')
        Wb = pb['head_proposal.weight'].astype(np.float64)
        Wa = pa['head_proposal.weight'].astype(np.float64)

        def basis(W):
            C = W - W.mean(0, keepdims=True)
            _, s, vt = np.linalg.svd(C, full_matrices=False)
            r = int(np.sum(s > max(s[0] * 1e-10, 1e-12)))
            return vt[:r].T

        Bb, Ba = basis(Wb), basis(Wa)
        cos = np.clip(np.linalg.svd(Bb.T @ Ba, compute_uv=False), -1, 1)
        ang = np.degrees(np.arccos(cos))
        heads[model] = Ba
        rows.append(dict(row_type='vs_initial', model=model, other_model='initial', angle_1_deg=float(ang[0]), angle_2_deg=float(ang[1]),
                         angle_3_deg=float(ang[2]), weight_change_frobenius=float(np.linalg.norm(Wa - Wb)),
                         bias_change_l2=float(np.linalg.norm(pa['head_proposal.bias'] - pb['head_proposal.bias'])),
                         source_file=paths.rel(d / 'parameters-after.npz')))
    for i, a in enumerate(maps):
        for b in maps[i + 1:]:
            cos = np.clip(np.linalg.svd(heads[a].T @ heads[b], compute_uv=False), -1, 1)
            ang = np.degrees(np.arccos(cos))
            rows.append(dict(row_type='pairwise_final', model=a, other_model=b, angle_1_deg=float(ang[0]), angle_2_deg=float(ang[1]),
                             angle_3_deg=float(ang[2]), weight_change_frobenius=None, bias_change_l2=None,
                             source_file='mechanisms-v1/runs/record-difficulty-*/parameters-after.npz'))
    df = pd.DataFrame(rows)
    saved = {r['model']: r['principal_angles_degrees'] for r in io.json_load(paths.M1 / 'tables' / 'readout-planes.json')}
    dev = []
    for _, r in df[df.row_type == 'vs_initial'].iterrows():
        if r['model'] in saved:
            dev.append(max(abs(a - b) for a, b in zip(sorted([r['angle_1_deg'], r['angle_2_deg'], r['angle_3_deg']]), sorted(saved[r['model']]))))
    df['max_abs_deviation_from_saved_deg'] = None
    df.loc[df.row_type == 'vs_initial', 'max_abs_deviation_from_saved_deg'] = max(dev) if dev else None
    csvout.write_table(df, relpath, title=title, describe={
        'row_type': 'vs_initial (final vs the run initial readout) or pairwise_final (final readouts of two maps)', 'other_model': 'Comparison model',
        'angle_1_deg': 'Smallest principal angle (degrees)', 'angle_2_deg': 'Second principal angle', 'angle_3_deg': 'Largest principal angle',
        'weight_change_frobenius': 'Frobenius norm of the readout weight change', 'bias_change_l2': 'L2 norm of the bias change',
        'max_abs_deviation_from_saved_deg': 'Maximum deviation from tables/readout-planes.json over all models (degrees)'},
        key=['row_type', 'model', 'other_model'])


def st4(groups, D, maps):
    rows = []
    jd = paths.M1 / 'runs' / 'record-remote-dynamic-M0-joint'
    b = _core(jd / 'parameters-before.npz')
    a = _core(jd / 'parameters-after.npz')
    runs = dict(D)
    runs['remote-dynamic-M0-joint'] = {k: a[k] - b[k] for k in registry.CORE_KEYS}
    base = groups[['group', 'group_name', 'n_cells', 'membership', 'superclass', 'transmitter']]
    frames = []
    for rid in maps + ['remote-dynamic-M0-joint']:
        f = base.copy()
        f.insert(0, 'run_id', rid)
        for k in registry.CORE_KEYS:
            f[f'delta_{k}'] = runs[rid][k]
        f['normalized_abs_change_sum'] = sum(np.abs(runs[rid][k]) / NORM[k] for k in registry.CORE_KEYS)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df['source_file'] = 'mechanisms-v1/runs/record-<run>/parameters-{before,after}.npz'
    csvout.write_table(df, 'supplementary/st4_parameter_change.csv', title='Parameter change per type group (24 maps + M0 joint)', describe={
        'group': 'Parameter group index (0-11769)', 'group_name': 'Group label (type:<cell type> or superclass:<superclass>)',
        'n_cells': 'Cells sharing the parameters', 'membership': 'Contains input cells, observed cells, both or neither',
        'superclass': 'Majority superclass of member cells', 'transmitter': 'Majority predicted transmitter',
        'delta_pre_log_gain': 'Final minus initial outgoing log-gain', 'delta_post_log_gain': 'Final minus initial incoming log-gain',
        'delta_threshold_mv': 'Final minus initial threshold offset (mV)', 'normalized_abs_change_sum': 'Sum of |changes| divided by their allowed ranges'},
        key=['run_id', 'group'])


def main():
    fig3ab()
    fig3c()
    fig3d()
    groups, maps = anatomy_inputs()
    groups, D, s3e, placement = fig3e_ed4(groups, maps)
    geo_title = 'Readout geometry: principal angles of centred readout row spaces'
    anat_title = 'Parameter-change anatomy across 24 maps (P05 -> final)'
    if placement['anatomy_in_main']:
        csvout.write_table(s3e, 'main/fig3/fig3e_parameter_change_anatomy.csv', title=anat_title, describe=ANATOMY_DESC,
                           key=['stratum_type', 'stratum', 'parameter'], sort=['stratum_type', 'stratum', 'parameter'])
        ed04b(maps, 'extended/ed04/ed04b_readout_geometry_maps.csv', geo_title)
        placement.update(fig3e_file='main/fig3/fig3e_parameter_change_anatomy.csv', ed04b_file='extended/ed04/ed04b_readout_geometry_maps.csv')
    else:
        ed04b(maps, 'main/fig3/fig3e_readout_geometry.csv', geo_title + ' (placed as Fig. 3e by the pre-registered rule)')
        csvout.write_table(s3e, 'extended/ed04/ed04b_parameter_change_anatomy.csv', title=anat_title + ' (moved to ED4b by the pre-registered rule)',
                           describe=ANATOMY_DESC, key=['stratum_type', 'stratum', 'parameter'], sort=['stratum_type', 'stratum', 'parameter'])
        placement.update(fig3e_file='main/fig3/fig3e_readout_geometry.csv', ed04b_file='extended/ed04/ed04b_parameter_change_anatomy.csv')
    csvout.write_table(pd.DataFrame([dict(decision='fig3e_placement', **placement)]), 'qa/placement_decisions.csv',
                       title='Pre-registered placement decisions and their outcome', describe={
                           'decision': 'Decision', 'rule': 'Rule fixed before computation', 'consistency_min_q': 'Smallest BH q in F3e.consistency',
                           'enrichment_min_q': 'Smallest BH q in F3e.enrichment', 'consistency_passed': 'Consistency beat the null at q < 0.05',
                           'enrichment_passed': 'Enrichment beat the null at q < 0.05', 'anatomy_in_main': 'Anatomy shown as Fig. 3e',
                           'fig3e_file': 'Deliverable placed as Fig. 3e', 'ed04b_file': 'Deliverable placed as ED4b'}, key=['decision'])
    st4(groups, D, maps)
    g = groups[['group', 'group_name', 'n_cells', 'superclass', 'transmitter', 'membership', 'input_cells', 'observed_cells', 'hops_from_input',
                'hops_to_observed', 'baseline_spike_probability', 'stratum']].copy()
    g.to_csv(paths.work('group_table.csv'), index=False)
