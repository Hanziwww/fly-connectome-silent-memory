"""P2: compute rebuild (gate), acquisition outcomes, learning curves, cost vs difficulty, stage costs,
zero-update transfer, methods table M4 and Extended Data 3."""
import math
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

from msa import compute, csvout, io, paths, registry, static_runs, stats, taxonomy

sys.path.insert(0, str(paths.GAME))
import game  # noqa: E402  (pure numpy board logic from sokoban/lib)

REPRESENTATIVE = ['difficulty-D1-T0', 'difficulty-D3-T0', 'difficulty-D5-T0', 'difficulty-D7-T0']
FEATURED = 'difficulty-D3-T0'
SENTINEL = 1e8
REMOTE_STATIC = ['remote-D7-head_only', 'remote-D7-demonstration']


def levels():
    return {l['id']: l for l in io.json_load(paths.TASKS / 'levels.json')}


def run_location(run_id):
    return (paths.R0 / run_id) if run_id in REMOTE_STATIC else (paths.RUNS / run_id)


def _jl(run_id, name):
    p = run_location(run_id) / name
    return io.jsonl_load(p) if p.exists() else []


def _js(run_id, name):
    p = run_location(run_id) / name
    return io.json_load(p) if p.exists() else None


# ---------------------------------------------------------------- compute rebuild (gate)

def compute_table():
    rows = []
    for rid in static_runs.run_ids() + REMOTE_STATIC:
        r = compute.rebuild(run_location(rid))
        cfg = _js(rid, 'config.json')
        res = _js(rid, 'result.json') or {}
        rows.append(dict(run_id=rid, condition=r['condition'], module=cfg.get('module'), task=cfg.get('task'), batch=r['batch'],
                         frozen_core=r['frozen_core'], interrupted=r['interrupted'], status=res.get('status'),
                         saved_forward_calls=r['saved_forward_calls'], saved_gradient_calls=r['saved_gradient_calls'],
                         rebuilt_forward_calls=r['rebuilt_forward_calls'], rebuilt_gradient_calls=r['rebuilt_gradient_calls'],
                         post_resume_rebuilt_calls=r.get('post_resume_rebuilt_calls'),
                         post_resume_rebuilt_gradient_calls=r.get('post_resume_rebuilt_gradient_calls'),
                         match=r['match'], uninterrupted_equivalent_calls=r.get('uninterrupted_equivalent_calls'),
                         resume_reassessment_calls=r.get('resume_reassessment_calls'), lost_update_calls=r.get('lost_update_calls'),
                         lifetime_calls_low=r.get('lifetime_calls_low'), lifetime_calls_high=r.get('lifetime_calls_high'),
                         source_file=paths.rel(run_location(rid))))
    df = pd.DataFrame(rows)
    for c in [c for c in df.columns if c.endswith('_calls') or c.endswith('_calls_low') or c.endswith('_calls_high')]:
        df[c] = df[c].astype('Int64')
    # Gate: every protocol run with saved counters (difficulty, controls, remote D7) must match exactly.
    # Benchmarks are reported but not gated (failed precision candidates stopped mid-computation).
    gated = df[(df['module'] != 'benchmark') & df['match'].notna()]
    failed = gated[~gated['match'].astype(bool)]
    need = df[(df['module'] == 'difficulty') & (~df['interrupted'])]
    if len(failed) or need['match'].isna().any() or len(need) != 22:
        raise RuntimeError(f'compute rebuild gate failed: {failed["run_id"].tolist()}')
    df['gated'] = (df['module'] != 'benchmark') & df['match'].notna()
    df['neuron_ms_per_call'] = df['batch'] * 167106 * 40
    df['compute_basis'] = np.where(df['rebuilt_forward_calls'].isna(), 'not_applicable',
                                   np.where(df['interrupted'], 'rebuilt_uninterrupted_equivalent', 'rebuilt_exact_match'))
    csvout.write_table(df, 'meta/compute_reconstruction.csv', title='Forward-call reconstruction from training logs (gate)', describe={
        'condition': 'Training condition', 'module': 'Study module', 'task': 'Map', 'frozen_core': 'Core frozen (persistent feature cache)',
        'interrupted': 'Run interrupted and resumed', 'status': 'Recorded end status',
        'saved_forward_calls': 'forward_calls counter saved in result.json (resumed process only for interrupted runs)',
        'saved_gradient_calls': 'gradient_forward_calls counter saved in result.json',
        'rebuilt_forward_calls': 'Forward calls rebuilt over the whole logged run (as if uninterrupted)',
        'rebuilt_gradient_calls': 'Gradient forward calls rebuilt over the whole logged run',
        'post_resume_rebuilt_calls': 'Rebuilt forward calls of the resumed process only (comparable with the saved counter)',
        'post_resume_rebuilt_gradient_calls': 'Rebuilt gradient calls of the resumed process only',
        'match': 'Rebuilt counters equal the saved counters exactly (post-resume segment for interrupted runs)',
        'uninterrupted_equivalent_calls': 'Forward calls of the logged run without the interruption overhead',
        'resume_reassessment_calls': 'Calls of the unlogged re-assessment at resume',
        'lost_update_calls': 'Calls of the updates computed before the crash and repeated after resume',
        'lifetime_calls_low': 'Lower bound of lifetime calls (equivalent + resume + lost updates)',
        'lifetime_calls_high': 'Upper bound (adds one possibly in-flight assessment)',
        'batch': 'Boards per simulator call', 'gated': 'Included in the exact-match gate (all protocol runs with saved counters)',
        'neuron_ms_per_call': 'Simulated neuron-milliseconds per forward call (batch x 167,106 cells x 40 ms)',
        'compute_basis': 'rebuilt_exact_match, rebuilt_uninterrupted_equivalent or not_applicable'},
        key=['run_id'])
    return df.set_index('run_id')


# ---------------------------------------------------------------- per-assessment and per-stage records

def assessment_rows(run_id, level):
    h = _jl(run_id, 'history.jsonl')
    cfg = _js(run_id, 'config.json')
    L = int(level['optimal_steps'])
    rows = []
    for i, x in enumerate(h):
        margins = np.asarray(x.get('margins') or [], float)
        valid = margins[np.abs(margins) < SENTINEL]
        rows.append(dict(run_id=run_id, condition=cfg['condition']['name'], map_id=cfg['task'], assessment=i + 1, event=x['event'],
                         depth=int(x['depth']), depth_fraction=x['depth'] / L, updates=int(x['updates']), elapsed_s=float(x['elapsed']),
                         time_segment=int(x['phase']), training_states=int(x['training_states']), loss=float(x['loss']),
                         entry_margin=float(x['bridge_margin']), min_dependency_margin=float(valid.min()) if len(valid) else None,
                         n_margins=len(margins), n_sentinel_margins=int(len(margins) - len(valid)), correct=int(x['correct']),
                         total=int(x['total']), lesson_solved=bool(x['lesson_solved']), retained=bool(x['retained']),
                         stage_passed=bool(x['stage_passed']), solved=bool(x['solved']),
                         original_status=taxonomy.behavior(x['original_status']), original_steps=int(x['original_steps']),
                         source_file=paths.rel(run_location(run_id) / 'history.jsonl')))
    return rows


def reference_entries(level):
    g = game.restore(level)
    s = game.state_from(level['start'])
    out = []
    for a in level['reference_actions']:
        player, boxes = s
        target = player + game.DELTAS[a]
        out.append(dict(state=s, action=int(a), push=target in boxes))
        s = game.step(g, s, a)
    if not game.solved(g, s):
        raise RuntimeError(f'reference solution of {level["id"]} does not solve')
    return out


def stage_rows(run_id, level):
    h = _jl(run_id, 'history.jsonl')
    b = _jl(run_id, 'behavior.jsonl')
    e = _jl(run_id, 'exposure.jsonl')
    res = _js(run_id, 'result.json') or {}
    cfg = _js(run_id, 'config.json')
    L = int(level['optimal_steps'])
    ref = reference_entries(level)
    dead = level.get('deadlocking_alternatives_on_witness') or [None] * L
    branch = level.get('witness_branch_counts') or [None] * L
    # Regressions: an earlier suffix solved at the previous assessment and unsolved now.
    last_status = {}
    reg_primary = [0] * len(h)
    for i, beh in enumerate(b):
        for r in beh['records']:
            tid = r['task']
            if '-suffix-' in tid and last_status.get(tid) == 'solved' and r['status'] != 'solved':
                reg_primary[i] += 1
        for r in beh['records']:
            last_status[r['task']] = r['status']
    # Secondary: margin sign flips (>0 to <=0) keyed by exposure state.
    reg_margin = [0] * len(h)
    prev = {}
    for i, (x, ex) in enumerate(zip(h, e)):
        cur = {}
        for s, m in zip(ex['states'], x.get('margins') or []):
            if abs(m) < SENTINEL:
                cur[(int(s[0]), tuple(sorted(s[1])))] = m
        reg_margin[i] = sum(1 for k, m in cur.items() if k in prev and prev[k] > 0 and m <= 0)
        prev = cur
    demonstration = cfg['condition']['schedule'] != 'suffix'
    depths = sorted({int(x['depth']) for x in h})
    final_updates = int(res.get('updates', h[-1]['updates'] if h else 0))
    rows = []
    for d in depths:
        idx = [i for i, x in enumerate(h) if int(x['depth']) == d]
        start_updates = int(h[idx[0]]['updates'])
        later = [i for i, x in enumerate(h) if int(x['depth']) > d]
        end_updates = int(h[later[0]]['updates']) if later else final_updates
        wi = L - d if not demonstration else None
        entry = ref[wi] if wi is not None and 0 <= wi < L else None
        passed = bool(later) or any(h[i]['solved'] for i in idx)
        rows.append(dict(run_id=run_id, condition=cfg['condition']['name'], map_id=cfg['task'], stage=d, stage_fraction=d / L,
                         witness_index=wi, entry_action=entry['action'] if entry else None, push_entry=entry['push'] if entry else None,
                         deadlock_alternatives_at_entry=(int(dead[wi]) if entry and dead[wi] is not None else None),
                         branch_count_at_entry=(int(branch[wi]) if entry and branch[wi] is not None else None),
                         updates_spent=end_updates - start_updates, zero_update=(end_updates - start_updates) == 0,
                         assessments=len(idx), failed_passes=sum(not h[i]['stage_passed'] for i in idx),
                         lesson_failures=sum(not h[i]['lesson_solved'] for i in idx), retention_failures=sum(not h[i]['retained'] for i in idx),
                         regressions_suffix=sum(reg_primary[i] for i in idx), regressions_margin=sum(reg_margin[i] for i in idx),
                         entry_margin_start=float(h[idx[0]]['bridge_margin']), entry_margin_end=float(h[idx[-1]]['bridge_margin']),
                         passed=passed, solved_at_stage=any(h[i]['solved'] for i in idx),
                         source_file=paths.rel(run_location(run_id))))
    return rows


# ---------------------------------------------------------------- outputs

ASSESS_DESC = {'condition': 'Training condition', 'assessment': 'Assessment index (1-based; one per behavioural evaluation)',
               'event': 'stage_baseline (first assessment of a stage) or update (after 4 optimizer steps)',
               'depth': 'Curriculum depth (suffix length in reference moves)', 'depth_fraction': 'Depth / certified optimal moves',
               'updates': 'Cumulative optimizer updates', 'elapsed_s': 'Cumulative training-loop seconds',
               'time_segment': 'Time segment (0, 1, 2)', 'training_states': 'Training states (entry + protected dependencies)',
               'loss': 'Training loss at the assessment', 'entry_margin': 'Allowed-vs-disallowed logit margin at the new entry state',
               'min_dependency_margin': 'Minimum margin over training states, excluding the no-competitor sentinel',
               'n_margins': 'Number of training-state margins', 'n_sentinel_margins': 'Margins equal to the 1e9 no-competitor sentinel (excluded)',
               'correct': 'Training states with the allowed action preferred', 'total': 'Training states',
               'lesson_solved': 'Current suffix solved autonomously', 'retained': 'All earlier suffixes still solved',
               'stage_passed': 'Stage passed (original solved, or lesson solved and retained)', 'solved': 'Original task solved autonomously',
               'original_status': 'End state of the rollout from the original start', 'original_steps': 'Moves in that rollout',
               'featured': 'Featured example in Fig. 1c'}

STAGE_DESC = {'condition': 'Training condition', 'stage': 'Curriculum stage (depth d)', 'stage_fraction': 'Stage / certified optimal moves',
              'witness_index': 'Index of the stage entry in the reference solution (L - d)', 'entry_action': 'Reference action at the entry',
              'push_entry': 'The entry action pushes a box', 'deadlock_alternatives_at_entry': 'Legal moves at the entry state that leave the solvable set',
              'branch_count_at_entry': 'Solvable branches at the entry state', 'updates_spent': 'Optimizer updates spent in this stage',
              'zero_update': 'Stage passed without any update', 'assessments': 'Logged assessments in this stage',
              'failed_passes': 'Assessments that did not pass the stage', 'lesson_failures': 'Assessments with the current suffix unsolved',
              'retention_failures': 'Assessments with at least one earlier suffix unsolved',
              'regressions_suffix': 'Earlier suffixes solved at the previous assessment and unsolved at this one (lower bound; assessments every 4 updates)',
              'regressions_margin': 'Training-state margins that flipped from >0 to <=0 since the previous assessment',
              'entry_margin_start': 'Entry margin at the first assessment of the stage', 'entry_margin_end': 'Entry margin at the last assessment of the stage',
              'passed': 'Stage was passed', 'solved_at_stage': 'The original task was solved during this stage'}


def main():
    lv = levels()
    comp = compute_table()
    diff_ids = static_runs.run_ids('difficulty-')
    all_ids = static_runs.run_ids() + REMOTE_STATIC

    # ED3a and Fig. 1c / 2b
    assess = []
    stages = []
    for rid in all_ids:
        cfg = _js(rid, 'config.json')
        level = lv[cfg['task']]
        if not _jl(rid, 'history.jsonl'):
            continue
        assess.extend(assessment_rows(rid, level))
        stages.extend(stage_rows(rid, level))
    A = pd.DataFrame(assess)
    S = pd.DataFrame(stages)
    csvout.write_table(A, 'extended/ed03/ed03a_learning_curves_all.csv', title='Learning curves for every static run',
                       describe=ASSESS_DESC, key=['run_id', 'assessment'])
    csvout.write_table(S, 'extended/ed03/ed03b_stage_costs_all.csv', title='Stage costs and regressions for every static run',
                       describe=STAGE_DESC, key=['run_id', 'stage'])
    c1 = A[A.run_id.isin(REPRESENTATIVE)].copy()
    c1['featured'] = c1['run_id'] == FEATURED
    csvout.write_table(c1, 'main/fig1/fig1c_curriculum_example.csv', title='Curriculum and loss on four representative maps',
                       describe=ASSESS_DESC, key=['run_id', 'assessment'])
    b2 = A[A.run_id.isin(diff_ids)].copy()
    b2['interrupted'] = b2['run_id'].map(lambda r: bool(comp.loc[r, 'interrupted']))
    csvout.write_table(b2, 'main/fig2/fig2b_learning_curves.csv', title='Learning curves, 24 difficulty maps',
                       describe={**ASSESS_DESC, 'interrupted': 'Run interrupted and resumed (history is continuous)'},
                       key=['run_id', 'assessment'])

    # Fig. 2a
    rows = []
    for rid in diff_ids:
        cfg = static_runs.config(rid)
        res = static_runs.result(rid)
        lev = lv[cfg['task']]
        bef = static_runs.before(rid) or {}
        aft = static_runs.after(rid) or {}
        ver = static_runs.verification(rid) or {}
        acc = static_runs.acceptance(rid) or {}
        vsteps = (ver.get('rollout') or {}).get('steps')
        rows.append(dict(run_id=rid, map_id=cfg['task'], tier=int(lev['tier']), board_size=int(lev['size']), boxes=int(lev['boxes']),
                         optimal_moves=int(lev['optimal_steps']), p05_status_before=taxonomy.behavior(bef.get('status')),
                         p05_steps_before=bef.get('steps'), training_status=res['status'], updates=int(res['updates']),
                         final_depth=int(res['depth']), solved_before_full_depth=int(res['depth']) < int(lev['optimal_steps']),
                         final_status=taxonomy.behavior(aft.get('status')), final_steps=aft.get('steps'),
                         reload_status=taxonomy.behavior((ver.get('rollout') or {}).get('status')), reload_steps=vsteps,
                         reload_verified=ver.get('status') == 'verified', accepted=bool(acc.get('accepted')),
                         excess_moves=(vsteps - int(lev['optimal_steps'])) if vsteps is not None else None,
                         source_file=paths.rel(paths.RUNS / rid)))
    a2 = pd.DataFrame(rows)
    k = int(((a2.reload_status == 'solved') & a2.reload_verified).sum())
    prop = stats.proportion(k, len(a2))
    a2['reload_solved_count'] = k
    a2['reload_solved_n'] = len(a2)
    a2['reload_solved_ci_low'] = prop['ci_low']
    a2['reload_solved_ci_high'] = prop['ci_high']
    csvout.write_table(a2, 'main/fig2/fig2a_acquisition_outcomes.csv', title='Per-map acquisition outcome', describe={
        'board_size': 'Board side (cells)', 'boxes': 'Boxes', 'optimal_moves': 'Certified optimal moves',
        'p05_status_before': 'Rollout end state of the P05 starting model before training', 'p05_steps_before': 'Moves in that rollout',
        'training_status': 'Training end status', 'updates': 'Optimizer updates', 'final_depth': 'Curriculum depth when training ended',
        'solved_before_full_depth': 'Original solved before the curriculum reached the full reference length',
        'final_status': 'End state of the champion rollout at the end of training', 'final_steps': 'Moves in that rollout',
        'reload_status': 'End state after independent reload', 'reload_steps': 'Moves after independent reload',
        'reload_verified': 'Independent reload verified', 'accepted': 'Accepted by the study rule',
        'excess_moves': 'Reload solution moves minus certified optimum',
        'reload_solved_count': 'Maps solved after independent reload (all maps)', 'reload_solved_n': 'Maps evaluated',
        'reload_solved_ci_low': 'Clopper-Pearson 95% lower limit of the solved proportion',
        'reload_solved_ci_high': 'Clopper-Pearson 95% upper limit'}, key=['run_id'], sort=['tier', 'map_id'])

    # Fig. 2c
    rows = []
    for rid in diff_ids:
        cfg = static_runs.config(rid)
        res = static_runs.result(rid)
        lev = lv[cfg['task']]
        st_r = S[S.run_id == rid]
        crow = comp.loc[rid]
        calls = int(crow['uninterrupted_equivalent_calls'])
        interrupted = bool(crow['interrupted'])
        rows.append(dict(run_id=rid, map_id=cfg['task'], tier=int(lev['tier']), board_size=int(lev['size']), boxes=int(lev['boxes']),
                         optimal_moves=int(lev['optimal_steps']), floor_cells=int(lev['floor_cells']), witness_pushes=int(lev['witness_pushes']),
                         deadlocking_alternatives_total=int(sum(lev.get('deadlocking_alternatives_on_witness') or [])),
                         shortest_dag_states=int(lev['shortest_dag_states']), updates=int(res['updates']), assessments=int(res['assessments']),
                         stages=len(st_r), updates_per_stage=int(res['updates']) / len(st_r), zero_update_stage_fraction=float(st_r.zero_update.mean()),
                         forward_calls=calls, neuron_ms=calls * compute.NEURON_MS_PER_CALL,
                         neuron_ms_lifetime_low=int(crow['lifetime_calls_low']) * compute.NEURON_MS_PER_CALL,
                         neuron_ms_lifetime_high=int(crow['lifetime_calls_high']) * compute.NEURON_MS_PER_CALL,
                         training_seconds=float(res['training_seconds']),
                         wall_seconds=None if interrupted else float(res['wall_seconds']), interrupted=interrupted,
                         compute_basis=crow['compute_basis'], source_file=paths.rel(paths.RUNS / rid)))
    c2 = pd.DataFrame(rows)
    csvout.write_table(c2, 'main/fig2/fig2c_cost_vs_difficulty.csv', title='Cost of acquisition against difficulty', describe={
        'board_size': 'Board side (cells)', 'boxes': 'Boxes', 'optimal_moves': 'Certified optimal moves (primary difficulty axis)',
        'floor_cells': 'Floor cells', 'witness_pushes': 'Pushes in the reference solution',
        'deadlocking_alternatives_total': 'Summed deadlocking alternatives along the reference solution',
        'shortest_dag_states': 'States in the shortest-path DAG', 'updates': 'Optimizer updates (co-primary cost)',
        'assessments': 'Logged assessments', 'stages': 'Curriculum stages reached', 'updates_per_stage': 'Updates / stages',
        'zero_update_stage_fraction': 'Fraction of stages passed without updates',
        'forward_calls': 'Simulator forward calls (rebuilt; uninterrupted-equivalent for interrupted runs)',
        'neuron_ms': 'Simulated neuron-milliseconds (co-primary cost) = forward_calls x 106,947,840',
        'neuron_ms_lifetime_low': 'Lifetime neuron-ms lower bound (includes interruption overhead)',
        'neuron_ms_lifetime_high': 'Lifetime neuron-ms upper bound', 'training_seconds': 'Training-loop seconds (cumulative; same GPU type for all maps)',
        'wall_seconds': 'Wall-clock seconds of the final process (empty for interrupted runs)', 'interrupted': 'Run interrupted and resumed',
        'compute_basis': 'Basis of forward_calls'}, key=['run_id'], sort=['optimal_moves', 'map_id'])
    _tests_2c(c2)

    # Fig. 2d
    d2 = S[S.run_id.isin(diff_ids)].copy()
    csvout.write_table(d2, 'main/fig2/fig2d_stage_costs.csv', title='Stage bottlenecks aligned to the reference solution',
                       describe=STAGE_DESC, key=['run_id', 'stage'])
    _tests_2d(d2)

    # ED3c
    desc_cols = ['optimal_moves', 'tier', 'boxes', 'board_size', 'floor_cells', 'witness_pushes', 'deadlocking_alternatives_total', 'shortest_dag_states']
    rows = []
    for i, a in enumerate(desc_cols):
        for b in desc_cols:
            rows.append(dict(descriptor_a=a, descriptor_b=b, spearman_rho=float(st.spearmanr(c2[a], c2[b]).statistic), n=len(c2)))
    csvout.write_table(pd.DataFrame(rows), 'extended/ed03/ed03c_difficulty_descriptors.csv', title='Collinearity of difficulty descriptors (24 maps)',
                       describe={'descriptor_a': 'Descriptor', 'descriptor_b': 'Descriptor', 'spearman_rho': 'Spearman correlation across maps'},
                       key=['descriptor_a', 'descriptor_b'], sort=[])

    fig2e(lv)
    m4(comp)


def _rec(result, test_id, family, contrast, unit='map', level='A', status='new_registered', **extra):
    stats.record(result, test_id=test_id, panel=extra.pop('panel', 'fig2c'), family=family, contrast=contrast, unit=unit, level=level,
                 analysis_status=status, history_set_id=extra.pop('history_set_id', None),
                 source_table=extra.pop('source_table', 'main/fig2/fig2c_cost_vs_difficulty.csv'), **extra)


def _tests_2c(c2):
    x = c2['optimal_moves'].to_numpy(float)
    _rec(stats.spearman(x, c2['updates'], 'fig2c.spearman.updates~optimum'), 'fig2c.spearman.updates~optimum', 'F2c.primary',
         'updates vs certified optimal moves')
    _rec(stats.spearman(x, c2['neuron_ms'], 'fig2c.spearman.neuron_ms~optimum'), 'fig2c.spearman.neuron_ms~optimum', 'F2c.primary',
         'simulated neuron-ms vs certified optimal moves')
    _rec(stats.jonckheere(c2['updates'], c2['tier'], 'fig2c.jt.updates~tier'), 'fig2c.jt.updates~tier', 'F2c.primary',
         'trend of updates over tiers 0-7')
    for col in ['tier', 'boxes', 'board_size', 'floor_cells', 'witness_pushes', 'deadlocking_alternatives_total']:
        tid = f'fig2c.spearman.updates~{col}'
        _rec(stats.spearman(c2[col], c2['updates'], tid), tid, 'F2c.secondary', f'updates vs {col}')
    _rec(stats.spearman(x, c2['training_seconds'], 'fig2c.spearman.training_seconds~optimum'), 'fig2c.spearman.training_seconds~optimum',
         'F2c.secondary', 'training seconds vs certified optimal moves')
    _rec(stats.spearman(x, c2['updates_per_stage'], 'fig2c.spearman.updates_per_stage~optimum'), 'fig2c.spearman.updates_per_stage~optimum',
         'F2c.secondary', 'updates per stage vs certified optimal moves')
    for col in ['updates', 'neuron_ms']:
        tid = f'fig2c.theilsen.log10_{col}~log10_optimum'
        _rec(stats.theil_sen(np.log10(x), np.log10(c2[col].to_numpy(float)), tid), tid, 'F2c.estimates',
             f'log-log slope (elasticity) of {col} on certified optimal moves')
    keep = ~c2['interrupted'].to_numpy(bool)
    sub = c2[keep]
    xs = sub['optimal_moves'].to_numpy(float)
    _rec(stats.spearman(xs, sub['updates'], 'fig2c.n22.spearman.updates~optimum'), 'fig2c.n22.spearman.updates~optimum',
         'F2c.sensitivity22', 'updates vs optimum without interrupted runs (n = 22)')
    _rec(stats.spearman(xs, sub['neuron_ms'], 'fig2c.n22.spearman.neuron_ms~optimum'), 'fig2c.n22.spearman.neuron_ms~optimum',
         'F2c.sensitivity22', 'neuron-ms vs optimum without interrupted runs (n = 22)')
    _rec(stats.jonckheere(sub['updates'], sub['tier'], 'fig2c.n22.jt.updates~tier'), 'fig2c.n22.jt.updates~tier',
         'F2c.sensitivity22', 'trend of updates over tiers without interrupted runs (n = 22)')


def _tests_2d(d2):
    excess, rhos = [], []
    for rid, g in d2.groupby('run_id'):
        g = g[g.push_entry.notna()]
        total = g['updates_spent'].sum()
        push = g[g.push_entry.astype(bool)]
        if total > 0 and len(g):
            excess.append(push['updates_spent'].sum() / total - len(push) / len(g))
        if g['updates_spent'].nunique() > 1 and g['deadlock_alternatives_at_entry'].nunique() > 1:
            rhos.append(float(st.spearmanr(g['updates_spent'], g['deadlock_alternatives_at_entry']).statistic))
    kw = dict(panel='fig2d', source_table='main/fig2/fig2d_stage_costs.csv', status='new_registered')
    _rec(stats.signed_rank(excess, 'fig2d.wilcoxon.push_excess_share'), 'fig2d.wilcoxon.push_excess_share', 'F2d',
         'per-map excess share of updates spent at push-entry stages (share of updates minus share of stages)', **kw)
    _rec(stats.sign_test(excess), 'fig2d.sign.push_excess_share', 'F2d.companion', 'sign of the per-map push excess share', **kw)
    _rec(stats.signed_rank(rhos, 'fig2d.wilcoxon.rho_deadlock'), 'fig2d.wilcoxon.rho_deadlock', 'F2d',
         'per-map Spearman rho between stage updates and deadlocking alternatives at the stage entry', **kw)
    _rec(stats.sign_test(rhos), 'fig2d.sign.rho_deadlock', 'F2d.companion', 'sign of the per-map rho(stage updates, deadlocking alternatives)', **kw)


def fig2e(lv):
    rows = []
    for k in ('D1', 'D3', 'D5', 'D7'):
        p = paths.R0 / f'generalization-{k}' / 'generalization.json'
        g = io.json_load(p)
        res = io.json_load(paths.R0 / f'generalization-{k}' / 'result.json')
        recs = g['records']
        held = [r for r in recs if r.get('split') == 'heldout']
        n_solved = sum(r['status'] == 'solved' for r in held)
        lo, hi = stats.cp_interval(n_solved, len(held))
        for r in recs:
            lev = lv.get(r['task'])
            rows.append(dict(row_type='record', model=f'difficulty-{k}-T0', task=r['task'], split=r.get('split'), family_id=r.get('family'),
                             tier=int(lev['tier']) if lev else None, optimal_moves=int(lev['optimal_steps']) if lev else None,
                             status=taxonomy.behavior(r['status']), solved=r['status'] == 'solved', steps=r.get('steps'),
                             numerator=None, denominator=None, estimate=None, ci_low=None, ci_high=None, ci_method=None,
                             source_file=paths.rel(p)))
        rows.append(dict(row_type='summary_heldout', model=f'difficulty-{k}-T0', task='all_heldout', split='heldout', family_id=None, tier=None,
                         optimal_moves=None, status=None, solved=None, steps=None, numerator=n_solved, denominator=len(held),
                         estimate=n_solved / len(held), ci_low=lo, ci_high=hi, ci_method='clopper_pearson_exact',
                         source_file=paths.rel(paths.R0 / f'generalization-{k}' / 'result.json')))
        if res.get('heldout_solved') != n_solved or res.get('heldout_total') != len(held):
            raise RuntimeError(f'generalization-{k}: summary disagrees with records')
    df = pd.DataFrame(rows)
    csvout.write_table(df, 'main/fig2/fig2e_zero_update_transfer.csv', title='Zero-update transfer of representative models', describe={
        'row_type': 'record (one evaluated layout) or summary_heldout (held-out proportion)', 'task': 'Evaluated layout',
        'split': 'heldout (8 untrained layouts), new_start or transform (variants of trained maps)', 'family_id': 'Wall-geometry family',
        'optimal_moves': 'Certified optimal moves', 'status': 'Rollout end state', 'solved': 'Solved', 'steps': 'Moves in the rollout'},
        key=['model', 'row_type', 'task'], sort=['model', 'row_type', 'split', 'task'])


def m4(comp):
    rows = []
    for rid, r in comp.iterrows():
        res = _js(rid, 'result.json') or {}
        env = _js(rid, 'environment.json') or {}
        calls = r['uninterrupted_equivalent_calls']
        rows.append(dict(run_id=rid, layer='remote_static' if rid in REMOTE_STATIC else 'local_static', condition=r['condition'],
                         task=r['task'], status=res.get('status'), updates=res.get('updates'), training_seconds=res.get('training_seconds'),
                         wall_seconds=None if r['interrupted'] else res.get('wall_seconds'), forward_calls=calls,
                         neuron_ms=(int(calls) * int(r['neuron_ms_per_call'])) if pd.notna(calls) else None,
                         compute_basis=r['compute_basis'], peak_allocated_gib=(res.get('resources') or {}).get('peak_allocated_gib'),
                         hardware=env.get('gpu'), source_file=r['source_file']))
    for base, layer in ((paths.R0, 'remote_memory'), (paths.RS, 'supplement_memory')):
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            if not (d / 'result.json').exists() or not d.name.startswith(('remote-dynamic', 'supp-S', 'supp-M0-visibility', 'supp-origin')):
                continue
            if d.name.startswith('supp-M0-visibility-'):
                continue
            res = io.json_load(d / 'result.json')
            tc = res.get('training_compute') or {}
            rows.append(dict(run_id=d.name, layer=layer, condition=(res['config'].get('arm') or res['config'].get('condition')),
                             task=res['config'].get('task'), status=res.get('status'), updates=res.get('updates'),
                             training_seconds=res.get('training_seconds'), wall_seconds=res.get('wall_seconds'),
                             forward_calls=tc.get('forward_calls'), neuron_ms=tc.get('neuron_ms'),
                             compute_basis='recorded' if tc else 'not_applicable',
                             peak_allocated_gib=(res.get('resources') or {}).get('peak_allocated_gib'),
                             hardware='NVIDIA GeForce RTX 5090 (remote)', source_file=paths.rel(d)))
    df = pd.DataFrame(rows)
    df['forward_calls'] = df['forward_calls'].astype('Int64')
    df['neuron_ms'] = df['neuron_ms'].astype('Int64')
    csvout.write_table(df, 'methods/m4_compute_per_run.csv', title='Compute per training run', describe={
        'layer': 'Evidence layer', 'condition': 'Condition or arm', 'task': 'Task', 'status': 'End status', 'updates': 'Optimizer updates',
        'training_seconds': 'Training-loop seconds', 'wall_seconds': 'Wall-clock seconds of the final process (empty when interrupted)',
        'forward_calls': 'Simulator forward calls (rebuilt for static runs; recorded for memory runs)',
        'neuron_ms': 'Simulated neuron-milliseconds', 'compute_basis': 'rebuilt_exact_match, rebuilt_uninterrupted_equivalent, recorded or not_applicable',
        'peak_allocated_gib': 'Peak allocated GPU memory (GiB)', 'hardware': 'GPU'}, key=['run_id'], sort=['layer', 'run_id'])
