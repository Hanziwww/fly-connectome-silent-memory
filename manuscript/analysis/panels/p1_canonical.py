"""P1: canonical tables — model specification, parameter budget, task library, memory design,
model registry, history sets, run registry (ST1), certificates (ST3), methods tables M1-M3, ED10."""
import json
import math

import numpy as np
import pandas as pd

from msa import csvout, io, paths, registry, static_runs, taxonomy

REP_RUN = 'difficulty-D3-T0'


def _src(*ps):
    return ';'.join(paths.rel(p) for p in ps)


# ---------------------------------------------------------------- Fig. 1a

def fig1a():
    b = static_runs.brain(REP_RUN)
    brain_p = paths.RUNS / REP_RUN / 'brain.json'
    dyn_p = paths.RUNS / REP_RUN / 'resolved-dynamics.npz'
    idx_p = paths.M1 / 'neurons' / 'index.npz'
    ann_p = paths.M1 / 'neurons' / 'annotations.parquet'
    w_p = paths.CONNECTOME / 'W_post_pre.npz'
    dyn = io.npz_load(dyn_p)
    idx = io.npz_load(idx_p)
    ann = io.parquet_load(ann_p)
    W = io.sparse_load(w_p)
    rows = []

    def add(section, quantity, value, unit, source, notes=''):
        rows.append(dict(section=section, quantity=quantity, value=value, unit=unit, source_file=source, notes=notes))

    s = _src(brain_p)
    add('network', 'neurons', int(b['neurons']), 'cells', s)
    add('network', 'stored_connectome_entries', int(b['stored_connectome_edges']), 'directed pairs', s)
    add('network', 'connectome_matrix_nonzero', int(W.nnz), 'directed pairs', _src(w_p), 'signed synapse-count matrix, rows = postsynaptic')
    add('network', 'excitatory_entries', int((W.data > 0).sum()), 'directed pairs', _src(w_p))
    add('network', 'inhibitory_entries', int((W.data < 0).sum()), 'directed pairs', _src(w_p))
    add('network', 'effective_simulated_edges', int(b['effective_stored_edges']), 'directed pairs', s,
        'after the simulator connection rules (cap, same-type gain, pruning)')
    add('network', 'shared_parameter_groups', int(b['groups']), 'groups', s, b.get('mapping', ''))
    add('network', 'input_cells', int(b['input_neurons']), 'cells', s, '400 board inputs (4 channels x 10 x 10) x 8 sensory cells')
    add('network', 'input_cells_index', int(idx['inputs'].sum()), 'cells', _src(idx_p))
    add('network', 'observed_cells', int(b['observed_neurons']), 'cells', s)
    add('network', 'observed_cells_index', int(idx['observed'].sum()), 'cells', _src(idx_p))
    add('network', 'input_observed_overlap', int((idx['inputs'] & idx['observed']).sum()), 'cells', _src(idx_p))
    add('network', 'degree_median', float(np.median(idx['degree'])), 'edges per cell', _src(idx_p), 'in + out edges of the effective graph')
    add('network', 'degree_max', int(idx['degree'].max()), 'edges per cell', _src(idx_p))
    add('readout', 'populations', int(b['populations']), 'pools', s)
    add('readout', 'cells_per_population', int(b['neurons_per_population']), 'cells', s)
    add('readout', 'feature_dim', int(b['feature_dim']), 'features', s, 'pooled synaptic current at 14 ms + pooled voltage relative to rest at 40 ms')
    add('readout', 'feature_snapshot_current_ms', int(b['snapshots_ms'][0]), 'ms', s)
    add('readout', 'feature_snapshot_voltage_ms', int(b['snapshots_ms'][1]), 'ms', s)
    add('readout', 'feature_clip', 10.0, 'SD units', _src(paths.DOCS / 'method.md'), 'standardized features clipped to [-10, 10]; normalization fixed')
    add('readout', 'actions', 4, 'actions', _src(paths.DOCS / 'method.md'), 'up, down, left, right; illegal moves masked')
    add('simulation', 'decision_window', int(b['simulation_ms']), 'ms', s, 'state reset before each decision in static tasks; persistent in memory tasks')
    add('simulation', 'time_step', float(b['lif']['dt']), 'ms', s)
    add('simulation', 'input_drive', 18.0, 'mV', _src(paths.CODE / 'backend.py'), 'board value x 18 mV injected into each of 8 sensory cells')
    add('simulation', 'precision', b['precision'], '', s, b.get('sparse_accumulation', ''))
    add('simulation', 'batch', int(b['batch']), 'boards', s)
    lif_units = dict(v_rest='mV', v_reset='mV', v_th='mV', tau_m='ms', tau_syn='ms', t_ref='ms', delay='ms', w_syn='mV per synapse',
                     adapt_jump='mV per spike', adapt_tau='ms', std_tau='ms', input_norm_ref='synapses', same_type_gain='',
                     conn_cap='synapses', rate_tau='ms')
    for k, unit in lif_units.items():
        add('lif', k, b['lif'][k], unit, s)
    add('lif', 'receptor_model', b['lif']['receptor_model'], '', s)
    add('lif', 'slow_currents_active', bool(b['slow_active']), '', s)
    add('lif', 'short_term_depression_active', bool(b['std_active']), '', s)
    add('lif', 'cells_with_nonzero_depression', int(np.count_nonzero(dyn['std_u'])), 'cells', _src(dyn_p))
    add('lif', 'max_depression_u', float(dyn['std_u'].max()), '', _src(dyn_p))
    add('lif', 'surrogate_gradient', bool(b['lif']['surrogate_grad']), '', s, 'hard spikes forward; surrogate derivative backward')
    a = _src(ann_p)
    add('annotation', 'cells_missing_type', int(ann['type'].isna().sum()), 'cells', a)
    add('annotation', 'cells_missing_superclass', int(ann['superclass'].isna().sum()), 'cells', a)
    add('annotation', 'cells_missing_soma_neuromere', int(ann['somaNeuromere'].isna().sum()), 'cells', a)
    add('annotation', 'cells_transmitter_unknown', int((ann['nt'] == 'unknown').sum()), 'cells', a)
    add('annotation', 'superclasses', int(ann['superclass'].nunique()), 'categories', a)
    add('annotation', 'transmitters', int(ann['nt'].nunique()), 'categories', a)
    add('annotation', 'cell_types', int(ann['type'].nunique()), 'categories', a)
    df = pd.DataFrame(rows)
    df['value'] = df['value'].map(lambda v: v if isinstance(v, str) else json.dumps(v) if isinstance(v, (bool, list)) else v)
    df['value'] = df['value'].astype(str)
    csvout.write_table(df, 'main/fig1/fig1a_model_specification.csv', title='Model specification',
                       describe={'section': 'Specification block', 'quantity': 'Quantity', 'value': 'Value',
                                 'unit': 'Unit'}, key=['section', 'quantity'], sort=[])


# ---------------------------------------------------------------- Fig. 1b

def fig1b():
    ck_p = paths.RUNS / REP_RUN / 'model.pt'
    ck = io.torch_load(ck_p)
    core, head, _ = registry.split_checkpoint(ck)
    lr_core, lr_head = 1e-4, 3e-4
    rows = []
    for k in registry.CORE_KEYS:
        n = int(core[k].numel())
        constraint = 'gain in [0.8, 1.25] (log-space clamp)' if 'gain' in k else 'offset in [-1, 1] mV'
        role = {'pre_log_gain': 'outgoing (presynaptic) gain per type group',
                'post_log_gain': 'incoming (postsynaptic) gain per type group',
                'threshold_mv': 'firing-threshold offset per type group'}[k]
        rows.append(dict(model_kind='connectome_linear', component='brain', parameter=k, shape=json.dumps(list(core[k].shape)),
                         n_elements=n, trainable_in_joint=True, constraint=constraint, learning_rate=lr_core, gradient_clip=1.0,
                         role=role, source_file=_src(ck_p, paths.ROOT / 'sokoban' / 'lib' / 'brain_model.py')))
    for k in registry.HEAD_KEYS:
        n = int(head[k].numel())
        trainable = k.startswith('proposal')
        rows.append(dict(model_kind='connectome_linear', component='readout', parameter=k, shape=json.dumps(list(head[k].shape)),
                         n_elements=n, trainable_in_joint=trainable, constraint='' if trainable else 'fixed (inherited normalization)',
                         learning_rate=lr_head if trainable else None, gradient_clip=5.0 if trainable else None,
                         role={'mean': 'feature normalization mean', 'scale': 'feature normalization scale',
                               'proposal.weight': 'affine readout weights (512 -> 4)', 'proposal.bias': 'readout bias'}[k],
                         source_file=_src(ck_p)))
    # Comparison learners (parameter counts read from their checkpoints).
    for kind, path in [('board_mlp', paths.RUNS / 'control-D1-board_mlp' / 'model.pt'),
                       ('gru', paths.R0 / 'remote-dynamic-M0-gru' / 'model.pt'),
                       ('fixed_trace_linear', paths.RS / 'supp-origin-S1-ring-leaky' / 'model.pt')]:
        c = io.torch_load(path)
        pair = c.get('pair', c)
        tensors = {}
        if kind == 'gru':
            tensors = pair['gru']
        else:
            tensors = pair.get('head') or {}
        for k, v in tensors.items():
            fixed = k in ('mean', 'scale')
            rows.append(dict(model_kind=kind, component='learner', parameter=k, shape=json.dumps(list(v.shape)),
                             n_elements=int(v.numel()), trainable_in_joint=not fixed,
                             constraint='fixed' if fixed else '', learning_rate=None, gradient_clip=None,
                             role='comparison learner parameter', source_file=_src(path)))
    df = pd.DataFrame(rows)
    totals = df.groupby(['model_kind', 'trainable_in_joint'])['n_elements'].sum().reset_index()
    for _, t in totals.iterrows():
        df = pd.concat([df, pd.DataFrame([dict(model_kind=t['model_kind'], component='total',
                                               parameter='trainable' if t['trainable_in_joint'] else 'fixed',
                                               shape='', n_elements=int(t['n_elements']), trainable_in_joint=bool(t['trainable_in_joint']),
                                               constraint='', learning_rate=None, gradient_clip=None,
                                               role='sum over parameters', source_file='')])], ignore_index=True)
    csvout.write_table(df, 'main/fig1/fig1b_parameter_budget.csv', title='Trainable parameters by component and constraint',
                       describe={'model_kind': 'Model architecture', 'component': 'Component (brain, readout, learner, total)',
                                 'parameter': 'Parameter tensor', 'shape': 'Tensor shape (JSON)', 'n_elements': 'Number of scalar parameters',
                                 'trainable_in_joint': 'Trainable in the reference (joint or learner) training condition',
                                 'constraint': 'Projection or bound applied after each step', 'learning_rate': 'Adam learning rate',
                                 'gradient_clip': 'Gradient-norm clip', 'role': 'Interpretation'},
                       key=['model_kind', 'component', 'parameter'], sort=[])


# ---------------------------------------------------------------- Fig. 1d / ST3

def fig1d_st3():
    lv_p = paths.TASKS / 'levels.json'
    cert_p = paths.TASKS / 'certificates.json'
    am_p = paths.STUDY / 'registration-amendments.json'
    levels = io.json_load(lv_p)
    certs = io.json_load(cert_p)
    amend = {c['task']: c for c in io.json_load(am_p)['changes']}
    rows = []
    for l in levels:
        c = certs[l['id']]
        dead = l.get('deadlocking_alternatives_on_witness') or []
        branch = l.get('witness_branch_counts') or []
        rows.append(dict(map_id=l['id'], family_id=l['family_id'], split=l['split'], tier=int(l['tier']), board_size=int(l['size']),
                         boxes=int(l['boxes']), optimal_moves=int(l['optimal_steps']), floor_cells=int(l['floor_cells']),
                         minimum_pushes_shortest=int(l['minimum_pushes_among_shortest_solutions']), witness_pushes=int(l['witness_pushes']),
                         shortest_path_count=int(l['shortest_path_count_capped']),
                         shortest_path_count_capped=int(l['shortest_path_count_capped']) >= int(l['shortest_path_count_cap']),
                         shortest_dag_states=int(l['shortest_dag_states']), floor_branch_cells=int(l['floor_branch_cells']),
                         articulation_cells=len(l.get('articulation_cells') or []),
                         deadlocking_alternatives_total=int(sum(dead)), witness_steps_with_deadlock_alternative=int(sum(1 for d in dead if d > 0)),
                         witness_branch_mean=float(np.mean(branch)) if branch else None,
                         certificate_exact=bool(c['exact']), certificate_complete=bool(c['complete']),
                         solvable_states=int(c['solvable_states']), theoretical_states=int(c['theoretical_states']),
                         maximum_distance=int(c['maximum_distance']), forward_verified=bool(c['forward']['verified']),
                         forward_optimal_moves=int(c['forward']['optimal_steps']), forward_visited_states=int(c['forward']['visited_states']),
                         family_replaced_before_evaluation=l['id'] in amend,
                         previous_family_id=amend[l['id']]['old_family'] if l['id'] in amend else None,
                         source_file=_src(lv_p, cert_p, am_p)))
    df = pd.DataFrame(rows)
    desc = {'family_id': 'Wall-geometry family identifier', 'split': 'train, heldout or calibration', 'board_size': 'Board side length (cells)',
            'boxes': 'Number of boxes', 'optimal_moves': 'Certified optimal number of moves', 'floor_cells': 'Number of floor cells',
            'minimum_pushes_shortest': 'Minimum pushes among shortest solutions', 'witness_pushes': 'Pushes in the reference (witness) solution',
            'shortest_path_count': 'Number of shortest solutions (capped)', 'shortest_path_count_capped': 'Count reached the cap',
            'shortest_dag_states': 'States in the shortest-path DAG', 'floor_branch_cells': 'Floor cells with three or more floor neighbours',
            'articulation_cells': 'Articulation cells of the floor graph', 'deadlocking_alternatives_total': 'Sum over witness steps of legal moves leaving the solvable set',
            'witness_steps_with_deadlock_alternative': 'Witness steps with at least one deadlocking alternative',
            'witness_branch_mean': 'Mean number of solvable branches along the witness', 'certificate_exact': 'Exact reverse search certificate',
            'certificate_complete': 'Reverse search complete', 'solvable_states': 'Solvable states found', 'theoretical_states': 'Theoretical state count',
            'maximum_distance': 'Maximum distance to solution among solvable states', 'forward_verified': 'Independent forward search verified the optimum',
            'forward_optimal_moves': 'Optimum found by forward search', 'forward_visited_states': 'States visited by forward search',
            'family_replaced_before_evaluation': 'Family replaced by the geometry amendment before any evaluation', 'previous_family_id': 'Replaced family id'}
    csvout.write_table(df, 'main/fig1/fig1d_task_library.csv', title='Static task library (34 maps)', describe=desc,
                       key=['map_id'], sort=['split', 'tier', 'map_id'])
    # ST3: certificates for static maps and memory cases.
    st3 = df[['map_id', 'split', 'optimal_moves', 'certificate_exact', 'certificate_complete', 'solvable_states', 'theoretical_states',
              'maximum_distance', 'forward_verified', 'forward_optimal_moves', 'forward_visited_states', 'source_file']].copy()
    st3.insert(0, 'task_kind', 'static_map')
    st3 = st3.rename(columns={'map_id': 'task_id'})
    mem_rows = []
    mem_p = paths.TASKS / 'memory.json'
    for t in io.json_load(mem_p):
        cert = t.get('certificates')
        for l in t['levels']:
            c = cert.get(l['id']) if isinstance(cert, dict) else None
            mem_rows.append(dict(task_kind='memory_case', task_id=l['id'], split='memory', optimal_moves=int(l['optimal_steps']),
                                 certificate_exact=None, certificate_complete=None, solvable_states=None, theoretical_states=None,
                                 maximum_distance=None, forward_verified=None, forward_optimal_moves=None, forward_visited_states=None,
                                 certificate_json=json.dumps(c, sort_keys=True) if c is not None else json.dumps(cert, sort_keys=True)[:2000],
                                 source_file=_src(mem_p)))
    rt_p = paths.RS_ROOT / 'tasks.json'
    for t in io.json_load(rt_p):
        for c in t['cases']:
            mem_rows.append(dict(task_kind='memory_case', task_id=f"{t['id']}:{c['id']}", split='supplement_memory',
                                 optimal_moves=int(c['level']['optimal_steps']), certificate_exact=None, certificate_complete=None,
                                 solvable_states=c.get('solvable_states'), theoretical_states=None, maximum_distance=None,
                                 forward_verified=None, forward_optimal_moves=None, forward_visited_states=None,
                                 certificate_json=json.dumps(c.get('certificate'), sort_keys=True), source_file=_src(rt_p)))
    st3['certificate_json'] = None
    st3 = pd.concat([st3, pd.DataFrame(mem_rows)], ignore_index=True)
    csvout.write_table(st3, 'supplementary/st3_task_certificates.csv', title='Task certificates', describe={
        **desc, 'task_kind': 'static_map or memory_case', 'task_id': 'Task identifier',
        'certificate_json': 'Certificate record for memory cases (irreversible-push and solvability certificates), JSON'},
        key=['task_kind', 'task_id'], sort=['task_kind', 'task_id'])


# ---------------------------------------------------------------- Fig. 1e and history sets

def fig1e():
    mem_p = paths.TASKS / 'memory.json'
    rt_p = paths.RS_ROOT / 'tasks.json'
    proto_p = paths.STUDY / 'protocol.json'
    clar_p = paths.STUDY / 'memory-protocol-clarification.json'
    proto = io.json_load(proto_p)
    io.json_load(clar_p)
    rows = []
    m0 = next(t for t in io.json_load(mem_p) if t['id'] == 'M0')
    for l in m0['levels']:
        cue = int(l['cue'])
        rows.append(dict(section='case', family='M0', case=f'c0-cue{cue}', cue=cue, context=0, visible=False, delay_ms=None,
                         prefix_windows=None, split=None, start_player=int(l['start'][0]), start_boxes=json.dumps(l['start'][1]),
                         goals=json.dumps(l['goals']), first_push_action=(2 if cue == 0 else 3), reference_moves=int(l['optimal_steps']),
                         description='hidden-goal cue task; goal shown only during the 40-ms cue', source_file=_src(mem_p)))
    for t in io.json_load(rt_p):
        fam = {'S1-ring': 'S1', 'S2-context': 'S2', 'M0-visibility': 'M0vis', 'M0-audit': 'M0audit'}[t['id']]
        for c in t['cases']:
            rows.append(dict(section='case', family=fam, case=c['id'], cue=int(c['cue']), context=int(c['context']), visible=bool(c['visible']),
                             delay_ms=None, prefix_windows=None, split=None, start_player=int(c['level']['start'][0]),
                             start_boxes=json.dumps(c['level']['start'][1]), goals=json.dumps(c['level']['goals']),
                             first_push_action=c.get('first_push_target'), reference_moves=int(c['level']['optimal_steps']),
                             description=t.get('context_protocol', ''), source_file=_src(rt_p)))
    for d in registry.GRID_DELAYS:
        for p in registry.GRID_PREFIXES:
            rows.append(dict(section='timing', family='all', case=None, cue=None, context=None, visible=None, delay_ms=d,
                             prefix_windows=p, split=registry.split_of(d, p), start_player=None, start_boxes=None, goals=None,
                             first_push_action=None, reference_moves=None,
                             description=('used for training and checkpoint selection' if registry.split_of(d, p) == 'train'
                                          else 'evaluation only'), source_file=_src(proto_p, clar_p)))
    frames = [('cue', 40, 'goal visible (hidden-goal tasks) or board with goal (visible-goal tasks)'),
              ('delay', None, 'blank board for the delay duration'), ('prefix_window', 40, 'cue-independent legal walk, per window'),
              ('decision', 40, 'one 40-ms window per autonomous decision; state persists across windows')]
    for name, ms, desc in frames:
        rows.append(dict(section='frame', family='all', case=name, cue=None, context=None, visible=None, delay_ms=ms, prefix_windows=None,
                         split=None, start_player=None, start_boxes=None, goals=None, first_push_action=None, reference_moves=None,
                         description=desc, source_file=_src(proto_p)))
    df = pd.DataFrame(rows)
    csvout.write_table(df, 'main/fig1/fig1e_memory_task_design.csv', title='Memory task design', describe={
        'section': 'case (task case), timing (delay x prefix grid) or frame (history structure)',
        'family': 'Task family: M0, S1, S2, M0vis (shared visible/hidden readout), M0audit (legacy prefix audit)',
        'case': 'Case identifier (context-cue[-visibility]) or frame name', 'cue': 'Cue index', 'context': 'Scene orientation (S2)',
        'visible': 'Goal visible at decision', 'delay_ms': 'Blank delay after the cue (timing rows) or frame duration (frame rows)',
        'prefix_windows': 'Common-prefix windows before the decision', 'split': 'train (used for training) or untrained (evaluation only)',
        'start_player': 'Player cell at the decision', 'start_boxes': 'Box cells at the decision (JSON)', 'goals': 'Goal cells (JSON)',
        'first_push_action': 'Action of the correct first irreversible push (0 up, 1 down, 2 left, 3 right)',
        'reference_moves': 'Reference solution length from the decision state', 'description': 'Description'},
        key=None, sort=['section', 'family', 'case', 'delay_ms', 'prefix_windows'])
    registry.register_protocol_sets()
    _ = proto


# ---------------------------------------------------------------- model registry

def models():
    rows = []

    def add(model_id, layer, task, condition, path, init_path=None):
        ck = io.torch_load(path) if path.exists() else None
        core, head, other = registry.split_checkpoint(ck)
        kind = registry.classify_kind(core, head, other) if ck is not None else 'missing'
        init_hash = None
        if init_path is not None and init_path.exists():
            ic, _, _ = registry.split_checkpoint(io.torch_load(init_path))
            init_hash = registry.core_hash(ic)
        trainable_head = 0
        if head:
            trainable_head = sum(int(v.numel()) for k, v in head.items() if k not in ('mean', 'scale'))
        rows.append(dict(model_id=model_id, layer=layer, task=task, condition=condition, kind=kind,
                         brain_hash=registry.core_hash(core), initial_brain_hash=init_hash, head_hash=registry.head_hash(head),
                         n_core_parameters=registry.n_elements(core), n_readout_trainable=trainable_head,
                         n_learner_parameters=registry.n_elements(other.get('gru')) if 'gru' in other else 0,
                         source_file=paths.rel(path)))

    for name in ('20261002-p03-01', '20261002-p05-01'):
        add('P' + name.split('-p')[1][:2], 'formal_lineage', name, 'joint', paths.LINEAGE / name / 'model.pt')
    for rid in static_runs.run_ids():
        c = static_runs.config(rid)
        add(rid, 'local_static', c['task'], c['condition']['name'], paths.RUNS / rid / 'model.pt', paths.RUNS / rid / 'initial.pt')
    for rid in ('remote-D7-head_only', 'remote-D7-demonstration'):
        add(rid, 'remote_static', 'D7-T0', rid.split('-')[-1], paths.R0 / rid / 'model.pt', paths.R0 / rid / 'initial.pt')
    for cond in ('carry_untrained', 'reset', 'head_only', 'joint', 'gru'):
        rid = f'remote-dynamic-M0-{cond}'
        add(rid, 'remote_memory', 'M0', cond, paths.R0 / rid / 'model.pt', paths.R0 / rid / 'initial.pt')
    for d in sorted(p.name for p in paths.RS.iterdir() if p.is_dir()):
        if (paths.RS / d / 'model.pt').exists():
            r = io.json_load(paths.RS / d / 'result.json')
            add(d, 'supplement_memory', r['config']['task'], r['config'].get('arm') or r['config'].get('policy_kind'),
                paths.RS / d / 'model.pt', paths.RS / d / 'initial.pt')
    df = pd.DataFrame(rows)
    p05 = df.loc[df.model_id == 'P05', 'brain_hash'].iloc[0]
    df['brain_equals_p05'] = df['brain_hash'] == p05
    first = {}
    for _, r in df.sort_values('model_id').iterrows():
        if r['brain_hash'] and r['brain_hash'] not in first:
            first[r['brain_hash']] = r['model_id']
    df['brain_state_group'] = df['brain_hash'].map(lambda h: first.get(h) if h else None)
    df['core_changed_in_run'] = [None if (i is None or b is None) else (i != b) for i, b in zip(df['initial_brain_hash'], df['brain_hash'])]
    csvout.write_table(df, 'meta/model_registry.csv', title='Model registry with brain and readout hashes', describe={
        'model_id': 'Model (run) identifier', 'layer': 'Evidence layer', 'task': 'Task or family', 'condition': 'Training condition or arm',
        'kind': 'Architecture: connectome_linear, board_mlp, gru, fixed_trace_linear',
        'brain_hash': 'SHA-256 of the final core parameters (pre-gain, post-gain, threshold; float32)',
        'initial_brain_hash': 'SHA-256 of the initial core parameters (initial.pt)', 'head_hash': 'SHA-256 of the readout tensors',
        'n_core_parameters': 'Core (brain) parameters in the checkpoint', 'n_readout_trainable': 'Trainable readout parameters',
        'n_learner_parameters': 'Recurrent learner parameters (GRU)', 'brain_equals_p05': 'Final core identical to the P05 core',
        'brain_state_group': 'First model (alphabetical) sharing this exact core; brain-level statistics are computed once per group',
        'core_changed_in_run': 'Final core differs from the run initial core'},
        key=['model_id'])
    return df


# ---------------------------------------------------------------- ST1 run registry

def _short(v):
    return v[:8] if isinstance(v, str) else None


def st1(models_df):
    rows = []
    gpu = {}

    def add(**kw):
        base = dict(layer=None, run_id=None, module=None, task=None, condition=None, parent=None, status=None, status_group=None,
                    updates=None, training_seconds=None, wall_seconds=None, final_solved=None, final_steps=None,
                    verification_status=None, accepted=None, hardware=None, code_version=None, interrupted=False,
                    n_interruptions=0, notes='', source_file='')
        base.update(kw)
        base['status_group'] = taxonomy.training_group(base['status'])
        rows.append(base)

    for name in ('20261002-p03-01', '20261002-p05-01'):
        p = paths.LINEAGE / name / 'result.json'
        r = io.json_load(p)
        tr = r.get('training', {})
        add(layer='formal_lineage', run_id=name, module='linear_joint', task=r.get('task') or name.split('-')[-2].upper(),
            condition='joint', status=tr.get('stop'), updates=tr.get('joint_updates'), training_seconds=tr.get('training_seconds'),
            hardware='local', source_file=paths.rel(p))
    for rid in static_runs.run_ids():
        c = static_runs.config(rid)
        r = static_runs.result(rid) or {}
        v = static_runs.verification(rid) or {}
        a = static_runs.acceptance(rid) or {}
        env = static_runs.environment(rid) or {}
        intr = static_runs.interruptions(rid)
        add(layer='local_static', run_id=rid, module=c.get('module'), task=c.get('task'), condition=c['condition']['name'],
            parent=c.get('parent'), status=static_runs.status(rid), updates=r.get('updates'), training_seconds=r.get('training_seconds'),
            wall_seconds=r.get('wall_seconds'), final_solved=r.get('final_solved'), final_steps=r.get('final_steps'),
            verification_status=v.get('status'), accepted=a.get('accepted'), hardware=env.get('gpu') or 'local (environment not recorded)',
            code_version=_short(c.get('code_version')), interrupted=bool(intr), n_interruptions=len(intr),
            notes=('counters cover only the resumed process' if intr else ''), source_file=paths.rel(paths.RUNS / rid))
    sched_p = paths.STUDY / 'schedule.json'
    present = set(static_runs.run_ids())
    for e in io.json_load(sched_p):
        if e['id'] not in present:
            cond = e['condition']['name'] if isinstance(e.get('condition'), dict) else e.get('condition')
            add(layer='registered_not_run', run_id=e['id'], module=e.get('module'), task=e.get('task'), condition=cond,
                parent=e.get('parent'), status='pending', notes='registered in schedule.json; never executed', source_file=paths.rel(sched_p))
    plan_p = paths.R0_ROOT / 'plan.json'
    for j in io.json_load(plan_p)['jobs']:
        d = paths.R0 / j['id']
        r = io.json_load(d / 'result.json') if (d / 'result.json').exists() else {}
        v = io.json_load(d / 'verification.json') if (d / 'verification.json').exists() else {}
        env = io.json_load(d / 'environment.json') if (d / 'environment.json').exists() else {}
        cfg = r.get('config') or {}
        add(layer='remote_batch1', run_id=j['id'], module=j.get('mode'), task=cfg.get('task') or j.get('reference'),
            condition=cfg.get('condition') if isinstance(cfg.get('condition'), str) else (cfg.get('condition') or {}).get('name'),
            status=r.get('status'), updates=r.get('updates'), training_seconds=r.get('training_seconds'), wall_seconds=r.get('wall_seconds'),
            final_solved=r.get('final_solved') if not isinstance(r.get('final_solved'), dict) else None,
            verification_status=v.get('status'), hardware=env.get('gpu') or 'NVIDIA GeForce RTX 5090 (remote)',
            code_version=_short(cfg.get('code_version')), source_file=paths.rel(d))
    an_p = paths.RS_ROOT / 'analysis.json'
    for r in io.json_load(an_p)['runs']:
        add(layer='remote_supplement', run_id=r['id'], module=r.get('mode'), task=r.get('task'), condition=r.get('arm') or r.get('policy_kind'),
            status=r.get('status'), updates=r.get('updates'), training_seconds=r.get('training_seconds'),
            verification_status=r.get('verification_status'), accepted=r.get('training_accepted'),
            hardware='NVIDIA GeForce RTX 5090 (remote)', notes=f"core parameter changes: {r.get('core_parameter_changes')}",
            source_file=paths.rel(an_p))
    for d in sorted(p for p in (paths.M1 / 'runs').iterdir() if p.is_dir()):
        r = io.json_load(d / 'result.json') if (d / 'result.json').exists() else {}
        add(layer='mechanisms_v1', run_id=d.name, module=d.name.split('-')[0], task=r.get('model'), condition='frozen_replay',
            status=r.get('status'), hardware='NVIDIA GeForce RTX 5090 (remote)',
            notes=('parameters unchanged' if (r.get('frozen_audit') or {}).get('parameters_unchanged') else ''), source_file=paths.rel(d))
    ja_p = paths.MS / 'analysis' / 'final-job-audit.csv'
    for _, j in io.csv_load(ja_p).iterrows():
        add(layer='mechanisms_supplement', run_id=j['job'], module=j['kind'], task=j['model'], condition='frozen_replay',
            status='complete', wall_seconds=float(j['wall_seconds']), hardware='NVIDIA GeForce RTX 5090 (remote)',
            notes=f"conditions {int(j['conditions'])}; invalid norm matches {int(j['invalid_norm_matches'])}", source_file=paths.rel(ja_p))
    df = pd.DataFrame(rows)
    desc = {'layer': 'Evidence layer', 'module': 'Study module or job kind', 'task': 'Task, map or model',
            'condition': 'Training condition, arm or job type', 'parent': 'Parent run for sequential branches', 'status': 'Recorded end status',
            'status_group': 'Grouped status (acquired, existing, stopped_rule, baseline, stopped_user, failed, check, evaluation, never_run)',
            'updates': 'Optimizer updates', 'training_seconds': 'Training-loop seconds (cumulative across resumes)',
            'wall_seconds': 'Wall-clock seconds of the final process', 'final_solved': 'Final rollout solved the original task',
            'final_steps': 'Moves in the final rollout', 'verification_status': 'Independent-reload verification status',
            'accepted': 'Accepted by the study acceptance rule', 'hardware': 'GPU used', 'code_version': 'Code snapshot hash (first 8 hex digits)',
            'interrupted': 'Run was interrupted and resumed', 'n_interruptions': 'Number of interruptions'}
    csvout.write_table(df, 'supplementary/st1_run_registry.csv', title='Complete run registry', describe=desc,
                       key=['layer', 'run_id'], sort=['layer', 'run_id'])
    summary = df.groupby(['layer', 'module', 'status', 'status_group'], dropna=False).size().reset_index(name='n_runs')
    summary['module'] = summary['module'].fillna('')
    csvout.write_table(summary, 'extended/ed10/ed10_run_registry_summary.csv', title='Run registry summary', describe={
        **desc, 'n_runs': 'Number of runs or jobs'}, key=['layer', 'module', 'status'], sort=['layer', 'module', 'status'])
    return df


# ---------------------------------------------------------------- Methods tables

def methods():
    proto_p = paths.STUDY / 'protocol.json'
    proto = io.json_load(proto_p)
    method_p = paths.DOCS / 'method.md'
    rows = []

    def add(scope, parameter, value, source, notes=''):
        rows.append(dict(scope=scope, parameter=parameter, value=json.dumps(value) if not isinstance(value, str) else value,
                         source_file=source, notes=notes))

    s = paths.rel(proto_p)
    for k in ('seed', 'core_lr', 'head_lr', 'steps_per_assessment', 'segments_seconds', 'minimum_margin_progress', 'allocator_gib',
              'minimum_free_gib', 'success', 'delay_ms', 'trained_delay_ms', 'heldout_delay_ms', 'prefix_windows',
              'trained_prefix_windows', 'heldout_prefix_windows', 'representative_tasks', 'discovery_families', 'confirmation_families',
              'independent_unit'):
        add('study_protocol', k, proto[k], s)
    m = paths.rel(method_p)
    add('optimization', 'optimizer', 'Adam (recreated at each curriculum stage)', m)
    add('optimization', 'gradient_clip_brain', 1.0, m)
    add('optimization', 'gradient_clip_readout', 5.0, m)
    add('optimization', 'batch_boards', 16, m)
    add('optimization', 'precision', 'mixed: float64 forward, float32 gradient sums', paths.rel(paths.STUDY / 'backend.json'))
    add('optimization', 'recompute_tolerance_mixed_fp64', 1e-5, paths.rel(paths.CODE / 'training.py'))
    add('optimization', 'recompute_tolerance_fp32', 1e-3, paths.rel(paths.CODE / 'training.py'))
    add('loss', 'entry_term', '0.25 * softplus((0.10 - m_entry) / 0.25)', m)
    add('loss', 'recovery_term', '0.5 * mean(v_i) + 0.5 * max(v_i), v_i = max(0, 0.03 - m_i)^2', m)
    add('loss', 'entry_only_variant', 'entry term only (no_recovery control)', paths.rel(paths.CODE / 'training.py'))
    add('loss', 'entry_ce_variant', 'recovery term + cross-entropy at the entry instead of the entry softplus', paths.rel(paths.CODE / 'training.py'))
    add('loss', 'demonstration_variant', 'mean cross-entropy over all witness states', paths.rel(paths.CODE / 'training.py'))
    add('curriculum', 'schedule', 'reverse suffix: start one move before the goal, extend by one reference move after an autonomous pass', m)
    add('curriculum', 'stage_pass', 'original solved, or current suffix solved and all earlier suffixes retained', paths.rel(paths.CODE / 'training.py'))
    add('curriculum', 'action_ceiling', 512, m)
    sched_p = paths.STUDY / 'schedule.json'
    seen = set()
    for e in io.json_load(sched_p):
        cond = e['condition']
        if isinstance(cond, dict) and cond['name'] not in seen:
            seen.add(cond['name'])
            for k, v in cond.items():
                if k != 'name':
                    add(f"condition:{cond['name']}", k, v, paths.rel(sched_p))
    for rid, path in [('remote-dynamic-M0-head_only', paths.R0 / 'remote-dynamic-M0-head_only' / 'result.json'),
                      ('supp-S1-ring-brain', paths.RS / 'supp-S1-ring-brain' / 'result.json')]:
        cfg = io.json_load(path)['config']
        for k, v in cfg.items():
            if k in ('id', 'source', 'code_version', 'task_lock_sha256', 'protocol_sha256', 'remote_code_sha256', 'task_sha256',
                     'source_sha256', 'code_sha256', 'tasks_sha256', 'pid'):
                continue
            add(f'memory_config:{rid}', k, v, paths.rel(path))
    df = pd.DataFrame(rows)
    csvout.write_table(df, 'methods/m1_hyperparameters.csv', title='Training hyperparameters and conditions', describe={
        'scope': 'Scope of the parameter', 'parameter': 'Parameter', 'value': 'Value (JSON for non-strings)'},
        key=['scope', 'parameter'], sort=[])
    stop = []
    for status, (group, desc) in taxonomy.TRAINING.items():
        stop.append(dict(rule_type='end_status', rule=status, group=group, definition=desc, source_file=paths.rel(paths.CODE / 'training.py')))
    stop += [
        dict(rule_type='segment', rule='segments_seconds', group='', definition='time segments of 1200, 2400 and 4800 s; a segment boundary '
             'allows extension only after a promotion in that segment or an entry-margin gain of at least 0.02 with retention', source_file=s),
        dict(rule_type='success', rule='autonomous_solution', group='', definition='autonomous solution from the original start within 512 actions; any route accepted', source_file=s),
        dict(rule_type='evaluation', rule='loop_detection', group='', definition='rollout ends on a repeated state, proven corner deadlock, blocked state or the 512-move ceiling', source_file=paths.rel(paths.CODE / 'behavior.py')),
        dict(rule_type='memory_acceptance', rule='all_training_histories', group='', definition='independent reload solves every registered training history within 512 actions', source_file=paths.rel(paths.RS_ROOT / 'tasks.json')),
        dict(rule_type='verification', rule='independent_reload', group='', definition='a fresh process reloads the saved checkpoint and re-runs the original task', source_file=paths.rel(paths.STUDY / 'README.md')),
    ]
    csvout.write_table(pd.DataFrame(stop), 'methods/m2_stopping_rules.csv', title='Stopping, success and verification rules', describe={
        'rule_type': 'Rule category', 'rule': 'Rule or status', 'group': 'Status group', 'definition': 'Definition'},
        key=['rule_type', 'rule'], sort=[])
    env_rows = []
    for scope, path in [('local_training', paths.RUNS / REP_RUN / 'environment.json'),
                        ('remote_batch1', paths.R0 / 'remote-D7-head_only' / 'environment.json'),
                        ('remote_supplement', paths.RS / 'supp-S1-ring-brain' / 'environment.json')]:
        env = io.json_load(path)
        for k, v in env.items():
            if k in ('pid', 'gpu_free_before_bytes', 'free_before_bytes'):
                continue
            env_rows.append(dict(scope=scope, item=k, value=str(v), source_file=paths.rel(path)))
    mig_p = paths.ROOT / 'MIGRATION' / 'environment-windows.json'
    mig = io.json_load(mig_p)
    env_rows.append(dict(scope='analysis_runtime', item='python', value=mig['python'].split()[0], source_file=paths.rel(mig_p)))
    for d in mig['distributions']:
        if d['name'] in ('numpy', 'scipy', 'pandas', 'pyarrow', 'torch'):
            env_rows.append(dict(scope='analysis_runtime', item=d['name'], value=d['version'], source_file=paths.rel(mig_p)))
    csvout.write_table(pd.DataFrame(env_rows), 'methods/m3_environment.csv', title='Hardware and software environment', describe={
        'scope': 'Environment scope', 'item': 'Item', 'value': 'Value'}, key=['scope', 'item'], sort=[])


def code_versions():
    rows = []
    for rid in static_runs.run_ids():
        c = static_runs.config(rid)
        cv = c.get('code_version')
        snap = paths.STUDY / 'code-snapshots' / (cv or '')
        rows.append(dict(run_id=rid, code_version=cv, snapshot_present=bool(cv) and snap.exists(),
                         protocol_sha256=c.get('protocol_sha256'), task_lock_sha256=c.get('task_lock_sha256'),
                         source_file=paths.rel(paths.RUNS / rid / 'config.json')))
    df = pd.DataFrame(rows)
    counts = df['code_version'].value_counts()
    df['runs_with_this_version'] = df['code_version'].map(counts)
    csvout.write_table(df, 'meta/code_versions.csv', title='Code version per local run', describe={
        'code_version': 'Code snapshot hash recorded in config.json', 'snapshot_present': 'Snapshot folder exists in code-snapshots/',
        'protocol_sha256': 'Protocol hash recorded at run start', 'task_lock_sha256': 'Task-lock hash recorded at run start',
        'runs_with_this_version': 'Number of local runs sharing this code version'}, key=['run_id'])


def main():
    fig1a()
    fig1b()
    fig1d_st3()
    fig1e()
    mdf = models()
    st1(mdf)
    methods()
    code_versions()
