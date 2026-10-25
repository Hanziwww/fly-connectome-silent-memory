"""P10: figure-ready tables in manuscript/results/figure_data and the panel index used by manuscript/figures.

These tables hold what the figures need beyond the panel source data: cell anatomy with the unfolded-CNS display
transform, the neuropil index, boards and reference solutions, per-cell cue-related state, graded coverage curves,
per-group parameter change, time-generalisation matrices, the neuron selection for skeleton tiles and skeleton
summaries. P10 runs before P9, so the manifests and the data dictionary include these tables.
"""
import math
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from msa import cnsview, csvout, io, paths
from panels.p3_brain_readout import NORM, TOP_FRACTION  # same normalization and top-1% rule as ED4c

sys.path.insert(0, str(paths.GAME))
import game  # noqa: E402  (pure numpy board logic from sokoban/lib)

ANATOMY = paths.ROOT / 'sokoban' / 'artifacts' / 'anatomy'
MORPHOLOGY = paths.ROOT / 'sokoban' / 'artifacts' / 'morphology'
SKELETON_CACHE = paths.ROOT / 'flybrain' / 'data' / 'morphology' / 'skeletons'
SKELETON_DOWNLOADS = paths.ROOT / 'manuscript' / 'figures' / 'assets' / 'skeletons'
SKELETON_URL = 'https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-precomputed/{}'
GROUP_NAMES = paths.ARCHIVE / 'experiments' / 'campaign' / 'run' / 'neuron-mapping.npz'
FD = 'figure_data/'

ACTION_NAMES = ('up', 'down', 'left', 'right')
# Distinct state sets (P6): the shared-readout hidden-goal set duplicates the M0 readout-only set and is left out.
STATE_SETS = [  # model, context, visible, column key, label
    ('remote-dynamic-M0-head_only', 0, 0, 'm0', 'M0 readout-only (hidden)'),
    ('remote-dynamic-M0-joint', 0, 0, 'm0_joint', 'M0 joint (hidden)'),
    ('supp-M0-visibility', 0, 1, 'm0vis_visible', 'M0 shared readout (visible)'),
    ('supp-S1-ring-brain', 0, 0, 's1', 'S1 readout-only (hidden)'),
    ('supp-S2-context-brain', 0, 0, 's2_o0', 'S2 readout-only, orientation 0 (hidden)'),
    ('supp-S2-context-brain', 1, 0, 's2_o180', 'S2 readout-only, orientation 180 (hidden)'),
]
SPIKE_KEYS = {'remote-dynamic-M0-head_only': 'm0', 'remote-dynamic-M0-joint': 'm0_joint', 'supp-M0-visibility': 'm0vis',
              'supp-S1-ring-brain': 's1', 'supp-S2-context-brain': 's2'}
VARIABLE_UNITS = {'v': 'mV', 'g': 'mV', 'adapt': 'mV', 'res': 'fraction'}
THRESHOLD_EXPONENTS = range(-60, 21)       # thresholds 10**(k/10): 1e-6 to 100 in tenth-decade steps


def _src(*ps):
    return ';'.join(paths.rel(p) for p in ps)


# ---------------------------------------------------------------- anatomy

def anatomy():
    meta = io.json_load(ANATOMY / 'metadata.json')
    raw = io.read_bytes(ANATOMY / 'regions.bin')
    xyz = np.frombuffer(io.read_bytes(ANATOMY / 'somata.bin'), '<f4').reshape(-1, 3).astype(np.float64)
    ids = np.frombuffer(io.read_bytes(ANATOMY / 'soma-body-ids.bin'), '<u4').astype(np.int64)
    if len(ids) != meta['soma_count'] or len(xyz) != len(ids) or len(np.unique(ids)) != len(ids):
        raise RuntimeError('soma table inconsistent with anatomy metadata')
    regions = []
    for r in meta['regions']:
        o = r['byte_offset']
        nv, nf = (int(v) for v in np.frombuffer(raw, '<u4', 2, o))
        if (nv, nf) != (r['vertices'], r['faces']) or 8 + 12 * (nv + nf) != r['byte_length']:
            raise RuntimeError(f'region {r["id"]}: binary block does not match metadata')
        regions.append((r, np.frombuffer(raw, '<f4', nv * 3, o + 8).reshape(-1, 3).astype(np.float64)))
    return meta, regions, xyz, ids


def fd_display_transform(regions, xyz):
    brain = np.vstack([v for r, v in regions if r['dataset'] == 'brain'])
    vnc = np.vstack([v for r, v in regions if r['dataset'] == 'vnc'])
    posterior = float(brain[:, 2].max())
    neck = vnc[(vnc[:, 2] > cnsview.NECK_Z) & (vnc[:, 2] <= cnsview.NECK_Z + 20)]
    centre_y = float((neck[:, 1].min() + neck[:, 1].max()) / 2)
    if abs(posterior - cnsview.NECK_Z) > 0.5 or abs(centre_y - cnsview.NECK_Y) > 1.0:
        raise RuntimeError(f'display transform constants drifted: posterior {posterior}, connective centre {centre_y}')
    u, side = cnsview.unfold(xyz)
    rows = [('neck_z_um', cnsview.NECK_Z, 'um', 'Hinge axis z (constant in msa/cnsview.py): points with z > neck_z are rotated'),
            ('neck_y_um', cnsview.NECK_Y, 'um', 'Hinge axis y (constant in msa/cnsview.py)'),
            ('brain_mesh_posterior_limit_um', posterior, 'um', 'Maximum z over brain neuropil mesh vertices (measured)'),
            ('vnc_mesh_anterior_limit_um', float(vnc[:, 2].min()), 'um', 'Minimum z over VNC neuropil mesh vertices (measured)'),
            ('connective_centre_y_um', centre_y, 'um', 'Mid-range y of VNC mesh vertices with neck_z < z <= neck_z + 20 (measured)'),
            ('somata_brain_side', float((~side).sum()), 'cells', 'Somata kept in frontal view'),
            ('somata_vnc_side', float(side.sum()), 'cells', 'Somata rotated into ventral view'),
            ('display_x_min_um', float(u[:, 0].min()), 'um', 'Soma extent in display coordinates'),
            ('display_x_max_um', float(u[:, 0].max()), 'um', 'Soma extent in display coordinates'),
            ('display_y_min_um', float(u[:, 1].min()), 'um', 'Soma extent in display coordinates'),
            ('display_y_max_um', float(u[:, 1].max()), 'um', 'Soma extent in display coordinates')]
    df = pd.DataFrame(rows, columns=['parameter', 'value', 'unit', 'description'])
    df['source_file'] = _src(ANATOMY / 'regions.bin', ANATOMY / 'somata.bin')
    csvout.write_table(df, FD + 'fd_display_transform.csv', title='Unfolded-CNS display transform (brain frontal, VNC ventral below the neck)', describe={
        'parameter': 'Parameter', 'value': 'Value', 'unit': 'Unit', 'description': 'Definition'}, key=['parameter'], sort=[],
        notes='X = x; brain side Y = y, depth = z; VNC side (z > neck_z) Y = neck_y + (z - neck_z), depth = neck_z - (y - neck_y).')


def fd_cells(xyz, ids):
    ann = io.parquet_load(paths.M1 / 'neurons' / 'annotations.parquet',
                          columns=['bodyId', 'type', 'superclass', 'class', 'somaSide', 'nt', 'in_syn', 'parameter_group',
                                   'direct_input', 'observed_by_policy'])
    idx = io.npz_load(paths.M1 / 'neurons' / 'index.npz', keys=['body_ids', 'populations', 'inputs', 'observed'])
    body = ann['bodyId'].to_numpy(np.int64)
    if not np.array_equal(body, idx['body_ids']) or not np.array_equal(ann['direct_input'].to_numpy(bool), idx['inputs']) \
            or not np.array_equal(ann['observed_by_policy'].to_numpy(bool), idx['observed']):
        raise RuntimeError('annotations and neuron index disagree')
    n = len(ann)
    pop = np.full(n, -1, np.int64)
    for p, members in enumerate(idx['populations']):
        pop[members] = p
    names = io.npz_load(GROUP_NAMES, keys=['group_names'])['group_names']
    level = np.array([s.split(':', 1)[0] for s in names])
    group = ann['parameter_group'].to_numpy(np.int64)
    soma_index = pd.Series(np.arange(len(ids)), index=ids)
    hit = soma_index.reindex(body).to_numpy()
    has = ~np.isnan(hit)
    loc = np.full((n, 3), np.nan)
    loc[has] = xyz[hit[has].astype(np.int64)]
    unf = np.full((n, 3), np.nan)
    u, side = cnsview.unfold(loc[has])
    unf[has] = u
    part = np.full(n, None, dtype=object)
    part[has] = np.where(side, 'vnc', 'brain')
    morph = io.json_load(MORPHOLOGY / 'metadata.json')
    arbor = pd.Series(np.arange(len(morph['body_ids'])), index=np.asarray(morph['body_ids'], np.int64)).reindex(body)
    role = np.where(ann['direct_input'].to_numpy(bool), 'input', np.where(ann['observed_by_policy'].to_numpy(bool), 'readout', 'other'))
    df = pd.DataFrame({
        'cell': np.arange(n), 'body_id': body, 'role': role,
        'readout_population': pd.array(np.where(pop >= 0, pop, None), dtype='Int64'),
        'superclass': ann['superclass'].fillna('unannotated').astype(object).to_numpy(),
        'cell_class': ann['class'].astype(object).to_numpy(), 'type': ann['type'].astype(object).to_numpy(),
        'transmitter': ann['nt'].fillna('unknown').astype(object).to_numpy(), 'soma_side': ann['somaSide'].astype(object).to_numpy(),
        'input_synapses': pd.array(ann['in_syn'].to_numpy(), dtype='Int64'), 'parameter_group': group, 'group_level': level[group],
        'soma_x_um': np.round(loc[:, 0], 3), 'soma_y_um': np.round(loc[:, 1], 3), 'soma_z_um': np.round(loc[:, 2], 3), 'cns_part': part,
        'unfold_x_um': np.round(unf[:, 0], 3), 'unfold_y_um': np.round(unf[:, 1], 3), 'unfold_depth_um': np.round(unf[:, 2], 3),
        'arbor_index': pd.array(arbor.to_numpy(), dtype='Int64'),
    })
    # One row per model cell; the sources are named in the title instead of a 167,106-times repeated column.
    csvout.write_table(df, FD + 'fd_cells.csv', title=('Every model cell: role, annotation, soma position and unfolded display position (sources: '
                                                      + _src(paths.M1 / 'neurons' / 'annotations.parquet', paths.M1 / 'neurons' / 'index.npz',
                                                             ANATOMY / 'somata.bin', ANATOMY / 'soma-body-ids.bin', MORPHOLOGY / 'metadata.json') + ')'),
                       describe={
        'cell': 'Model cell index (row of the connectome matrix)', 'body_id': 'MaleCNS v1.0 body ID',
        'role': 'input (3,200 visual input cells), readout (4,096 cells read by the policy) or other',
        'readout_population': 'Readout population (0-255, 16 cells each) for readout cells',
        'superclass': 'MaleCNS superclass (unannotated when missing)', 'cell_class': 'MaleCNS class', 'type': 'MaleCNS cell type',
        'transmitter': 'Predicted transmitter (unknown when missing)', 'soma_side': 'Soma side annotation',
        'input_synapses': 'Input synapse count (annotation in_syn)', 'parameter_group': 'Shared parameter group (0-11,769)',
        'group_level': 'type (one cell type) or superclass (cells without a usable type)',
        'soma_x_um': 'Soma x (um; +x = fly left)', 'soma_y_um': 'Soma y (um; +y = ventral)', 'soma_z_um': 'Soma z (um; +z = posterior)',
        'cns_part': 'brain or vnc side of the display hinge (see fd_display_transform)',
        'unfold_x_um': 'Display X (um)', 'unfold_y_um': 'Display Y (um, downwards)', 'unfold_depth_um': 'Display depth (um, away from the viewer)',
        'arbor_index': 'Index into the cached arbor sample (sokoban/artifacts/morphology) when available'}, key=['cell'])
    return df


def fd_neuropils(meta):
    groups = {g['id']: g for g in meta['groups']}
    rows = []
    for r in meta['regions']:
        (x0, y0, z0), (x1, y1, z1) = r['bounds_um']
        rows.append(dict(region_id=r['id'], label=r['label'], dataset=r['dataset'], region_group=r['group'],
                         region_group_label=groups[r['group']]['label'], group_colour=groups[r['group']]['color'],
                         centre_x_um=r['center_um'][0], centre_y_um=r['center_um'][1], centre_z_um=r['center_um'][2],
                         min_x_um=x0, min_y_um=y0, min_z_um=z0, max_x_um=x1, max_y_um=y1, max_z_um=z1, vertices=r['vertices'], faces=r['faces'],
                         byte_offset=r['byte_offset'], byte_length=r['byte_length']))
    df = pd.DataFrame(rows)
    df['source_file'] = _src(ANATOMY / 'metadata.json', ANATOMY / 'regions.bin')
    csvout.write_table(df, FD + 'fd_neuropils.csv', title='Neuropil surface index (114 MaleCNS regions, display meshes in um)', describe={
        'region_id': 'Region identifier', 'label': 'Neuropil label', 'dataset': 'brain or vnc ROI set', 'region_group': 'Anatomical group',
        'region_group_label': 'Anatomical group label', 'group_colour': 'Group colour used by the interactive viewer',
        'centre_x_um': 'Centre x (um)', 'centre_y_um': 'Centre y (um)', 'centre_z_um': 'Centre z (um)', 'min_x_um': 'Bounds min x (um)',
        'min_y_um': 'Bounds min y (um)', 'min_z_um': 'Bounds min z (um)', 'max_x_um': 'Bounds max x (um)', 'max_y_um': 'Bounds max y (um)',
        'max_z_um': 'Bounds max z (um)', 'vertices': 'Display mesh vertices', 'faces': 'Display mesh triangles',
        'byte_offset': 'Block offset in regions.bin (uint32 nv, uint32 nf, float32 vertices, uint32 faces)', 'byte_length': 'Block length (bytes)'},
        key=['region_id'], sort=[])


# ---------------------------------------------------------------- boards

def fd_boards():
    cells, steps = [], []

    def add(board_id, kind, family, level, src, solve):
        walls, goals = set(level['walls']), set(level['goals'])
        player, boxes = int(level['start'][0]), set(level['start'][1])
        for c in range(100):
            cells.append(dict(board_id=board_id, board_kind=kind, family=family, cell=c, row=c // 10, col=c % 10, wall=c in walls,
                              goal=c in goals, box_start=c in boxes, player_start=c == player, source_file=src))
        acts = level.get('reference_actions') if solve else None
        if not acts:
            return
        g = game.restore(level)
        s = game.state_from(level['start'])
        for i, a in enumerate(acts, 1):
            push = s[0] + game.DELTAS[a] in s[1]
            s = game.step(g, s, a)
            if s is None:
                raise RuntimeError(f'{board_id}: reference action {i} is illegal')
            steps.append(dict(board_id=board_id, step=i, action=int(a), action_name=ACTION_NAMES[a], push=bool(push), player_cell=int(s[0]),
                              boxes=[int(b) for b in s[1]], solved=bool(game.solved(g, s)), source_file=src))
        if not game.solved(g, s):
            raise RuntimeError(f'{board_id}: reference solution does not solve')

    lp = paths.TASKS / 'levels.json'
    for level in io.json_load(lp):
        add(level['id'], 'static_map', f"tier{level['tier']}", level, paths.rel(lp), True)
    mp = paths.TASKS / 'memory.json'
    for task in io.json_load(mp):
        if task['id'] == 'M0':                     # M1/M2 were never trained or evaluated
            for level in task['levels']:
                add(f"M0:c0-cue{int(level['cue'])}", 'memory_case', 'M0', level, paths.rel(mp), True)
    rp = paths.RS_ROOT / 'tasks.json'
    fam = {'S1-ring': 'S1', 'S2-context': 'S2', 'M0-visibility': 'M0vis'}
    for task in io.json_load(rp):
        if task['id'] not in fam:
            continue
        for case in task['cases']:
            add(f"{fam[task['id']]}:{case['id']}", 'memory_case', fam[task['id']], case['level'], paths.rel(rp), True)
            add(f"{fam[task['id']]}:{case['id']}:cue_frame", 'memory_cue_frame', fam[task['id']], case['cue_level'], paths.rel(rp), False)
    csvout.write_table(pd.DataFrame(cells), FD + 'fd_boards.csv', title='Board layouts on the common 10 x 10 canvas (static maps and memory cases)', describe={
        'board_id': 'Board (map id, or family:case for memory tasks; :cue_frame = board shown during the cue)',
        'board_kind': 'static_map, memory_case or memory_cue_frame', 'family': 'Tier (static maps) or memory family', 'cell': 'Cell index (row * 10 + col)',
        'row': 'Row (0 = top)', 'col': 'Column (0 = left)', 'wall': 'Wall cell', 'goal': 'Goal cell', 'box_start': 'Box at the start',
        'player_start': 'Player at the start'}, key=['board_id', 'cell'])
    csvout.write_table(pd.DataFrame(steps), FD + 'fd_solutions.csv', title='Reference solutions replayed with the game rules (certified optimal for static maps)', describe={
        'board_id': 'Board', 'step': 'Move number', 'action': 'Action code (0 up, 1 down, 2 left, 3 right)', 'action_name': 'Action',
        'push': 'Move pushes a box', 'player_cell': 'Player cell after the move', 'boxes': 'Box cells after the move (JSON)',
        'solved': 'All boxes on goals after the move'}, key=['board_id', 'step'])


# ---------------------------------------------------------------- per-cell memory state

def _cellsets(df):
    inp = df['direct_input'].astype(bool).to_numpy()
    obs = df['observed_by_policy'].astype(bool).to_numpy()
    return {'all': np.ones(len(df), bool), 'input': inp, 'readout': obs, 'other': ~inp & ~obs}


def fd_soma_state(cells):
    frames = {}
    out = pd.DataFrame({'cell': cells['cell'].to_numpy(), 'body_id': cells['body_id'].to_numpy()})
    describe = {}
    for model, ctx, vis, key, label in STATE_SETS:
        df = frames.get(model)
        if df is None:
            df = frames[model] = io.parquet_load(paths.M1 / 'interpretation' / f'{model}-memory-cells-d160-p4.parquet')
            if not np.array_equal(df['bodyId'].to_numpy(np.int64), out['body_id'].to_numpy()):
                raise RuntimeError(f'{model}: per-cell table not in model-cell order')
        out[f'dv_{key}_mv'] = np.round(df[f'c{ctx}-visible{vis}-delta-v'].to_numpy(np.float64), 6)
        describe[f'dv_{key}_mv'] = f'{label}: voltage, cue 1 minus cue 0 (mV, last window at 160 ms delay and 4 prefix windows; rounded to 1e-6)'
    for model, key in SPIKE_KEYS.items():
        out[f'spikes_{key}'] = np.round(frames[model]['spikes_mean'].to_numpy(np.float64), 6)
        describe[f'spikes_{key}'] = f'{model}: spike probability per ms in the same window, mean over cases (0 = no spike)'
    csvout.write_table(out, FD + 'fd_soma_state.csv', title=('Per-cell cue-related voltage difference and spiking in the six distinct state sets (source: '
                                                            'studies/brain-sokoban-v1/mechanisms-v1/interpretation/*-memory-cells-d160-p4.parquet)'),
                       describe={'cell': 'Model cell index', 'body_id': 'MaleCNS body ID', **describe}, key=['cell'])
    return frames


def fd_coverage_graded(frames):
    thresholds = np.array([10.0 ** (k / 10) for k in THRESHOLD_EXPONENTS])
    rows = []
    for model, ctx, vis, key, label in STATE_SETS:
        df = frames[model]
        for var, unit in VARIABLE_UNITS.items():
            d = np.abs(df[f'c{ctx}-visible{vis}-delta-{var}'].to_numpy(np.float64))
            for pop, mask in _cellsets(df).items():
                x = np.sort(d[mask])
                above = len(x) - np.searchsorted(x, thresholds, side='right')
                for t, a in zip(thresholds, above):
                    rows.append(dict(state_set=label, state_key=key, model=model, context=ctx, visible=bool(vis), variable=var, unit=unit,
                                     population=pop, threshold=float(t), cells=int(len(x)), cells_above=int(a), fraction_above=float(a / len(x))))
    df = pd.DataFrame(rows)
    # Gate: the 1e-6 rung reproduces Fig. 6b (P6) for every state set, variable and population.
    f6 = pd.read_csv(paths.out('main/fig6/fig6b_cue_difference_coverage.csv'), low_memory=False)
    f6 = f6[f6['duplicate_of'].isna() & (f6['variable'] != 'all')].replace({'population': {'observed': 'readout'}})
    chk = df[df['threshold'] == 1e-6].merge(f6, on=['model', 'context', 'visible', 'variable', 'population'], how='inner')
    if len(chk) != len(STATE_SETS) * 16 or (chk['cells_above'] != chk['cells_above_threshold']).any():
        raise RuntimeError('fd_coverage_graded: 1e-6 rung disagrees with fig6b')
    csvout.write_table(df, FD + 'fd_coverage_graded.csv', title='Graded coverage: share of cells whose cue-related state difference exceeds each threshold', describe={
        'state_set': 'Distinct state set', 'state_key': 'Column key used in fd_soma_state', 'variable': 'v voltage, g synaptic current, adapt adaptation, res synaptic resource',
        'unit': 'Unit of the variable and threshold', 'population': 'all, input, readout or other cells', 'threshold': 'Threshold on |cue 1 - cue 0| (10^(k/10))',
        'cells': 'Cells in the population', 'cells_above': 'Cells with |difference| > threshold', 'fraction_above': 'cells_above / cells'},
        key=['state_key', 'variable', 'population', 'threshold'])
    return df


# ---------------------------------------------------------------- parameter change by group

def fd_group_change(cells):
    st4 = pd.read_csv(paths.out('supplementary/st4_parameter_change.csv'), low_memory=False)
    maps = sorted(r for r in st4['run_id'].unique() if r.startswith('difficulty-'))
    if len(maps) != 24:
        raise RuntimeError(f'expected 24 difficulty maps in ST4, found {len(maps)}')
    d = st4[st4['run_id'].isin(maps)].sort_values(['run_id', 'group'], kind='mergesort')
    n = int(d['group'].nunique())
    comb_check = sum(d[f'delta_{p}'].abs() / NORM[p] for p in NORM)
    if not np.allclose(comb_check.to_numpy(), d['normalized_abs_change_sum'].to_numpy(), rtol=0, atol=1e-9):
        raise RuntimeError('ST4 normalized change does not match the P3 normalization')
    comb = d['normalized_abs_change_sum'].to_numpy().reshape(len(maps), n)
    score = (rankdata(comb, axis=1) / n).mean(0)          # P3 / ED4c: mean within-map percentile of combined change
    k = int(math.ceil(TOP_FRACTION * n))
    top = np.zeros(n, bool)
    top[np.argsort(-score, kind='mergesort')[:k]] = True
    first = d[d['run_id'] == maps[0]].reset_index(drop=True)
    out = first[['group', 'group_name', 'n_cells', 'membership', 'superclass', 'transmitter']].copy()
    out['group_level'] = out['group_name'].str.split(':').str[0]
    out['consistency_score'] = score
    out['consistency_rank'] = rankdata(-score, method='ordinal').astype(np.int64)
    out['top_1pct'] = top
    for p in NORM:
        x = d[f'delta_{p}'].to_numpy().reshape(len(maps), n)
        out[f'mean_delta_{p}'] = x.mean(0)
        out[f'maps_increased_{p}'] = (x > 0).sum(0)
        out[f'maps_decreased_{p}'] = (x < 0).sum(0)
        out[f'mean_normalized_abs_{p}'] = (np.abs(x) / NORM[p]).mean(0)
    c = cells.dropna(subset=['unfold_x_um'])
    g = c.groupby('parameter_group').agg(cells_with_soma=('cell', 'size'), soma_x_um=('soma_x_um', 'mean'), soma_y_um=('soma_y_um', 'mean'),
                                         soma_z_um=('soma_z_um', 'mean'), unfold_x_um=('unfold_x_um', 'mean'), unfold_y_um=('unfold_y_um', 'mean'))
    out = out.merge(g.round(3), left_on='group', right_index=True, how='left')
    out['cells_with_soma'] = out['cells_with_soma'].fillna(0).astype(np.int64)
    out['n_maps'] = len(maps)
    out['source_file'] = 'manuscript/results/supplementary/st4_parameter_change.csv;manuscript/results/figure_data/fd_cells.csv'
    describe = {'group': 'Shared parameter group', 'group_name': 'Group name (type:<type> or superclass:<superclass>)', 'n_cells': 'Cells in the group',
                'membership': 'input, observed, input_and_observed or neither', 'superclass': 'Majority superclass', 'transmitter': 'Majority transmitter',
                'group_level': 'type or superclass', 'consistency_score': 'Mean within-map percentile of combined normalized |change| over 24 maps (ED4c score)',
                'consistency_rank': 'Rank by consistency score (1 = most consistently changed)', 'top_1pct': f'Among the top {k} groups (1%)',
                'cells_with_soma': 'Member cells with a soma position', 'soma_x_um': 'Mean member soma x (um)', 'soma_y_um': 'Mean member soma y (um)',
                'soma_z_um': 'Mean member soma z (um)', 'unfold_x_um': 'Mean member display X (um)', 'unfold_y_um': 'Mean member display Y (um)', 'n_maps': 'Maps'}
    for p in NORM:
        describe[f'mean_delta_{p}'] = f'Mean final minus initial {p} over maps'
        describe[f'maps_increased_{p}'] = f'Maps in which {p} increased'
        describe[f'maps_decreased_{p}'] = f'Maps in which {p} decreased'
        describe[f'mean_normalized_abs_{p}'] = f'Mean |change of {p}| / allowed range over maps'
    csvout.write_table(out, FD + 'fd_group_change.csv', title='Parameter change per shared group across the 24 maps, with member soma centroids',
                       describe=describe, key=['group'])
    return out


# ---------------------------------------------------------------- time generalisation

def fd_time_generalisation():
    rows = []
    for model in sorted({s[0] for s in STATE_SETS}):
        p = paths.M1 / 'tables' / f'{model}-time-generalization.npz'
        z = io.npz_load(p)
        acc = z['accuracy']
        for i in range(acc.shape[0]):
            for j in range(acc.shape[1]):
                rows.append(dict(model=model, train_ms=i, test_ms=j, accuracy=float(acc[i, j]), n_train_histories=int(z['samples_train']),
                                 n_test_histories=int(z['samples_held']), source_file=paths.rel(p)))
    csvout.write_table(pd.DataFrame(rows), FD + 'fd_time_generalisation.csv', title='Cue decoding across time within the decision window (train time x test time)', describe={
        'train_ms': 'Decoder training time (ms from the window start)', 'test_ms': 'Decoder test time (ms from the window start)',
        'accuracy': 'Cue decoding accuracy on held-out histories', 'n_train_histories': 'Training histories', 'n_test_histories': 'Held-out histories'},
        key=['model', 'train_ms', 'test_ms'])


# ---------------------------------------------------------------- neuron selection and skeletons

def _cached_ids():
    return {int(p.name) for p in SKELETON_CACHE.iterdir() if p.is_file() and p.name.isdigit()}


def fd_neuron_selection(cells, frames, groups):
    cached = _cached_ids()
    c = cells.set_index('body_id', drop=False)
    rows = []

    def add(set_name, rank, label, body_id, score, reason):
        r = c.loc[int(body_id)]
        rows.append(dict(selection_set=set_name, rank=rank, label=label, body_id=int(body_id), cell=int(r['cell']), type=r['type'],
                         superclass=r['superclass'], role=r['role'], has_soma=not pd.isna(r['soma_x_um']), cached=int(body_id) in cached,
                         download=int(body_id) not in cached, score=score, reason=reason))

    def pick(members, count):
        m = cells.loc[members].assign(has_soma=lambda x: x['soma_x_um'].notna(), is_cached=lambda x: x['body_id'].isin(cached),
                                      syn=lambda x: x['input_synapses'].fillna(-1).astype(np.int64))
        m = m.sort_values(['has_soma', 'is_cached', 'syn', 'body_id'], ascending=[False, False, False, True], kind='mergesort')
        return m['body_id'].tolist()[:count]

    for line in ('learning', 'memory'):
        gp = paths.M1 / 'runs' / f'discover-{line}' / 'groups.json'
        for g in io.json_load(gp)['groups']:
            for i, b in enumerate(g['body_ids']):
                add('A_candidate', i, g['id'], b, float(g['score']), f'{line} candidate group ({",".join(g["type_labels"])})')
            for i, b in enumerate(g['control_body_ids']):
                add('A_matched', i, g['id'], b, float(g['match_mean_distance']), f'matched control for {g["id"]}')
    inp = cells[cells['role'] == 'input']
    counts = inp.groupby('type').size().reset_index(name='n').sort_values(['n', 'type'], ascending=[False, True], kind='mergesort')
    for rank, (t, nt) in enumerate(counts.head(12).itertuples(index=False)):
        for b in pick(inp.index[inp['type'] == t], 1):
            add('B_input_type', rank, t, b, float(nt), 'one cell of each of the 12 input types with the most input cells')
    dv = np.abs(frames['remote-dynamic-M0-head_only']['c0-visible0-delta-v'].to_numpy(np.float64))
    typed = cells.assign(dv=dv)[cells['type'].notna()]
    tm = typed.groupby('type').agg(n=('dv', 'size'), mean_dv=('dv', 'mean')).reset_index()
    tm = tm[tm['n'] >= 2].sort_values(['mean_dv', 'type'], ascending=[False, True], kind='mergesort').head(8)
    for rank, r in enumerate(tm.itertuples(index=False)):
        m = typed[typed['type'] == r.type].sort_values(['dv', 'body_id'], ascending=[False, True], kind='mergesort')
        for b in m['body_id'].head(2):
            add('C_cue_type', rank, r.type, b, float(r.mean_dv), 'two most cue-responsive cells (M0 |dv|) of the 8 types with the largest mean |dv|')
    tg = groups[groups['group_level'] == 'type'].sort_values('consistency_rank', kind='mergesort').head(8)
    for rank, r in enumerate(tg.itertuples(index=False)):
        for b in pick(cells.index[cells['parameter_group'] == r.group], 2):
            add('D_change_group', rank, r.group_name, b, float(r.consistency_score), 'two cells of the 8 most consistently changed type-level groups')
    ro = cells[cells['role'] == 'readout']
    sc = ro.groupby('superclass').size().reset_index(name='n').sort_values(['n', 'superclass'], ascending=[False, True], kind='mergesort')
    for rank, (s, ns) in enumerate(sc.itertuples(index=False)):
        for b in pick(ro.index[ro['superclass'] == s], 2):
            add('E_readout_exemplar', rank, s, b, float(ns), 'two readout cells per readout superclass (largest input synapse count)')
    df = pd.DataFrame(rows)
    df['source_file'] = _src(paths.M1 / 'runs' / 'discover-learning' / 'groups.json', paths.M1 / 'runs' / 'discover-memory' / 'groups.json')
    csvout.write_table(df, FD + 'fd_neuron_selection.csv', title='Neurons selected for skeleton tiles (selection fixed before any download)', describe={
        'selection_set': 'A_candidate, A_matched, B_input_type, C_cue_type, D_change_group or E_readout_exemplar', 'rank': 'Rank within the set',
        'label': 'Group, type or superclass that defines the selection', 'body_id': 'MaleCNS body ID', 'cell': 'Model cell index',
        'type': 'Cell type', 'superclass': 'Superclass', 'role': 'input, readout or other', 'has_soma': 'Soma position available',
        'cached': 'Skeleton already in flybrain/data/morphology/skeletons', 'download': 'Skeleton to download from the MaleCNS v1.0 release',
        'score': 'Selection score (group score, match distance, input cells, mean |dv| or readout cells)', 'reason': 'Selection rule'},
        key=['selection_set', 'label', 'body_id'], sort=['selection_set', 'rank', 'label', 'body_id'])
    return df


def _skeleton_stats(path, soma):
    data = io.read_bytes(path)
    nv, ne = (int(v) for v in np.frombuffer(data, '<u4', 2))
    if len(data) != 8 + 12 * nv + 8 * ne:
        raise RuntimeError(f'{path.name}: size does not match the precomputed skeleton layout')
    v = np.frombuffer(data, '<f4', nv * 3, 8).reshape(-1, 3).astype(np.float64) / 1000.0
    e = np.frombuffer(data, '<u4', ne * 2, 8 + 12 * nv).reshape(-1, 2)
    if ne and int(e.max()) >= nv:
        raise RuntimeError(f'{path.name}: edge index out of range')
    cable = float(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).sum()) if ne else 0.0
    dist = float(np.linalg.norm(v - soma, axis=1).min()) if soma is not None and nv else None
    return dict(n_vertices=nv, n_edges=ne, cable_um=cable, min_x_um=float(v[:, 0].min()), min_y_um=float(v[:, 1].min()),
                min_z_um=float(v[:, 2].min()), max_x_um=float(v[:, 0].max()), max_y_um=float(v[:, 1].max()), max_z_um=float(v[:, 2].max()),
                soma_distance_um=dist, file_bytes=len(data))


def fd_skeletons(cells, selection):
    import hashlib
    c = cells.set_index('body_id')
    downloaded = {}
    if SKELETON_DOWNLOADS.exists():
        receipt_p = SKELETON_DOWNLOADS / 'receipt.csv'
        receipt = io.csv_load(receipt_p) if receipt_p.exists() else pd.DataFrame(columns=['body_id', 'sha256'])
        sha = dict(zip(receipt['body_id'].astype(np.int64), receipt['sha256']))
        for p in sorted(SKELETON_DOWNLOADS.iterdir()):
            if p.is_file() and re.fullmatch(r'\d+', p.name):
                downloaded[int(p.name)] = (p, sha.get(int(p.name)))
    rows = []
    for b in sorted(set(selection['body_id'])):
        r = c.loc[b]
        soma = None if pd.isna(r['soma_x_um']) else np.array([r['soma_x_um'], r['soma_y_um'], r['soma_z_um']], float)
        cache_p = SKELETON_CACHE / str(b)
        if cache_p.is_file():
            src, p, expected = 'cache', cache_p, None
        elif b in downloaded:
            src, (p, expected) = 'download', downloaded[b]
        else:
            rows.append(dict(body_id=b, status='not_downloaded', url=SKELETON_URL.format(b), source_file=None))
            continue
        st = _skeleton_stats(p, soma)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if expected is not None and expected != digest:
            raise RuntimeError(f'skeleton {b}: SHA-256 differs from the download receipt')
        rows.append(dict(body_id=b, status=src, url=SKELETON_URL.format(b), **st, sha256=digest, source_file=paths.rel(p)))
    df = pd.DataFrame(rows)
    for col in ('n_vertices', 'n_edges', 'file_bytes'):
        df[col] = df[col].astype('Int64') if col in df else pd.array([None] * len(df), dtype='Int64')
    for col in ('cable_um', 'min_x_um', 'min_y_um', 'min_z_um', 'max_x_um', 'max_y_um', 'max_z_um', 'soma_distance_um', 'sha256'):
        if col not in df:
            df[col] = None
    df = df[['body_id', 'status', 'url', 'n_vertices', 'n_edges', 'cable_um', 'min_x_um', 'min_y_um', 'min_z_um', 'max_x_um', 'max_y_um',
             'max_z_um', 'soma_distance_um', 'file_bytes', 'sha256', 'source_file']]
    bad = df[(df['status'] != 'not_downloaded') & (df['soma_distance_um'] > 5.0)]
    if len(bad):
        raise RuntimeError(f'skeletons far from their soma: {bad.body_id.tolist()[:5]}')
    csvout.write_table(df, FD + 'fd_skeletons.csv', title='Skeletons of the selected neurons (MaleCNS v1.0, CC BY 4.0)', describe={
        'body_id': 'MaleCNS body ID', 'status': 'cache, download or not_downloaded', 'url': 'Release URL of the precomputed skeleton',
        'n_vertices': 'Skeleton vertices', 'n_edges': 'Skeleton edges', 'cable_um': 'Total cable length (um)', 'min_x_um': 'Bounds min x (um)',
        'min_y_um': 'Bounds min y (um)', 'min_z_um': 'Bounds min z (um)', 'max_x_um': 'Bounds max x (um)', 'max_y_um': 'Bounds max y (um)',
        'max_z_um': 'Bounds max z (um)', 'soma_distance_um': 'Distance from the soma to the nearest skeleton vertex (um)',
        'file_bytes': 'File size (bytes)', 'sha256': 'SHA-256 of the file'}, key=['body_id'])
    return df


# ---------------------------------------------------------------- panel index

M, E, S, F = 'main/', 'extended/', 'supplementary/', FD
PANELS = [  # figure, panel, content, tables, test-id prefixes
    ('Fig1', 'a', 'Whole-CNS anchor render; input, readout and exemplar cells',
     [F + 'fd_cells.csv', F + 'fd_neuropils.csv', F + 'fd_display_transform.csv', F + 'fd_skeletons.csv', F + 'fd_neuron_selection.csv'], []),
    ('Fig1', 'b', 'Model schematic with trained terms', [M + 'fig1/fig1a_model_specification.csv', M + 'fig1/fig1b_parameter_budget.csv'], []),
    ('Fig1', 'c', 'Parameter budget against fixed connectome entries', [M + 'fig1/fig1b_parameter_budget.csv', M + 'fig1/fig1a_model_specification.csv'], []),
    ('Fig1', 'd', '34 boards by tier with certified optimum', [M + 'fig1/fig1d_task_library.csv', F + 'fd_boards.csv', F + 'fd_solutions.csv'], []),
    ('Fig1', 'e', 'D3-T0 curriculum staircase', [M + 'fig1/fig1c_curriculum_example.csv'], []),
    ('Fig1', 'f', 'Memory task timeline and timing grid', [M + 'fig1/fig1e_memory_task_design.csv', F + 'fd_boards.csv'], []),
    ('Fig2', 'a', 'Learning curves for 24 maps', [M + 'fig2/fig2b_learning_curves.csv', M + 'fig2/fig2a_acquisition_outcomes.csv'], []),
    ('Fig2', 'b', 'Updates and neuron-ms vs certified optimum (log-log)', [M + 'fig2/fig2c_cost_vs_difficulty.csv'],
     ['fig2c.spearman', 'fig2c.theilsen', 'fig2c.jt', 'fig2c.n22']),
    ('Fig2', 'c', 'Outcome card and zero-update transfer', [M + 'fig2/fig2a_acquisition_outcomes.csv', M + 'fig2/fig2e_zero_update_transfer.csv'], []),
    ('Fig2', 'd', 'Stage barcodes (push and deadlock structure)', [M + 'fig2/fig2d_stage_costs.csv'], []),
    ('Fig2', 'e', 'Per-map push excess share and deadlock correlation', [M + 'fig2/fig2d_stage_costs.csv'], ['fig2d.wilcoxon', 'fig2d.sign']),
    ('Fig3', 'a', 'Brain x readout swap matrix', [M + 'fig3/fig3c_four_combination.csv'], ['fig3c.mcnemar']),
    ('Fig3', 'b', 'Margin decomposition per map', [M + 'fig3/fig3d_margin_decomposition.csv'], ['fig3d.wilcoxon']),
    ('Fig3', 'c', 'Training-control matrix', [M + 'fig3/fig3a_training_controls.csv', M + 'fig3/fig3b_control_costs.csv'], []),
    ('Fig3', 'd', 'Readout principal angles', [M + 'fig3/fig3e_readout_geometry.csv'], []),
    ('Fig3', 'e', 'Map x map consistency of parameter change vs stratified null',
     [S + 'st4_parameter_change.csv', E + 'ed04/ed04a_parameter_change_consistency.csv'], ['ed04a.mean_pairwise_spearman']),
    ('Fig3', 'f', 'Parameter change by synaptic distance from the inputs and to the readout', [E + 'ed04/ed04b_parameter_change_anatomy.csv'], []),
    ('Fig3', 'g', 'Whole-CNS map of the consistency of parameter change', [F + 'fd_group_change.csv', F + 'fd_cells.csv', F + 'fd_display_transform.csv'],
     []),
    ('Fig3', 'h', 'Neurons of the eight most consistently changed types, with per-map percentiles',
     [F + 'fd_neuron_selection.csv', F + 'fd_skeletons.csv', F + 'fd_group_change.csv', S + 'st4_parameter_change.csv'], []),
    ('Fig4', 'a', 'Unfolded-CNS maps of cue-related voltage difference and spiking',
     [F + 'fd_soma_state.csv', F + 'fd_cells.csv', F + 'fd_coverage_graded.csv', F + 'fd_display_transform.csv'], []),
    ('Fig4', 'b', 'Task and cue-pair logic', [M + 'fig1/fig1e_memory_task_design.csv', F + 'fd_boards.csv'], []),
    ('Fig4', 'c', 'Carried vs reset vs untrained readout', [M + 'fig4/fig4b_carry_vs_reset.csv'], ['fig4b.mcnemar', 'fig4b.mcnemar_pairs']),
    ('Fig4', 'd', 'Graded coverage by state variable', [F + 'fd_coverage_graded.csv', M + 'fig6/fig6b_cue_difference_coverage.csv'], []),
    ('Fig4', 'e', 'Whole-state intervention barcode', [M + 'fig4/fig4c_state_interventions.csv'], ['fig4c.mcnemar', 'fig4c.wilcoxon_margin']),
    ('Fig5', 'a', 'Board triptych and directional-transfer discordance', [M + 'fig4/fig4e_directional_transfer.csv', F + 'fd_boards.csv'],
     ['fig4e.mcnemar', 'fig4e.stratified', 'fig4e.clustered', 'fig4e.sensitivity']),
    ('Fig5', 'b', 'State interpolation schematic', [M + 'fig4/fig4f_dose_response.csv'], []),
    ('Fig5', 'c', 'First-decision margin vs interpolation dose', [M + 'fig4/fig4f_dose_response.csv'], ['fig4f.page', 'fig4f.wilcoxon']),
    ('Fig5', 'd', 'Hidden vs visible whole-state interventions', [M + 'fig4/fig4d_visibility_contrast.csv'], ['fig4d.sign']),
    ('Fig6', 'a', 'Cue ensemble (top 1% by |dv|) and its connections on the unfolded CNS',
     [F + 'fd_soma_state.csv', F + 'fd_cells.csv', F + 'fd_ensemble.csv', F + 'fd_ensemble_edges.csv', F + 'fd_display_transform.csv'], []),
    ('Fig6', 'b', 'Neurons of the eight most cue-responsive types, with type means per state set',
     [F + 'fd_neuron_selection.csv', F + 'fd_skeletons.csv', E + 'ed08/ed08a_type_atlas.csv'], []),
    ('Fig6', 'c', 'Cue difference by synaptic distance from the inputs and to the readout', [F + 'fd_cue_by_distance.csv', F + 'fd_cell_distance.csv'],
     []),
    ('Fig6', 'd', 'Wiring within the ensemble vs matched random sets', [F + 'fd_ensemble.csv'], ['fig_circuit.ensemble']),
    ('Fig6', 'e', 'Sign motif: excitatory share by cue preference vs preference shuffles', [F + 'fd_sign_motif.csv'], ['fig_circuit.sign']),
    ('Fig6', 'f', 'Wiring between the two preference groups (M0)', [F + 'fd_ensemble_edges.csv', F + 'fd_soma_state.csv', F + 'fd_sign_motif.csv'], []),
    ('Fig6', 'g', 'Superclass and transmitter enrichment of the ensemble', [E + 'ed08/ed08b_atlas_enrichment.csv'], ['ed08b.enrichment']),
    ('Fig6', 'h', 'Inhibitory transmitters in the larger and smaller preference groups', [F + 'fd_soma_state.csv', F + 'fd_cells.csv'], []),
    ('Fig7', 'a', 'Every cue pair of 21 learner configurations at trained and untrained timings',
     [E + 'ed05/ed05a_memory_behaviour_all.csv', E + 'ed05/ed05c_cue_pair_classes.csv'], ['ed05c.memory_use']),
    ('Fig7', 'b', 'Dense delay scan (M0): behaviour and cue decoding per cue pair', [E + 'ed05/ed05d_dense_delay_scan.csv'],
     ['ed05d.behaviour_pairs', 'ed05d.decoder_pairs']),
    ('Fig7', 'c', 'First-decision margins at trained and untrained timings', [E + 'ed05/ed05a_memory_behaviour_all.csv'], []),
    ('Fig7', 'd', 'Cue decoding on held-out timings and held-out inputs', [M + 'fig6/fig6f_decoding_dynamics.csv'], ['fig6f.decoder_pairs']),
    ('Fig7', 'e', 'Cue-difference energy in the readout subspace vs a random subspace', [M + 'fig6/fig6c_readout_geometry.csv'], ['fig6c.sign']),
    ('Fig7', 'f', 'State prediction by horizon: persistence, autonomous affine, input and phase', [M + 'fig6/fig6f_decoding_dynamics.csv'], []),
    ('Fig7', 'g', 'Matched-duration control: neutral input vs blank delays', [M + 'fig5/fig5c_matched_duration.csv'], ['fig5c.mcnemar']),
    ('ED1', '', 'Method development and lineage', [E + 'ed01/ed01a_method_development.csv', E + 'ed01/ed01b_formal_lineage.csv'], []),
    ('ED2', '', 'Numerical validation and compute reconstruction', [E + 'ed02/ed02a_precision_calibration.csv', E + 'ed02/ed02b_validation_checks.csv',
                                                                    'meta/compute_reconstruction.csv'], []),
    ('ED3', '', 'All learning curves, stage costs, descriptors', [E + 'ed03/ed03a_learning_curves_all.csv', E + 'ed03/ed03b_stage_costs_all.csv',
                                                                  E + 'ed03/ed03c_difficulty_descriptors.csv'], []),
    ('ED4', '', 'Parameter change by group: soma maps per parameter, strata, enrichment, consistency through training',
     [F + 'fd_group_change.csv', F + 'fd_cells.csv', E + 'ed04/ed04a_parameter_change_consistency.csv', E + 'ed04/ed04b_parameter_change_anatomy.csv',
      E + 'ed04/ed04c_parameter_change_enrichment.csv'], ['ed04a.spearman', 'ed04c.enrichment']),
    ('ED5', '', 'Memory histories, delays, matched duration, transfer, S2 context',
     [E + 'ed05/ed05a_memory_behaviour_all.csv', E + 'ed05/ed05b_memory_training_runs.csv', E + 'ed05/ed05c_cue_pair_classes.csv',
      E + 'ed05/ed05d_dense_delay_scan.csv', M + 'fig5/fig5c_matched_duration.csv', M + 'fig5/fig5d_cross_layout_transfer.csv',
      M + 'fig5/fig5e_context_s2.csv'], ['ed05d.behaviour_pairs', 'ed05d.decoder_pairs', 'fig5c.mcnemar', 'fig5d.decoder_pairs', 'fig5e.mcnemar']),
    ('ED6', '', 'Intervention factorial, component x window map (null), norm audit, margin change vs outcome, decision-window edits '
                'in carried, joint, untrained-readout and reset models',
     [E + 'ed06/ed06a_intervention_factorial.csv', E + 'ed06/ed06b_norm_audit.csv', M + 'fig6/fig6d_component_interventions.csv'],
     ['fig4c.mcnemar', 'fig6d.mcnemar']),
    ('ED7', '', 'Candidate vs matched groups (skeleton tiles, null outcomes), routes, pathways',
     [M + 'fig6/fig6e_candidate_groups.csv', E + 'ed07/ed07a_group_selection.csv', E + 'ed07/ed07b_group_matching.csv',
      E + 'ed07/ed07c_signed_pathways.csv', F + 'fd_neuron_selection.csv', F + 'fd_skeletons.csv'], ['fig6e.mcnemar', 'fig6e.wilcoxon_routes']),
    ('ED8', '', 'Type and transmitter atlas; soma maps per distinct state set; agreement between state sets; type activity before vs '
                'after training',
     [E + 'ed08/ed08a_type_atlas.csv', E + 'ed08/ed08b_atlas_enrichment.csv', F + 'fd_soma_state.csv', F + 'fd_cells.csv'], ['ed08b.enrichment']),
    ('ED9', '', 'Decoding and dynamics at every horizon; time generalisation; affine eigenvalues and gain; demixed components; '
                'cross-map static decoding; decoder vs behaviour',
     [E + 'ed09/ed09a_decoding_all.csv', E + 'ed09/ed09b_dynamics_all.csv', F + 'fd_time_generalisation.csv'], []),
    ('ED10', '', 'Run registry summary', [E + 'ed10/ed10_run_registry_summary.csv', S + 'st1_run_registry.csv'], []),
]


def panel_index():
    tests = pd.read_csv(paths.out('stats/statistical_tests.csv'), usecols=['test_id'], low_memory=False)['test_id']
    rows = []
    for fig, panel, content, tables, prefixes in PANELS:
        missing = [t for t in tables if not paths.out(t).exists()]
        if missing:
            raise RuntimeError(f'panel index {fig}{panel}: missing tables {missing}')
        counts = [int(tests.str.startswith(p + '.').sum()) for p in prefixes]
        if any(k == 0 for k in counts):
            raise RuntimeError(f'panel index {fig}{panel}: test prefix without tests {prefixes}')
        rows.append(dict(figure=fig, panel=panel, content=content, tables=';'.join(tables), test_id_prefixes=';'.join(prefixes) or None,
                         tests=int(sum(counts))))
    csvout.write_table(pd.DataFrame(rows), FD + 'panel_index.csv', title='Panel index: figure panel to source tables and statistical tests', describe={
        'figure': 'Figure (Fig1-Fig7 main, ED1-ED10 Extended Data)', 'panel': 'Panel letter (empty = whole Extended Data figure)', 'content': 'Panel content',
        'tables': 'Source tables under manuscript/results (separated by ";")', 'test_id_prefixes': 'Test-id prefixes in stats/statistical_tests.csv (separated by ";")',
        'tests': 'Tests matching the prefixes'}, key=['figure', 'panel'], sort=[])


def main():
    meta, regions, xyz, ids = anatomy()
    fd_display_transform(regions, xyz)
    cells = fd_cells(xyz, ids)
    fd_neuropils(meta)
    fd_boards()
    frames = fd_soma_state(cells)
    cov = fd_coverage_graded(frames)
    groups = fd_group_change(cells)
    fd_time_generalisation()
    selection = fd_neuron_selection(cells, frames, groups)
    fd_skeletons(cells, selection)
    panel_index()
    ladder = cov[(cov['variable'] == 'v') & (cov['population'] == 'all') & cov['threshold'].isin([0.01, 0.1, 1.0])]
    for key, g in ladder.groupby('state_key', sort=False):
        print(f'  coverage ladder {key}: ' + ', '.join(f'>{t:g} mV {100 * f:.1f}%' for t, f in zip(g['threshold'], g['fraction_above'])), flush=True)
    sel = selection.drop_duplicates('body_id')
    print(f'  neuron selection: {len(selection)} rows, {len(sel)} cells, {int(sel["download"].sum())} to download', flush=True)
