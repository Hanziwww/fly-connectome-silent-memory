"""P12: the plan v2 study results routed into manuscript/results/plan_v2 for the v4 (Cell) figures, so every number in a
figure is read from manuscript/results (manuscript/NARRATIVE_V4_OUTLINE.md, section 12; plan of 2026-10-22).

Inputs are the hash-checked result tables of the two plan v2 studies (studies/brain-supplement-v2/results,
studies/brain-pathology-v1/results), one plan v2 run summary (the state transplant, g4-linear-response/result.json) and,
for Fig. 2, the frozen dense 1-ms records of the mechanism study that P11 reads for fu4. Tests are not recomputed: the
adjusted p values come from the collected test table (collect_tests.py; Holm or BH within each declared family). Fig. 2's
per-cell delay-end state is a descriptive readout of the same records, gated against fu4's energies. Class (c) results
(RESULTS_INVENTORY_V4.md) are not routed. Every table carries reader-facing names (NARRATIVE_V4_OUTLINE.md §12). Tables
grow figure by figure.
"""
import io as _io
import re

import numpy as np
import pandas as pd

from msa import csvout, io, paths, stats
from panels.p11_followup import FIELDS, _first_decision, _npz_rows, _record_meta, _record_pairs  # the fu4 records and reader

PV = 'plan_v2/'
SUPP = paths.ROOT / 'studies' / 'brain-supplement-v2' / 'results'
PATH = paths.ROOT / 'studies' / 'brain-pathology-v1' / 'results'
DECLARED = paths.ANALYSIS / 'declared'      # recorded from the study's complete tables (family sizes, a generator state)
# Reader-facing names (NARRATIVE_V4_OUTLINE.md §12): the memory-task layouts M0, S1, S2 are scenes 1-3; head_only trains the
# readout with the brain fixed, joint trains brain and readout. The repository code stays in the 'code' column only.
X1_LEARNERS = {'M0-head_only': ('scene 1', 'readout'), 'M0-joint': ('scene 1', 'brain and readout'),
               'S1-head_only': ('scene 2', 'readout'), 'S2-head_only': ('scene 3', 'readout')}


def _read(path):
    return io.csv_load(path, low_memory=False)


def fig1_tables():
    x10 = _read(SUPP / 'x10_maps.csv')
    x10 = x10[['map', 'replicate', 'optimal_moves', 'status', 'updates', 'final_solved']].copy()
    x10['source_file'] = paths.rel(SUPP / 'x10_maps.csv')
    csvout.write_table(x10, PV + 'x10_maps.csv', title='Sokoban ladder learned in six replicate sets (set 0 = the original set, the analysed model)', describe={
        'map': 'Map identifier (difficulty tier and family)', 'replicate': 'Replicate set (0 = the original set)',
        'optimal_moves': 'Certified optimal number of moves', 'status': 'Run status', 'updates': 'Parameter updates to the solve',
        'final_solved': 'The map was solved from the original start'}, key=['map', 'replicate'])
    x9 = _read(SUPP / 'x9_pairings.csv')
    x9 = x9[['map', 'combination', 'brain', 'readout', 'status', 'steps']].copy()
    x9['solved'] = x9['status'].eq('solved')
    x9['source_file'] = paths.rel(SUPP / 'x9_pairings.csv')
    csvout.write_table(x9, PV + 'x9_pairings.csv', title='Brain x readout checkpoint pairings on 24 maps', describe={
        'map': 'Map identifier', 'combination': 'Brain and readout checkpoint (0 before, m mid, 1 after training)',
        'brain': 'Brain checkpoint', 'readout': 'Readout checkpoint', 'status': 'Outcome of the rollout from the original start',
        'steps': 'Steps taken', 'solved': 'The rollout solved the map'}, key=['map', 'combination'])
    c = _read(SUPP / 'campaign_runs.csv')
    c = c[c['job'].str.startswith('x1-')].copy()
    parts = c['job'].str.extract(r'x1-(?P<learner>.+)-r(?P<replicate>\d+)')
    names = parts['learner'].map(X1_LEARNERS)
    if names.isna().any():
        raise ValueError(f"X1 learners without a reader-facing name: {sorted(parts['learner'][names.isna()].unique())}")
    # run status and update counts are not routed: the figure shows memory use only, and the success counts are class (c)
    # (RESULTS_INVENTORY_V4.md, L9; STAR Methods and Table S2)
    x1 = pd.DataFrame(dict(scene=names.str[0].to_numpy(), trained=names.str[1].to_numpy(), code=parts['learner'].to_numpy(),
                           replicate=parts['replicate'].astype(int).to_numpy(), trained_net_use=c['trained_net_use'].to_numpy(),
                           untrained_net_use=c['untrained_net_use'].to_numpy()))
    x1['lock'] = x1['trained_net_use'] - x1['untrained_net_use']
    x1['source_file'] = paths.rel(SUPP / 'campaign_runs.csv')
    csvout.write_table(x1, PV + 'x1_runs.csv', title='Delayed-cue memory in 34 replicates of four learners (replicate training)', describe={
        'scene': 'Memory-task scene (board layout)', 'trained': 'Trained parameters (readout only, with the brain fixed; or brain and readout)',
        'code': 'Repository identifier of the learner (code and test tables only)',
        'replicate': 'Replicate (0 = the run of the analysed model)', 'trained_net_use': 'Net memory use at the 9 trained timings (cue-pair classes)',
        'untrained_net_use': 'Net memory use at the 26 untrained timings', 'lock': 'Trained minus untrained net use'},
        key=['scene', 'trained', 'replicate'])
    for name, rel in (('graph_paths', 'phase4_graph_paths.csv'), ('graph_flow', 'phase4_graph_flow.csv')):
        g = _read(PATH / rel)
        g['source_file'] = paths.rel(PATH / rel)
        if name == 'graph_paths':
            csvout.write_table(g, PV + 'graph_paths.csv', title='Walks from the visual input cells to the readout cells by length', describe={
                'pathway': 'Source and target cell sets', 'k': 'Walk length (synapses)', 'walks': 'Number of walks of length k',
                'log10_walks': 'log10 of the number of walks', 'targets_reached_within_k': 'Share of target cells reached within k synapses'},
                key=['pathway', 'k'])
        else:
            csvout.write_table(g, PV + 'graph_flow.csv', title='Maximum flow (synapse counts) between cell sets, intact and lesioned', describe={
                'pathway': 'Source and target cell sets', 'lesion': 'Lesion applied', 'cells_removed': 'Cells removed',
                'max_flow': 'Maximum flow in synapses', 'relative': 'Flow relative to the intact network'}, key=['pathway', 'lesion'])


# ---------------------------------------------------------------- reader-facing names (the glossary as data)

R, B = 'readout', 'brain and readout'
NAMES = [  # code, kind, scene, trained, condition
    ('remote-dynamic-M0-head_only', 'model', 'scene 1', R, ''), ('remote-dynamic-M0-joint', 'model', 'scene 1', B, ''),
    ('supp-M0-visibility', 'model', 'scene 1', R, 'readout shared by hidden- and visible-goal training'),
    ('supp-S1-ring-brain', 'model', 'scene 2', R, ''), ('supp-S2-context-brain', 'model', 'scene 3', R, ''),
    ('M0 readout-only', 'model label', 'scene 1', R, ''), ('M0 joint', 'model label', 'scene 1', B, ''),
    ('M0 shared readout', 'model label', 'scene 1', R, 'readout shared by hidden- and visible-goal training'),
    ('S1 readout-only', 'model label', 'scene 2', R, ''), ('S2 readout-only', 'model label', 'scene 3', R, ''),
    ('M0-head_only', 'replicate-training code', 'scene 1', R, ''), ('M0-joint', 'replicate-training code', 'scene 1', B, ''),
    ('S1-head_only', 'replicate-training code', 'scene 2', R, ''), ('S2-head_only', 'replicate-training code', 'scene 3', R, ''),
    ('M0', 'swap unit', 'scene 1', R, ''), ('M0joint', 'swap unit', 'scene 1', B, ''),
    ('M0vis.hidden', 'swap unit', 'scene 1', R, 'shared readout, goal hidden'),
    ('M0vis.visible', 'swap unit', 'scene 1', R, 'shared readout, goal visible'),
    ('S1', 'swap unit', 'scene 2', R, ''), ('S2', 'swap unit', 'scene 3', R, ''),
    ('m0', 'state key', 'scene 1', R, ''), ('m0_joint', 'state key', 'scene 1', B, ''),
    ('m0vis_visible', 'state key', 'scene 1', R, 'shared readout, goal visible'), ('s1', 'state key', 'scene 2', R, ''),
    ('s2_o0', 'state key', 'scene 3', R, 'context 1'), ('s2_o180', 'state key', 'scene 3', R, 'context 2')]


def names_table():
    n = pd.DataFrame(NAMES, columns=['code', 'kind', 'scene', 'trained', 'condition'])
    n['label'] = n['scene'] + ', ' + n['trained'] + ' trained' + n['condition'].map(lambda c: f' ({c})' if c else '')
    csvout.write_table(n, PV + 'names.csv', title='Reader-facing names of the internal model codes (NARRATIVE_V4_OUTLINE.md section 12)', describe={
        'code': 'Internal code (repository, result tables)', 'kind': 'Where the code is used',
        'scene': 'Memory-task scene (board layout): M0, S1, S2 are scenes 1, 2, 3', 'trained': 'Trained parameters (readout only, with the brain fixed; or brain and readout)',
        'condition': 'Further condition, if any', 'label': 'Reader-facing label'}, key=['code', 'kind'])
    return n


# ---------------------------------------------------------------- Fig. 2: the activity-silent trace

# The four learners of Fig. 1G as recorded in the frozen mechanism study (dense 1-ms records, the files and reader of fu4 in
# P11), at a trained timing; scene 3 in its first context (its second gives the same fu4 values).
FIG2_LEARNERS = (('remote-dynamic-M0-head_only', 's1_readout'), ('remote-dynamic-M0-joint', 's1_brain'),
                 ('supp-S1-ring-brain', 's2_readout'), ('supp-S2-context-brain', 's3_readout'))
FIG2_TIMING = (160, 4)                                     # delay (ms), prefix windows: as fd_soma_state and fu4
FIG2_THRESHOLDS = 10.0 ** (np.arange(-30, 11) / 10)       # |dv| thresholds, 0.001 to 10 mV
# The full-state transplant is defined from a 20-ms delay on (at 0 ms there is no delay to carry the cue). Adaptation alone
# is routed where it suffices (320-640 ms); at the trained 160-ms timing it does not (class c, inventory T10; STAR Methods).
TRANSPLANT_MIN_DELAY, ADAPT_ONLY_DELAYS = 20, (320, 640)


def _dense_pair(folder, meta, i0, i1, kinds):
    """Chunks (segment kind, step stamps, cue 1 minus cue 0 in float64 as (ms, field, cell)) of the dense records of one cue
    pair, for the segments of the given kinds up to the end of the first decision, in time order (as P11's _dense_pair)."""
    first = meta['events'][_first_decision(meta)]
    end = first['start_ms'] + first['ms']
    seg = lambda i: sorted((e for e in meta['dense_files'] if e['case'] == i and e['start_ms'] < end and e['kind'] in kinds),  # noqa: E731
                           key=lambda e: e['start_ms'])
    s0, s1 = seg(i0), seg(i1)
    if not s0 or [(e['start_ms'], e['ms']) for e in s0] != [(e['start_ms'], e['ms']) for e in s1]:
        raise RuntimeError('dense segments of the cue pair disagree')
    for e0, e1 in zip(s0, s1):
        d0, d1 = io.read_bytes(folder / e0['file']), io.read_bytes(folder / e1['file'])
        for dd in (d0, d1):
            with np.load(_io.BytesIO(dd), allow_pickle=False) as z:
                if tuple(z['fields'].astype(str)) != FIELDS:
                    raise RuntimeError('unexpected dense fields')
                ms = z['milliseconds'].astype(np.int64)
            if ms[0] != e0['start_ms'] + 1 or len(ms) != e0['ms']:
                raise RuntimeError('dense milliseconds disagree with the segment list')
        for (start, a), (_, b) in zip(_npz_rows(d0, 'values', 20), _npz_rows(d1, 'values', 20)):
            yield e0['kind'], ms[start:start + len(a)], b.astype(np.float64) - a.astype(np.float64)


def fig2_tables(names):
    d, p = FIG2_TIMING
    label = names[names['kind'] == 'model'].set_index('code')
    fu4 = io.csv_load(paths.out('followup/fu4_state_dynamics.csv'), low_memory=False)
    cells, summary, coverage, raster = {}, [], [], []
    for model, key in FIG2_LEARNERS:
        folder, meta = _record_meta(model, d, p)
        i0, i1 = _record_pairs(meta)[(0, 'hidden')]
        delay = next(e for e in meta['events'] if e['kind'] == 'delay')
        cue_end, end_delay = delay['start_ms'], delay['start_ms'] + delay['ms']        # stamps: the delay covers cue_end+1..end_delay
        late = cue_end + delay['ms'] // 2                                             # the second half of the delay: late+1..end_delay
        kinds = ('cue', 'delay', 'prefix', 'decision') if key == 's1_readout' else ('delay',)
        n_delay = n_late = dv_end = None
        energy_spikes, energy_v_end, last_diff = {}, None, 0
        for kind, ms, D in _dense_pair(folder, meta, i0, i1, kinds):
            dsp = D[:, 0]                                                             # spikes, cue 1 minus cue 0: -1, 0 or +1
            for k, t in enumerate(ms):
                energy_spikes[int(t)] = float(dsp[k] @ dsp[k])
            if kind == 'delay':
                diff = dsp != 0
                n_delay = diff.sum(0) if n_delay is None else n_delay + diff.sum(0)
                n_late = (diff[ms > late].sum(0) if n_late is None else n_late + diff[ms > late].sum(0))
                if diff.any():
                    last_diff = max(last_diff, int(ms[diff.any(1)].max()))
                if ms[-1] == end_delay:
                    dv_end, dadapt_end = D[-1, 2].copy(), D[-1, 3].copy()
                    energy_v_end = float(dv_end @ dv_end)
            if key == 's1_readout':
                t_i, c_i = np.nonzero(dsp)
                raster.append(pd.DataFrame(dict(ms=ms[t_i], cell=c_i, sign=dsp[t_i, c_i].astype(int))))
        # gates: this readout reproduces fu4 (P11) on the same records
        ref = fu4[(fu4['model'] == model) & (fu4['context'] == 0) & (fu4['visibility'] == 'hidden') & (fu4['delay_ms'] == d)
                  & (fu4['prefix_windows'] == p)].set_index('ms')
        if not np.isclose(energy_v_end, float(ref.loc[end_delay, 'energy_v']), rtol=1e-9, atol=0):
            raise RuntimeError(f'{model}: delay-end voltage energy disagrees with fu4')
        e_ref = ref.loc[sorted(energy_spikes), 'energy_spikes'].to_numpy(float)
        if not np.allclose(np.array([energy_spikes[t] for t in sorted(energy_spikes)]), e_ref, rtol=0, atol=1e-9):
            raise RuntimeError(f'{model}: differential spike counts disagree with fu4')
        cells[f'dv_{key}_mv'] = np.round(dv_end, 6)
        cells[f'dadapt_{key}'] = np.round(dadapt_end, 6)
        cells[f'delay_spike_ms_{key}'] = n_delay.astype(int)
        name = label.loc[model]
        x = np.sort(np.abs(dv_end))
        above = len(x) - np.searchsorted(x, FIG2_THRESHOLDS, side='right')
        coverage += [dict(learner=key, scene=name['scene'], trained=name['trained'], threshold_mv=float(t), cells=len(x),
                          cells_above=int(a), fraction_above=float(a / len(x))) for t, a in zip(FIG2_THRESHOLDS, above)]
        summary.append(dict(learner=key, scene=name['scene'], trained=name['trained'], code=model, delay_ms=d, prefix_windows=p,
                            cue_offset_ms=cue_end, end_of_delay_ms=end_delay,
                            cells=len(x), cells_spiking_differently_in_delay=int((n_delay > 0).sum()),
                            cells_spiking_differently_late_delay=int((n_late > 0).sum()),
                            last_differential_spike_ms_after_cue_offset=int(last_diff - cue_end) if last_diff else 0,
                            energy_v_end_of_delay=energy_v_end))
    out = pd.DataFrame({'cell': np.arange(len(next(iter(cells.values())))), **cells})
    src = paths.rel(paths.M1 / 'runs')
    csvout.write_table(out, PV + 'fig2_delay_end_cells.csv', title=f'Per-cell cue difference at the end of the delay ({d}-ms delay, {p} prefix windows) in four learners', describe={
        'cell': 'Model cell index (as fd_cells)',
        **{f'dv_{k}_mv': f'{k}: membrane potential, cue 1 minus cue 0, last ms of the delay (mV, rounded to 1e-6)' for _, k in FIG2_LEARNERS},
        **{f'dadapt_{k}': f'{k}: adaptation variable (subtracted from the target potential, mV), cue 1 minus cue 0, last ms of the delay (rounded to 1e-6)' for _, k in FIG2_LEARNERS},
        **{f'delay_spike_ms_{k}': f'{k}: delay steps (ms) in which the cell spikes differently after the two cues' for _, k in FIG2_LEARNERS}},
        key=['cell'], notes=f'Source: dense records under {src}/record-<model>/dense (free-after-d{d}-p{p}); gated against followup/fu4_state_dynamics.csv.')
    s = pd.DataFrame(summary)
    s['source_file'] = src
    csvout.write_table(s, PV + 'fig2_delay_end_summary.csv', title='The silent trace at the end of the delay: differential spiking and voltage energy per learner', describe={
        'learner': 'Learner key (columns of fig2_delay_end_cells)', 'scene': 'Memory-task scene', 'trained': 'Trained parameters',
        'code': 'Repository identifier of the model', 'delay_ms': 'Delay (ms)', 'prefix_windows': 'Prefix windows',
        'cue_offset_ms': 'Step of cue offset (the cue covers steps 1 to this one)', 'end_of_delay_ms': 'Last step of the delay', 'cells': 'Cells',
        'cells_spiking_differently_in_delay': 'Cells that spike differently after the two cues at any delay step',
        'cells_spiking_differently_late_delay': 'Same, second half of the delay',
        'last_differential_spike_ms_after_cue_offset': 'Last delay step with any differential spike, ms after cue offset',
        'energy_v_end_of_delay': 'Sum of squared voltage differences at the last delay step (equals fu4 energy_v)'}, key=['learner'])
    c = pd.DataFrame(coverage)
    csvout.write_table(c, PV + 'fig2_coverage.csv', title='Coverage of the silent trace: cells whose voltage differs by more than a threshold at the end of the delay', describe={
        'learner': 'Learner key', 'scene': 'Memory-task scene', 'trained': 'Trained parameters', 'threshold_mv': 'Threshold on |cue 1 - cue 0| (mV)',
        'cells': 'Cells', 'cells_above': 'Cells above the threshold', 'fraction_above': 'cells_above / cells'}, key=['learner', 'threshold_mv'])
    r = pd.concat(raster, ignore_index=True)
    csvout.write_table(r, PV + 'fig2_raster.csv', title=f'Differential spikes of scene 1 (readout trained), cue onset to the end of the first decision ({d}-ms delay, {p} prefix windows)', describe={
        'ms': 'Step from cue onset (1 ms each)', 'cell': 'Model cell index', 'sign': '+1 the cell spikes only after cue 1, -1 only after cue 0'},
        key=['ms', 'cell'], notes='Same records and gate as fig2_delay_end_cells.')
    # state transplant (plan v2 run summary)
    g = io.json_load(paths.ROOT / 'studies' / 'brain-supplement-v2' / 'runs' / 'g4-linear-response' / 'result.json')
    rows = []
    for row in g['rows']:
        if row['delay'] < TRANSPLANT_MIN_DELAY:
            continue
        rows.append(dict(prefix_windows=int(row['prefix']), delay_ms=int(row['delay']), full_state_cosine=float(row['all3_cos']),
                         full_state_margin=float(row['all3_margin']), native_margin=float(row['margin_cue1']),
                         adaptation_only_cosine=float(row['adapt_eps1_cos']) if row['delay'] in ADAPT_ONLY_DELAYS and row['prefix'] == 4 else np.nan))
    tr = pd.DataFrame(rows)
    tr['scene'], tr['trained'], tr['code'] = label.loc[g['model'], 'scene'], label.loc[g['model'], 'trained'], g['model']
    tr['source_file'] = 'studies/brain-supplement-v2/runs/g4-linear-response/result.json'
    csvout.write_table(tr, PV + 'fig2_transplant.csv', title='State transplant at the end of the delay: the decision-time cue difference it evokes', describe={
        'prefix_windows': 'Prefix windows', 'delay_ms': 'Delay (ms)',
        'full_state_cosine': 'Cosine between the decision-time difference evoked by transplanting voltage, synaptic current and adaptation and the native one',
        'full_state_margin': 'Readout margin after the full-state transplant', 'native_margin': 'Native readout margin of the donor cue',
        'adaptation_only_cosine': 'Same cosine for adaptation alone; routed for 320-640 ms with 4 prefix windows, where adaptation alone suffices',
        'scene': 'Memory-task scene', 'trained': 'Trained parameters', 'code': 'Repository identifier of the model'}, key=['prefix_windows', 'delay_ms'])
    # causal swaps (X5): the four learners of Fig. 1G. The shared hidden/visible-goal readout units are not routed: their
    # comparison is class (c) (inventory T12, T13; Figure S2 and Table S2).
    x5 = _read(SUPP / 'x5_histories.csv')
    units = names[names['kind'] == 'swap unit'].set_index('code')
    x5 = x5[x5['unit'].isin(['M0', 'M0joint', 'S1', 'S2'])].copy()
    sw = pd.DataFrame(dict(scene=x5['unit'].map(units['scene']).to_numpy(), trained=x5['unit'].map(units['trained']).to_numpy(),
                           code=x5['unit'].to_numpy(), case=x5['case'].to_numpy(), cue=x5['cue'].to_numpy(), context=x5['context'].to_numpy(),
                           delay_ms=x5['delay_ms'].to_numpy(), prefix_windows=x5['prefix_windows'].to_numpy(),
                           baseline_solved=x5['baseline_solved'].to_numpy(), swapped_solved=x5['opposite_cue_solved'].to_numpy()))
    sw['source_file'] = paths.rel(SUPP / 'x5_histories.csv')
    csvout.write_table(sw, PV + 'fig2_swaps.csv', title='Opposite-cue whole-state swaps at the trained timings, per history, four learners', describe={
        'scene': 'Memory-task scene', 'trained': 'Trained parameters', 'code': 'Repository identifier of the learner (test tables)',
        'case': 'History case (context and cue)', 'cue': 'Cue', 'context': 'Context', 'delay_ms': 'Delay (ms)', 'prefix_windows': 'Prefix windows',
        'baseline_solved': 'Solved without intervention', 'swapped_solved': 'Solved after the state was swapped for the opposite cue'},
        key=['code', 'case', 'delay_ms', 'prefix_windows'])
    # what the trace costs: spikes per trial and decoded cue information of the healthy model
    e = _read(PATH / 'phase4_energy.csv')
    e = e[(e['source'] == 'phase3-regime-a') & (e['condition'] == 'healthy')][['task_mean_hz', 'spikes_per_trial', 'cue_bits_per_decision',
                                                                             'first_push_correct_share']].copy()
    if len(e) != 1:
        raise RuntimeError('expected one healthy energy record')
    e['scene'], e['trained'] = 'scene 1', 'readout'
    e['source_file'] = paths.rel(PATH / 'phase4_energy.csv')
    csvout.write_table(e, PV + 'fig2_energy.csv', title='Spikes per trial and decoded cue information per decision (scene 1, readout trained)', describe={
        'task_mean_hz': 'Mean firing rate over the task (Hz)', 'spikes_per_trial': 'Spikes per trial, all cells',
        'cue_bits_per_decision': 'Cue information in the first push (bits per decision)', 'first_push_correct_share': 'Share of first pushes that follow the cue',
        'scene': 'Memory-task scene', 'trained': 'Trained parameters'})


# ---------------------------------------------------------------- Fig. 3: no circuit is special

X67_SCENE = {'m0': 'scene 1', 's1': 'scene 2'}                 # X6/X7 ensemble sources and test scenes
X67_TEST = {'x6': 'cells silenced at the decision', 'x7': 'internal synapses removed'}
POPULATIONS = {'kc': 'Kenyon cells', 'alln_gaba': 'GABAergic antennal-lobe local neurons', 'descending': 'descending neurons',
               'monoamine': 'monoaminergic cells'}
REMOVAL_KIND = {'target': 'whole population', 'anchor': 'control population', 'healthy': 'healthy'}
# connection-rule variants (flybrain defaults: receptor-table signs, same-type gain 0.1, synapse cap 60). receptor-none is
# receptor_model None (studies/brain-pathology-v1/code/variants.py): every synapse takes its presynaptic transmitter's sign
VARIANTS = {'default': ('as built', 'default'), 'receptor-none': ('presynaptic signs', 'receptor rule'),
            'same-type-0.05': ('same-type gain 0.05', 'same-type gain'), 'same-type-0.2': ('same-type gain 0.2', 'same-type gain'),
            'same-type-off': ('same-type gain 1', 'same-type gain'), 'conn-cap-30': ('synapse cap 30', 'synapse cap'),
            'conn-cap-120': ('synapse cap 120', 'synapse cap'), 'conn-cap-off': ('no synapse cap', 'synapse cap')}
FIG3_RENDERED_NULLS = 3                                         # matched sets drawn beside the ensemble in the band


def _margin(logits, action, legal):
    """As analyze_x4_x5.margin: the reference first action's logit minus the strongest other legal action's."""
    if logits is None or action is None:
        return np.nan
    others = [logits[a] for a in range(len(logits)) if legal[a] and a != action]
    return logits[action] - max(others) if others else np.nan


def fig3_tables(names):
    import scipy.sparse as sp
    # X6/X7 per set: the analysis reports the ensemble and the null mean, minimum and maximum; the figure shows all 20
    # matched sets, recomputed with the analysis's margin rule and gated against x6_x7_effects
    tasks = {t['id']: t for t in io.json_load(paths.M1 / 'inputs' / 'memory-tasks.json')}
    runs = SUPP.parent / 'runs' / 'x6-x7'
    rows = []
    for key in X67_SCENE:
        rec = io.json_load(runs / f'{key}.json')
        ref = {c['id']: int(c['level']['reference_actions'][0]) for c in tasks[rec['task']]['cases']}
        conds = rec['conditions']
        eff, solved = [[] for _ in conds], [0] * len(conds)
        for entry in rec['results']:
            for i, b in enumerate(entry['baseline']):
                m0 = _margin(b['first_logits'], ref[b['case']], b['first_legal'])
                for j, outs in enumerate(entry['outcomes']):
                    eff[j].append(_margin(outs[i]['first_logits'], ref[outs[i]['case']], outs[i]['first_legal']) - m0)
                    solved[j] += outs[i]['status'] == 'solved'
        eff = np.array(eff)
        for j, c in enumerate(conds):
            rows.append(dict(test=X67_TEST[c['kind']], kind=c['kind'], ensemble_scene=X67_SCENE[c['source']], test_scene=X67_SCENE[rec['scene']],
                             set='ensemble' if c['set'] == 'ensemble' else f"matched {int(c['set'][4:]) + 1}",
                             mean_margin_effect=float(eff[j].mean()), histories=int(eff.shape[1]), solved=int(solved[j])))
    x = pd.DataFrame(rows)
    for _, r in _read(SUPP / 'x6_x7_effects.csv').iterrows():
        q = x[(x['kind'] == r['kind']) & (x['ensemble_scene'] == X67_SCENE[r['source']]) & (x['test_scene'] == X67_SCENE[r['scene']])]
        ens, nul = q.loc[q['set'] == 'ensemble', 'mean_margin_effect'].iloc[0], q.loc[q['set'] != 'ensemble', 'mean_margin_effect']
        want = (r['ensemble_mean_effect'], r['null_mean_effect'], r['null_min'], r['null_max'])
        if len(nul) != 20 or not np.allclose((ens, nul.mean(), nul.min(), nul.max()), want, rtol=0, atol=1e-9):
            raise RuntimeError(f"X6/X7 per-set effects disagree with x6_x7_effects ({r['kind']}, {r['source']}, {r['scene']})")
    x['source_file'] = paths.rel(runs)
    csvout.write_table(x, PV + 'fig3_ensemble_sets.csv', title='Silencing the cue ensemble or its internal synapses vs 20 matched sets, per set', describe={
        'test': 'Intervention', 'kind': 'Repository code of the intervention (test tables)', 'ensemble_scene': 'Scene whose top-1% cue ensemble (and matched sets) is used',
        'test_scene': 'Scene in which the intervention is tested', 'set': 'The ensemble or one of 20 matched random sets',
        'mean_margin_effect': 'Mean change of the first-decision margin over the trained histories (reference action minus the strongest other)',
        'histories': 'Trained histories', 'solved': 'Histories still solved'}, key=['kind', 'ensemble_scene', 'test_scene', 'set'])
    # internal connections of the ensemble and of each matched set (the X6/X7 connectome); gated against X7's ensemble counts
    W = io.sparse_load(paths.ROOT / 'flybrain' / 'cache' / 'W_post_pre.npz').tocsr()
    W.eliminate_zeros()
    A = (W != 0).tocsr()
    sets = io.json_load(SUPP.parent / 'tasks' / 'x6-x7-sets.json')['sets']
    wiring, members = [], []
    x7 = _read(SUPP / 'x6_x7_effects.csv')
    for key, s in sets.items():
        groups = [('ensemble', np.asarray(s['cells']))] + [(f'matched {i + 1}', np.asarray(n['cells'])) for i, n in enumerate(s['nulls'])]
        for name, cells in groups:
            sub = A[cells][:, cells]
            wiring.append(dict(ensemble_scene=X67_SCENE[key], set=name, cells=len(cells), internal_connections=int(sub.nnz)))
            if key == 'm0' and (name == 'ensemble' or int(name.split()[-1]) <= FIG3_RENDERED_NULLS):
                members.append(pd.DataFrame(dict(ensemble_scene=X67_SCENE[key], set=name, cell=np.sort(cells))))
        want = int(x7[(x7['kind'] == 'x7') & (x7['source'] == key)]['requested_synapses'].iloc[0])
        if wiring[-len(groups)]['internal_connections'] != want:
            raise RuntimeError(f'{key}: ensemble internal connections disagree with X7')
    w = pd.DataFrame(wiring)
    w['source_file'] = 'studies/brain-supplement-v2/tasks/x6-x7-sets.json; flybrain/cache/W_post_pre.npz'
    csvout.write_table(w, PV + 'fig3_set_wiring.csv', title='Connections within the cue ensemble and within each of its 20 matched sets', describe={
        'ensemble_scene': 'Scene of the ensemble', 'set': 'The ensemble or a matched set', 'cells': 'Cells in the set',
        'internal_connections': 'Directed cell pairs within the set with at least one synapse'}, key=['ensemble_scene', 'set'])
    csvout.write_table(pd.concat(members, ignore_index=True), PV + 'fig3_band_sets.csv', title='Cells of the scene-1 cue ensemble and of the matched sets drawn in the band',
                       describe={'ensemble_scene': 'Scene of the ensemble', 'set': 'The ensemble or a matched set', 'cell': 'Model cell index'}, key=['set', 'cell'])
    # removal of whole populations vs matched random cells (G3a)
    d = _read(PATH / 'dependency_conditions.csv')
    d['population'] = d['population'].fillna('healthy')
    rm = pd.DataFrame(dict(condition=d['condition'], population=d['population'].map(lambda p: POPULATIONS.get(p, 'none')),
                           fraction_removed=d['level'], kind=d['kind'].map(REMOVAL_KIND).fillna('matched random cells'), cells=d['cells'],
                           solved=d['solved'], histories=d['histories'], solved_only_after_removal=d['only_lesion'],
                           solved_only_healthy=d['only_healthy'], giant_fibre_spikes=d['gf_spikes'], task_mean_hz=d['task_mean_hz']))
    rm['source_file'] = paths.rel(PATH / 'dependency_conditions.csv')
    csvout.write_table(rm, PV + 'fig3_removals.csv', title='Removing whole populations vs the same numbers of matched random cells (all memory models, 126 trained histories)', describe={
        'condition': 'Condition (repository code)', 'population': 'Population removed', 'fraction_removed': 'Fraction of the population removed',
        'kind': 'Whole population, matched random cells, control population or healthy', 'cells': 'Cells removed',
        'solved': 'Trained histories solved', 'histories': 'Trained histories', 'solved_only_after_removal': 'Histories solved only after the removal',
        'solved_only_healthy': 'Histories solved only by the healthy network', 'giant_fibre_spikes': 'Giant fibre spikes in the looming test',
        'task_mean_hz': 'Mean firing rate over the task (Hz)'}, key=['condition'])
    # where the cue difference lives, by circuit
    st = _read(PATH / 'phase4_structure.csv')
    label = names[names['kind'] == 'model'].set_index('code')
    ce = st.assign(scene=st['model'].map(label['scene']), trained=st['model'].map(label['trained'])).rename(columns={'model': 'code'})
    ce['source_file'] = paths.rel(PATH / 'phase4_structure.csv')
    csvout.write_table(ce, PV + 'fig3_cue_energy.csv', title='Share of the cue difference carried by each circuit, against its share of cells', describe={
        'code': 'Repository identifier of the model', 'circuit': 'Circuit (cell population)', 'cells': 'Cells', 'share_of_cells': 'Share of all cells',
        'share_of_cue_energy': 'Share of the cue-difference energy', 'enrichment': 'Share of cue energy / share of cells',
        'median_hops_from_input': 'Median synapses from the visual input cells', 'median_hops_to_readout': 'Median synapses to the readout cells',
        'scene': 'Memory-task scene', 'trained': 'Trained parameters'}, key=['code', 'circuit'])
    # connection-rule variants (O14): learning and the Kv4-lesion memory effect, the declared conclusions that hold in every
    # run. Untrained-timing use and the strict lock are not routed: the strict lock fails under same-type gain 0.05 (class c,
    # inventory D9; STAR Methods and Table S2).
    o = _read(PATH / 'o14_runs.csv')
    var = o['variant'].map(VARIANTS)
    if var.isna().any():
        raise RuntimeError('connection-rule variant without a reader-facing name')
    so = pd.DataFrame(dict(variant=var.str[0], rule=var.str[1], code=o['variant'], replicate=o['replicate'],
                           healthy_network_stable=o['healthy_network_stable'], learned=o['status'].eq('learned_solved'), updates=o['updates'],
                           trained_net_use=o['trained_net_use'], solved_healthy=o['trained_solved_healthy'], solved_lesion=o['trained_solved_c01'],
                           lesion_effect=o['c01_effect_present']))
    so['source_file'] = paths.rel(PATH / 'o14_runs.csv')
    csvout.write_table(so, PV + 'fig3_structure.csv', title='Training under seven connection-rule variants: learning and the Kv4-lesion memory effect per run', describe={
        'variant': 'Connection rule changed (defaults: receptor-table signs, same-type gain 0.1, synapse cap 60)', 'rule': 'Rule family',
        'code': 'Repository code of the variant', 'replicate': 'Replicate (as built: the replicate-training runs 0-2)',
        'healthy_network_stable': 'The healthy network stays stable under the variant', 'learned': 'The run learned the trained timings',
        'updates': 'Parameter updates', 'trained_net_use': 'Net memory use at the 9 trained timings',
        'solved_healthy': 'Trained histories solved, healthy (18)', 'solved_lesion': 'Trained histories solved under the pan-neuronal Kv4 lesion at dose 1',
        'lesion_effect': 'Fewer histories solved under the lesion'}, key=['code', 'replicate'])


# ---------------------------------------------------------------- Fig. 3 maps: where the cue sits at the decision, where the rules act

CUE_MAP_MODEL = 'remote-dynamic-M0-head_only'
OLFACTORY = ('Kenyon cells', 'uniglomerular PNs (ACh ALPN)', 'GABAergic ALLNs', 'MBONs', 'APL', 'ORNs')
RULE_CAP = 60.0
# the receptor rule of the round-3 default (flybrain LIFParams): entries whose fast sign differs from the presynaptic one
RECEPTOR_CHANGES = {'entries': 48295, 'glutamate': 30916, 'histamine': 17379}


def _circuits(a):
    """The circuits of the phase-4 structure table (studies/brain-pathology-v1/code/phase4_structure.py)."""
    t, c, nt = (a[k].fillna('').astype(str) for k in ('type', 'class', 'nt'))
    return {'Kenyon cells': t.str.startswith('KC'), 'uniglomerular PNs (ACh ALPN)': (c == 'ALPN') & (nt == 'acetylcholine'),
            'GABAergic ALLNs': (c == 'ALLN') & (nt == 'gaba'), 'MBONs': c == 'MBON', 'APL': t == 'APL', 'ORNs': c == 'olfactory',
            'visual input cells': a['direct_input'].astype(bool), 'readout cells': a['observed_by_policy'].astype(bool)}


def fig3_maps():
    """Per cell, for the dark maps of Fig. 3F-G. F: the cue difference at the decision (scene 1, readout trained; the source of
    the phase-4 structure table, gated against fig3_cue_energy) with the circuit memberships. G: the share of each cell's input
    synapses that each connection rule acts on - the receptor table's signs (entries whose sign differs from the presynaptic
    transmitter's; gated against the documented 48,295 entries), the same-type gain (synapses from cells of the cell's own
    type) and the synapse cap (synapses beyond 60 per connection)."""
    from flybrain import connectome as cn
    a = io.parquet_load(paths.M1 / 'neurons' / 'annotations.parquet')
    src = paths.M1 / 'interpretation' / f'{CUE_MAP_MODEL}-memory-cells-d160-p4.parquet'
    dv = io.parquet_load(src, columns=['c0-visible0-delta-v'])['c0-visible0-delta-v'].to_numpy(np.float64)
    groups = {k: v.to_numpy() for k, v in _circuits(a).items()}
    ce = io.csv_load(paths.out(PV + 'fig3_cue_energy.csv'))
    ce = ce[ce['code'] == CUE_MAP_MODEL].set_index('circuit')
    dv2 = dv ** 2
    for name, mask in groups.items():
        if not np.isclose(dv2[mask].sum() / dv2.sum(), ce.loc[name, 'share_of_cue_energy'], rtol=1e-9, atol=0):
            raise RuntimeError(f'fig3 maps: the cue share of {name} disagrees with fig3_cue_energy')
    olf, olf_name = np.zeros(len(a), bool), np.full(len(a), '', dtype=object)
    for name in OLFACTORY:
        olf_name[groups[name] & ~olf] = name
        olf |= groups[name]
    cue = pd.DataFrame(dict(cell=np.arange(len(a)), superclass=a['superclass'].fillna('unknown').astype(str) if 'superclass' in a else '',
                            dv_decision_mv=np.round(dv, 6), readout_cell=groups['readout cells'], visual_input_cell=groups['visual input cells'],
                            olfactory_cell=olf, olfactory_circuit=olf_name))
    cue['source_file'] = paths.rel(src)
    csvout.write_table(cue, PV + 'fig3_cue_cells.csv', title='The cue difference at the decision per cell, with the circuits of Fig. 3F (scene 1, readout trained)', describe={
        'cell': 'Model cell index', 'superclass': 'Superclass', 'dv_decision_mv': 'Membrane potential, difference between the cues in the last recorded window (the first decision; mV)',
        'readout_cell': 'Read out by the policy', 'visual_input_cell': 'Receives the visual input', 'olfactory_cell': 'In one of the six olfactory circuits of Fig. 3F',
        'olfactory_circuit': 'The olfactory circuit (Kenyon cells, uniglomerular PNs, GABAergic ALLNs, MBONs, APL, ORNs)'}, key=['cell'],
        notes='Squared differences summed per circuit equal the shares in fig3_cue_energy (gated).')
    c = cn.load(paths.ROOT / 'flybrain' / 'cache', verbose=False)
    if not np.array_equal(c.neurons['bodyId'].to_numpy(np.int64), a['bodyId'].to_numpy(np.int64)):
        raise RuntimeError('fig3 maps: the connectome and the annotations differ in cell order')
    W = c.W.tocsr()
    rec = cn.receptor_signs(c, net_rule='abs')
    coo = W.tocoo()
    post, pre = coo.row, coo.col
    count = np.abs(coo.data).astype(np.float64)
    pre_sign = c.neurons['sign'].to_numpy(np.float32)[pre] if 'sign' in c.neurons else np.sign(coo.data).astype(np.float32)
    nt = c.neurons['nt'].fillna('').astype(str).to_numpy()[pre]
    changed = (rec.fast_sign != pre_sign) & (count > 0)
    found = {'entries': int(changed.sum()), 'glutamate': int((changed & (nt == 'glutamate') & (rec.fast_sign > 0)).sum()),
             'histamine': int((changed & (nt == 'histamine') & (rec.fast_sign == 0)).sum())}
    if found != RECEPTOR_CHANGES:
        raise RuntimeError(f'fig3 maps: receptor-rule changes {found} differ from the documented {RECEPTOR_CHANGES}')
    types = c.neurons['type'].fillna('').astype(str).to_numpy()
    same = (types[post] == types[pre]) & (types[post] != '') & (count > 0)
    beyond = np.maximum(count - RULE_CAP, 0.0)
    n = len(a)
    inp = np.bincount(post, weights=count, minlength=n)

    def share(w):
        return np.round(np.divide(np.bincount(post, weights=w, minlength=n), inp, out=np.zeros(n), where=inp > 0), 6)

    rc = pd.DataFrame(dict(cell=np.arange(n), input_synapses=inp.astype(np.int64), receptor_sign_share=share(count * changed),
                           same_type_share=share(count * same), beyond_cap_share=share(beyond)))
    rc['source_file'] = 'flybrain/cache/W_post_pre.npz; flybrain/cache/neurons.parquet; flybrain/flybrain/data/receptors_by_type.csv'
    csvout.write_table(rc, PV + 'fig3_rule_cells.csv', title='Where the connection rules act: the share of each cell\'s input synapses that each rule changes', describe={
        'cell': 'Model cell index', 'input_synapses': 'Input synapses of the cell',
        'receptor_sign_share': 'Share of input synapses whose sign the receptor table sets differently from the presynaptic transmitter',
        'same_type_share': 'Share of input synapses from cells of the same type (scaled by the same-type gain)',
        'beyond_cap_share': f'Share of input synapses beyond {RULE_CAP:.0f} per connection (removed by the default cap)'}, key=['cell'])
    total = count.sum()
    rows = []
    for rule, sel, syn in (('receptor rule', changed, count * changed), ('same-type gain', same, count * same), ('synapse cap', count > RULE_CAP, beyond)):
        rows.append(dict(rule=rule, entries=int(sel.sum()), synapses=float(syn.sum()), share_of_synapses=float(syn.sum() / total),
                         cells=int((np.bincount(post, weights=syn, minlength=n) > 0).sum())))
    rs = pd.DataFrame(rows)
    rs['source_file'] = rc['source_file'].iloc[0]
    csvout.write_table(rs, PV + 'fig3_rules.csv', title='The three connection rules varied in Fig. 3G: connections, synapses and cells they act on', describe={
        'rule': 'Rule family', 'entries': 'Connections (cell pairs) the rule acts on', 'synapses': 'Synapses changed (receptor rule, same-type gain) or removed (cap)',
        'share_of_synapses': 'Share of all synapses', 'cells': 'Cells with at least one input the rule acts on'}, key=['rule'],
        notes=f'Receptor rule gated against the documented {RECEPTOR_CHANGES["entries"]:,} entries ({RECEPTOR_CHANGES["glutamate"]:,} glutamate to excitatory, '
              f'{RECEPTOR_CHANGES["histamine"]:,} histamine silenced).')


# ---------------------------------------------------------------- Figure S1: bit-exact replay on every GPU; wiring at the hardest map

G0_HOSTS = (('g0-report.json', 'host 1', 1), ('g0-report-x6-gpu{}.json', 'host 2', 6), ('g0-report-x4-gpu{}.json', 'host 3', 4))
PLANNING_MODEL = 'the analysed planning model, mixed vs double precision'
# The registered hardest-map controls that change the wiring (studies/brain-sokoban-v1/code/registry.py, backend.py): all start
# from the neutral initialisation; 'neutral' keeps the real wiring, 'weight_shuffle' permutes the weights within sign on the real
# topology, 'rewired' swaps edges keeping every cell's in/out degree and sign
WIRING_CONTROLS = {'neutral': 'real wiring', 'weight_shuffle': 'weights shuffled, wiring kept', 'rewired': 'rewired, degrees kept'}
RUN_STATUS = {'learned_solved': 'learned', 'stagnation': 'stopped without progress'}


def s1_tables(names):
    label = names[names['kind'] == 'model'].set_index('code')['label']
    root = paths.ROOT / 'studies' / 'brain-supplement-v2'
    rows = []
    for pattern, host, n_gpu in G0_HOSTS:
        for k in range(n_gpu):
            f = root / (pattern if n_gpu == 1 else pattern.format(k))
            d = io.json_load(f)
            if not d['passed']:
                raise RuntimeError(f'{f.name}: the replay did not pass')
            p = d['p05']
            rows.append(dict(host=host, gpu=k + 1, model=PLANNING_MODEL, histories=1, histories_identical=int(bool(p['same_actions_mixed_float64'])),
                             max_abs_logit_difference=float(p['max_abs_logit_difference_mixed_float64']), passed=bool(p['ok']), source_file=paths.rel(f)))
            for code, m in d['memory'].items():
                rows.append(dict(host=host, gpu=k + 1, model=label[code], histories=int(m['histories']), histories_identical=int(m['ok_histories']),
                                 max_abs_logit_difference=float(m['max_abs_logit_difference']), passed=bool(m['ok']), source_file=paths.rel(f)))
    rp_ = pd.DataFrame(rows)
    csvout.write_table(rp_, PV + 's1_replay.csv', title='Replay of every analysed model on every GPU of the three training hosts (gate before any run)', describe={
        'host': 'Training host', 'gpu': 'GPU on the host', 'model': 'Analysed model replayed',
        'histories': 'Histories replayed (the planning model: one solve of its map)', 'histories_identical': 'Histories reproducing the recorded actions and logits',
        'max_abs_logit_difference': 'Largest |logit difference| from the recording', 'passed': 'Within the gate tolerance'}, key=['host', 'gpu', 'model'])
    c = _read(SUPP / 'x9_d7_controls.csv')
    c = c[(c['task'] == 'D7-T0') & c['condition'].isin(WIRING_CONTROLS)]
    x = _read(SUPP / 'x10_maps.csv')
    x = x[x['map'] == 'D7-T0']
    if len(c) != len(WIRING_CONTROLS) or len(x) != 6:
        raise RuntimeError('S1: the hardest-map runs are incomplete')
    w = pd.concat([pd.DataFrame(dict(condition='analysed lineage, real wiring', run=x['replicate'].astype(int), status=x['status'].map(RUN_STATUS),
                                     updates=x['updates'].astype(int), solved=x['final_solved'].astype(str).str.lower().eq('true'),
                                     source_file=paths.rel(SUPP / 'x10_maps.csv'))),
                   pd.DataFrame(dict(condition=c['condition'].map(WIRING_CONTROLS), run=0, status=c['status'].map(RUN_STATUS), updates=c['updates'].astype(int),
                                     solved=c['final_solved'].astype(str).str.lower().eq('true'), source_file=paths.rel(SUPP / 'x9_d7_controls.csv')))],
                  ignore_index=True)
    if w['status'].isna().any():
        raise RuntimeError('S1: a hardest-map run without a reader-facing status')
    csvout.write_table(w, PV + 's1_wiring.csv', title='The hardest map (60 optimal moves): the analysed lineage in six sets and the wiring controls from a neutral start', describe={
        'condition': 'Brain: the analysed lineage, or a control from the neutral initialisation', 'run': 'Replicate set (0: the original set) or the one control run',
        'status': 'Outcome of the run', 'updates': 'Parameter updates (to solving, or to the stop)', 'solved': 'The map is solved at the end'}, key=['condition', 'run'])


# ---------------------------------------------------------------- Figure S2: scene 3's state carries its context with the cue

def s2_tables(names):
    """Scene 3 (two board orientations as contexts): every trained history with its own state, with the opposite cue's state
    and with the state of the other context after the same cue (X5 cross; the swap and cross families of tests.csv)."""
    x = _read(SUPP / 'x5_histories.csv')
    x = x[x['unit'] == 'S2']
    sc3 = names[(names['kind'] == 'model') & (names['code'] == 'supp-S2-context-brain')]['label'].iloc[0]
    t = pd.DataFrame(dict(learner=sc3, case=x['case'], cue=x['cue'], context=x['context'], delay_ms=x['delay_ms'], prefix_windows=x['prefix_windows'],
                          own_state_solved=x['baseline_solved'].astype(str).str.lower().eq('true'),
                          opposite_cue_state_solved=x['opposite_cue_solved'].astype(str).str.lower().eq('true'),
                          other_context_same_cue_solved=x['cross_same_cue_solved'].astype(str).str.lower().eq('true')))
    t['source_file'] = paths.rel(SUPP / 'x5_histories.csv')
    csvout.write_table(t, PV + 's2_context.csv', title='Scene 3: trained histories solved with their own state, the opposite cue\'s state and the other context\'s state after the same cue', describe={
        'learner': 'Learner', 'case': 'History (repository case code)', 'cue': 'Cue', 'context': 'Board orientation (context)', 'delay_ms': 'Delay (ms)',
        'prefix_windows': 'Prefix windows', 'own_state_solved': 'Solved with its own state', 'opposite_cue_state_solved': 'Solved with the opposite cue\'s state swapped in',
        'other_context_same_cue_solved': 'Solved with the other context\'s state after the same cue swapped in'}, key=['case', 'context', 'delay_ms', 'prefix_windows'])


# ---------------------------------------------------------------- Figure S3: the scene-2 ensemble; healthy-circuit signatures under the variants

# C4 healthy-circuit signatures of the disease study under the connection-rule variants (phase2_variants_c4). The mushroom-body
# signature is not routed: it fails at same-type gain 1 (class c, inventory D9; STAR Methods)
S3_SIGNATURES = {'c4_network': 'stable network', 'c4_al': 'antennal-lobe inhibition', 'c4_loom': 'looming escape', 'c4_noop': 'monoamine no-op'}


def s3_tables():
    sets = io.json_load(SUPP.parent / 'tasks' / 'x6-x7-sets.json')['sets']['s1']
    groups = [('ensemble', sets['cells'])] + [(f'matched {i + 1}', n['cells']) for i, n in enumerate(sets['nulls'][:FIG3_RENDERED_NULLS])]
    m = pd.concat([pd.DataFrame(dict(ensemble_scene='scene 2', set=name, cell=np.sort(np.asarray(c, np.int64)))) for name, c in groups], ignore_index=True)
    w = io.csv_load(paths.out(PV + 'fig3_set_wiring.csv'))
    w = w[w['ensemble_scene'] == 'scene 2'].set_index('set')
    for name, c in groups:
        if len(c) != int(w.loc[name, 'cells']):
            raise RuntimeError(f'S3: {name} of scene 2 differs from fig3_set_wiring')
    csvout.write_table(m, PV + 's3_band_sets.csv', title='Cells of the scene-2 cue ensemble and of three of its matched sets (Figure S3 band)',
                       describe={'ensemble_scene': 'Scene of the ensemble', 'set': 'The ensemble or a matched set', 'cell': 'Model cell index'}, key=['set', 'cell'])
    v = _read(PATH / 'phase2_variants_c4.csv')
    var = v['variant'].map(VARIANTS)
    if var.isna().any():
        raise RuntimeError('S3: connection-rule variant without a reader-facing name')
    t = pd.DataFrame(dict(variant=var.str[0], rule=var.str[1], code=v['variant'], task_hz=v['task_hz'],
                          **{lab: v[col].astype(str).str.lower().eq('true') for col, lab in S3_SIGNATURES.items()},
                          pn_hz=v['pn_hz'], pn_hz_ln_removed=v['pn_hz_ln_removed'], gf_loom_spikes=v['gf_loom_spikes']))
    t['source_file'] = paths.rel(PATH / 'phase2_variants_c4.csv')
    csvout.write_table(t, PV + 's3_variants.csv', title='The healthy-circuit signatures of the disease study under the seven connection-rule variants', describe={
        'variant': 'Connection rule changed (as built: the defaults)', 'rule': 'Rule family', 'code': 'Repository code of the variant',
        'task_hz': 'Healthy network: mean rate in the task (Hz)', 'stable network': 'Signature reproduced: the healthy network stays stable',
        'antennal-lobe inhibition': 'Signature reproduced: removing antennal-lobe local neurons broadens projection-neuron responses',
        'looming escape': 'Signature reproduced: a looming stimulus drives the giant fibre', 'monoamine no-op': 'Signature reproduced: removing monoaminergic cells leaves the task unchanged',
        'pn_hz': 'Projection-neuron rate to odours (Hz)', 'pn_hz_ln_removed': 'The same with the antennal-lobe local neurons removed (Hz)',
        'gf_loom_spikes': 'Giant fibre spikes to the looming stimulus'}, key=['code'])


# ---------------------------------------------------------------- Figure S4: single-gene stains, mapping QC, SK at strict mapping

ATLAS_NAMES = {'davis2020': 'Davis 2020', 'ozel2021': 'Özel 2021', 'davie2018': 'Davie 2018', 'afca2023': 'AFCA 2023', 'adfca2025': 'AD-FCA 2025'}
S4_MAPPINGS = {'all': 'all cells, any tier', 'all-strict': 'all cells, without hemilineage pairs', 'vnc': 'VNC, type and hemilineage',
               'vnc-strict': 'VNC, without hemilineage pairs'}


def s4_tables():
    """Per cell, the z of each channel gene and the adaptation index of its mapped atlas unit (the single-gene stains); the mapping
    checks of sc_mapping_qc as numbers; and SK enrichment of the scene-1 carriers with and without the hemilineage-pair tier."""
    cells = mapping_v2()
    vals = _read(PATH / 'sc_unit_values_v2.csv')
    vals['unit'] = vals['unit'].astype(str)
    mapped = cells['tier'].ne('unmapped').to_numpy()
    key = (cells['source'].astype(str) + '|' + cells['unit'].astype(str)).where(mapped)
    vk = vals.assign(k=vals['source'] + '|' + vals['unit']).set_index('k')
    g = pd.DataFrame(dict(cell=np.arange(len(cells)), tier=cells['tier'].to_numpy()))
    for gene in CHANNELS:
        g[f'z_{gene}'] = np.where(mapped, vk[f'z_{gene}'].reindex(key).to_numpy(), np.nan).round(4)
    g['adaptation_index'] = np.where(mapped, vk['A'].reindex(key).to_numpy(), np.nan).round(4)
    f4 = io.csv_load(paths.out(PV + 'fig4_cell_expression.csv'))
    for col in ('z_SK', 'z_Shal', 'z_slo', 'adaptation_index'):
        if not np.allclose(g[col].to_numpy(float), f4[col].to_numpy(float), equal_nan=True, rtol=0, atol=1e-9):
            raise RuntimeError(f'S4: {col} disagrees with fig4_cell_expression')
    g['source_file'] = 'runs/sc-mapping/cell_map.parquet; results/sc_unit_values_v2.csv'
    csvout.write_table(g, PV + 's4_cell_genes.csv', title='Per-cell atlas expression of the five channel genes and the adaptation index (mapping v2)', describe={
        'cell': 'Model cell index', 'tier': 'Mapping tier', **{f'z_{x}': f'{x}: z-score of the unit mean among the mapped units of its atlas' for x in CHANNELS},
        'adaptation_index': 'Mean z of the adaptation-index genes (SK, slo, KCNQ, Shab)'}, key=['cell'], notes='SK, Shal, slo and the index gated against fig4_cell_expression.')
    qc_ = _read(PATH / 'sc_mapping_qc.csv').set_index('check')['value'].astype(str)
    rows = []
    for check, val in qc_.items():
        m = re.match(r'transmitter_concordance_(\w+)$', check)
        if m:
            a, b = re.match(r'(\d+)/(\d+)', val).groups()
            rows.append(dict(check=check, kind='transmitter concordance', source=ATLAS_NAMES[m.group(1)], agree=int(a), total=int(b)))
            continue
        m = re.match(r'map_units_present_(\w+)$', check)
        if m:
            a, b, u = re.match(r'(\d+)/(\d+) map rows; (\d+) units', val).groups()
            rows.append(dict(check=check, kind='map rows present', source=ATLAS_NAMES[m.group(1)], agree=int(a), total=int(b), units=int(u)))
            continue
        m = re.match(r'cross_source_(\w+?)_vs_(\w+)$', check)
        if m:
            mp, mr, xp, xr = re.match(r'(\d+) matched unit pairs: median rho ([\d.]+); (\d+) mismatched pairs: median ([\d.]+)', val).groups()
            rows.append(dict(check=check, kind='cross-source agreement', source=ATLAS_NAMES[m.group(1)], compared_with=ATLAS_NAMES[m.group(2)], matched_pairs=int(mp),
                             matched_rho=float(mr), mismatched_pairs=int(xp), mismatched_rho=float(xr)))
            continue
        m = re.match(r'marker_(.+)_(\w+)$', check)
        if m and m.group(2) in ATLAS_NAMES:
            genes, u, z = re.match(r"mean z of \[(.*)\] over (\d+) mapped units: ([\d.-]+)", val).groups()
            rows.append(dict(check=check, kind='marker genes', source=ATLAS_NAMES[m.group(2)], population=m.group(1), genes=genes.replace("'", ''), units=int(u), mean_z=float(z)))
            continue
        if check == 'davie_vs_flybrain_expression_central':
            u, gn, d = re.match(r'(\d+)/\d+ units x (\d+) genes: max abs diff ([\d.e-]+)', val).groups()
            rows.append(dict(check=check, kind='reproduction', source=ATLAS_NAMES['davie2018'], units=int(u), genes=gn, max_abs_diff=float(d)))
    q = pd.DataFrame(rows)
    if (q['kind'] == 'transmitter concordance').sum() != len(ATLAS_NAMES):
        raise RuntimeError('S4: transmitter concordance missing for an atlas')
    q['source_file'] = paths.rel(PATH / 'sc_mapping_qc.csv')
    csvout.write_table(q, PV + 's4_mapping_qc.csv', title='Checks of the atlas-to-connectome mapping (sc_mapping_qc, parsed)', describe={
        'check': 'Check (repository name)', 'kind': 'Check', 'source': 'Atlas', 'compared_with': 'Second atlas (cross-source agreement)', 'population': 'Cell population (marker genes)',
        'agree': 'Map rows agreeing (transmitter) or present', 'total': 'Map rows checked', 'units': 'Atlas units', 'genes': 'Genes (marker genes; reproduction: gene count)',
        'matched_pairs': 'Unit pairs mapped to the same cell type', 'matched_rho': 'Median expression correlation of matched pairs',
        'mismatched_pairs': 'Unit pairs mapped to different cell types', 'mismatched_rho': 'Median correlation of mismatched pairs',
        'mean_z': 'Mean z of the marker genes in the mapped units', 'max_abs_diff': 'Largest difference from the published central-brain expression'},
        key=['check'], sort=['kind', 'source', 'compared_with', 'population'])
    st = _read(PATH / 'sc_trace_v2_strict.csv')
    an = _read(PATH / 'sc_trace_v2_tests.csv')
    an = an[an['test'].astype(str).str.startswith(('m0:all:any:', 'm0:vnc:type_lineage:'))]
    rows = []
    for _, r in pd.concat([an, st]).iterrows():
        model, comp, level, gene, null = str(r['test']).split(':')
        mapping = comp if comp in S4_MAPPINGS else ('vnc' if comp == 'vnc' else comp)
        rows.append(dict(mapping=S4_MAPPINGS[mapping], gene='SK' if gene == 'SK' else 'adaptation index', null='matched sets' if null == 'matched-sets' else 'mapping null',
                         carriers=int(r['n']), coverage=float(r['coverage']), statistic=float(r['statistic']), null_mean=float(r['mean_reference']),
                         null_sd=float(r['null_sd']), z=float((r['statistic'] - r['mean_reference']) / r['null_sd'])))
    s = pd.DataFrame(rows)
    s = s[s['gene'] == 'SK']                                     # the index against the mapping null is class c (M12); SK carries the claim
    s['source_file'] = paths.rel(PATH / 'sc_trace_v2_tests.csv') + '; ' + paths.rel(PATH / 'sc_trace_v2_strict.csv')
    csvout.write_table(s, PV + 's4_sk_mapping.csv', title='SK enrichment of the scene-1 trace carriers with and without the hemilineage-pair tier', describe={
        'mapping': 'Cells and mapping tiers', 'gene': 'Gene', 'null': 'Null distribution', 'carriers': 'Mapped carriers', 'coverage': 'Share of carriers mapped',
        'statistic': 'Mean SK z of the mapped carriers', 'null_mean': 'Mean of the null', 'null_sd': 'SD of the null', 'z': 'Carriers above the null mean (null SDs)'},
        key=['mapping', 'null'])


# ---------------------------------------------------------------- Figure S5: the decision-time pattern at all seven delays

S5_DELAYS, S5_PREFIX = (0, 20, 40, 80, 160, 320, 640), 4


def s5_tables():
    """The per-cell cue difference at the first decision at all seven delays (prefix 4; scene 1, readout trained), from the
    free-run records of fu4 as Fig. 5A, gated against fu4's decision energies and the four columns of fig5_decision_cells."""
    model = 'remote-dynamic-M0-head_only'
    fe = io.csv_load(paths.out('followup/fu4_decision_energy.csv'), low_memory=False)
    cells = {}
    for d in S5_DELAYS:
        folder, meta = _record_meta(model, d, S5_PREFIX)
        z = io.npz_load(folder / f'free-after-d{d}-p{S5_PREFIX}.npz', keys=['mean'])['mean']
        s = _first_decision(meta)
        i0, i1 = _record_pairs(meta)[(0, 'hidden')]
        dv = z[s, i1, 2].astype(np.float64) - z[s, i0, 2].astype(np.float64)
        want = float(fe[(fe['model'] == model) & (fe['visibility'] == 'hidden') & (fe['delay_ms'] == d) & (fe['prefix_windows'] == S5_PREFIX)]['energy_v'].iloc[0])
        if not np.isclose(float(dv @ dv), want, rtol=1e-9, atol=0):
            raise RuntimeError(f'S5: the decision-time cue difference at d{d} disagrees with fu4')
        cells[f'dv_d{d}_p{S5_PREFIX}_mv'] = np.round(dv, 6)
    dc = pd.DataFrame({'cell': np.arange(len(next(iter(cells.values())))), **cells})
    f5 = io.csv_load(paths.out(PV + 'fig5_decision_cells.csv'))
    for col in f5.columns.drop('cell'):
        if not np.allclose(dc[col].to_numpy(float), f5[col].to_numpy(float), rtol=0, atol=1e-9):
            raise RuntimeError(f'S5: {col} disagrees with fig5_decision_cells')
    csvout.write_table(dc, PV + 's5_decision_cells.csv', title='Per-cell cue difference of the membrane potential at the first decision at all seven delays (prefix 4; scene 1, readout trained)', describe={
        'cell': 'Model cell index', **{k: f'Delay {k.split("_")[1][1:]} ms, {S5_PREFIX} prefix windows: cue 1 minus cue 0, first decision window mean (mV)' for k in cells}},
        key=['cell'], notes='Gated against followup/fu4_decision_energy.csv and the four columns of fig5_decision_cells.csv.')
    ref = dc[f'dv_d160_p{S5_PREFIX}_mv'].to_numpy(float)
    cos = pd.DataFrame([dict(delay_ms=d, cosine_with_160=float(dc[f'dv_d{d}_p{S5_PREFIX}_mv'] @ ref / np.sqrt((dc[f'dv_d{d}_p{S5_PREFIX}_mv'] @ dc[f'dv_d{d}_p{S5_PREFIX}_mv']) * (ref @ ref))),
                              cells_above_0p3_mv=int((dc[f'dv_d{d}_p{S5_PREFIX}_mv'].abs() > 0.3).sum())) for d in S5_DELAYS])
    csvout.write_table(cos, PV + 's5_decision_cosines.csv', title='The decision-time pattern at each delay against the trained 160-ms pattern (prefix 4)', describe={
        'delay_ms': 'Delay (ms)', 'cosine_with_160': 'Cosine with the 160-ms pattern over all cells', 'cells_above_0p3_mv': 'Cells with |cue difference| > 0.3 mV'},
        key=['delay_ms'])
    # per superclass, as sums: summed over any set of superclasses they give that set's cosine with the 160-ms pattern
    sup = io.parquet_load(paths.M1 / 'neurons' / 'annotations.parquet', columns=['superclass'])['superclass'].fillna('unknown').astype(str).to_numpy()
    if len(sup) != len(dc):
        raise RuntimeError('S5: the annotations and the cell table differ in length')
    whole, rows = cos.set_index('delay_ms')['cosine_with_160'], []
    for d in S5_DELAYS:
        v = dc[f'dv_d{d}_p{S5_PREFIX}_mv'].to_numpy(float)
        s = pd.DataFrame(dict(superclass=sup, dot=v * ref, sq=v * v)).groupby('superclass').agg(cells=('dot', 'size'), dot=('dot', 'sum'), sq=('sq', 'sum'))
        if not np.isclose(s['dot'].sum() / np.sqrt(s['sq'].sum() * (ref @ ref)), whole.loc[d], rtol=1e-9, atol=0):
            raise RuntimeError(f'S5: the superclass sums at d{d} do not rebuild the whole-CNS cosine')
        rows += [dict(superclass=k, delay_ms=d, cells=int(r['cells']), dot_with_160_mv2=float(r['dot']), sum_sq_mv2=float(r['sq'])) for k, r in s.iterrows()]
    csvout.write_table(pd.DataFrame(rows), PV + 's5_decision_superclass.csv', title='The decision-time cue difference per superclass at each delay, as sums that rebuild cosines with the 160-ms pattern (prefix 4)', describe={
        'superclass': 'Superclass (neuron annotations)', 'delay_ms': 'Delay (ms)', 'cells': 'Cells in the superclass',
        'dot_with_160_mv2': 'Sum over the superclass of the cue difference at this delay times the cue difference at 160 ms (mV^2)',
        'sum_sq_mv2': 'Sum over the superclass of the squared cue difference at this delay (mV^2)'},
        key=['superclass', 'delay_ms'], notes='Summed over all superclasses they rebuild s5_decision_cosines.csv (gated).')


# ---------------------------------------------------------------- Figure S6: routes at every level, the robust-training draws

S6_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)        # the renders: draw 0 and hubs first at every 10%
DY2_BOOT, DY2_SEED = 10000, 20261020                             # analyze_dy2.py: bootstrap of the geometric-mean ratios


def _dy2_bootstrap(models, cond):
    """The bootstrap of the two ratios of training across threshold states, replayed in analyze_dy2.py's draw order (C01
    ED50, then removal ED50; a ratio with no positive ED50 on either side is skipped) and gated against dy2_conditions.csv.
    A table that starts at the reported condition resumes the generator from its state at that condition's first draw
    (declared/dy2_bootstrap_state.json, recorded from the complete table), so that both give the same draws."""
    x1 = models[models['group'] == 'x1']
    rng = np.random.default_rng(DY2_SEED)
    if cond['condition'].iloc[0] == ROBUST_CONDITION:
        rng.bit_generator.state = io.json_load(DECLARED / 'dy2_bootstrap_state.json')
    boots = {}
    for c in cond['condition']:
        g = models[(models['group'] == 'dy2') & (models['condition'] == c)]
        for name, col in (('access window', 'ed50_c01_mv'), ('cell-loss tolerance', 'ed50_removal_share')):
            x, y = g[col].astype(float).to_numpy(), x1[col].astype(float).to_numpy()
            x, y = x[x > 0], y[y > 0]
            if not len(x) or not len(y):
                continue
            b = np.array([np.exp(np.log(rng.choice(x, len(x))).mean() - np.log(rng.choice(y, len(y))).mean()) for _ in range(DY2_BOOT)])
            if c == ROBUST_CONDITION:
                want = cond[cond['condition'] == c].iloc[0]
                pre = 'ed50_c01_ratio' if col == 'ed50_c01_mv' else 'ed50_removal_ratio'
                est = float(np.exp(np.log(x).mean() - np.log(y).mean()))
                got = (est, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
                if not np.allclose(got, [float(want[pre]), float(want[pre + '_lo']), float(want[pre + '_hi'])], rtol=1e-9, atol=0):
                    raise RuntimeError(f'S6: the replayed {name} bootstrap disagrees with dy2_conditions')
                boots[name] = b
        if c == ROBUST_CONDITION:
            break
    return boots


def s6_tables():
    """Figure S6, three parts.
    (1) The route network at every removal level of phase4_graph.py, random (each of its three seeded draws) and hubs first
        (its degree order): reachability and three-step walks gated against phase4_graph_percolation.csv row by row; the
        route cells and the readout cells by shortest route; per cell, the route traffic of draw 0 and of the hubs-first order
        at every 10% (gated against fig6_routes at 30, 60 and 90%).
    (2) Memory under each of the two held-out jitter draws per replicate, gated against fig6_robust.
    (3) The bootstrap distributions of the access-window and cell-loss ratios, replayed from analyze_dy2.py's seed and draw
        order and gated against dy2_conditions.csv."""
    W = io.sparse_load(paths.ROOT / 'flybrain' / 'cache' / 'W_post_pre.npz').tocsr()
    W.eliminate_zeros()
    n = W.shape[0]
    A = (W != 0).astype(np.float64).tocsr()                     # rows post: A @ x moves activity downstream
    AT = A.T.tocsr()
    ann = io.parquet_load(paths.M1 / 'neurons' / 'annotations.parquet', columns=['direct_input', 'observed_by_policy'])
    inputs = np.flatnonzero(ann['direct_input'].to_numpy(bool))
    readout = np.flatnonzero(ann['observed_by_policy'].to_numpy(bool))
    s, r = np.zeros(n), np.zeros(n)
    s[inputs], r[readout] = 1, 1

    def route(alive):
        al = alive.astype(np.float64)
        f1 = (A @ s) * al
        f2 = (A @ f1) * al
        b1 = (AT @ r) * al
        b2 = (AT @ b1) * al
        t = f1 * b1 + f1 * b2 + f2 * b1                          # walks of length 2 and 3 through the cell (as an intermediate)
        t[inputs], t[readout] = 0, 0
        return t, float(((A @ f2) * al)[readout].sum())

    def level(order, draw, q, m, alive, t, w3):
        h = _hops(A, inputs, alive)[readout]
        reach = _reach(A, inputs, readout, alive)
        if not np.isclose(float((h >= 0).mean()), reach, rtol=0, atol=1e-12):
            raise RuntimeError('S6: route lengths disagree with reachability')
        return dict(order=order, draw=draw, share_removed=q, cells_removed=m, readout_reachable=reach, walks3_relative=w3 / base3,
                    route_cells=int((t > 0).sum()), readout_direct=int((h == 1).sum()), readout_two=int((h == 2).sum()),
                    readout_three=int((h == 3).sum()), readout_four_plus=int((h >= 4).sum()), readout_unreachable=int((h < 0).sum()))

    perc = _read(PATH / 'phase4_graph_percolation.csv')
    rng = np.random.default_rng(GRAPH_SEED)
    pool = np.setdiff1d(np.arange(n), np.concatenate([inputs, readout]))
    degree = np.asarray(A.sum(0)).ravel() + np.asarray(A.sum(1)).ravel()
    hubs = pool[np.argsort(-degree[pool], kind='mergesort')]
    intact, base3 = route(np.ones(n, bool))
    maps, rows = {'intact': intact}, [level('intact', -1, 0.0, 0, np.ones(n, bool), intact, base3)]
    for q in GRAPH_SHARES:
        m = int(round(q * len(pool)))
        draws = [rng.choice(pool, m, replace=False) for _ in range(GRAPH_DRAWS)]
        for order, draw, removed in [('random', d, draws[d]) for d in range(GRAPH_DRAWS)] + [('hubs first', -1, hubs[:m])]:
            alive = np.ones(n, bool)
            alive[removed] = False
            t, w3 = route(alive)
            row = level(order, draw, q, m, alive, t, w3)
            want = perc[(perc['order'] == ('random' if order == 'random' else 'degree (hubs first)')) & np.isclose(perc['share_removed'].astype(float), q)]
            if order == 'random':
                want = want[want['draw'].astype(str).isin([str(draw), f'{draw}.0'])]
            if len(want) != 1 or not (np.isclose(row['readout_reachable'], float(want['readout_reachable'].iloc[0]), rtol=0, atol=1e-12)
                                      and np.isclose(row['walks3_relative'], float(want['walks3_relative'].iloc[0]), rtol=1e-9)):
                raise RuntimeError(f'S6: {order} draw {draw} at {q} disagrees with phase4_graph_percolation')
            rows.append(row)
            if (draw == 0 or order == 'hubs first') and np.isclose(q, S6_LEVELS).any():
                maps[f'{"random" if order == "random" else "hubs"}_{int(round(100 * q))}'] = t
    lv = pd.DataFrame(rows)
    lv['source_file'] = paths.rel(PATH / 'phase4_graph_percolation.csv')
    csvout.write_table(lv, PV + 's6_routes_levels.csv', title='The route network from the visual input cells to the readout cells at every removal level: random (three seeded draws) and hubs first', describe={
        'order': 'intact, random or hubs first (cells in descending order of connections)', 'draw': 'Random draw (0-2); -1 for the intact graph and the hubs-first order',
        'share_removed': 'Share of the other cells removed', 'cells_removed': 'Cells removed', 'readout_reachable': 'Share of readout cells reachable from the inputs',
        'walks3_relative': 'Three-step walks relative to the intact graph', 'route_cells': 'Cells on an input-to-readout walk of length 2-3',
        'readout_direct': 'Readout cells with a direct input synapse', 'readout_two': 'Readout cells whose shortest route has 2 synapses',
        'readout_three': 'Readout cells whose shortest route has 3 synapses', 'readout_four_plus': 'Readout cells whose shortest route has 4 or more synapses',
        'readout_unreachable': 'Readout cells not reachable'}, key=['order', 'draw', 'share_removed'],
        notes='Seed, draw order and degree order of phase4_graph.py; reachability and three-step walks gated against phase4_graph_percolation.csv for every row.')
    keys = ['intact'] + [f'random_{int(round(100 * q))}' for q in S6_LEVELS] + [f'hubs_{int(round(100 * q))}' for q in S6_LEVELS]
    ids = np.arange(n)
    cells = pd.DataFrame(dict(cell=ids, role=np.where(np.isin(ids, inputs), 'input', np.where(np.isin(ids, readout), 'readout', 'other'))))
    for k in keys:
        whole = np.rint(maps[k])
        if not np.array_equal(whole, maps[k]):
            raise RuntimeError('S6: route traffic is not a whole number of walks')
        cells[f'traffic_{k}'] = whole.astype(np.int64)
    f6 = io.csv_load(paths.out(PV + 'fig6_routes.csv')).set_index('cell')
    for a, b in (('intact', '0'), ('random_30', '30'), ('random_60', '60'), ('random_90', '90')):
        if not np.array_equal(cells.loc[f6.index, f'traffic_{a}'].to_numpy(float), f6[f'traffic_{b}'].to_numpy(float)):
            raise RuntimeError(f'S6: traffic {a} disagrees with fig6_routes')
    cells = cells[(cells[[f'traffic_{k}' for k in keys]].sum(axis=1) > 0) | cells['role'].ne('other')].reset_index(drop=True)
    cells['source_file'] = 'flybrain/cache/W_post_pre.npz; studies/brain-sokoban-v1/mechanisms-v1/neurons/annotations.parquet'
    csvout.write_table(cells, PV + 's6_routes.csv', title='Route traffic per cell, intact and at every 10% of removal: random draw 0 and hubs first', describe={
        'cell': 'Model cell index', 'role': 'input (visual input cell), readout (observed by the readout) or other',
        'traffic_intact': 'Input-to-readout walks of length 2 and 3 through the cell, intact graph',
        **{f'traffic_random_{k}': f'The same, {k}% of the other cells removed at random (draw 0)' for k in (int(round(100 * q)) for q in S6_LEVELS)},
        **{f'traffic_hubs_{k}': f'The same, {k}% removed hubs first' for k in (int(round(100 * q)) for q in S6_LEVELS)}},
        key=['cell'], notes='Gated against fig6_routes.csv (intact, 30, 60, 90%). Only cells on a route or with a role are listed.')
    # the two held-out jitter draws per replicate
    models = _read(PATH / 'dy2_models.csv')
    panel = _read(PATH / 'dy2_panel.csv')
    rb = io.csv_load(paths.out(PV + 'fig6_robust.csv')).set_index(['training', 'replicate'])
    keep = models[(models['group'] == 'x1') | ((models['group'] == 'dy2') & (models['condition'] == ROBUST_CONDITION))]
    held = []
    for _, m in keep.iterrows():
        tr = ONE_STATE if m['group'] == 'x1' else ACROSS
        p = panel[panel['model'] == m['model']].set_index('evaluation')['trained_net_use']
        v = [float(p[f'{ROBUST_CONDITION}-s{k}']) for k in range(2)]
        if not np.isclose(np.mean(v), float(rb.loc[(tr, int(m['replicate'])), 'memory_under_jitter']), rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"S6: held-out draws of {m['model']} disagree with fig6_robust")
        held += [dict(training=tr, replicate=int(m['replicate']), heldout_draw=k + 1, memory_under_jitter=v[k]) for k in range(2)]
    hd = pd.DataFrame(held)
    hd['source_file'] = paths.rel(PATH / 'dy2_panel.csv')
    csvout.write_table(hd, PV + 's6_heldout_draws.csv', title='Memory under each held-out threshold-jitter draw (sigma 1 uV), per replicate: training at one state and across states', describe={
        'training': 'one state, or across states (threshold jitter, sigma 1 uV)', 'replicate': 'Replicate',
        'heldout_draw': 'Held-out jitter draw (1: thresholds centred on the lower quartile of the training jitter; 2: the upper)',
        'memory_under_jitter': 'Trained net use under the draw'}, key=['training', 'replicate', 'heldout_draw'],
        notes='Their mean per replicate is memory_under_jitter of fig6_robust.csv (gated).')
    # the bootstrap of the two ratios, replayed
    boots = _dy2_bootstrap(models, _read(PATH / 'dy2_conditions.csv'))
    bt = pd.DataFrame([dict(quantity=k, resample=i, ratio=float(v)) for k, b in boots.items() for i, v in enumerate(b)])
    bt['source_file'] = paths.rel(PATH / 'dy2_models.csv')
    csvout.write_table(bt, PV + 's6_bootstrap.csv', title='Bootstrap of the across-states / one-state ratios (geometric means): access window (C01 ED50) and cell-loss tolerance (removal ED50)', describe={
        'quantity': 'access window or cell-loss tolerance', 'resample': 'Bootstrap resample (10,000; replicates resampled within each training arm)',
        'ratio': 'Geometric-mean ratio, across states / one state'}, key=['quantity', 'resample'],
        notes='Replayed from analyze_dy2.py (seed 20261020, its draw order); estimate and 95% interval gated against dy2_conditions.csv.')


# ---------------------------------------------------------------- Figure S7: disease in detail

S7_BRAINS = {'p05': 'the model of the disease study', 'm0joint': 'scene 1, brain and readout trained'}   # names as fig7_aging
S7_LEARNERS = {'M0-head_only': 'readout', 'M0-joint': 'brain and readout'}
VULNERABLE_SHARE = 0.01                                                # sc_annotation.py
DAVIE_NON_NEURONAL = ('Ensheathing_glia', 'Astrocyte-like', 'Perineurial_glia', 'Cortex_glia', 'Subperineurial_glia', 'Chiasm_glia',
                      'Plasmatocytes', 'Photoreceptors', 'Hsp')        # sc_aging.py


def s7_tables():
    """Figure S7. Every table is gated against the record Fig. 7 or the study already uses.
    (1) Healthy-circuit signatures (C4) on both brains, per odour (gated against the G2 shortlist's means).
    (2) Regime A for every retained case and the healthy brain (trained histories; the matched random lesions are not routed:
        the targeted-versus-random contrast is class c, P15), gated against fig7_lost_access.
    (3) Olfactory memory by delay for every case (readout accuracies; random lesions not routed), gated against fig7_profiles.
    (4) Regime B per dose and learner with the healthy replicates, gated against fig7_lost_access; the design's power curve.
    (5) The C01-vulnerable cells as sc_annotation.py defines them, gated against sc_annotation_sets.
    (6) Task rates per cell at both doses of the uniform lesion and the Shal pattern at dose 1, gated against fig7_cells.
    (7) The two Shal weightings per cell (class means, the primary; global mean, the variant), gated against fig7_cells.
    (8) Leave-one-superclass-out of the Shal pattern (descriptive; its equivalence test is class c, P17).
    (9) The ageing operator's expression source, Davie 2018 per unit (AFCA is not routed: its Kenyon-cell direction is class c, P18)."""
    # (1) signatures
    g2 = io.json_load(PATH.parent / 'g2-shortlist.json')['c4']
    sig = []
    for b, name in S7_BRAINS.items():
        c4 = io.json_load(PATH.parent / 'runs' / 'phase2-credibility' / b / 'c4.json')
        sm, apl, ln = c4['smell'], c4['apl_removed'], c4['ln_removed']
        got = dict(kc_fraction=np.mean(sm['kc_fraction_responding']), kc_fraction_apl_removed=np.mean(apl['kc_fraction_responding']),
                   kc_overlap=sm['kc_pair_overlap'], kc_overlap_apl_removed=apl['kc_pair_overlap'], pn_hz=np.mean(sm['pn_mean_hz']),
                   pn_hz_ln_removed=np.mean(ln['pn_mean_hz']), pn_ls=sm['pn_lifetime_sparseness_median'], pn_ls_ln_removed=ln['pn_lifetime_sparseness_median'])
        for k, v in got.items():
            if not np.isclose(float(v), float(g2[b][k]), rtol=1e-9, atol=0):
                raise RuntimeError(f'S7: C4 {k} of {b} disagrees with the G2 shortlist')
        for cond, rec in (('intact', sm), ('APL removed', apl), ('local neurons removed', ln)):
            for i, od in enumerate(rec['odours']):
                sig += [dict(brain=name, condition=cond, measure='KC fraction responding', odour=od, value=float(rec['kc_fraction_responding'][i])),
                        dict(brain=name, condition=cond, measure='PN mean rate (Hz)', odour=od, value=float(rec['pn_mean_hz'][i]))]
            sig += [dict(brain=name, condition=cond, measure='KC pair overlap', odour='all', value=float(rec['kc_pair_overlap'])),
                    dict(brain=name, condition=cond, measure='PN lifetime sparseness (median)', odour='all', value=float(rec['pn_lifetime_sparseness_median']))]
        for r in c4['loom']:
            sig.append(dict(brain=name, condition=r['condition'], measure='giant-fibre spikes', odour='none', value=float(r['gf_spikes'])))
    sg = pd.DataFrame(sig)
    sg['source_file'] = 'studies/brain-pathology-v1/runs/phase2-credibility/<brain>/c4.json'
    csvout.write_table(sg, PV + 's7_signatures.csv', title='Healthy-circuit signatures (C4) on both brains, per odour', describe={
        'brain': 'Model brain', 'condition': 'intact, APL removed, local neurons removed, or the looming / blank stimulus',
        'measure': 'KC fraction responding, PN mean rate, KC pair overlap, PN lifetime sparseness or giant-fibre spikes',
        'odour': 'Odour (all: one value per condition; none: looming)', 'value': 'Value'}, key=['brain', 'condition', 'measure', 'odour'],
        notes='Means gated against g2-shortlist.json (c4).')
    # (2) regime A
    la = io.csv_load(paths.out(PV + 'fig7_lost_access.csv'))
    a = _read(PATH / 'phase3_regime_a.csv')
    a = a[(a['split'] == 'train') & ~a['condition'].astype(str).str.endswith('-random')].copy()
    a['dose'] = np.where(a['condition'] == 'healthy', 0.0, a['dose'].astype(float))
    a['case'] = np.where(a['condition'] == 'healthy', 'healthy', a['case'].astype(str))
    for tp, model in (('readout', 'remote-dynamic-M0-head_only'), ('brain and readout', 'remote-dynamic-M0-joint')):
        want = la[(la['regime'] == 'lesion after learning') & (la['trained_parameters'] == tp)].set_index('dose')['trained_solved']
        got = a[(a['model'] == model) & a['case'].isin(['C01', 'healthy'])].set_index('dose')['solved']
        if not all(int(got.loc[d]) == int(want.loc[d]) for d in want.index):
            raise RuntimeError(f'S7: regime A of {model} disagrees with fig7_lost_access')
    ra_ = a[['condition', 'case', 'dose', 'model', 'histories', 'solved', 'net_memory_use']].copy()
    ra_['source_file'] = paths.rel(PATH / 'phase3_regime_a.csv')
    csvout.write_table(ra_, PV + 's7_regime_a.csv', title='Regime A for every retained case: trained histories solved after the lesion, five memory models', describe={
        'condition': 'Lesion and dose (healthy: none)', 'case': 'Retained case (C01, C04, C13, C16) or healthy', 'dose': 'Dose (0: healthy)', 'model': 'Memory model',
        'histories': 'Trained histories', 'solved': 'Trained histories solved', 'net_memory_use': 'Net memory use over the trained timings'},
        key=['condition', 'model'], notes='The matched random lesions are not routed (class c, P15). C01 and healthy gated against fig7_lost_access.')
    # (3) olfactory memory by delay
    o = _read(PATH / 'phase3_olfactory.csv')
    o = o[~o['condition'].astype(str).str.endswith('-random')].copy()
    pf = io.csv_load(paths.out(PV + 'fig7_profiles.csv'))
    h = o[(o['condition'] == 'healthy') & (o['readout'] == 'mbon')].set_index(['pair', 'delay_ms'])['accuracy']
    for r in pf.itertuples():
        q = o[(o['case'] == r.code) & np.isclose(o['dose'].astype(float), float(r.dose)) & (o['readout'] == 'mbon')].set_index(['pair', 'delay_ms'])['accuracy']
        if len(q) != len(h) or not np.isclose(float((q - h.loc[q.index]).mean()), float(r.olfaction_mbon_accuracy_change), rtol=1e-9, atol=1e-12):
            raise RuntimeError(f'S7: olfactory memory of {r.code} dose {r.dose} disagrees with fig7_profiles')
    o['dose'] = np.where(o['condition'] == 'healthy', 0.0, o['dose'].astype(float))
    o['case'] = np.where(o['condition'] == 'healthy', 'healthy', o['case'].astype(str))
    ol = o[['condition', 'case', 'dose', 'pair', 'delay_ms', 'readout', 'accuracy', 'dprime']].copy()
    ol['source_file'] = paths.rel(PATH / 'phase3_olfactory.csv')
    csvout.write_table(ol, PV + 's7_olfactory.csv', title='Olfactory memory by delay for every retained case and the healthy brain: odour-pair discrimination by four readouts', describe={
        'condition': 'Lesion and dose (healthy: none)', 'case': 'Retained case or healthy', 'dose': 'Dose (0: healthy)', 'pair': 'Odour pair', 'delay_ms': 'Delay (ms)',
        'readout': 'Readout (mbon: mushroom-body output neurons; kc; pn; readout512: the task readout)', 'accuracy': 'Discrimination accuracy', 'dprime': "d'"},
        key=['condition', 'pair', 'delay_ms', 'readout'], notes='The matched random lesions are not routed (class c, P15). MBON means gated against fig7_profiles.')
    # (4) regime B per dose and learner, with the healthy replicates, and the power of the design
    c = _read(SUPP / 'campaign_runs.csv')
    rb = []
    for r in c.itertuples():
        mb = re.match(r'b-(C\d+)-dose([\d.]+)-(M0-head_only|M0-joint)-r(\d+)$', r.job)
        mx = re.match(r'x1-(M0-head_only|M0-joint)-r(\d+)$', r.job)
        if mb:
            rb.append(dict(case=mb.group(1), dose=float(mb.group(2)), trained_parameters=S7_LEARNERS[mb.group(3)], replicate=int(mb.group(4)),
                           trained_solved=int(r.trained_solved), trained_histories=int(r.trained_histories)))
        elif mx:
            rb.append(dict(case='healthy', dose=0.0, trained_parameters=S7_LEARNERS[mx.group(1)], replicate=int(mx.group(2)),
                           trained_solved=int(r.trained_solved), trained_histories=int(r.trained_histories)))
    rb = pd.DataFrame(rb)
    for tp in S7_LEARNERS.values():
        want = la[(la['regime'] == 'learning under the lesion') & (la['trained_parameters'] == tp)]
        for d, q in want.groupby('dose'):
            got = rb[(rb['case'] == 'C01') & (rb['trained_parameters'] == tp) & np.isclose(rb['dose'], d)]
            if sorted(got['trained_solved']) != sorted(q['trained_solved'].astype(int)):
                raise RuntimeError(f'S7: regime B ({tp}, dose {d}) disagrees with fig7_lost_access')
    rb['source_file'] = paths.rel(SUPP / 'campaign_runs.csv')
    csvout.write_table(rb, PV + 's7_regime_b.csv', title='Regime B per dose and learner: trained histories solved after learning under the lesion, with the healthy replicates', describe={
        'case': 'Retained case or healthy', 'dose': 'Dose (0: healthy)', 'trained_parameters': 'readout or brain and readout', 'replicate': 'Replicate',
        'trained_solved': 'Trained histories solved', 'trained_histories': 'Trained histories'}, key=['case', 'dose', 'trained_parameters', 'replicate'],
        notes='C01 gated against fig7_lost_access; the per-dose tests are class c (P15) and not routed.')
    pw = io.json_load(PATH.parent / 'amendments' / 'regime-b-n.json')
    cur = pd.DataFrame(pw['curves']['solved']['curve'])[['n', 'power', 'power_ci_low', 'power_ci_high', 'n_sim', 'alpha']]
    cur['target_drop_histories'] = float(pw['target_solved_drop'])
    cur['target_power'] = float(pw['curves']['solved']['target'])
    cur['source_file'] = 'studies/brain-pathology-v1/amendments/regime-b-n.json'
    csvout.write_table(cur, PV + 's7_power.csv', title='Regime B design: power to detect a 2.7-history (15-percentage-point) drop by replicates per dose and learner', describe={
        'n': 'Replicates per dose and learner', 'power': 'Simulated power', 'power_ci_low': '95% interval, low', 'power_ci_high': '95% interval, high',
        'n_sim': 'Simulations', 'alpha': 'Significance level', 'target_drop_histories': 'Drop to detect (trained histories of 18)', 'target_power': 'Target power'},
        key=['n'])
    # (5) the C01-vulnerable cells
    cm_ = io.parquet_load(SC_RUNS / 'sc-mapping' / 'cell_map.parquet')
    n = len(cm_)
    ens = io.json_load(SUPP.parent / 'tasks' / 'x6-x7-sets.json')['sets']
    typ = cm_['type'].fillna('').astype(str)
    relevant = cm_['delta_adapt_m0'].fillna(0).ne(0).to_numpy() | cm_['observed_by_policy'].to_numpy(bool) | typ.str.match(r'ER\d').to_numpy()
    for k in ('m0', 's1'):
        relevant[np.asarray(ens[k]['cells'], int)] = True
    idx = pd.Series(np.arange(n), index=cm_['bodyId'].to_numpy(np.int64))
    rates = io.parquet_load(SC_RUNS / 'sc-vulnerable' / 'rates.parquet')
    if not np.array_equal(rates['bodyId'].to_numpy(np.int64), cm_['bodyId'].to_numpy(np.int64)):
        raise RuntimeError('S7: the rate table is not in model cell order')
    change = np.abs(rates['rate_c01_dose1'].to_numpy() - rates['rate_healthy'].to_numpy())
    k = int(round(VULNERABLE_SHARE * relevant.sum()))
    top = np.flatnonzero(relevant)[np.argsort(-change[relevant], kind='mergesort')[:k]]
    sets = _read(PATH / 'sc_annotation_sets.csv').set_index('cell_set')
    want = sets.loc['C01-vulnerable cells (top 1% of memory-relevant)']
    comp = '; '.join(f'{s} {c}' for s, c in cm_.iloc[np.sort(top)]['superclass'].fillna('NA').value_counts().head(6).items())   # cell order, as sc_annotation
    if int(want['cells']) != len(top) or comp != str(want['superclasses']):
        raise RuntimeError('S7: the C01-vulnerable cells disagree with sc_annotation_sets')
    vu = pd.DataFrame(dict(cell=top, superclass=cm_.iloc[top]['superclass'].fillna('unknown').to_numpy(), type=typ.iloc[top].to_numpy(),
                           transmitter=cm_.iloc[top]['nt'].fillna('').astype(str).to_numpy(), rate_healthy_hz=rates['rate_healthy'].to_numpy()[top],
                           rate_uniform_dose1_hz=rates['rate_c01_dose1'].to_numpy()[top])).sort_values('cell').reset_index(drop=True)
    vu['source_file'] = 'runs/sc-mapping/cell_map.parquet; runs/sc-vulnerable/rates.parquet; studies/brain-supplement-v2/tasks/x6-x7-sets.json'
    csvout.write_table(vu, PV + 's7_vulnerable.csv', title='The C01-vulnerable cells: the 1% of memory-relevant cells with the largest rate change under the uniform lesion at dose 1', describe={
        'cell': 'Model cell index', 'superclass': 'Superclass', 'type': 'Cell type', 'transmitter': 'Transmitter (EM prediction)', 'rate_healthy_hz': 'Task rate, healthy (Hz)',
        'rate_uniform_dose1_hz': 'Task rate, uniform lesion dose 1 (Hz)'}, key=['cell'],
        notes='Memory-relevant: trace carriers (M0), the scene-1 and scene-2 cue ensembles, readout cells and ring neurons (sc_annotation.py); count and composition gated against sc_annotation_sets.')
    # (6) task rates per cell and (7) the two Shal weightings
    f7 = io.csv_load(paths.out(PV + 'fig7_cells.csv'))
    w = {}
    for kind in ('class', 'global'):
        arr = np.full(n, np.nan)
        for val, cells_ in io.json_load(PATH.parent / 'tasks' / 'sc-kv4-conditions.json')['weights'][kind]:
            arr[np.asarray(cells_, int)] = float(val)
        if np.isnan(arr).any() or not np.isclose(arr.mean(), 1.0, atol=1e-6):
            raise RuntimeError(f'S7: {kind} weights incomplete or not mean 1')
        w[kind] = arr
    checks = ((rates['rate_healthy'], f7['rate_healthy_hz']), (rates['rate_c01_dose1'], f7['rate_uniform_dose1_hz']),
              (rates['rate_kv4_class_dose1'], f7['rate_shal_dose1_hz']), (w['class'], f7['kv4_weight']))
    if not all(np.allclose(np.asarray(x, float), np.asarray(y, float), rtol=1e-12, atol=0) for x, y in checks):
        raise RuntimeError('S7: per-cell rates or weights disagree with fig7_cells')
    pc = pd.DataFrame(dict(cell=np.arange(n), rate_healthy_hz=rates['rate_healthy'].to_numpy(), rate_uniform_dose0p5_hz=rates['rate_c01_dose0.5'].to_numpy(),
                           rate_uniform_dose1_hz=rates['rate_c01_dose1'].to_numpy(), rate_shal_dose1_hz=rates['rate_kv4_class_dose1'].to_numpy(),
                           weight_class_means=w['class'], weight_global_mean=w['global'], vulnerable=np.isin(np.arange(n), top)))
    pc['source_file'] = 'runs/sc-vulnerable/rates.parquet; tasks/sc-kv4-conditions.json'
    csvout.write_table(pc, PV + 's7_cells.csv', title='Per model cell: task rates healthy, under the uniform Kv4-loss lesion at 0.5 and 1, and the Shal-patterned lesion at 1; the two Shal weightings', describe={
        'cell': 'Model cell index', 'rate_healthy_hz': 'Task rate, healthy (Hz)', 'rate_uniform_dose0p5_hz': 'Task rate, uniform lesion dose 0.5 (Hz)',
        'rate_uniform_dose1_hz': 'Task rate, uniform lesion dose 1 (Hz)', 'rate_shal_dose1_hz': 'Task rate, Shal-patterned lesion (class means) dose 1 (Hz)',
        'weight_class_means': 'Lesion weight from Shal, unmapped cells at their class mean (primary; mean 1)',
        'weight_global_mean': 'Lesion weight from Shal, unmapped cells at the global mean (variant; mean 1)', 'vulnerable': 'A C01-vulnerable cell'},
        key=['cell'], notes='Healthy, uniform dose 1, Shal dose 1 and the class weights gated against fig7_cells.')
    # (8) leave-one-superclass-out (descriptive)
    kc_ = _read(PATH / 'sc_kv4_conditions.csv')
    lo = kc_[kc_['condition'].astype(str).str.startswith('loo-') | (kc_['condition'] == 'kv4-class-dose1')]   # with the whole pattern as reference
    lo = lo[['condition', 'uniform_superclass', 'dose', 'task_mean_hz', 'fraction_above_100hz'] + [c_ for c_ in kc_.columns if c_.startswith('trained_solved_')]].copy()
    lo['uniform_superclass'] = lo['uniform_superclass'].fillna('none')
    lo['source_file'] = paths.rel(PATH / 'sc_kv4_conditions.csv')
    csvout.write_table(lo, PV + 's7_loo.csv', title='The Shal pattern at dose 1 with one superclass made uniform (descriptive)', describe={
        'condition': 'Condition', 'uniform_superclass': 'Superclass given the uniform lesion', 'dose': 'Dose', 'task_mean_hz': 'Mean task rate (Hz)',
        'fraction_above_100hz': 'Share of cells above 100 Hz', **{c_: f'Trained histories solved, {c_[len("trained_solved_"):]}' for c_ in lo.columns if c_.startswith('trained_solved_')}},
        key=['condition'], notes='Descriptive detail; the equivalence test of the pattern is class c (P17) and not routed.')
    # (9) the ageing operator's expression source
    ex = _read(PATH / 'sc_aging_expression.csv')
    ex = ex[ex['source'] == 'davie2018'].copy()
    ex['unit'] = ex['unit'].astype(str)
    ex['neuronal'] = ~ex['unit'].isin(DAVIE_NON_NEURONAL)
    ex = ex[['unit', 'neuronal', 'cells_young', 'cells_old', 'log2f'] + [c_ for c_ in ex.columns if c_.startswith('log2f_') and c_ not in ('log2f_female', 'log2f_male')]]
    ex['source_file'] = paths.rel(PATH / 'sc_aging_expression.csv')
    csvout.write_table(ex, PV + 's7_ageing_expression.csv', title='The ageing operator\'s source: K+ channel expression, old over young, per Davie 2018 unit', describe={
        'unit': 'Davie 2018 cluster or annotation', 'neuronal': 'Neuronal unit (glia, plasmatocytes, photoreceptors and the Hsp cluster are not)',
        'cells_young': 'Cells, young (1-9 days)', 'cells_old': 'Cells, old (30-50 days)', 'log2f': 'log2 old/young, summed K+ channel genes',
        **{c_: f'log2 old/young, {c_[len("log2f_"):]}' for c_ in ex.columns if c_.startswith('log2f_')}}, key=['unit'],
        notes='AFCA rows are not routed (their Kenyon-cell direction is class c, P18).')


# ---------------------------------------------------------------- Fig. 4: the molecular signature

SC_RUNS = PATH.parent / 'runs'
LIN_SOURCE = 'allen2020vnc'
CHANNELS = ('SK', 'slo', 'KCNQ', 'Shab', 'Shal')
TYPE_TIERS, LINEAGE_TIERS = ('exact', 'alias', 'fuzzy'), ('lineage', 'lineage_pair')
CARRIER_KEYS = {'m0': 'remote-dynamic-M0-head_only', 'm0_joint': 'remote-dynamic-M0-joint', 's1': 'supp-S1-ring-brain',
                's2': 'supp-S2-context-brain'}
# carrier types with genetic access drawn in Fig. 4I: driver lines in sc_annotation_types and a carrier cell with a local skeleton
ACCESS_TYPES = ('LC10d', 'LPLC2', 'LPLC1', 'LC16')


def mapping_v2():
    """The mapping v2 per cell, as sc_trace_v2.py declares it (amendments 0008-0010, 0013): a cell whose v1 tier is class or
    unmapped and whose hemilineage label is in the v3 label map takes that lineage unit and its tier. Model cell order."""
    cells = io.parquet_load(SC_RUNS / 'sc-mapping' / 'cell_map.parquet')
    order = io.parquet_load(paths.M1 / 'neurons' / 'annotations.parquet', columns=['bodyId'])['bodyId'].to_numpy(np.int64)
    if not np.array_equal(cells['bodyId'].to_numpy(np.int64), order):
        raise RuntimeError('cell_map not in model cell order')
    gates = io.json_load(SC_RUNS / 'sc-lineage-vnc-v3' / 'gates.json')
    if not gates['passed']:
        raise RuntimeError('the v3 bridge gates did not pass; no lineage tier')
    lm = io.csv_load(SC_RUNS / 'sc-lineage-vnc-v3' / 'label_map.csv').set_index('malecns_label')
    hl = cells['trumanHl'].astype(str).to_numpy()
    up = cells['tier'].isin(['class', 'unmapped']).to_numpy() & np.isin(hl, lm.index.to_numpy())
    cells['tier'] = cells['tier'].astype('object')
    cells['source'] = cells['source'].astype('object')
    cells['unit'] = cells['unit'].astype('object')
    cells.loc[up, 'tier'] = lm.loc[hl[up], 'tier'].to_numpy()
    cells.loc[up, 'source'] = LIN_SOURCE
    cells.loc[up, 'unit'] = lm.loc[hl[up], 'unit'].to_numpy()
    return cells


def fig4_tables(names):
    cells = mapping_v2()
    carriers = {k: cells[f'delta_adapt_{k}'].fillna(0).ne(0).to_numpy() for k in CARRIER_KEYS}
    # gate: the coverage of mapping v2 (sc_mapping_v2_coverage, written to 6 significant digits) for all cells and the M0 carriers
    cov = _read(PATH / 'sc_mapping_v2_coverage.csv').set_index('cell_set')
    for name, mask in (('all', np.ones(len(cells), bool)), ('trace carriers (m0)', carriers['m0'])):
        t = cells.loc[mask, 'tier']
        got = (t.isin(TYPE_TIERS).mean(), (t == 'lineage').mean(), (t == 'lineage_pair').mean(), t.isin(TYPE_TIERS + LINEAGE_TIERS).mean())
        want = tuple(cov.loc[name, ['type_or_fuzzy', 'lineage', 'lineage_pair', 'type_lineage']])
        if int(cov.loc[name, 'cells']) != int(mask.sum()) or not np.allclose(got, want, rtol=1e-5, atol=1e-7):
            raise RuntimeError(f'mapping v2 coverage disagrees with sc_mapping_v2_coverage ({name})')
    vals = _read(PATH / 'sc_unit_values_v2.csv')
    vals['unit'] = vals['unit'].astype(str)
    mapped = cells['tier'].ne('unmapped').to_numpy()
    key = cells['source'].astype(str) + '|' + cells['unit'].astype(str)
    vk = vals.assign(k=vals['source'] + '|' + vals['unit']).set_index('k')
    if not key[mapped].isin(vk.index).all():
        raise RuntimeError('a mapped cell has no unit value')
    expr = pd.DataFrame(dict(cell=np.arange(len(cells)), body_id=cells['bodyId'].to_numpy(np.int64), tier=cells['tier'].to_numpy(),
                             source=cells['source'].where(mapped).to_numpy(), unit=cells['unit'].where(mapped).to_numpy()))
    for g in ('SK', 'Shal', 'slo'):
        expr[f'z_{g}'] = np.where(mapped, vk[f'z_{g}'].reindex(key.where(mapped)).to_numpy(), np.nan).round(4)
    expr['adaptation_index'] = np.where(mapped, vk['A'].reindex(key.where(mapped)).to_numpy(), np.nan).round(4)
    expr['carrier_scene1'] = carriers['m0']
    expr['hemilineage'] = cells['trumanHl'].to_numpy()
    csvout.write_table(expr, PV + 'fig4_cell_expression.csv', title='Per-cell atlas expression (mapping v2): the unit z-scores of SK, Shal and slo and the adaptation index, with the mapping tier', describe={
        'cell': 'Model cell index', 'body_id': 'MaleCNS body ID', 'tier': 'Mapping tier: exact, alias or fuzzy (cell type), lineage or lineage_pair (VNC hemilineage), class, unmapped',
        'source': 'Atlas of the unit', 'unit': 'Atlas unit (cell type, hemilineage or class)', 'z_SK': 'SK, z-score of the unit mean among the mapped units of its atlas',
        'z_Shal': 'Shal, same', 'z_slo': 'slo, same', 'adaptation_index': 'Mean z of the adaptation-index genes (SK, slo, KCNQ, Shab)',
        'carrier_scene1': 'Trace carrier in scene 1, readout trained (adaptation differs between cues at the decision)',
        'hemilineage': 'MaleCNS hemilineage label'}, key=['cell'], notes='Rebuilt from runs/sc-mapping/cell_map.parquet and the v3 label map; gated against sc_mapping_v2_coverage.csv and sc_unit_values_v2.csv.')
    # the 21 VNC hemilineage units: carriers, expression and detection of the channel panel, transmitters and markers
    units = _read(PATH / 'sc_lineage_vnc_v3_units.csv').set_index('unit')
    frac = io.parquet_load(SC_RUNS / 'sc-lineage-vnc-v3' / 'units_allen2020vnc_frac.parquet')[list(CHANNELS)]   # whole table: keeps the unit index
    frac.index = frac.index.astype(str)
    gates = io.json_load(SC_RUNS / 'sc-lineage-vnc-v3' / 'gates.json')
    markers = {d['unit']: d for d in gates['V3_detail']}
    lin = vals[vals['source'] == LIN_SOURCE].set_index('unit')
    rows = []
    for u in gates['units']:
        m = (cells['source'] == LIN_SOURCE).to_numpy() & (cells['unit'].astype(str) == u).to_numpy()
        r = dict(unit=u, tier=units.loc[u, 'tier'], atlas_cells=int(units.loc[u, 'cells']), malecns_cells=int(m.sum()),
                 carriers_scene1=int((m & carriers['m0']).sum()), carrier_share=float((m & carriers['m0']).sum() / max(m.sum(), 1)),
                 em_transmitter=units.loc[u, 'em_transmitter'], transcriptome_transmitter=units.loc[u, 'transcriptome_transmitter'],
                 markers=int(markers[u]['markers']), marker_mean_z=float(markers[u]['mean_z']), adaptation_index=float(lin.loc[u, 'A']))
        for g in CHANNELS:
            r[f'z_{g}'] = float(lin.loc[u, f'z_{g}'])
            r[f'detected_{g}'] = float(frac.loc[u, g])
        rows.append(r)
    h = pd.DataFrame(rows)
    h['source_file'] = 'runs/sc-lineage-vnc-v3 (units, gates, label map); results/sc_unit_values_v2.csv'
    csvout.write_table(h, PV + 'fig4_hemilineages.csv', title='The 21 VNC hemilineage units of the validated bridge: carriers, channel expression and detection, transmitters and markers', describe={
        'unit': 'Hemilineage unit (pairs merged where the atlas does not separate them)', 'tier': 'lineage or lineage_pair', 'atlas_cells': 'Atlas cells in the unit',
        'malecns_cells': 'MaleCNS cells mapped to the unit', 'carriers_scene1': 'Of these, trace carriers in scene 1 (readout trained)', 'carrier_share': 'carriers_scene1 / malecns_cells',
        'em_transmitter': 'Transmitter predicted from EM (MaleCNS)', 'transcriptome_transmitter': 'Transmitter from the atlas transcriptome', 'markers': 'Published hemilineage markers enriched in the unit',
        'marker_mean_z': 'Mean z of those markers', 'adaptation_index': 'Mean z of SK, slo, KCNQ and Shab',
        **{f'z_{g}': f'{g}: z-score of the unit mean among the 21 units' for g in CHANNELS},
        **{f'detected_{g}': f'{g}: share of the unit\'s atlas cells with the transcript detected' for g in CHANNELS}}, key=['unit'])
    # the VNC atlas in UMAP space (the v3 bridge per-cell export, amendment 0014; its check reproduces the declared gates)
    chk = io.json_load(SC_RUNS / 'sc-lineage-vnc-v3-export' / 'check.json')
    if not (chk['same_gates']['passed'] and chk['same_cluster_recovery']):
        raise RuntimeError('the VNC export does not reproduce the v3 bridge')
    um = io.parquet_load(SC_RUNS / 'sc-lineage-vnc-v3-export' / 'cells.parquet')
    um = um[['cell', 'umap1', 'umap2', 'unit']].copy()
    um['umap1'], um['umap2'] = um['umap1'].round(4), um['umap2'].round(4)
    csvout.write_table(um, PV + 'fig4_vnc_atlas_umap.csv', title='The 26,775 adult VNC atlas cells in UMAP space with their hemilineage unit (v3 bridge)', describe={
        'cell': 'Atlas cell index (after the authors\' QC)', 'umap1': 'UMAP coordinate 1', 'umap2': 'UMAP coordinate 2', 'unit': 'Hemilineage unit (empty: not in a unit)'},
        key=['cell'], notes='Source: runs/sc-lineage-vnc-v3-export/cells.parquet (check.json: same gates, same cluster recovery as the declared run).')
    # validation of the bridge: cluster sizes, transmitters, markers; mapping coverage
    cl = _read(PATH / 'sc_lineage_vnc_v3_clusters.csv')
    cl['source_file'] = paths.rel(PATH / 'sc_lineage_vnc_v3_clusters.csv')
    csvout.write_table(cl, PV + 'fig4_bridge_clusters.csv', title='Published vs recovered atlas cluster sizes (v3 bridge)', describe={
        'cluster': 'Published cluster', 'published_cells': 'Cells in the published cluster', 'neuronal': 'Neuronal cluster', 'published_transmitter': 'Published transmitter',
        'unit': 'Hemilineage unit the cluster belongs to', 'recovered_cells': 'Cells the reproduced clustering assigns to it'}, key=['cluster'])
    cv = _read(PATH / 'sc_mapping_v2_coverage.csv')
    cv['source_file'] = paths.rel(PATH / 'sc_mapping_v2_coverage.csv')
    csvout.write_table(cv, PV + 'fig4_coverage.csv', title='Mapping coverage at the cell-type and hemilineage tiers', describe={
        'cell_set': 'Cell set (repository label; m0 = scene 1 readout trained, m0_joint = scene 1 brain and readout, s1 = scene 2, s2 = scene 3)', 'cells': 'Cells',
        'type_or_fuzzy': 'Share mapped at the cell-type tier', 'lineage': 'Share mapped at the hemilineage tier', 'lineage_pair': 'Share mapped to a hemilineage pair',
        'type_lineage': 'Share mapped at the type or hemilineage tier', 'any_tier': 'Share mapped at any tier (class included)'}, key=['cell_set'])
    # H1 v2: carriers against matched sets and the mapping null. The index against the mapping null is class c (inventory M12).
    tr = _read(PATH / 'sc_trace_v2_tests.csv')
    adj = _read(PATH / 'all_tests.csv')
    adj = adj[adj['family'].astype(str).str.startswith('F14.SC.trace.v2')].set_index(['family', 'test'])['p_adjusted_family']
    tr['p_adjusted'] = [float(adj.loc[(f, t)]) for f, t in zip(tr['family'], tr['test'])]
    parts = tr['test'].str.split(':', expand=True)
    tr['model_key'], tr['compartment'], tr['level'], tr['statistic_name'], tr['null'] = parts[0], parts[1], parts[2], parts[3], parts[4]
    tr = tr[~((tr['statistic_name'] == 'A') & (tr['null'] == 'mapping-null'))].copy()
    lab = names[names['kind'] == 'model'].set_index('code')
    tr['scene'] = tr['model_key'].map(lambda k: lab.loc[CARRIER_KEYS[k], 'scene'])
    tr['trained'] = tr['model_key'].map(lambda k: lab.loc[CARRIER_KEYS[k], 'trained'])
    tr = tr[['family', 'test', 'scene', 'trained', 'compartment', 'level', 'statistic_name', 'null', 'n', 'coverage', 'statistic', 'mean_reference',
             'null_sd', 'p_value', 'p_adjusted']]
    tr['source_file'] = paths.rel(PATH / 'sc_trace_v2_tests.csv')
    csvout.write_table(tr, PV + 'fig4_trace_tests.csv', title='Trace carriers vs matched sets and the mapping null: mean SK z and mean adaptation index', describe={
        'family': 'Declared family', 'test': 'Test (repository label)', 'scene': 'Memory-task scene', 'trained': 'Trained parameters', 'compartment': 'all cells or the VNC',
        'level': 'Mapping level used', 'statistic_name': 'SK (mean SK z) or A (mean adaptation index)', 'null': 'matched-sets (superclass x atlas strata) or mapping-null (values permuted across mapped units)',
        'n': 'Mapped carriers', 'coverage': 'Share of carriers mapped at the level', 'statistic': 'Mean over the mapped carriers', 'mean_reference': 'Null mean',
        'null_sd': 'Null standard deviation', 'p_value': 'One-sided p (10,000 draws)', 'p_adjusted': 'p adjusted within the family (Holm or BH)'}, key=['family', 'test'])
    # carrier types with genetic access
    an = _read(PATH / 'sc_annotation_types.csv')
    an = an[(an['cell_set'] == 'trace carriers (M0)') & an['type'].isin(ACCESS_TYPES)][['type', 'cells', 'superclass', 'source', 'unit', 'tier', 'driver_lines', 'A']
                                                                                       + [f'z_{g}' for g in CHANNELS]].copy()
    if sorted(an['type']) != sorted(ACCESS_TYPES) or an['driver_lines'].isna().any():
        raise RuntimeError('a carrier type with genetic access is missing or has no driver line')
    an = an.rename(columns={'cells': 'carriers_scene1', 'A': 'adaptation_index'})
    an['source_file'] = paths.rel(PATH / 'sc_annotation_types.csv')
    csvout.write_table(an, PV + 'fig4_access_types.csv', title='Trace-carrier cell types with split-GAL4 driver lines and their channel expression', describe={
        'type': 'Cell type', 'carriers_scene1': 'Trace carriers of the type (scene 1, readout trained)', 'superclass': 'Superclass', 'source': 'Atlas', 'unit': 'Atlas unit',
        'tier': 'Mapping tier', 'driver_lines': 'Split-GAL4 lines', 'adaptation_index': 'Mean z of SK, slo, KCNQ and Shab',
        **{f'z_{g}': f'{g} z-score of the unit' for g in CHANNELS}}, key=['type'])
    # heterogeneous adaptation (T-SC6): scene-1 runs; the scene-2 runs (BH 0.057) and the type-permuted null brains are class c
    vr = _read(PATH / 'sc_variant_runs.csv')
    vr = vr[(vr['brain'] == 'variant') & (vr['family'] == 'M0')][['replicate', 'status', 'trained_net_use', 'untrained_net_use', 'lock', 'adaptation_drop',
                                                                  'swap_drop', 'ed50_c01_mv', 'ed50_c01_kind']].copy()
    vr.insert(0, 'trained', 'readout')
    vr.insert(0, 'scene', 'scene 1')
    vr['source_file'] = paths.rel(PATH / 'sc_variant_runs.csv')
    csvout.write_table(vr, PV + 'fig4_heterogeneity.csv', title='Brains with transcriptome-informed adaptation heterogeneity: memory, lock and mechanism per run (scene 1)', describe={
        'scene': 'Memory-task scene', 'trained': 'Trained parameters', 'replicate': 'Replicate', 'status': 'Run status',
        'trained_net_use': 'Net memory use at the 9 trained timings', 'untrained_net_use': 'Net memory use at the 26 untrained timings', 'lock': 'Trained minus untrained net use',
        'adaptation_drop': 'Drop in net use when the adaptation difference is removed at the trained timings', 'swap_drop': 'Drop in net use after an opposite-cue state swap',
        'ed50_c01_mv': 'Kv4-lesion dose (mV) that halves memory', 'ed50_c01_kind': 'How the ED50 was obtained'}, key=['replicate'])


# ---------------------------------------------------------------- Fig. 5: access locked to the timing of learning

TRAINED_TIMINGS = {(d, p) for d in (0, 40, 160) for p in (0, 1, 4)}
JITTER_MAX_DELAY = 320                                          # X3 jittered training covers delays 0-320 ms (prefixes 0-8)
BAND_TIMINGS = ((40, 4), (160, 4), (320, 4), (640, 4))         # two trained, two untrained
X2_SCENES = {'M1': 'scene 4', 'M2': 'scene 5', 'S1-mirror': 'scene 6', 'S2-mirror': 'scene 7', 'S1-minus45': 'scene 8', 'S2-minus48': 'scene 9'}
X2_DESCRIBE = {'scene 4': 'scene 1 with one added floor square', 'scene 5': 'scene 1 with two added floor squares',
               'scene 6': 'scene 2 mirrored left-right', 'scene 7': 'scene 3 mirrored left-right',
               'scene 8': 'scene 2 with one floor square removed', 'scene 9': 'scene 3 with one floor square removed'}


def _pair_use(rollouts):
    """Net memory use per timing of one cue pair: +1 both cues solved, 0 one, -1 neither."""
    out = {}
    for r in rollouts:
        k = (int(r['delay_ms']), int(r['prefix_windows']))
        out.setdefault(k, []).append(r['status'] == 'solved')
    return {k: (1 if all(v) else (-1 if not any(v) else 0)) for k, v in out.items() if len(v) == 2}


def _run_timings(folder):
    v = io.json_load(folder / 'verification.json')
    return _pair_use(v['results']['rollouts'])


def fig5_tables(names):
    runs = SUPP.parent / 'runs'
    camp = _read(SUPP / 'campaign_runs.csv').set_index('job')
    rows = []
    for r in range(10):
        job = f'x1-M0-head_only-r{r}'
        use = _run_timings(runs / job)
        tr = np.mean([u for k, u in use.items() if k in TRAINED_TIMINGS])
        un = np.mean([u for k, u in use.items() if k not in TRAINED_TIMINGS])
        if not (np.isclose(tr, camp.loc[job, 'trained_net_use']) and np.isclose(un, camp.loc[job, 'untrained_net_use']) and len(use) == 35):
            raise RuntimeError(f'{job}: per-timing use disagrees with campaign_runs')
        rows += [dict(arm='fixed timings', replicate=r, delay_ms=d, prefix_windows=p, trained=(d, p) in TRAINED_TIMINGS, net_use=u)
                 for (d, p), u in sorted(use.items())]
    x3 = _read(SUPP / 'x3_runs.csv')
    for _, q in x3[x3['arm'] == 'jittered'].iterrows():
        use = _run_timings(runs / q['job'])
        inside = np.mean([u for k, u in use.items() if k not in TRAINED_TIMINGS and k[0] <= JITTER_MAX_DELAY])
        outside = np.mean([u for k, u in use.items() if k not in TRAINED_TIMINGS and k[0] > JITTER_MAX_DELAY])
        if not (np.isclose(inside, q['untrained_inside_net_use']) and np.isclose(outside, q['untrained_outside_net_use'])):
            raise RuntimeError(f"{q['job']}: per-timing use disagrees with x3_runs")
        # the jittered arm's trained timings are not routed: their cost is class c (inventory A7; STAR Methods, Table S2)
        rows += [dict(arm='jittered timings', replicate=int(q['replicate']), delay_ms=d, prefix_windows=p, trained=False, net_use=u)
                 for (d, p), u in sorted(use.items()) if (d, p) not in TRAINED_TIMINGS]
    t = pd.DataFrame(rows)
    t.insert(0, 'trained_parameters', 'readout')
    t.insert(0, 'scene', 'scene 1')
    t['inside_jitter_range'] = t['delay_ms'] <= JITTER_MAX_DELAY
    t['source_file'] = 'studies/brain-supplement-v2/runs/x1-M0-head_only-r*, x3-M0-head_only-r* (verification.json)'
    csvout.write_table(t, PV + 'fig5_timings.csv', title='Net memory use at each delay x prefix timing, per replicate, fixed-timing (replicate training) and jittered-timing training (scene 1)', describe={
        'scene': 'Memory-task scene', 'trained_parameters': 'Trained parameters', 'arm': 'Training regime', 'replicate': 'Replicate', 'delay_ms': 'Delay (ms)',
        'prefix_windows': 'Prefix windows', 'trained': 'One of the 9 trained timings of fixed-timing training', 'net_use': '+1 both cues solved, 0 one, -1 neither',
        'inside_jitter_range': 'Delay within the jittered-training range (0-320 ms)'}, key=['arm', 'replicate', 'delay_ms', 'prefix_windows'],
        notes='Jittered runs: untrained timings only. Gated against campaign_runs.csv and x3_runs.csv.')
    # the lock in nine scenes: originals (replicate training) and the six new scenes (X2)
    lab = names[names['kind'] == 'replicate-training code'].set_index('code')
    sc_rows = []
    for code, n in (('M0-head_only', 10), ('S1-head_only', 7), ('S2-head_only', 7)):
        for r in range(n):
            job = f'x1-{code}-r{r}'
            sc_rows.append(dict(scene=lab.loc[code, 'scene'], origin='original', replicate=r, trained_net_use=float(camp.loc[job, 'trained_net_use']),
                                lock=float(camp.loc[job, 'trained_net_use'] - camp.loc[job, 'untrained_net_use']), code=code))
    x2 = _read(SUPP / 'x2_runs.csv')
    for _, q in x2.iterrows():
        sc_rows.append(dict(scene=X2_SCENES[q['scene']], origin=X2_DESCRIBE[X2_SCENES[q['scene']]], replicate=int(str(q['job']).rsplit('-r', 1)[1]),
                            trained_net_use=float(q['trained_net_use']), lock=float(q['trained_net_use'] - q['untrained_net_use']), code=q['scene']))
    sc = pd.DataFrame(sc_rows)
    sc['source_file'] = 'studies/brain-supplement-v2/results/campaign_runs.csv; x2_runs.csv'
    csvout.write_table(sc, PV + 'fig5_scenes.csv', title='The timing lock (trained minus untrained net use) per replicate in nine scenes', describe={
        'scene': 'Memory-task scene', 'origin': 'original, or how the new scene derives from an original', 'replicate': 'Replicate',
        'trained_net_use': 'Net memory use at the 9 trained timings', 'lock': 'Trained minus untrained net use (declared lock statistic)',
        'code': 'Repository identifier (test tables)'}, key=['scene', 'replicate'])
    # code collapse at untrained timings (fu1); the scene-2 and scene-3 tests are not significant after Holm (class c, A7)
    fu1 = io.csv_load(paths.out('followup/fu1_timing_lock.csv'), low_memory=False)
    keep = {'remote-dynamic-M0-head_only': ('scene 1', 'readout'), 'remote-dynamic-M0-joint': ('scene 1', 'brain and readout')}
    fu1 = fu1[fu1['model'].isin(keep) & (fu1['visibility'] == 'hidden')].copy()
    fu1['scene'] = fu1['model'].map(lambda m: keep[m][0])
    fu1['trained_parameters'] = fu1['model'].map(lambda m: keep[m][1])
    fu1 = fu1[['scene', 'trained_parameters', 'delay_ms', 'prefix_windows', 'trained', 'differential', 'common', 'cosine_with_trained_difference']]
    # Holm p of the declared family F11.timing.differential, computed as P8 does (stats.holm over the family's tests registered by
    # P11); P8 runs after this phase and writes the same values to stats/statistical_tests.csv, which a clean run does not yet hold
    st = pd.DataFrame([r for r in stats.load_all() if r['family'] == 'F11.timing.differential'])
    ok = st['p_value'].notna()
    st.loc[ok, 'p_adjusted'] = stats.holm(st.loc[ok, 'p_value'].astype(float).to_numpy())
    st = st.set_index('test_id')
    for m, (scn, trn) in keep.items():
        tid = f'fu1.differential.{m}|c0|hidden'
        fu1.loc[(fu1['scene'] == scn) & (fu1['trained_parameters'] == trn), 'holm_p_differential'] = float(st.loc[tid, 'p_adjusted'])
    fu1['source_file'] = 'manuscript/results/followup/fu1_timing_lock.csv; stats/statistical_tests.csv'
    csvout.write_table(fu1, PV + 'fig5_code.csv', title='Decision-time cue code at all 35 timings: differential and common margins, and its cosine with the trained code (scene 1)', describe={
        'scene': 'Memory-task scene', 'trained_parameters': 'Trained parameters', 'delay_ms': 'Delay (ms)', 'prefix_windows': 'Prefix windows', 'trained': 'Trained timing',
        'differential': 'Cue-differential part of the first-decision margin', 'common': 'Cue-independent part', 'cosine_with_trained_difference': 'Cosine of the decision-time cue difference with the trained one',
        'holm_p_differential': 'Holm p of the declared test: differential margin lower at untrained timings (F11.timing.differential)'},
        key=['scene', 'trained_parameters', 'delay_ms', 'prefix_windows'])
    # operator drift (state transplant run): one adaptation pattern injected at different delays
    g = io.json_load(paths.ROOT / 'studies' / 'brain-supplement-v2' / 'runs' / 'g4-linear-response' / 'result.json')
    dr = []
    for p in (4, 0):
        for k, cosv in g['transfer'][f'p{p}_cosine'].items():
            a, b = (int(x) for x in k.split('-'))
            dr += [dict(prefix_windows=p, delay_a_ms=a, delay_b_ms=b, cosine=float(cosv)), dict(prefix_windows=p, delay_a_ms=b, delay_b_ms=a, cosine=float(cosv))]
    dr = pd.DataFrame(dr)
    dr['source_file'] = 'studies/brain-supplement-v2/runs/g4-linear-response/result.json'
    csvout.write_table(dr, PV + 'fig5_drift.csv', title='Operator drift: cosine between the decision-time responses to one adaptation pattern injected at two delays (scene 1, readout trained)', describe={
        'prefix_windows': 'Prefix windows', 'delay_a_ms': 'Delay of one injection (ms)', 'delay_b_ms': 'Delay of the other (ms)',
        'cosine': 'Cosine between the two decision-time responses'}, key=['prefix_windows', 'delay_a_ms', 'delay_b_ms'],
        notes='The pattern is the unit-norm adaptation difference at the end of the 160-ms delay, scaled to each delay\'s own trace norm and added to the cue-0 state.')
    mg = pd.DataFrame([dict(prefix_windows=p, delay_ms=int(d), trained_delay=int(d) in {0, 40, 160}, margin_shift=float(v))
                       for p in (4, 0) for d, v in g['transfer'][f'p{p}_margin_shift'].items()])
    mg['source_file'] = 'studies/brain-supplement-v2/runs/g4-linear-response/result.json'
    csvout.write_table(mg, PV + 'fig5_drift_margin.csv', title='Operator drift: the push one adaptation pattern gives when injected at each delay (scene 1, readout trained)', describe={
        'prefix_windows': 'Prefix windows', 'delay_ms': 'Delay of the injection (ms)', 'trained_delay': 'One of the three trained delays (0, 40, 160 ms)',
        'margin_shift': 'Change in the readout\'s first-decision preference for the cue-1 push over the cue-0 push when the cue-1 minus cue-0 pattern is added to the cue-0 state (positive: toward the push the injected cue requires)'},
        key=['prefix_windows', 'delay_ms'])
    # the decision-time cue difference per cell at four timings (the band), from the free-run records of fu4
    model = 'remote-dynamic-M0-head_only'
    fe = io.csv_load(paths.out('followup/fu4_decision_energy.csv'), low_memory=False)
    cells = {}
    for d, p in BAND_TIMINGS:
        folder, meta = _record_meta(model, d, p)
        z = io.npz_load(folder / f'free-after-d{d}-p{p}.npz', keys=['mean'])['mean']
        s = _first_decision(meta)
        i0, i1 = _record_pairs(meta)[(0, 'hidden')]
        dv = z[s, i1, 2].astype(np.float64) - z[s, i0, 2].astype(np.float64)
        want = float(fe[(fe['model'] == model) & (fe['visibility'] == 'hidden') & (fe['delay_ms'] == d) & (fe['prefix_windows'] == p)]['energy_v'].iloc[0])
        if not np.isclose(float(dv @ dv), want, rtol=1e-9, atol=0):
            raise RuntimeError(f'decision-time cue difference at d{d} p{p} disagrees with fu4')
        cells[f'dv_d{d}_p{p}_mv'] = np.round(dv, 6)
    dc = pd.DataFrame({'cell': np.arange(len(next(iter(cells.values())))), **cells})
    csvout.write_table(dc, PV + 'fig5_decision_cells.csv', title='Per-cell cue difference of the membrane potential at the first decision, four timings (scene 1, readout trained)', describe={
        'cell': 'Model cell index', **{k: f'Delay {k.split("_")[1][1:]} ms, {k.split("_")[2][1:]} prefix windows: cue 1 minus cue 0, first decision window mean (mV)' for k in cells}},
        key=['cell'], notes='Gated against followup/fu4_decision_energy.csv.')
    keys = list(cells)
    cos = [dict(timing_a=a, timing_b=b, cosine=float(dc[a] @ dc[b] / np.sqrt((dc[a] @ dc[a]) * (dc[b] @ dc[b])))) for a in keys for b in keys]
    csvout.write_table(pd.DataFrame(cos), PV + 'fig5_decision_cosines.csv', title='Cosine between the per-cell decision-time cue differences of the four band timings', describe={
        'timing_a': 'Column of fig5_decision_cells', 'timing_b': 'Column of fig5_decision_cells', 'cosine': 'Cosine over all cells'}, key=['timing_a', 'timing_b'])
    # the recorded learners' own outcome at every timing (behaviour files of the records, gated against fu1), with the trace
    # at the decision (fu4) and the cue difference in the readout's input (fu1)
    full = io.csv_load(paths.out('followup/fu1_timing_lock.csv'), low_memory=False)
    rec = []
    for m, (scn, trn) in keep.items():
        en = fe[(fe['model'] == m) & (fe['visibility'] == 'hidden')].set_index(['context', 'delay_ms', 'prefix_windows'])
        for _, q in full[(full['model'] == m) & (full['visibility'] == 'hidden')].iterrows():
            ctx, d, p = int(q['context']), int(q['delay_ms']), int(q['prefix_windows'])
            beh = io.json_load(paths.M1 / 'runs' / f'record-{m}' / f'free-after-d{d}-p{p}-behavior.json')
            cases = {int(c['cue']): c for c in beh if not c['visible'] and int(c['context']) == ctx}
            solved = [cases[c]['status'] == 'solved' for c in (0, 1)]
            if all(solved) != (str(q['both_solved']).lower() == 'true'):
                raise RuntimeError(f'{m} d{d} p{p}: behaviour disagrees with fu1')
            rec.append(dict(scene=scn, trained_parameters=trn, delay_ms=d, prefix_windows=p, trained=(d, p) in TRAINED_TIMINGS,
                            cue0_solved=solved[0], cue1_solved=solved[1], net_use=(1 if all(solved) else (-1 if not any(solved) else 0)),
                            first_pushes_correct=int(sum(bool(cases[c]['first_push_correct']) for c in (0, 1))),
                            adaptation_energy_decision=float(en.loc[(ctx, d, p), 'energy_adapt']), readout_input_cue_difference=float(q['cue_difference_norm'])))
    rec = pd.DataFrame(rec)
    rec['source_file'] = 'studies/brain-sokoban-v1/mechanisms-v1/runs/record-*/free-after-*-behavior.json; manuscript/results/followup/fu4_decision_energy.csv, fu1_timing_lock.csv'
    csvout.write_table(rec, PV + 'fig5_record_outcomes.csv', title='The recorded scene-1 learners at all 35 timings: outcome per cue, the adaptation trace at the first decision and the cue difference in the readout input', describe={
        'scene': 'Memory-task scene', 'trained_parameters': 'Trained parameters', 'delay_ms': 'Delay (ms)', 'prefix_windows': 'Prefix windows', 'trained': 'Trained timing',
        'cue0_solved': 'Cue-0 history solved', 'cue1_solved': 'Cue-1 history solved', 'net_use': '+1 both cues solved, 0 one, -1 neither',
        'first_pushes_correct': 'First pushes towards the correct target (of 2)', 'adaptation_energy_decision': 'Cue-difference energy of adaptation at the first decision (fu4)',
        'readout_input_cue_difference': 'Norm of the z-scored cue difference of the readout input features at the first decision (fu1)'},
        key=['scene', 'trained_parameters', 'delay_ms', 'prefix_windows'], notes='Hidden goal, context 0. Gated against fu1 both_solved.')


# ---------------------------------------------------------------- Fig. 6: access locked to the excitability of learning

ONE_STATE, ACROSS = 'one state', 'across states'              # replicate training; training under a threshold jitter of sigma 1 uV
ROBUST_CONDITION = 'th0.001'                                   # training across threshold states: offsets with an SD of 1 uV
RUNAWAY_SHARE = 0.01                                           # runaway: >= 1% of cells above 100 Hz in the mean-field sweep
REMOVAL_SHARES = (0.0002, 0.0005, 0.001, 0.002, 0.005)


def _ed50(start, points, low, high, threshold=0.5):
    """analyze_dy2.ed50: where trained net use first falls to <= 0.5 along the grid, log-linear interpolation; low if
    already there, high if never, 0 if there is no memory to lose."""
    if start is None or start <= threshold:
        return 0.0, 'none'
    prev_m, prev_u = None, start
    for m, u in points:
        if u <= threshold:
            if prev_m is None:
                return low, 'left'
            f = (prev_u - threshold) / (prev_u - u)
            return float(10 ** (np.log10(prev_m) + f * (np.log10(m) - np.log10(prev_m)))), 'interpolated'
        prev_m, prev_u = m, u
    return high, 'right'


def fig6_tables(names):
    """Fig. 6 after the user's correction of 2026-10-22: susceptibility to perturbations is an unfavourable outcome (class c),
    so only the positive results are routed: single-spike kicks absorbed, stable activity, redundant wiring, and training
    across states widening access with the same mechanism. The ED50 gate still recomputes every window from the curves."""
    models = _read(PATH / 'dy2_models.csv')
    panel = _read(PATH / 'dy2_panel.csv')
    keep = models[(models['group'] == 'x1') | ((models['group'] == 'dy2') & (models['condition'] == ROBUST_CONDITION))].copy()
    keep['training'] = np.where(keep['group'] == 'x1', ONE_STATE, ACROSS)
    robust = []
    for _, m in keep.iterrows():
        p = panel[panel['model'] == m['model']].set_index('evaluation')['trained_net_use']
        grid = sorted(-float(e[len('shift'):]) for e in p.index if e.startswith('shift-'))
        pts = [(g, float(p[f'shift{-g:+g}'])) for g in grid]
        val, kind = _ed50(float(p['as-trained']), pts, grid[0], grid[-1])
        rem = []
        for q in REMOVAL_SHARES:
            draws = [k for k in p.index if k.startswith('remove-random-') and abs(float(k.split('-')[2]) - q) < 1e-12]
            if len(draws) != 3:
                raise RuntimeError(f"{m['model']}: {len(draws)} removal draws at {q}")
            rem.append((q, float(np.mean([p[k] for k in draws]))))
        rval, rkind = _ed50(float(p['as-trained']), rem, REMOVAL_SHARES[0], REMOVAL_SHARES[-1])
        if (kind, rkind) != (m['ed50_c01_kind'], m['ed50_removal_kind']) or not np.isclose(val, float(m['ed50_c01_mv']), rtol=1e-9, atol=0) \
                or not np.isclose(rval, float(m['ed50_removal_share']), rtol=1e-9, atol=0):
            raise RuntimeError(f"{m['model']}: ED50 recomputed from dy2_panel disagrees with dy2_models")
        robust.append(dict(training=m['training'], replicate=int(m['replicate']), ed50_shift_uv=1000 * val, ed50_kind=kind,
                           ed50_removal_percent=100 * rval, ed50_removal_kind=rkind, memory_under_jitter=float(m[f'perturbed_{ROBUST_CONDITION}']),
                           adaptation_drop=float(m['adaptation_drop']), swap_drop=float(m['swap_drop'])))
    rb = pd.DataFrame(robust)
    rb.insert(0, 'trained_parameters', 'readout')
    rb.insert(0, 'scene', 'scene 1')
    cond = _read(PATH / 'dy2_conditions.csv').set_index('condition').loc[ROBUST_CONDITION]
    one, acr = rb[rb['training'] == ONE_STATE], rb[rb['training'] == ACROSS]
    checks = [(acr['memory_under_jitter'].mean(), cond['own_perturbation_mean']), (one['memory_under_jitter'].mean(), cond['x1_under_same_mean']),
              (acr['adaptation_drop'].mean(), cond['adaptation_drop']), (one['adaptation_drop'].mean(), cond['adaptation_drop_x1']),
              (acr['swap_drop'].mean(), cond['swap_drop']), (one['swap_drop'].mean(), cond['swap_drop_x1'])]
    if not all(np.isclose(a, b, rtol=1e-9, atol=1e-12) for a, b in checks):
        raise RuntimeError('fig6 robust table disagrees with dy2_conditions')
    rb['access_window_fold'] = rb['ed50_shift_uv'] / float(np.median(one['ed50_shift_uv']))
    rb['cell_loss_tolerance_fold'] = rb['ed50_removal_percent'] / float(np.median(one['ed50_removal_percent']))
    rb['source_file'] = paths.rel(PATH / 'dy2_models.csv') + ', ' + paths.rel(PATH / 'dy2_panel.csv')
    csvout.write_table(rb, PV + 'fig6_robust.csv', title='Training at one state vs across states (threshold jitter, sigma 1 uV): access window, robustness and mechanism per replicate (scene 1, readout trained)', describe={
        'scene': 'Memory-task scene', 'trained_parameters': 'Trained parameters', 'training': 'Training regime', 'replicate': 'Replicate',
        'ed50_shift_uv': 'Uniform threshold lowering at which trained net use first falls to 0.5 (uV)', 'ed50_kind': 'interpolated, left (below the grid) or right (above it)',
        'ed50_removal_percent': 'Random removal at which trained net use first falls to 0.5 (% of cells; mean of 3 draws)', 'ed50_removal_kind': 'interpolated, left or right (censored at 0.5%)',
        'memory_under_jitter': 'Net use at the trained timings under two held-out sigma-1-uV threshold-jitter draws', 'adaptation_drop': 'Drop in net use when the adaptation difference is cleared (prefix-0 trained timings)',
        'swap_drop': "Drop in net use when the other cue's state is swapped in", 'access_window_fold': 'ED50 relative to the median of training at one state',
        'cell_loss_tolerance_fold': 'Removal ED50 relative to the median of training at one state'}, key=['training', 'replicate'],
        notes='Trained and untrained net use, the lock and the update count of the jitter-trained runs are class c (inventory E8) and not routed. Gated against dy2_conditions.csv.')
    summ = pd.DataFrame([dict(quantity='ED50 ratio, across states / one state', estimate=float(cond['ed50_c01_ratio']), ci_low=float(cond['ed50_c01_ratio_lo']),
                              ci_high=float(cond['ed50_c01_ratio_hi'])),
                         dict(quantity='removal ED50 ratio, across states / one state', estimate=float(cond['ed50_removal_ratio']),
                              ci_low=float(cond['ed50_removal_ratio_lo']), ci_high=float(cond['ed50_removal_ratio_hi']))])
    summ['source_file'] = paths.rel(PATH / 'dy2_conditions.csv')
    csvout.write_table(summ, PV + 'fig6_robust_summary.csv', title='Geometric-mean ratios of the access window and the removal tolerance, training across states vs at one state', describe={
        'quantity': 'Ratio', 'estimate': 'Geometric-mean ratio', 'ci_low': '95% bootstrap interval, low', 'ci_high': '95% bootstrap interval, high'}, key=['quantity'])
    # stable activity: the network's firing rate under a uniform threshold change (sweep and memory task), and where runaway starts
    mf = io.json_load(PATH.parent / 'runs' / 'g4-meanfield' / 'result.json')
    rate = pd.DataFrame([dict(source='uniform-offset sweep', shift_mv=float(r['offset_mv']), mean_rate_hz=float(r['mean_rate_hz']),
                              share_above_100hz=float(r['fraction_over_100hz'])) for r in mf['rows']])
    mem = _read(PATH / 'dy1_memory.csv')
    mem = mem[mem['source'] != 'regime A'].drop_duplicates('shift_mv')
    rate = pd.concat([rate, pd.DataFrame(dict(source='memory task, analysed learner', shift_mv=mem['shift_mv'].astype(float),
                                              mean_rate_hz=mem['task_mean_hz'].astype(float), share_above_100hz=np.nan))], ignore_index=True)
    sweep = rate[rate['source'] == 'uniform-offset sweep'].sort_values('shift_mv', ascending=False)
    first = float(sweep[sweep['share_above_100hz'] >= RUNAWAY_SHARE]['shift_mv'].iloc[0])
    rate['runaway_from_mv'] = first
    rate['last_stable_mv'] = float(sweep[sweep['shift_mv'] > first]['shift_mv'].min())
    rate['source_file'] = paths.rel(PATH.parent / 'runs' / 'g4-meanfield' / 'result.json') + '; ' + paths.rel(PATH / 'dy1_memory.csv')
    csvout.write_table(rate, PV + 'fig6_rate.csv', title='Network firing rate under a uniform threshold change', describe={
        'source': 'uniform-offset sweep (mean-field run, healthy brain, 800-ms window) or the memory task of the analysed learner',
        'shift_mv': 'Uniform threshold shift (mV)', 'mean_rate_hz': 'Mean firing rate over cells (Hz)', 'share_above_100hz': 'Share of cells above 100 Hz (sweep only)',
        'runaway_from_mv': f'First sweep offset with >= {RUNAWAY_SHARE:.0%} of cells above 100 Hz', 'last_stable_mv': 'The offset before it'}, key=['source', 'shift_mv'])
    # transient perturbations: single extra spikes
    div = _read(PATH / 'dy1_divergence.csv')
    kicks = div[div['kind'] == 'spike'][['delay_ms', 'prefix_windows', 'cue', 'size', 'hamming_decision', 'growth_per_s']].rename(columns={'size': 'kicked_cell'})
    kicks['kicked_cell'] = kicks['kicked_cell'].astype(int)
    kicks['source_file'] = paths.rel(PATH / 'dy1_divergence.csv')
    csvout.write_table(kicks, PV + 'fig6_kicks.csv', title='Single extra spikes: cells still differing at the decision, two timings and both cues', describe={
        'delay_ms': 'Delay (ms)', 'prefix_windows': 'Prefix windows', 'cue': 'Cue', 'kicked_cell': 'Model cell given the extra spike',
        'hamming_decision': 'Cells with different spikes at the decision (window mean)', 'growth_per_s': 'Fitted growth rate of the difference (1/s)'},
        key=['delay_ms', 'prefix_windows', 'cue', 'kicked_cell'])
    # redundant wiring: graph reachability and flow under removal
    perc = _read(PATH / 'phase4_graph_percolation.csv').groupby(['order', 'share_removed'], as_index=False).agg(
        draws=('draw', 'size'), readout_reachable=('readout_reachable', 'mean'), readout_reachable_min=('readout_reachable', 'min'),
        walks3_relative=('walks3_relative', 'mean'))
    perc['order'] = perc['order'].map({'random': 'random cells', 'degree (hubs first)': 'hubs first'})
    perc['source_file'] = paths.rel(PATH / 'phase4_graph_percolation.csv')
    csvout.write_table(perc, PV + 'fig6_wiring_graph.csv', title='The connectome under cell removal: readout reachable from the visual inputs and three-step walks', describe={
        'order': 'Removal order', 'share_removed': 'Share of cells removed', 'draws': 'Random draws', 'readout_reachable': 'Share of readout cells reachable from the visual input cells (mean)',
        'readout_reachable_min': 'Minimum over draws', 'walks3_relative': 'Three-step walks from input to readout, relative to the intact graph'}, key=['order', 'share_removed'])
    flow = _read(PATH / 'phase4_graph_flow.csv')
    flow['source_file'] = paths.rel(PATH / 'phase4_graph_flow.csv')
    flow['lesion'] = flow['lesion'].str.replace(r'C04 dose (\d) \((\d+)% of KCs, regime A set\)', r'\2% of Kenyon cells', regex=True)
    flow['lesion'] = flow['lesion'].str.replace(r'50% of KCs \(G3a set\)', '50% of Kenyon cells', regex=True).str.replace('all KCs', 'all Kenyon cells')
    csvout.write_table(flow, PV + 'fig6_wiring_flow.csv', title='Maximum flow from the visual inputs to the readout and from the olfactory receptor neurons to the mushroom-body output, with Kenyon cells removed', describe={
        'pathway': 'Pathway', 'lesion': 'Cells removed', 'cells_removed': 'Number removed', 'max_flow': 'Maximum flow (synapses)', 'relative': 'Relative to the intact graph'},
        key=['pathway', 'lesion'])
    # the tables of the first Fig. 6 build that carried the susceptibility results (now class c) leave plan_v2
    for stale in ('fig6_shift_curves', 'fig6_landmarks', 'fig6_dynamics', 'fig6_rerouting', 'fig6_divergence_cells', 'fig6_wiring_behaviour', 'fig6_wiring_fit'):
        for f in paths.out(PV).glob(f'{stale}.*'):
            f.unlink()


GRAPH_SEED, GRAPH_SHARES, GRAPH_DRAWS = 20261019, (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), 3   # phase4_graph.py
ROUTE_SHARES = (0.3, 0.6, 0.9)                                  # the band: draw 0 of these random removals, and the intact graph


def _hops(A, sources, alive):
    """Shortest route length (synapses) from the sources to every cell through alive cells; -1 if unreachable."""
    hop = np.full(A.shape[0], -1, np.int64)
    hop[sources] = 0
    frontier = np.zeros(A.shape[0], np.float32)
    frontier[sources] = 1
    k = 0
    while frontier.any():
        k += 1
        new = ((A @ frontier) > 0) & alive & (hop < 0)
        hop[new] = k
        frontier = new.astype(np.float32)
    return hop


def _reach(A, sources, targets, alive):
    """phase4_graph.reach_all: share of targets reachable from the sources through alive cells (any length)."""
    seen = np.zeros(A.shape[0], bool)
    frontier = np.zeros(A.shape[0], np.float32)
    frontier[sources] = 1
    seen[sources] = True
    while frontier.any():
        new = ((A @ frontier) > 0) & alive & ~seen
        seen |= new
        frontier = new.astype(np.float32)
    return float(seen[targets].mean())


def fig6_routes():
    """Route traffic from the visual input cells to the readout cells, per cell, in the intact connectome and after the
    random removals of phase4_graph.py (same seed and draw order; draw 0 of 30, 60 and 90%): the number of input-to-readout
    walks of length 2 and 3 passing through each cell. Gated: reachability and three-step walks equal the percolation table."""
    W = io.sparse_load(paths.ROOT / 'flybrain' / 'cache' / 'W_post_pre.npz').tocsr()
    W.eliminate_zeros()
    n = W.shape[0]
    A = (W != 0).astype(np.float64).tocsr()                     # rows post: A @ x moves activity downstream
    ann = io.parquet_load(paths.M1 / 'neurons' / 'annotations.parquet', columns=['bodyId', 'direct_input', 'observed_by_policy'])
    inputs = np.flatnonzero(ann['direct_input'].to_numpy(bool))
    readout = np.flatnonzero(ann['observed_by_policy'].to_numpy(bool))
    rng = np.random.default_rng(GRAPH_SEED)
    pool = np.setdiff1d(np.arange(n), np.concatenate([inputs, readout]))
    removed = {0.0: np.array([], np.int64)}
    for q in GRAPH_SHARES:
        m = int(round(q * len(pool)))
        draws = [rng.choice(pool, m, replace=False) for _ in range(GRAPH_DRAWS)]
        if q in ROUTE_SHARES:
            removed[q] = draws[0]
    perc = _read(PATH / 'phase4_graph_percolation.csv')
    perc = perc[(perc['order'] == 'random') & (perc['draw'].astype(str).isin(['0', '0.0']))].set_index('share_removed')
    s = np.zeros(n)
    s[inputs] = 1
    r = np.zeros(n)
    r[readout] = 1
    cells = pd.DataFrame(dict(cell=np.arange(n), role=np.where(np.isin(np.arange(n), inputs), 'input', np.where(np.isin(np.arange(n), readout), 'readout', 'other'))))
    summary, base3, alives, traffics = [], None, {}, {}
    for q in sorted(removed):
        alive = np.ones(n, bool)
        alive[removed[q]] = False
        al = alive.astype(np.float64)
        f1 = (A @ s) * al
        f2 = (A @ f1) * al
        b1 = (A.T @ r) * al
        b2 = (A.T @ b1) * al
        w3 = float(((A @ f2) * al)[readout].sum())
        base3 = w3 if q == 0 else base3
        reach = _reach(A, inputs, readout, alive)
        if q > 0:
            want = perc.loc[q]
            if not (np.isclose(reach, float(want['readout_reachable']), rtol=0, atol=1e-12) and np.isclose(w3 / base3, float(want['walks3_relative']), rtol=1e-9)):
                raise RuntimeError(f'route computation at {q} disagrees with phase4_graph_percolation (draw 0)')
        key = f'{int(round(100 * q))}'
        traffic = f1 * b1 + f1 * b2 + f2 * b1                    # walks of length 2 and 3 through the cell (as an intermediate)
        traffic[inputs] = 0
        traffic[readout] = 0
        hop = _hops(A, inputs, alive)
        if not np.isclose(float((hop[readout] >= 0).mean()), reach, rtol=0, atol=1e-12):
            raise RuntimeError('route lengths disagree with reachability')
        cells[f'alive_{key}'] = alive
        cells[f'traffic_{key}'] = traffic
        alives[key], traffics[key] = alive, traffic
        cells[f'hops_{key}'] = hop
        summary.append(dict(share_removed=q, cells_removed=int(len(removed[q])), readout_reachable=reach, walks3_relative=w3 / base3,
                            route_cells=int((traffic > 0).sum()), readout_cells=int(len(readout)), input_cells=int(len(inputs))))
    keep = (cells[[c for c in cells if c.startswith('traffic_')]].sum(axis=1) > 0) | cells['role'].ne('other')
    cells = cells[keep].reset_index(drop=True)
    cells['source_file'] = 'flybrain/cache/W_post_pre.npz; studies/brain-sokoban-v1/mechanisms-v1/neurons/annotations.parquet'
    csvout.write_table(cells, PV + 'fig6_routes.csv', title='Routes from the visual input cells to the readout cells: input-to-readout walks of length 2-3 through each cell, intact and after random removal of 30, 60 and 90% of the other cells', describe={
        'cell': 'Model cell index', 'role': 'input (visual input cell), readout (observed by the readout) or other',
        **{f'alive_{k}': f'Cell kept in the {k}% removal draw' for k in ('0', '30', '60', '90')},
        **{f'traffic_{k}': f'Input-to-readout walks of length 2 and 3 through the cell, {k}% removed' for k in ('0', '30', '60', '90')},
        **{f'hops_{k}': f'Shortest route from the visual input cells, in synapses ({k}% removed; -1 unreachable)' for k in ('0', '30', '60', '90')}},
        key=['cell'], notes='Seed and draw order of phase4_graph.py (draw 0); reachability and three-step walks gated against phase4_graph_percolation.csv. Only cells on a route or with a role are listed.')
    sm = pd.DataFrame(summary)
    fig6_route_detail(A, alives, traffics, s, r, sm)
    sm['source_file'] = paths.rel(PATH / 'phase4_graph_percolation.csv')
    csvout.write_table(sm, PV + 'fig6_routes_summary.csv', title='The route network at each removal level (draw 0)', describe={
        'share_removed': 'Share of the other cells removed', 'cells_removed': 'Cells removed', 'readout_reachable': 'Share of readout cells reachable from the inputs',
        'walks3_relative': 'Three-step walks relative to the intact graph', 'route_cells': 'Cells on an input-to-readout walk of length 2-3',
        'readout_cells': 'Readout cells', 'input_cells': 'Visual input cells'}, key=['share_removed'])
    # the five single-spike kicks: each kicked cell's direct postsynaptic partners
    div = _read(PATH / 'dy1_divergence.csv')
    kicked = sorted(div.loc[div['kind'] == 'spike', 'size'].astype(int).unique())
    kp = []
    for c in kicked:
        col = W[:, c].tocoo()
        kp += [dict(kicked_cell=int(c), partner_cell=int(i), synapses=int(abs(v)), sign=int(np.sign(v))) for i, v in zip(col.row, col.data)]
    kp = pd.DataFrame(kp)
    kp['source_file'] = 'flybrain/cache/W_post_pre.npz'
    csvout.write_table(kp, PV + 'fig6_kick_partners.csv', title='The direct postsynaptic partners of the five cells given a single extra spike', describe={
        'kicked_cell': 'Model cell given the extra spike', 'partner_cell': 'Postsynaptic partner', 'synapses': 'Synapses (count)', 'sign': '+1 excitatory, -1 inhibitory'},
        key=['kicked_cell', 'partner_cell'])


# ---------------------------------------------------------------- Figs 1-2 upgrade: one memory trial, ms by ms (dark filmstrips)

TRIAL_MODEL, TRIAL_TIMING = 'remote-dynamic-M0-head_only', (160, 4)
TRIAL_PHASES = {'cue': (1, 40), 'delay': (41, 200), 'input': (201, 360), 'decision': (361, 400)}       # ms stamps
TRIAL_SNAPSHOTS = (40, 80, 200, 240, 400)                                                           # cue end, +40, delay end, +40 input, decision end
TRIAL_WINDOWS = {40: (1, 40), 80: (41, 80), 200: (161, 200), 240: (201, 240), 400: (361, 400)}       # the 40 ms before each snapshot
TRIAL_DV_MV = 0.01


def _dense_case(folder, meta, i, kinds):
    """Chunks (segment kind, step stamps, values in float64 as (ms, field, cell)) of the dense records of one case, for the
    segments of the given kinds up to the end of the first decision, in time order."""
    first = meta['events'][_first_decision(meta)]
    end = first['start_ms'] + first['ms']
    segs = sorted((e for e in meta['dense_files'] if e['case'] == i and e['start_ms'] < end and e['kind'] in kinds), key=lambda e: e['start_ms'])
    for e in segs:
        d = io.read_bytes(folder / e['file'])
        with np.load(_io.BytesIO(d), allow_pickle=False) as z:
            if tuple(z['fields'].astype(str)) != FIELDS:
                raise RuntimeError('unexpected dense fields')
            ms = z['milliseconds'].astype(np.int64)
        for start, a in _npz_rows(d, 'values', 20):
            yield e['kind'], ms[start:start + len(a)], a.astype(np.float64)


def trial_tables():
    """Scene 1 (readout trained), 160-ms delay, 4 prefix windows, both cues: the CNS through one memory trial. Per ms and
    superclass, the cells spiking after each cue, the cells carrying an adaptation difference (the trace) and those with a
    voltage difference above 0.01 mV; per cell, spikes by phase and the cue difference at five snapshots. Gated against fu4
    (differential spikes per ms) and fig2_delay_end_cells (the delay-end difference)."""
    d, p = TRIAL_TIMING
    folder, meta = _record_meta(TRIAL_MODEL, d, p)
    i0, i1 = _record_pairs(meta)[(0, 'hidden')]
    kinds = ('cue', 'delay', 'prefix', 'decision')
    order = io.parquet_load(paths.M1 / 'neurons' / 'annotations.parquet', columns=['bodyId'])['bodyId'].to_numpy(np.int64)
    cmap = io.parquet_load(SC_RUNS / 'sc-mapping' / 'cell_map.parquet', columns=['bodyId', 'superclass'])
    sup = cmap.set_index('bodyId').loc[order, 'superclass'].fillna('unknown').astype(str).to_numpy()
    classes, cls = np.unique(sup, return_inverse=True)
    n = len(order)
    phase_spikes = {(c, ph): np.zeros(n, np.int64) for c in (0, 1) for ph in TRIAL_PHASES}
    spiking = {}
    for c, i in ((0, i0), (1, i1)):
        counts = []
        for _, ms, A in _dense_case(folder, meta, i, kinds):
            spk = A[:, 0] > 0
            for ph, (a, b) in TRIAL_PHASES.items():
                sel = (ms >= a) & (ms <= b)
                if sel.any():
                    phase_spikes[(c, ph)] += spk[sel].sum(0)
            for k, t in enumerate(ms):
                counts.append((int(t), np.bincount(cls, weights=spk[k], minlength=len(classes))))
        spiking[c] = dict(counts)
    snaps = {t: None for t in TRIAL_SNAPSHOTS}
    win_diff = {t: np.zeros(n, np.int64) for t in TRIAL_SNAPSHOTS}
    rows, e_spk, end_delay = [], {}, None
    for kind, ms, D in _dense_pair(folder, meta, i0, i1, kinds):
        dsp = D[:, 0] != 0
        for k, t in enumerate(ms):
            t = int(t)
            e_spk[t] = float(D[k, 0] @ D[k, 0])
            trace = D[k, 3] != 0
            volt = np.abs(D[k, 2]) > TRIAL_DV_MV
            tr_c = np.bincount(cls, weights=trace, minlength=len(classes))
            vo_c = np.bincount(cls, weights=volt, minlength=len(classes))
            ds_c = np.bincount(cls, weights=dsp[k], minlength=len(classes))
            sp0, sp1 = spiking[0][t], spiking[1][t]
            rows += [dict(ms=t, superclass=classes[j], spiking_cue0=int(sp0[j]), spiking_cue1=int(sp1[j]), trace_cells=int(tr_c[j]),
                          voltage_cells=int(vo_c[j]), differential_spike_cells=int(ds_c[j])) for j in range(len(classes))]
            if t in snaps:
                snaps[t] = (D[k, 3].copy(), D[k, 2].copy())
            for s, (a, b) in TRIAL_WINDOWS.items():
                if a <= t <= b:
                    win_diff[s] += dsp[k]
        if kind == 'delay':
            end_delay = int(ms[-1])
    fu4 = io.csv_load(paths.out('followup/fu4_state_dynamics.csv'), low_memory=False)
    ref = fu4[(fu4['model'] == TRIAL_MODEL) & (fu4['context'] == 0) & (fu4['visibility'] == 'hidden') & (fu4['delay_ms'] == d)
              & (fu4['prefix_windows'] == p)].set_index('ms')
    if not np.allclose([e_spk[t] for t in sorted(e_spk)], ref.loc[sorted(e_spk), 'energy_spikes'].to_numpy(float), rtol=0, atol=1e-9):
        raise RuntimeError('trial: differential spikes disagree with fu4')
    f2 = io.csv_load(paths.out(PV + 'fig2_delay_end_cells.csv'), low_memory=False)
    da_end, dv_end = snaps[end_delay]
    if not (np.allclose(np.round(dv_end, 6), f2['dv_s1_readout_mv'].to_numpy(float), rtol=0, atol=1e-9)
            and np.allclose(np.round(da_end, 6), f2['dadapt_s1_readout'].to_numpy(float), rtol=0, atol=1e-9)):
        raise RuntimeError('trial: the delay-end difference disagrees with fig2_delay_end_cells')
    tc = pd.DataFrame(rows)
    tc['cells'] = tc['superclass'].map(pd.Series(np.bincount(cls, minlength=len(classes)), index=classes))
    tc['source_file'] = f'studies/brain-sokoban-v1/mechanisms-v1/runs/record-{TRIAL_MODEL}/dense (free-after-d{d}-p{p})'
    csvout.write_table(tc, PV + 'trial_timecourse.csv', title=f'One memory trial (scene 1, readout trained; {d}-ms delay, {p} prefix windows), ms by ms and by superclass', describe={
        'ms': 'Step (ms from cue onset; cue 1-40, delay 41-200, input 201-360, first decision 361-400)', 'superclass': 'Superclass',
        'spiking_cue0': 'Cells spiking in this ms, cue 0', 'spiking_cue1': 'Cells spiking in this ms, cue 1',
        'trace_cells': 'Cells whose adaptation differs between the cues (the trace)', 'voltage_cells': f'Cells whose membrane potential differs by > {TRIAL_DV_MV} mV',
        'differential_spike_cells': 'Cells spiking differently after the two cues in this ms', 'cells': 'Cells in the superclass'}, key=['ms', 'superclass'],
        notes='Gated against fu4 (differential spikes per ms) and fig2_delay_end_cells (the delay-end difference).')
    cells = pd.DataFrame(dict(cell=np.arange(n), superclass=sup))
    for c in (0, 1):
        for ph in TRIAL_PHASES:
            cells[f'spikes_{ph}_cue{c}'] = phase_spikes[(c, ph)]
    for t in TRIAL_SNAPSHOTS:
        cells[f'dadapt_t{t}'] = np.round(snaps[t][0], 6)
        cells[f'dv_t{t}_mv'] = np.round(snaps[t][1], 6)
        cells[f'diffspike_w{t}'] = win_diff[t]
    cells['source_file'] = f'studies/brain-sokoban-v1/mechanisms-v1/runs/record-{TRIAL_MODEL}/dense (free-after-d{d}-p{p})'
    csvout.write_table(cells, PV + 'trial_cells.csv', title=f'One memory trial per cell (scene 1, readout trained; {d}-ms delay, {p} prefix windows)', describe={
        'cell': 'Model cell index', 'superclass': 'Superclass',
        **{f'spikes_{ph}_cue{c}': f'Spikes in the {ph} phase ({a}-{b} ms), cue {c}' for c in (0, 1) for ph, (a, b) in TRIAL_PHASES.items()},
        **{f'dadapt_t{t}': f'Adaptation, cue 1 minus cue 0, at ms {t}' for t in TRIAL_SNAPSHOTS},
        **{f'dv_t{t}_mv': f'Membrane potential, cue 1 minus cue 0, at ms {t} (mV)' for t in TRIAL_SNAPSHOTS},
        **{f'diffspike_w{t}': f'ms with a differential spike in {TRIAL_WINDOWS[t][0]}-{TRIAL_WINDOWS[t][1]}' for t in TRIAL_SNAPSHOTS}},
        key=['cell'], notes='Rounded to 1e-6; the ms-200 columns equal fig2_delay_end_cells (scene 1).')


# ---------------------------------------------------------------- Fig. 7: disease steals access, not memories

CASES = {'C01': 'pan-neuronal Aβ42 (Kv4 loss)', 'C04': 'late Aβ42 (Kenyon-cell loss)', 'C13': 'fragile X (weaker antennal-lobe inhibition)',
         'C16': 'tau (less excitable Kenyon cells)'}
SCENE1_LEARNERS = {'remote-dynamic-M0-head_only': 'readout', 'remote-dynamic-M0-joint': 'brain and readout'}
RESCUE_LEARNERS = {'M0-head_only': 'readout', 'M0-joint': 'brain and readout'}
# AD Fly Cell Atlas head annotation -> model cells (Phase 4 data need 4): by cell type where the names correspond, else none
# (those cells, like every unmatched neuron, take the pan-neuronal value)
ADFCA_CROSSWALK = [('lamina monopolar neuron L1', 'type', r'^L1$'), ('lamina monopolar neuron L2', 'type', r'^L2$'),
                   ('lamina monopolar neuron L3', 'type', r'^L3$'), ('lamina monopolar neuron L4', 'type', r'^L4$'),
                   ('lamina monopolar neuron L5', 'type', r'^L5$'), ('lamina wide-field 1 neuron', 'type', r'^Lawf1$'),
                   ('lamina wide-field 2 neuron', 'type', r'^Lawf2$'), ('lamina intrinsic neuron', 'type', r'^Lai$'),
                   ('columnar neuron T1', 'type', r'^T1$'), ('T neuron T2', 'type', r'^T2$'), ('T neuron T2a', 'type', r'^T2a$'),
                   ('T neuron T3', 'type', r'^T3$'), ('T neuron T4/T5', 'type', r'^T[45][a-d]$'), ('centrifugal neuron C2', 'type', r'^C2$'),
                   ('centrifugal neuron C3', 'type', r'^C3$'), ('distal medullary amacrine neuron Dm2', 'type', r'^Dm2$'),
                   ('distal medullary amacrine neuron Dm3', 'type', r'^Dm3[a-c]?$'), ('distal medullary amacrine neuron Dm8', 'type', r'^Dm8[ab]?$'),
                   ('distal medullary amacrine neuron Dm10', 'type', r'^Dm10$'), ('medullary intrinsic neuron Mi1', 'type', r'^Mi1$'),
                   ('medullary intrinsic neuron Mi4', 'type', r'^Mi4$'), ('medullary intrinsic neuron Mi9', 'type', r'^Mi9$'),
                   ('medullary intrinsic neuron Mi15', 'type', r'^Mi15$'), ('transmedullary neuron Tm1', 'type', r'^Tm1$'),
                   ('transmedullary neuron Tm2', 'type', r'^Tm2$'), ('transmedullary neuron Tm20', 'type', r'^Tm20$'),
                   ('transmedullary neuron Tm29', 'type', r'^Tm29$'), ('transmedullary neuron Tm3a', 'type', r'^Tm3$'),
                   ('transmedullary neuron Tm4', 'type', r'^Tm4$'), ('transmedullary neuron Tm5ab', 'type', r'^Tm5[ab]$'),
                   ('transmedullary neuron Tm9', 'type', r'^Tm9$'), ('transmedullary Y neuron TmY14', 'type', r'^TmY14$'),
                   ('transmedullary Y neuron TmY3', 'type', r'^TmY3$'), ('transmedullary Y neuron TmY4', 'type', r'^TmY4$'),
                   ('transmedullary Y neuron TmY5a', 'type', r'^TmY5a$'), ('transmedullary Y neuron TmY8', 'type', r'^TmY8$'),
                   ('lobula columnar neuron LC10a', 'type', r'^LC10a$'), ('lobula columnar neuron LC12', 'type', r'^LC12$'),
                   ('lobula columnar neuron LC16', 'type', r'^LC16$'), ('lobula columnar neuron LC17', 'type', r'^LC17$'),
                   ('lobula plate-lobula columnar LPLC1', 'type', r'^LPLC1$'), ('lobula plate-lobula columnar LPLC2', 'type', r'^LPLC2$'),
                   ('alpha/beta Kenyon cell', 'type', r'^KCab'), ("alpha'/beta' Kenyon cell", 'type', r"^KCa'b'"), ('gamma Kenyon cell', 'type', r'^KCg'),
                   ('olfactory projection neuron', 'class', r'^ALPN$')]
ADFCA_PAN = 'pan-neuronal (CNS neuron)'


def fig7_tables(names):
    # A: the credibility pipeline (G2), counts only for the attrition (inventory P16)
    ct = _read(PATH.parent / 'credibility-table.csv')
    g2 = io.json_load(PATH.parent / 'g2-shortlist.json')
    retained = sorted(r['candidate'] for r in g2['retained'])
    if sorted(ct.loc[ct['decision'] == 'retained', 'candidate']) != retained or retained != sorted(CASES):
        raise RuntimeError('credibility table and G2 shortlist disagree')
    truth = lambda s: s.astype(str).str.lower().eq('true')  # noqa: E731
    stages = [('fly pathologies in the literature', len(ct)), ('evidence, mechanism and operator (C1–C3)', int(truth(ct['c1_c3_pass']).sum())),
              ('healthy circuits match the fly (C4)', int(truth(ct['c4_pass']).sum())), ('reproduces the fly phenotype (C5)', int(truth(ct['c5_pass']).sum())),
              ('robust in two brains (C6)', int(truth(ct['c6_pass']).sum())), ('retained (ranked, at most 4)', len(retained))]
    fun = pd.DataFrame([dict(stage=s, order=i, cases=n) for i, (s, n) in enumerate(stages)])
    fun['source_file'] = paths.rel(PATH.parent / 'credibility-table.csv') + ', ' + paths.rel(PATH.parent / 'g2-shortlist.json')
    csvout.write_table(fun, PV + 'fig7_funnel.csv', title='The credibility pipeline: literature fly pathologies passing each criterion', describe={
        'stage': 'Criterion passed (C1-C6 as declared before scoring)', 'order': 'Order', 'cases': 'Cases'}, key=['order'],
        notes='Per-case reasons are in credibility-table.csv (Table S1).')
    cas = pd.DataFrame([dict(code=c, case=n, condition=ct.set_index('candidate').loc[c, 'condition'], operator=ct.set_index('candidate').loc[c, 'operator'])
                        for c, n in CASES.items()])
    cas['source_file'] = paths.rel(PATH.parent / 'credibility-table.csv')
    csvout.write_table(cas, PV + 'fig7_cases.csv', title='The four retained disease cases', describe={
        'code': 'Repository identifier', 'case': 'Reader-facing name', 'condition': 'Fly condition as published', 'operator': 'Model operator'}, key=['code'])
    # B, C: lesion after learning (regime A, the analysed learners) and learning under the lesion (regime B, 6 replicates per dose)
    ra = _read(PATH / 'phase3_regime_a.csv')
    ra = ra[(ra['split'] == 'train') & ra['model'].isin(SCENE1_LEARNERS) & (ra['condition'].str.startswith('C01-') | (ra['condition'] == 'healthy'))]
    rows = [dict(regime='lesion after learning', trained_parameters=SCENE1_LEARNERS[r['model']], dose=float(r['dose']) if r['condition'] != 'healthy' else 0.0,
                 replicate=0, trained_solved=int(r['solved']), trained_histories=int(r['histories'])) for _, r in ra.iterrows()]
    camp = _read(SUPP / 'campaign_runs.csv')
    rb = camp[camp['job'].str.match(r'^b-C01-dose[0-9.]+-M0-(head_only|joint)-r\d$')]
    for _, r in rb.iterrows():
        m = re.match(r'^b-C01-dose([0-9.]+)-M0-(head_only|joint)-r(\d)$', r['job'])
        rows.append(dict(regime='learning under the lesion', trained_parameters='readout' if m.group(2) == 'head_only' else 'brain and readout',
                         dose=float(m.group(1)), replicate=int(m.group(3)), trained_solved=int(r['trained_solved']), trained_histories=int(r['trained_histories'])))
    la = pd.DataFrame(rows)
    if len(la[la['regime'] == 'learning under the lesion']) != 48:
        raise RuntimeError('regime B: expected 48 runs')
    la['source_file'] = paths.rel(PATH / 'phase3_regime_a.csv') + '; ' + paths.rel(SUPP / 'campaign_runs.csv')
    csvout.write_table(la, PV + 'fig7_lost_access.csv', title='Trained histories solved: the pan-neuronal Kv4-loss lesion after learning vs learning under it (scene 1)', describe={
        'regime': 'Lesion after learning (the analysed learners) or learning under the lesion (retrained, 6 replicates per dose)',
        'trained_parameters': 'Trained parameters', 'dose': 'Lesion dose (x calibrated)', 'replicate': 'Replicate (0: the analysed learner)',
        'trained_solved': 'Trained-timing histories solved', 'trained_histories': 'Trained-timing histories'}, key=['regime', 'trained_parameters', 'dose', 'replicate'])
    # D: relearning after the lesion (regime C), readout trained; the joint runs' recovery is class c (P15)
    rc = _read(PATH / 'regime_c.csv')
    rc = rc[rc['mode'] == 'head_only'][['dose', 'replicate', 'solved_0', 'solved_50', 'solved_100', 'total', 'recovery_fraction']].copy()
    rc.insert(0, 'trained_parameters', 'readout')
    rc['source_file'] = paths.rel(PATH / 'regime_c.csv')
    csvout.write_table(rc, PV + 'fig7_relearning.csv', title='Relearning after the lesion: trained histories solved before, halfway and after retraining the readout (scene 1)', describe={
        'trained_parameters': 'Trained parameters', 'dose': 'Lesion dose', 'replicate': 'Replicate', 'solved_0': 'Solved after the lesion, before relearning',
        'solved_50': 'After 50 relearning updates', 'solved_100': 'After 100 updates', 'total': 'Trained histories',
        'recovery_fraction': '(solved after 100 - before) / (total - before)'}, key=['dose', 'replicate'],
        notes='Joint (brain and readout) relearning is class c (inventory P15) and not routed.')
    # E: late restoration of excitability after learning under the lesion (T19), with its tests
    rs = _read(PATH / 'rescue.csv')
    rs = rs[rs['learner'].isin(RESCUE_LEARNERS)].copy()
    rs['trained_parameters'] = rs['learner'].map(RESCUE_LEARNERS)
    rs = rs[['trained_parameters', 'dose', 'replicate', 'trained_solved_as_trained', 'trained_solved_restored']]
    rs['replicate'] = rs['replicate'].astype(int)
    rs['source_file'] = paths.rel(PATH / 'rescue.csv')
    csvout.write_table(rs, PV + 'fig7_restoration.csv', title='Late restoration: memories formed under the lesion, tested under it (as trained) and with excitability restored', describe={
        'trained_parameters': 'Trained parameters', 'dose': 'Lesion dose the run learned under', 'replicate': 'Replicate',
        'trained_solved_as_trained': 'Trained histories solved under the lesion', 'trained_solved_restored': 'Solved with the lesion removed'},
        key=['trained_parameters', 'dose', 'replicate'])
    # F: case profiles (relative to healthy), without the random-lesion controls (they belong to Fig. 3; P15)
    pr = _read(PATH / 'phase3_profiles.csv')
    pr = pr[~pr['condition'].str.endswith('random')].copy()
    pr['code'] = pr['condition'].str.split('-').str[0]
    pr['case'] = pr['code'].map(CASES)
    pr['dose'] = pr['condition'].str.extract(r'dose([0-9.]+)')[0].astype(float)
    pr = pr[['case', 'code', 'dose', 'memory_relative', 'movement_gf_relative', 'olfaction_mbon_accuracy_change', 'activity_relative']]
    pr['source_file'] = paths.rel(PATH / 'phase3_profiles.csv')
    csvout.write_table(pr, PV + 'fig7_profiles.csv', title='Phenotype profiles of the four cases, relative to the healthy brain (lesion after learning)', describe={
        'case': 'Disease case', 'code': 'Repository identifier', 'dose': 'Dose (x calibrated)', 'memory_relative': 'Memory (trained histories solved, relative)',
        'movement_gf_relative': 'Escape: giant-fibre spikes to looming, relative', 'olfaction_mbon_accuracy_change': 'Change in odour decoding accuracy from mushroom-body output',
        'activity_relative': 'Mean firing rate in the task, relative'}, key=['code', 'dose'])
    # H: the Shal-patterned lesion vs the uniform one (regime A protocol, the analysed scene-1 readout learner)
    kv = _read(PATH / 'sc_kv4_conditions.csv')
    kv = kv[(kv['protocol'] == 'regime-a') & kv['weights'].isin(['class', 'global'])]
    hrows = [dict(pattern='Shal-patterned (type, class means)' if r['weights'] == 'class' else 'Shal-patterned (type, global mean)', dose=float(r['dose']),
                  task_mean_hz=float(r['task_mean_hz']), share_above_100hz=float(r['fraction_above_100hz']),
                  trained_solved=int(r['trained_solved_remote-dynamic-M0-head_only'])) for _, r in kv.iterrows()]
    hm = ra[ra['model'] == 'remote-dynamic-M0-head_only']
    hrows += [dict(pattern='uniform', dose=float(r['dose']) if r['condition'] != 'healthy' else 0.0, task_mean_hz=float(r['task_mean_hz']), share_above_100hz=np.nan,
                   trained_solved=int(r['solved'])) for _, r in hm.iterrows()]
    hk = pd.DataFrame(hrows)
    chk = _read(PATH / 'sc_kv4_conditions.csv').set_index('condition').loc['check-uniform-dose0.5']
    u05 = hk[(hk['pattern'] == 'uniform') & (hk['dose'] == 0.5)].iloc[0]
    if not (np.isclose(chk['task_mean_hz'], u05['task_mean_hz'], rtol=1e-6) and int(chk['trained_solved_remote-dynamic-M0-head_only']) == int(u05['trained_solved'])):
        raise RuntimeError('the weighted code path does not reproduce the uniform lesion')
    hk['source_file'] = paths.rel(PATH / 'sc_kv4_conditions.csv') + '; ' + paths.rel(PATH / 'phase3_regime_a.csv')
    csvout.write_table(hk, PV + 'fig7_shal_runaway.csv', title='Uniform vs Shal-patterned Kv4-loss lesion of the same mean: task rate, runaway and memory (scene 1, readout trained)', describe={
        'pattern': 'Lesion pattern (uniform, or weighted per cell by its type\'s Shal expression; unmapped cells take their class or the global mean)',
        'dose': 'Mean dose (x calibrated)', 'task_mean_hz': 'Mean firing rate in the task (Hz)', 'share_above_100hz': 'Share of cells above 100 Hz (Shal-patterned runs)',
        'trained_solved': 'Trained histories solved (of 18)'}, key=['pattern', 'dose'], notes='The weighted code path reproduces the uniform lesion at dose 0.5 (checked).')
    # I: memories formed under the uniform lesion, tested under the Shal pattern of the same mean and with the lesion removed
    pl = _read(PATH / 'sc_kv4_regime_b.csv')
    pl['trained_parameters'] = np.where(pl['run'].str.contains('head_only'), 'readout', 'brain and readout')
    pl['replicate'] = pl['run'].str.extract(r'-r(\d)$')[0].astype(int)
    if not truth(pl['control_reproduces_verification']).all():
        raise RuntimeError('a pattern-lock control does not reproduce its verification')
    pl = pl[['trained_parameters', 'replicate', 'trained_as_trained', 'trained_kv4', 'trained_restored']]
    pl['source_file'] = paths.rel(PATH / 'sc_kv4_regime_b.csv')
    csvout.write_table(pl, PV + 'fig7_pattern_lock.csv', title='Memories formed under the uniform lesion (dose 0.5): trained histories solved under it, under the Shal pattern of the same mean, and with no lesion', describe={
        'trained_parameters': 'Trained parameters', 'replicate': 'Replicate', 'trained_as_trained': 'Under the uniform lesion it learned with',
        'trained_kv4': 'Under the Shal-patterned lesion of the same mean', 'trained_restored': 'With the lesion removed'}, key=['trained_parameters', 'replicate'])
    # J: the transcriptome-derived ageing operator (class-resolved) on two brains
    ag = _read(PATH / 'sc_aging.csv')
    ag = ag[(ag['operator'] == 'aging_class') & (ag['dose'] != 'verdict')].copy()
    ag['dose'] = ag['dose'].astype(float)
    ag['brain'] = ag['brain'].map({'p05': 'the model of the disease study', 'm0joint': 'scene 1, brain and readout trained'})
    ag = ag[['brain', 'dose', 'kc_ratio', 'task_mean_hz', 'fraction_over_100hz']]
    ag['source_file'] = paths.rel(PATH / 'sc_aging.csv')
    csvout.write_table(ag, PV + 'fig7_aging.csv', title='The transcriptome-derived ageing operator: Kenyon-cell odour response relative to the young brain, and network stability', describe={
        'brain': 'Brain', 'dose': 'Operator dose (x the measured old-vs-young change)', 'kc_ratio': 'Kenyon-cell odour response, relative to dose 0',
        'task_mean_hz': 'Mean firing rate in the task (Hz)', 'fraction_over_100hz': 'Share of cells above 100 Hz'}, key=['brain', 'dose'],
        notes='Passes C5 (fly Kenyon-cell odour responses flat or lower with age) and C6 on both brains.')
    # K: spikes per trial against cue information per decision
    en = _read(PATH / 'phase4_energy.csv')
    en = en[(en['source'] == 'phase3-regime-a') & ~en['condition'].str.endswith('random')].copy()
    en['code'] = np.where(en['condition'] == 'healthy', 'healthy', en['condition'].str.split('-').str[0])
    en['case'] = en['code'].map(CASES).fillna('healthy')
    en = en[['case', 'code', 'dose', 'task_mean_hz', 'spikes_per_trial', 'cue_bits_per_decision', 'first_push_correct_share']]
    en['source_file'] = paths.rel(PATH / 'phase4_energy.csv')
    csvout.write_table(en, PV + 'fig7_energy.csv', title='Spikes per trial and cue information per decision under the disease cases (lesion after learning)', describe={
        'case': 'Disease case or healthy', 'code': 'Repository identifier', 'dose': 'Dose', 'task_mean_hz': 'Mean firing rate (Hz)',
        'spikes_per_trial': 'Spikes per trial, all cells', 'cue_bits_per_decision': 'Mutual information between cue and first push (bits)',
        'first_push_correct_share': 'First pushes correct'}, key=['code', 'dose'])
    # G: the band, per cell; AD-FCA crosswalk, lesion rates (uniform and Shal-patterned, dose 1), ageing operator
    cm_ = io.parquet_load(SC_RUNS / 'sc-mapping' / 'cell_map.parquet')[['bodyId', 'type', 'class', 'superclass']]
    order = io.parquet_load(paths.M1 / 'neurons' / 'annotations.parquet', columns=['bodyId'])['bodyId'].to_numpy(np.int64)
    idx = pd.Series(np.arange(len(order)), index=order)
    cm_['cell'] = idx.loc[cm_['bodyId'].to_numpy(np.int64)].to_numpy()
    cm_ = cm_.sort_values('cell').reset_index(drop=True)
    desc = _read(PATH / 'sc_adfca_descriptive.csv')
    shal = desc[(desc['contrast'] == 'Abeta42') & (desc['outcome'] == 'Shal')].set_index('unit')
    pan = float(shal.loc[ADFCA_PAN, 'log2fc'])
    tst = _read(PATH / 'all_tests.csv')
    pan_test = tst[(tst['family'] == 'F14.SC.adfca.pan') & (tst['test'] == f'Abeta42:{ADFCA_PAN}:Shal')]
    if len(pan_test) != 1 or not np.isclose(float(pan_test['estimate'].iloc[0]), pan, rtol=1e-5):
        raise RuntimeError('pan-neuronal Shal change disagrees with the declared test')
    unit = pd.Series('', index=cm_.index, dtype=object)
    trows = []
    for u, field, pat in ADFCA_CROSSWALK:
        hit = cm_[field].fillna('').astype(str).str.match(pat) & unit.eq('')
        unit[hit] = u
        r = shal.loc[u]
        trows.append(dict(adfca_type=u, model_cells=int(hit.sum()), model_types=';'.join(sorted(cm_.loc[hit, 'type'].dropna().astype(str).unique())),
                          shal_log2fc=float(r['log2fc']), se=float(r['se']), p_value=float(r['p'])))
    listed = set(shal.index) - {ADFCA_PAN}
    for u in sorted(listed - {u for u, _, _ in ADFCA_CROSSWALK}):
        r = shal.loc[u]
        trows.append(dict(adfca_type=u, model_cells=0, model_types='', shal_log2fc=float(r['log2fc']), se=float(r['se']), p_value=float(r['p'])))
    tt = pd.DataFrame(trows)
    tt['source_file'] = paths.rel(PATH / 'sc_adfca_descriptive.csv') + '; ' + paths.rel(SC_RUNS / 'sc-mapping' / 'cell_map.parquet')
    csvout.write_table(tt, PV + 'fig7_adfca_types.csv', title='Aβ42-induced Shal mRNA change per AD Fly Cell Atlas cell type and the model cells it maps to', describe={
        'adfca_type': 'AD Fly Cell Atlas head annotation', 'model_cells': 'Model cells mapped by name (0: no corresponding model type)',
        'model_types': 'Model cell types mapped', 'shal_log2fc': 'Aβ42 vs control, adjusted log2 fold change of Shal', 'se': 'Standard error',
        'p_value': 'Unadjusted p (descriptive)'}, key=['adfca_type'], notes=f'Every other neuron takes the pan-neuronal change ({pan:+.3f} log2).')
    rates = io.parquet_load(SC_RUNS / 'sc-vulnerable' / 'rates.parquet')
    rates['cell'] = idx.loc[rates['bodyId'].to_numpy(np.int64)].to_numpy()
    rates = rates.sort_values('cell').reset_index(drop=True)
    want = {'rate_healthy': float(ra[(ra['condition'] == 'healthy')]['task_mean_hz'].iloc[0]),
            'rate_c01_dose1': float(ra[(ra['condition'] == 'C01-dose1')]['task_mean_hz'].iloc[0]),
            'rate_kv4_class_dose1': float(_read(PATH / 'sc_kv4_conditions.csv').set_index('condition').loc['kv4-class-dose1', 'task_mean_hz'])}
    for col, w in want.items():
        if not np.isclose(rates[col].mean(), w, rtol=1e-4):
            raise RuntimeError(f'{col}: mean {rates[col].mean()} disagrees with the condition table ({w})')
    kv4w = np.full(len(cm_), np.nan)
    for val, cells_ in io.json_load(PATH.parent / 'tasks' / 'sc-kv4-conditions.json')['weights']['class']:
        kv4w[np.asarray(cells_, int)] = float(val)
    if np.isnan(kv4w).any() or not np.isclose(kv4w.mean(), 1.0, atol=1e-6):
        raise RuntimeError('class Kv4 weights incomplete or not mean 1')
    ag_c = io.parquet_load(SC_RUNS / 'sc-aging' / 'cell_f.parquet')
    ag_c['cell'] = idx.loc[ag_c['bodyId'].to_numpy(np.int64)].to_numpy()
    ag_c = ag_c.sort_values('cell').reset_index(drop=True)
    band = pd.DataFrame(dict(cell=cm_['cell'], adfca_type=unit.to_numpy(), shal_log2fc_abeta42=np.where(unit.eq(''), pan, unit.map(shal['log2fc']).fillna(pan)),
                             adfca_tier=np.where(unit.eq(''), 'pan-neuronal', 'cell type'),
                             rate_healthy_hz=rates['rate_healthy'], rate_uniform_dose1_hz=rates['rate_c01_dose1'], rate_shal_dose1_hz=rates['rate_kv4_class_dose1'],
                             kv4_weight=kv4w, ageing_log2_factor=ag_c['log2f'], kenyon_cell=cm_['type'].fillna('').astype(str).str.startswith('KC')))
    band['source_file'] = 'runs/sc-mapping/cell_map.parquet; runs/sc-vulnerable/rates.parquet; tasks/sc-kv4-conditions.json; runs/sc-aging/cell_f.parquet'
    csvout.write_table(band, PV + 'fig7_cells.csv', title='Disease painted on the brain, per model cell: Aβ42 Shal change of its mapped type, task rates under the uniform and Shal-patterned Kv4-loss lesion (dose 1), its Shal weight and the ageing operator', describe={
        'cell': 'Model cell index', 'adfca_type': 'AD Fly Cell Atlas type mapped by name (empty: pan-neuronal value)', 'shal_log2fc_abeta42': 'Aβ42 vs control Shal log2 fold change',
        'adfca_tier': 'cell type or pan-neuronal', 'rate_healthy_hz': 'Task rate, healthy (Hz)', 'rate_uniform_dose1_hz': 'Task rate, uniform lesion dose 1 (Hz)',
        'rate_shal_dose1_hz': 'Task rate, Shal-patterned lesion of the same mean (Hz)', 'kv4_weight': 'Per-cell lesion weight from Shal expression (mean 1)',
        'ageing_log2_factor': 'Ageing operator: log2 old/young factor on the cell\'s K+ conductances', 'kenyon_cell': 'Kenyon cell'}, key=['cell'],
        notes='Mean rates gated against the condition tables; weights complete with mean 1; the pan-neuronal Shal change equals the declared test.')


# X8 (F13.X8, readout-only vs joint training from a neutral brain) is class (c) since the Fig. 1 review of 2026-10-22: it
# is reported in STAR Methods and Table S2 only, so no figure table carries it.
TEST_FAMILIES = ('F13.X1.', 'F13.X2.', 'F13.X3.', 'F13.X5.swap', 'F13.X5.cross', 'F13.X6.', 'F13.X7.', 'F13.X9.pairing',
                 'F13.X10.', 'F12.', 'F14.SC.', 'F14.DY.robust', 'F15.link')


# ---------------------------------------------------------------- Revision 3 (MaleCNS-style figure elements)
# The cue's flow by cell type (Figs 2H, 2K, 4I, 4J), the cue ensemble's wiring in detail (Fig. 3B) and the route network by class
# and by type (Figs 6G-H, S6G). Each table is gated against the record the figures already use.

CUE_FLOW_BRAIN = paths.M1.parent / 'mechanisms-supplement-v1' / 'runs' / f'preflight-{TRIAL_MODEL}' / 'brain.json'
ROUTE_KEYS = ('0', '30', '60', '90')
ROUTE_TRANSMITTERS = ('acetylcholine', 'gaba', 'glutamate')
# the route map's classes (Fig. 6G), one node each, placed by anatomy (brain, neck, VNC); other central-brain superclasses
# ('cb_*') are 'central brain'
ROUTE_CLASS = {'visual_projection': 'visual projection', 'visual_projection_tbc': 'visual projection', 'visual_centrifugal': 'visual projection',
               'ol_intrinsic': 'optic lobe', 'ol_sensory': 'optic lobe',
               'descending_neuron': 'descending', 'efferent_descending': 'descending', 'sensory_descending': 'descending',
               'ascending_neuron': 'ascending', 'efferent_ascending': 'ascending', 'sensory_ascending': 'ascending', 'sensory_ascending_tbc': 'ascending',
               'vnc_intrinsic': 'VNC', 'vnc_sensory': 'VNC', 'vnc_sensory_tbc': 'VNC',
               'vnc_motor': 'motor', 'vnc_efferent': 'motor', 'vnc_endocrine': 'motor'}
ROUTE_PAIR_STATES = (('0', 'intact'), ('60', '60% removed'))


def _route_class(superclass):
    s = str(superclass)
    return ROUTE_CLASS.get(s, 'central brain' if s.startswith('cb') else 'other')


def _cell_table():
    """Every model cell in annotation order (as figure_data/fd_cells, which P10 writes after this phase): type, superclass,
    transmitter and role (visual input, readout or other)."""
    ann = io.parquet_load(paths.M1 / 'neurons' / 'annotations.parquet', columns=['bodyId', 'type', 'superclass', 'nt', 'direct_input', 'observed_by_policy'])
    role = np.where(ann['direct_input'].to_numpy(bool), 'input', np.where(ann['observed_by_policy'].to_numpy(bool), 'readout', 'other'))
    return pd.DataFrame(dict(cell=np.arange(len(ann)), body_id=ann['bodyId'].to_numpy(np.int64), type=ann['type'].fillna('untyped').astype(str).to_numpy(),
                             superclass=ann['superclass'].fillna('unannotated').astype(str).to_numpy(),
                             transmitter=ann['nt'].fillna('unknown').astype(str).to_numpy(), role=role))


def _effective_weights(cells):
    """The fast synaptic weights of the scene-1 learner's brain as the simulator builds them (flybrain.brain: receptor signs,
    synapse cap, path gains and same-type gain, then fan-in normalization and w_syn), with the parameters recorded in its
    brain.json, in mV per presynaptic spike (rows postsynaptic). The learned per-group gains and thresholds are not applied.
    Gated: the matrix's SHA-256 equals the learner's recorded effective connectome (base_graph_sha256), and the receptor-rule
    changes equal the documented entries."""
    import dataclasses
    import hashlib
    import numba
    import scipy.sparse as sp
    jit = numba.config.DISABLE_JIT
    numba.config.DISABLE_JIT = True                               # no numba cache in the read-only package; only the weight rules are used
    try:
        from flybrain import brain as fb, connectome as cn
    finally:
        numba.config.DISABLE_JIT = jit
    meta = io.json_load(CUE_FLOW_BRAIN)
    fields = {f.name for f in dataclasses.fields(fb.LIFParams)}
    p = fb.LIFParams(**{k: v for k, v in meta['lif'].items() if k in fields})
    c = cn.load(paths.ROOT / 'flybrain' / 'cache', verbose=False)
    if not np.array_equal(c.neurons['bodyId'].to_numpy(np.int64), cells['body_id'].to_numpy(np.int64)):
        raise RuntimeError('cue flow: the connectome and the annotations differ in cell order')
    rec = fb._receptor(c, p)
    coo = c.W.tocsr().tocoo()                                    # the order of receptor_signs (as fig3_maps)
    pre_sign = c.neurons['sign'].to_numpy(np.float32)[coo.col] if 'sign' in c.neurons else np.sign(coo.data).astype(np.float32)
    if int(((rec.fast_sign != pre_sign) & (np.abs(coo.data) > 0)).sum()) != RECEPTOR_CHANGES['entries']:
        raise RuntimeError('cue flow: receptor-rule changes differ from the documented entries')
    W = fb._shaped_weights(c, p, rec)                            # as flybrain.brain.Brain.__init__ (full connectome: its own reference)
    tot = np.asarray(abs(W).sum(axis=1)).ravel()
    scale = np.clip((p.input_norm_ref / np.maximum(tot, 1.0)) ** p.input_norm_alpha, 0.02, 1.0).astype(np.float32)
    W = (sp.diags(scale) @ W).tocsr()
    W = W.copy()
    W.data = W.data * np.float32(p.w_syn)
    h = hashlib.sha256()
    for v in (W.indptr, W.indices, W.data):
        h.update(v.tobytes())
    if (h.hexdigest() != meta['base_graph_sha256'] or c.W.nnz != int(meta['stored_connectome_edges'])
            or int(np.count_nonzero(W.data)) != int(meta['effective_nonzero_edges'])):
        raise RuntimeError("cue flow: the rebuilt effective connectome differs from the learner's brain.json")
    W = W.astype(np.float64)
    W.eliminate_zeros()
    return W, c.W.tocsr(), p


def fig2_cue_flow(cells):
    """How the cue travels in the scene-1 learner (readout trained) at its trained timing (160-ms delay, 4 prefix windows), by
    cell type, from the per-cell record of one trial (trial_cells; the dense 1-ms record, gated against fu4): written (spiking
    difference between the cues in the cue phase), delayed (in the delay), held (the adaptation difference at the end of the
    delay: the trace) and read (spiking difference in the first decision). Per type pair, the cue drive at the first decision:
    the effective synaptic weight times the presynaptic spiking difference, summed over the pair's connections. Gated: the
    trace cells equal trial_timecourse at the end of the delay."""
    W, Wraw, p = _effective_weights(cells)
    tc = io.csv_load(paths.out(PV + 'trial_cells.csv'), low_memory=False).set_index('cell').reindex(cells['cell'])
    tt = io.csv_load(paths.out(PV + 'trial_timecourse.csv'), low_memory=False)

    def diff(ph):
        return (tc[f'spikes_{ph}_cue1'] - tc[f'spikes_{ph}_cue0']).to_numpy(np.float64)
    write, delay, read = diff('cue'), diff('delay'), diff('decision')
    end = TRIAL_PHASES['delay'][1]
    hold = tc[f'dadapt_t{end}'].to_numpy(np.float64)
    if int((hold != 0).sum()) != int(tt.loc[tt['ms'] == end, 'trace_cells'].sum()):
        raise RuntimeError('cue flow: trace cells disagree with trial_timecourse at the end of the delay')
    c = cells.assign(write=np.abs(write), delay=np.abs(delay), hold=np.abs(hold), read=np.abs(read))
    ty = c.groupby('type').agg(cells=('cell', 'size'), superclass=('superclass', 'first'), role=('role', lambda r: ','.join(sorted(set(r)))),
                               transmitter=('transmitter', 'first'), write=('write', 'sum'), delay=('delay', 'sum'), hold=('hold', 'sum'),
                               hold_cells=('hold', lambda v: int((v > 0).sum())), read=('read', 'sum')).reset_index()
    pre = np.flatnonzero(read != 0)
    t_ = cells['type'].to_numpy()
    E = W[:, pre].tocoo()
    contrib = E.data * read[pre][E.col]
    e = pd.DataFrame(dict(pre=t_[pre][E.col], post=t_[E.row], drive=contrib, absd=np.abs(contrib)))
    edges = e.groupby(['pre', 'post']).agg(drive=('drive', 'sum'), abs_drive=('absd', 'sum'), connections=('drive', 'size')).reset_index()
    R = Wraw[:, pre].tocoo()
    syn = pd.DataFrame(dict(pre=t_[pre][R.col], post=t_[R.row], synapses=np.abs(R.data).astype(np.float64))).groupby(['pre', 'post'])['synapses'].sum()
    edges['synapses'] = syn.reindex(pd.MultiIndex.from_frame(edges[['pre', 'post']])).fillna(0.0).to_numpy()
    src = f'studies/brain-sokoban-v1/mechanisms-v1/runs/record-{TRIAL_MODEL}/dense (free-after-d160-p4) via trial_cells; flybrain/cache; ' + paths.rel(CUE_FLOW_BRAIN)
    ty['source_file'] = src
    edges['source_file'] = src
    csvout.write_table(ty, PV + 'fig2_cue_flow_types.csv', title='How the cue travels through each cell type in one trial (scene 1, readout trained; 160-ms delay, 4 prefix windows)', describe={
        'type': 'Cell type (untyped cells together)', 'cells': 'Cells of the type', 'superclass': 'Superclass', 'role': 'Roles of its cells (input, readout, other)',
        'transmitter': 'Transmitter', 'write': 'Spikes that differ between the cues in the cue phase (1-40 ms), summed over the type\'s cells',
        'delay': 'Spikes that differ in the delay (41-200 ms)', 'hold': 'Adaptation difference at the end of the delay (ms 200), |cue 1 - cue 0| summed: the trace',
        'hold_cells': 'Cells of the type that hold the trace', 'read': 'Spikes that differ in the first decision (361-400 ms)'}, key=['type'],
        notes='From trial_cells (gated against fu4); trace cells gated against trial_timecourse.')
    csvout.write_table(edges, PV + 'fig2_cue_flow_edges.csv', title='Cue drive at the first decision between cell types (scene 1, readout trained; 160-ms delay, 4 prefix windows)', describe={
        'pre': 'Presynaptic cell type', 'post': 'Postsynaptic cell type',
        'drive': 'Effective synaptic weight x presynaptic spiking difference, summed over connections (mV)',
        'abs_drive': 'The same with each connection\'s contribution taken absolute (mV)', 'connections': 'Connections from a cell that spikes differently',
        'synapses': 'Their synapses (stored connectome)'}, key=['pre', 'post'],
        notes=(f'Effective weights: the simulator\'s connection rules (receptor signs "{p.receptor_net_rule}", cap {p.conn_cap:g}, path gains, same-type gain '
               f'{p.same_type_gain:g}), fan-in normalization and w_syn {p.w_syn:g} mV, as recorded for the learner; learned gains not applied. Gated against '
               'brain.json (stored and effective connections) and the documented receptor-rule entries.'))
    return ty, edges


def fig4_access_tiles(cells, ty, edges):
    """Per carrier type with genetic access (Fig. 4I): its cells, trace carriers, readout and input cells, and the share of its
    cue drive (fig2_cue_flow_edges) that goes into readout types. Gated: carriers equal fig4_access_types."""
    a = io.csv_load(paths.out(PV + 'fig4_access_types.csv'))
    x = io.csv_load(paths.out(PV + 'fig4_cell_expression.csv'), low_memory=False)[['cell', 'carrier_scene1']]
    c = cells.merge(x, on='cell', how='left')
    car = c['carrier_scene1'].astype(str).str.lower().eq('true')
    ro = set(ty.loc[ty['role'].str.contains('readout'), 'type'])
    rows = []
    for tname in a['type']:
        m = (c['type'] == tname).to_numpy()
        o = edges[(edges['pre'] == tname) & (edges['post'] != tname)]
        rows.append(dict(type=tname, cells=int(m.sum()), carriers=int((m & car).sum()), readout_cells=int((m & (c['role'] == 'readout')).sum()),
                         input_cells=int((m & (c['role'] == 'input')).sum()),
                         drive_to_readout_share=float(o.loc[o['post'].isin(ro), 'abs_drive'].sum() / o['abs_drive'].sum())))
    df = pd.DataFrame(rows)
    if not (df.set_index('type')['carriers'] == a.set_index('type')['carriers_scene1']).all():
        raise RuntimeError('fig4 access tiles: carriers disagree with fig4_access_types')
    df['source_file'] = 'plan_v2/fig4_access_types.csv; plan_v2/fig4_cell_expression.csv; plan_v2/fig2_cue_flow_edges.csv'
    csvout.write_table(df, PV + 'fig4_access_tiles.csv', title='Carrier types with genetic access: cells, trace carriers and where their cue drive goes', describe={
        'type': 'Cell type', 'cells': 'Cells of the type', 'carriers': 'Scene-1 trace carriers among them', 'readout_cells': 'Readout cells among them',
        'input_cells': 'Visual input cells among them', 'drive_to_readout_share': 'Share of the type\'s cue drive at the first decision (to other types) that goes into readout types'},
        key=['type'], notes='Carriers gated against fig4_access_types.')


def fig3_wiring_detail(cells):
    """The cue ensembles' internal connections by the source cell's superclass, for the ensemble and each of its 20 matched sets
    (scenes 1 and 2), and the scene-1 ensemble's connections themselves (the matrix of Fig. 3B). Gated against fig3_set_wiring."""
    A = io.sparse_load(paths.ROOT / 'flybrain' / 'cache' / 'W_post_pre.npz').tocsr()
    A.eliminate_zeros()
    A = (A != 0).tocsr()
    sets = io.json_load(SUPP.parent / 'tasks' / 'x6-x7-sets.json')['sets']
    wiring = io.csv_load(paths.out(PV + 'fig3_set_wiring.csv')).set_index(['ensemble_scene', 'set'])['internal_connections']
    sup = cells['superclass'].to_numpy()
    rows, edges = [], None
    for key, s in sets.items():
        members = [('ensemble', np.asarray(s['cells']))] + [(f'matched {i + 1}', np.asarray(nl_['cells'])) for i, nl_ in enumerate(s['nulls'])]
        for name, cl in members:
            sub = A[cl][:, cl].tocoo()                             # rows postsynaptic, columns presynaptic
            if sub.nnz != int(wiring.loc[(X67_SCENE[key], name)]):
                raise RuntimeError(f'fig3 wiring detail: {X67_SCENE[key]} {name} disagrees with fig3_set_wiring')
            for sc_, k in pd.Series(sup[cl[sub.col]]).value_counts().items():
                rows.append(dict(ensemble_scene=X67_SCENE[key], set=name, source_superclass=sc_, connections=int(k)))
            if key == 'm0' and name == 'ensemble':
                edges = pd.DataFrame(dict(pre=cl[sub.col], post=cl[sub.row]))
    g = pd.DataFrame(rows)
    g['source_file'] = 'studies/brain-supplement-v2/tasks/x6-x7-sets.json; flybrain/cache/W_post_pre.npz'
    csvout.write_table(g, PV + 'fig3_set_groups.csv', title='Connections within the cue ensembles and their 20 matched sets, by the source cell\'s superclass', describe={
        'ensemble_scene': 'Scene of the ensemble', 'set': 'The ensemble or a matched set', 'source_superclass': 'Superclass of the presynaptic cell',
        'connections': 'Directed cell pairs within the set with at least one synapse, from cells of this superclass'},
        key=['ensemble_scene', 'set', 'source_superclass'], notes='Sums per set equal fig3_set_wiring (gated).')
    edges = edges.sort_values(['pre', 'post']).reset_index(drop=True)
    edges['source_file'] = g['source_file'].iloc[0]
    csvout.write_table(edges, PV + 'fig3_ensemble_edges.csv', title='The scene-1 cue ensemble\'s internal connections (the matrix of Fig. 3B)', describe={
        'pre': 'Presynaptic model cell', 'post': 'Postsynaptic model cell'}, key=['pre', 'post'],
        notes='Count equal to fig3_set_wiring (scene 1, ensemble; gated).')


def fig6_route_detail(A, alives, traffics, s, r, summary):
    """The route network by class and by type, from the walks of fig6_routes (same draws). Per level, the walks over each hop -
    input -> first relay, first relay -> readout (two-step walks), first -> second relay and second relay -> readout (three-step
    walks) - by the classes of the hop's two cells and its presynaptic transmitter (Fig. 6G); per cell type, the route traffic
    (Fig. 6H); and, intact and with 60% removed, the typed first and last hops (Fig. S6G). Gated: the hop sums rebuild the two-
    and three-step walks; three-step walks relative to intact equal fig6_routes_summary; type traffic sums to the cell traffic."""
    cells = _cell_table()
    role = cells['role'].to_numpy()
    cls = np.array([_route_class(x) for x in cells['superclass']], dtype=object)
    tr = cells['transmitter'].str.lower()
    tr = np.where(tr.isin(ROUTE_TRANSMITTERS), tr, 'other')
    types = cells['type'].to_numpy()
    C = A.tocoo()
    rows, base3 = [], None
    want = summary.set_index('share_removed')['walks3_relative']
    for key in ROUTE_KEYS:
        live = alives[key].astype(np.float64)
        f1 = (A @ s) * live
        b1 = (A.T @ r) * live
        b2 = (A.T @ b1) * live
        f2 = (A @ f1) * live
        walks2 = float((f1 * b1).sum())
        walks3 = float(((A @ f2) * live)[r > 0].sum())
        base3 = walks3 if key == '0' else base3
        hops = []
        m = (s[C.col] > 0) & (live[C.row] > 0)
        x = C.row[m]
        hops.append(('input', 'relay 1', np.full(int(m.sum()), 'visual input', object), cls[x], tr[C.col[m]], C.data[m] * (b1[x] + b2[x])))
        m = (live[C.col] > 0) & (role[C.row] == 'readout')
        hops.append(('relay 1', 'readout', cls[C.col[m]], cls[C.row[m]], tr[C.col[m]], C.data[m] * f1[C.col[m]]))
        m = (live[C.col] > 0) & (live[C.row] > 0)
        hops.append(('relay 1', 'relay 2', cls[C.col[m]], cls[C.row[m]], tr[C.col[m]], C.data[m] * f1[C.col[m]] * b1[C.row[m]]))
        m = (live[C.col] > 0) & (role[C.row] == 'readout')
        hops.append(('relay 2', 'readout', cls[C.col[m]], cls[C.row[m]], tr[C.col[m]], C.data[m] * f2[C.col[m]]))
        sums = {}
        for sa, sb, a_, b_, t_, w_ in hops:
            sums[(sa, sb)] = float(w_.sum())
            d = pd.DataFrame(dict(a=a_, b=b_, t=t_, w=w_)).groupby(['a', 'b', 't'])['w'].sum()
            rows += [dict(level=int(key), share_removed=int(key) / 100, stage_a=sa, class_a=ca, stage_b=sb, class_b=cb, transmitter=tt, walks=float(v))
                     for (ca, cb, tt), v in d.items() if v > 0]
        ok = (np.isclose(sums[('input', 'relay 1')], walks2 + walks3, rtol=1e-9) and np.isclose(sums[('relay 1', 'readout')], walks2, rtol=1e-9)
              and np.isclose(sums[('relay 1', 'relay 2')], walks3, rtol=1e-9) and np.isclose(sums[('relay 2', 'readout')], walks3, rtol=1e-9)
              and np.isclose(walks3 / base3, float(want.loc[int(key) / 100]), rtol=1e-9))
        if not ok:
            raise RuntimeError(f'route detail: the hop sums at {key}% disagree with the walks of fig6_routes')
    rc = pd.DataFrame(rows)
    rc['source_file'] = 'flybrain/cache/W_post_pre.npz; studies/brain-sokoban-v1/mechanisms-v1/neurons/annotations.parquet; draws of plan_v2/fig6_routes.csv'
    csvout.write_table(rc, PV + 'fig6_route_classes.csv', title='The route network by cell class: input-to-readout walks of length 2-3 over each hop, intact and after random removal of 30, 60 and 90% of the other cells', describe={
        'level': 'Share of the other cells removed (%)', 'share_removed': 'Share of the other cells removed',
        'stage_a': 'Hop start: visual input, first relay or second relay', 'class_a': 'Class of the presynaptic cell',
        'stage_b': 'Hop end: first relay, second relay or readout', 'class_b': 'Class of the postsynaptic cell',
        'transmitter': 'Transmitter of the presynaptic cell (acetylcholine, GABA, glutamate or other)', 'walks': 'Walks that use the hop'},
        key=['level', 'stage_a', 'class_a', 'stage_b', 'class_b', 'transmitter'],
        notes='Gated: per level the hop sums rebuild the two- and three-step walks, and three-step walks relative to intact equal fig6_routes_summary.')
    other = (role == 'other') & (types != 'untyped')
    tt = pd.DataFrame(dict(type=types[other], superclass=cells['superclass'].to_numpy()[other], **{f'traffic_{k}': traffics[k][other] for k in ROUTE_KEYS}))
    agg = tt.groupby('type').agg(**{f'traffic_{k}': (f'traffic_{k}', 'sum') for k in ROUTE_KEYS}, cells=('type', 'size'), superclass=('superclass', 'first')).reset_index()
    agg = agg[agg[[f'traffic_{k}' for k in ROUTE_KEYS]].sum(axis=1) > 0].reset_index(drop=True)
    for k in ROUTE_KEYS:
        rest = float(traffics[k][(role == 'other') & (types == 'untyped')].sum())
        if not np.isclose(float(agg[f'traffic_{k}'].sum()) + rest, float(traffics[k].sum()), rtol=1e-12):
            raise RuntimeError(f'route types: type traffic does not sum to the cell traffic at {k}%')
    agg['source_file'] = 'plan_v2/fig6_routes.csv (traffic); studies/brain-sokoban-v1/mechanisms-v1/neurons/annotations.parquet'
    csvout.write_table(agg, PV + 'fig6_route_types.csv', title='Route traffic per cell type, intact and after random removal of 30, 60 and 90% of the other cells', describe={
        **{f'traffic_{k}': f'Input-to-readout walks of length 2 and 3 through the type\'s cells, {k}% removed' for k in ROUTE_KEYS},
        'type': 'Cell type (typed cells other than the input and readout cells)', 'cells': 'Cells of the type', 'superclass': 'Superclass'},
        key=['type'], notes='With the untyped cells, sums to the cell traffic of fig6_routes at every level (gated).')
    pairs = []
    for key, state in ROUTE_PAIR_STATES:
        live = alives[key].astype(np.float64)
        f1 = (A @ s) * live
        f2 = (A @ f1) * live
        b1 = (A.T @ r) * live
        b2 = (A.T @ b1) * live
        relay = (role == 'other') & alives[key]
        inp, rdo = np.flatnonzero(s > 0), np.flatnonzero(r > 0)
        Ai = A[:, inp].tocoo()                                    # first hop: input i -> relay m, the walks continuing from m
        keep = relay[Ai.row]
        m_, i_ = Ai.row[keep], inp[Ai.col[keep]]
        first = pd.DataFrame(dict(pre=types[i_], post=types[m_], walks=(b1 + b2)[m_])).groupby(['pre', 'post'])['walks'].sum().reset_index()
        first['hop'] = 'first'
        Ar = A[rdo, :].tocoo()                                    # last hop: relay m -> readout, the walks arriving at m
        keep = relay[Ar.col]
        rr, m_ = rdo[Ar.row[keep]], Ar.col[keep]
        last = pd.DataFrame(dict(pre=types[m_], post=types[rr], walks=(f1 + f2)[m_])).groupby(['pre', 'post'])['walks'].sum().reset_index()
        last['hop'] = 'last'
        rel = relay.astype(np.float64)
        if not np.isclose(float(first['walks'].sum() + last['walks'].sum()), float((traffics[key] * rel).sum() + (f1 * b1 * rel).sum()), rtol=1e-9):
            raise RuntimeError(f'route pairs: first and last hops disagree with the cell traffic at {key}%')
        e = pd.concat([first, last])
        e = e[e['walks'] > 0]
        e['state'] = state
        pairs.append(e)
    sp_ = pd.concat(pairs, ignore_index=True)
    sp_['source_file'] = rc['source_file'].iloc[0]
    csvout.write_table(sp_, PV + 's6_route_pairs.csv', title='Route walks over typed first and last hops, intact and with 60% of the other cells removed', describe={
        'pre': 'Presynaptic cell type', 'post': 'Postsynaptic cell type', 'walks': 'Input-to-readout walks of length 2-3 that use the hop',
        'hop': 'first (input -> relay) or last (relay -> readout)', 'state': 'intact or 60% removed (draw 0)'}, key=['state', 'hop', 'pre', 'post'],
        notes='Gated: first plus last hop walks equal the relays\' traffic plus their two-step walks counted twice.')


def revision3_tables():
    cells = _cell_table()
    ty, edges = fig2_cue_flow(cells)
    fig4_access_tiles(cells, ty, edges)
    fig3_wiring_detail(cells)


# Table S2 lists the declared tests this paper reports. Of the robust-readout families it reports training across threshold
# states (offsets with an SD of 1 uV); every adjusted p stays the value adjusted over the complete declared family, whose
# declared size is listed, so no correction is recomputed over a subset.
PAPER_ROBUST = ROBUST_CONDITION + ':'


def _family_sizes(a):
    """Tests per declared family: the sizes recorded from the study's complete table (declared/family_sizes.csv), so that a
    table that holds only the reported tests of a family still lists the size its correction used. A table may hold fewer
    tests of a declared family than were declared, never more, and no other family."""
    rows = a.groupby('family')['test'].size()
    dec = io.csv_load(DECLARED / 'family_sizes.csv').set_index('family')['tests'].astype(int)
    if not set(rows.index) <= set(dec.index) or (rows > dec.reindex(rows.index)).any():
        raise RuntimeError('tests: all_tests.csv disagrees with the declared family sizes')
    return dec


def tests_table():
    a = _read(PATH / 'all_tests.csv')
    keep = a['family'].astype(str).apply(lambda f: f.startswith(TEST_FAMILIES))
    size = _family_sizes(a)
    t = a.loc[keep, ['family', 'test', 'n', 'estimate', 'p_value', 'p_adjusted_family', 'correction', 'primary']].copy()
    t = t.rename(columns={'p_adjusted_family': 'p_adjusted'})
    robust = t['family'].astype(str).str.startswith('F14.DY.robust')
    t = t[~robust | t['test'].astype(str).str.startswith(PAPER_ROBUST)].copy()
    t.insert(t.columns.get_loc('correction'), 'family_tests', t['family'].map(size).astype(int))
    t['source_file'] = paths.rel(PATH / 'all_tests.csv')
    csvout.write_table(t, PV + 'tests.csv', title='Declared tests reported in this paper, with p values adjusted within each complete declared family', describe={
        'family': 'Declared test family', 'test': 'Test',
        'estimate': 'Point estimate of the tested quantity where the test records one (e.g. Spearman rho for F13.X10.fig2)',
        'p_value': 'Unadjusted p value',
        'p_adjusted': 'p value adjusted within the complete declared family (Holm or BH, see correction)',
        'family_tests': 'Tests in the complete declared family, all of which enter its adjustment',
        'correction': 'Multiple-comparison correction of the family (holm, bh or none)', 'primary': 'Primary family'},
        key=['family', 'test'])


def main():
    names = names_table()
    fig1_tables()
    fig2_tables(names)
    trial_tables()
    fig3_tables(names)
    fig3_maps()
    fig4_tables(names)
    fig5_tables(names)
    fig6_tables(names)
    fig6_routes()
    revision3_tables()
    fig7_tables(names)
    s1_tables(names)
    s2_tables(names)
    s3_tables()
    s4_tables()
    s5_tables()
    s6_tables()
    s7_tables()
    from panels.p12_link import link_tables         # the lesion link (amendment 0015)
    from panels.p12_video import video_tables       # the supplemental videos
    link_tables()
    video_tables()
    tests_table()
