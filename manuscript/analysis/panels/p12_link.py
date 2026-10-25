"""P12, the lesion link (amendment 0015; F15.link): where the cue trace is written and where it is read, routed from the study's
result tables (analyze_lesion_link.py, which checks every untouched condition against the earlier records) and the host's
per-cell record, with reader-facing names (called from p12_plan_v2.main).

Brain conditions: the trace written and held in one state and read in another, the full state of each history moved at the
end of its delay. States: healthy, and the disease-like loss of potassium current (C01 in the repository) at four doses.
Readouts: trained healthy (10 replicates), trained under the lesion (6 per dose), relearned under the lesion after healthy
training (3 per dose).
"""
import numpy as np
import pandas as pd

from msa import csvout, io, paths

PV = 'plan_v2/'
PATH = paths.ROOT / 'studies' / 'brain-pathology-v1' / 'results'
LINK = paths.ROOT / 'studies' / 'brain-pathology-v1' / 'runs' / 'lesion-link'
KIND = {'healthy-trained': 'trained healthy', 'lesion-trained': 'trained under the lesion', 'relearned': 'relearned under the lesion'}
SHOWCASE_DOSE, SHOWCASE = 1.0, (160, 4)


def _state(s):
    return 'healthy' if s == 'H' else 'lesion'


def _dose(s):
    return None if s == 'H' else float(s[1:])


def link_tables():
    out = io.csv_load(PATH / 'lesion_link_outcomes.csv')
    solved = out['statuses'].str.count(':solved')
    if not ((solved == out['solved']) & (out['histories'] == 18)).all():
        raise RuntimeError('link: solved counts disagree with the per-history statuses')
    t = out[['readout', 'kind', 'dose', 'replicate', 'condition', 'written_in', 'read_in', 'solved', 'push_mean', 'alignment_mean']].copy()
    t['readout_kind'] = t['kind'].map(KIND)
    t['trace_written'], t['trace_read'] = t['written_in'].map(_state), t['read_in'].map(_state)
    t['lesion_dose'] = [_dose(w) if w != 'H' else _dose(r) for w, r in zip(t['written_in'], t['read_in'])]
    t = t.rename(columns={'readout': 'code', 'condition': 'condition_code'}).drop(columns=['kind', 'written_in', 'read_in'])
    t['source_file'] = paths.rel(PATH / 'lesion_link_outcomes.csv')
    csvout.write_table(t, PV + 'link_outcomes.csv', title='Where the trace is written and where it is read: trained histories solved per readout and brain condition (scene 1, readout trained)', describe={
        'code': 'Repository identifier of the readout', 'dose': 'Lesion dose the readout learned or relearned under (empty: trained healthy)',
        'replicate': 'Replicate', 'condition_code': 'Repository code of the condition (written-read)', 'solved': 'Trained-timing histories solved (of 18)',
        'push_mean': 'First-decision push toward the cued target (mean over timings; logit of the cued choice minus the other)',
        'alignment_mean': "Cosine of the readout-feature cue difference with the readout's choice axis (mean over timings)",
        'readout_kind': 'Readout', 'trace_written': 'State in which the trace was written and held', 'trace_read': 'State in which it was read',
        'lesion_dose': 'Dose of the lesion state in the condition (empty: healthy throughout)'},
        key=['code', 'condition_code'], notes='Gated: solved equals the per-history statuses of the study table.')
    # the brain-level measures: per condition and timing, and the cosines between conditions
    st = io.csv_load(PATH / 'lesion_link_states.csv')
    pr = io.csv_load(PATH / 'lesion_link_pairs.csv')
    cells, describe = {}, {}
    show = [('H', 'H'), (f'L{SHOWCASE_DOSE:g}', f'L{SHOWCASE_DOSE:g}'), ('H', f'L{SHOWCASE_DOSE:g}'), (f'L{SHOWCASE_DOSE:g}', 'H')]
    z = io.npz_load(LINK / 'states.npz', keys=[f'{w}-{r}|{q}' for w, r in show for q in ('trace', 'window_v')])
    for w, r in show:
        c = f'{w}-{r}'
        for q, col in (('trace', 'trace_norm'), ('window_v', 'decision_v_norm')):
            v = z[f'{c}|{q}'].astype(np.float64)
            want = st[(st['condition'] == c) & (st['delay_ms'] == SHOWCASE[0]) & (st['prefix_windows'] == SHOWCASE[1])][col].iloc[0]
            if not np.isclose(np.linalg.norm(v), float(want), rtol=1e-5, atol=1e-9):
                raise RuntimeError(f'link: the per-cell {q} of {c} disagrees with the study table')
            k = f"{'trace' if q == 'trace' else 'decision_v'}_{_state(w)}_written_{_state(r)}_read"
            cells[k] = np.round(v, 6)
            describe[k] = (('Adaptation at the end of the delay' if q == 'trace' else 'Membrane potential over the first decision window (mV)')
                           + f', cue 1 minus cue 0; trace written {_state(w)}, read {_state(r)}')
    a, b = cells['decision_v_healthy_written_healthy_read'], cells['decision_v_lesion_written_lesion_read']
    want = pr[(pr['quantity'] == 'window_v') & (pr['delay_ms'] == SHOWCASE[0]) & (pr['prefix_windows'] == SHOWCASE[1])
              & (pr['condition_a'] == 'H-H') & (pr['condition_b'] == f'L{SHOWCASE_DOSE:g}-L{SHOWCASE_DOSE:g}')]['cosine'].iloc[0]
    if not np.isclose(a @ b / np.sqrt((a @ a) * (b @ b)), float(want), rtol=1e-5, atol=1e-6):
        raise RuntimeError('link: the per-cell cosine disagrees with the study table')
    cl = pd.DataFrame({'cell': np.arange(len(a)), **cells})
    csvout.write_table(cl, PV + 'link_cells.csv', title=f'Per-cell cue differences at the {SHOWCASE[0]}-ms delay with {SHOWCASE[1]} prefix windows, dose {SHOWCASE_DOSE:g}: the stored trace and the decision-time membrane potential in the four combinations of writing and reading state', describe={
        'cell': 'Model cell index', **describe},
        key=['cell'], notes='Source: studies/brain-pathology-v1/runs/lesion-link/states.npz. Gated: the norms equal lesion_link_states and '
        'the cosine between the two matched conditions equals lesion_link_pairs.')
    st['trace_written'] = st['condition'].str.split('-').str[0].map(_state)
    st['trace_read'] = st['condition'].str.split('-').str[1].map(_state)
    st['lesion_dose'] = [next((_dose(x) for x in c.split('-') if x != 'H'), None) for c in st['condition']]
    st = st.rename(columns={'condition': 'condition_code'})
    st['source_file'] = paths.rel(PATH / 'lesion_link_states.csv')
    csvout.write_table(st, PV + 'link_states.csv', title='Where the trace is written and where it is read: the stored trace, the decision-time state and the readout input per condition and timing', describe={
        'condition_code': 'Repository code of the condition (written-read)', 'delay_ms': 'Delay (ms)', 'prefix_windows': 'Prefix windows',
        'trace_norm': 'Norm of the cue difference of adaptation at the end of the delay (the stored trace that is read)',
        'trace_cells': 'Cells with a cue difference of adaptation there', 'decision_v_norm': 'Norm of the cue difference of the membrane potential over the first decision window (mV)',
        'decision_v_cells_over_0p01': 'Cells with a cue difference above 0.01 mV there', 'decision_adapt_norm': 'Norm of the cue difference of adaptation over the first decision window',
        'features_norm': 'Norm of the cue difference of the readout input', 'delay_differ_cells': 'Cells spiking differently between the cues in the second half of the delay (own writing state only)',
        'delay_rate_hz_cue0': 'Mean firing rate over all cells in the delay, cue 0 (Hz)', 'delay_rate_hz_cue1': 'The same, cue 1',
        'trace_written': 'State in which the trace was written and held', 'trace_read': 'State in which it was read', 'lesion_dose': 'Lesion dose in the condition'},
        key=['condition_code', 'delay_ms', 'prefix_windows'])
    pr['source_file'] = paths.rel(PATH / 'lesion_link_pairs.csv')
    pr = pr.rename(columns={'condition_a': 'condition_code_a', 'condition_b': 'condition_code_b'})
    csvout.write_table(pr, PV + 'link_cosines.csv', title='Cosines between brain conditions of the per-cell cue differences (stored trace, decision-time membrane potential and adaptation, readout input), per timing', describe={
        'quantity': 'trace (adaptation at the end of the delay), window_v and window_adapt (first decision window), features (readout input)',
        'delay_ms': 'Delay (ms)', 'prefix_windows': 'Prefix windows', 'condition_code_a': 'Condition (written-read)', 'condition_code_b': 'Condition (written-read)',
        'cosine': 'Cosine over cells (or readout features)'}, key=['quantity', 'delay_ms', 'prefix_windows', 'condition_code_a', 'condition_code_b'])
