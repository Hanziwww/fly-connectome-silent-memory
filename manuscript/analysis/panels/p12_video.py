"""P12, video tables: the per-move and per-millisecond activity the supplemental videos draw, routed from the frozen records
into manuscript/results/plan_v2 and gated against the tables the figures already use (called from p12_plan_v2.main).

- Video S1 (Figure 1), the hardest planning map: the final brain with the final readout solves it in its 60 certified moves.
  Per move the board, the action and the four logits (behaviors.json; gated: the record's matched logits choose the same
  actions), and per move the cells that spike in the move's 40-ms window with their spike counts (matched-after.npz: the
  window means x 40). The map's walls and goals come from the frozen task file.
- Videos S2 (Figure 2) and S3 (Figure 5): one memory trial at the trained 160-ms delay and at the untrained 640-ms delay
  (scene 1, readout trained, 4 prefix windows, hidden goal), both cues: every spike of every cell from the cue to the end of
  the first decision (the 1-ms records that fu4 reads) and each cell's adaptation jump, from which the trace (the adaptation
  difference between the cues) follows exactly, a(t) = a(t - 1) exp(-dt / tau) + jump x spike(t) with the record's LIF
  parameters (brain.json: tau 200 ms, one jump for every cell). Gates: the
  adaptation rebuilt from the spikes equals the recorded one at every ms; the 160-ms trial's spikes per cell and phase equal
  trial_cells.
"""
import numpy as np
import pandas as pd

from msa import csvout, io, paths
from panels.p11_followup import _record_meta, _record_pairs

PV = 'plan_v2/'
M1 = paths.M1 / 'runs'
PLANNING, PLANNING_MAP = 'record-difficulty-D7-T0', 'D7-T0'
TRIAL_MODEL, TRIAL_DELAYS, TRIAL_PREFIX = 'remote-dynamic-M0-head_only', (160, 640), 4
REBUILD_TOL = 1e-4                              # float64 rebuild against the float32 record


def planning():
    folder = M1 / PLANNING
    beh = io.json_load(folder / 'behaviors.json')
    final = [b for b in beh if b['brain_phase'] == 1 and b['head_phase'] == 1]
    if len(final) != 1 or final[0]['status'] != 'solved' or final[0]['steps'] != 60:
        raise RuntimeError('planning video: the final pairing must solve the hardest map in 60 moves')
    dec = final[0]['decisions']
    z = io.npz_load(folder / 'matched-after.npz', keys=['mean', 'logits'])
    if z['mean'].shape[0] != len(dec):
        raise RuntimeError('planning video: the matched record and the rollout disagree in length')
    moves = []
    for i, d in enumerate(dec):
        legal = np.asarray(d['legal'], bool)
        if int(np.where(legal, z['logits'][i].astype(np.float64), -np.inf).argmax()) != int(d['action']):
            raise RuntimeError(f'planning video: move {i}: the matched logits choose another action')
        moves.append(dict(move=i, player=int(d['state'][0]), boxes=';'.join(str(int(b)) for b in d['state'][1]), action=int(d['action']),
                          **{f'logit_{a}': float(d['logits'][a]) for a in range(4)}, **{f'legal_{a}': bool(legal[a]) for a in range(4)}))
    mv = pd.DataFrame(moves)
    mv['source_file'] = paths.rel(folder / 'behaviors.json')
    csvout.write_table(mv, PV + 'video_planning_moves.csv', title='Video S1: the 60 moves of the final brain and readout on the hardest planning map', describe={
        'move': 'Move (0-59)', 'player': 'Agent cell before the move (row x 10 + column)', 'boxes': 'Box cells before the move',
        'action': 'Chosen action (0 up, 1 down, 2 left, 3 right)', **{f'logit_{a}': f'Readout logit of action {a}' for a in range(4)},
        **{f'legal_{a}': f'Action {a} legal' for a in range(4)}}, key=['move'],
        notes='Gated: the matched record (matched-after.npz) chooses the same action at every move.')
    level = next(lv for lv in io.json_load(paths.M1.parent / 'tasks' / 'levels.json') if lv['id'] == PLANNING_MAP)
    if [int(level['start'][0]), [int(b) for b in level['start'][1]]] != [moves[0]['player'], [int(b) for b in moves[0]['boxes'].split(';')]]:
        raise RuntimeError('planning video: the map does not start where the rollout starts')
    if level['size'] != 10:
        raise RuntimeError('planning video: the hardest map is 10 x 10')
    mp = pd.DataFrame(dict(cell=np.arange(100)))
    mp['wall'], mp['goal'] = mp['cell'].isin(level['walls']), mp['cell'].isin(level['goals'])
    mp['source_file'] = paths.rel(paths.M1.parent / 'tasks' / 'levels.json')
    csvout.write_table(mp, PV + 'video_planning_map.csv', title='Video S1: the hardest planning map (10 x 10 cells)', describe={
        'cell': 'Board cell (row x 10 + column)', 'wall': 'Wall', 'goal': 'Goal'}, key=['cell'])
    rows = []
    for i in range(len(dec)):
        n = np.rint(z['mean'][i, 0].astype(np.float64) * 40).astype(np.int64)
        on = np.flatnonzero(n > 0)
        rows.append(pd.DataFrame(dict(move=i, cell=on, spikes=n[on])))
    cl = pd.concat(rows, ignore_index=True)
    csvout.write_table(cl, PV + 'video_planning_cells.csv', title="Video S1: cells that spike in each move's 40-ms window and their spike counts", describe={
        'move': 'Move (0-59)', 'cell': 'Model cell index', 'spikes': 'Spikes in the 40-ms decision window'}, key=['move', 'cell'],
        notes=f'Source: {paths.rel(folder / "matched-after.npz")} (window means x 40).')


def trial():
    from panels.p12_plan_v2 import TRIAL_PHASES, _dense_case
    kinds = ('cue', 'delay', 'prefix', 'decision')
    rows, phases, jump = [], [], None
    for d in TRIAL_DELAYS:
        folder, meta = _record_meta(TRIAL_MODEL, d, TRIAL_PREFIX)
        i0, i1 = _record_pairs(meta)[(0, 'hidden')]
        lif = io.json_load(folder / 'brain.json')['lif']
        if lif['adapt_by_type'] is not None or lif['dt'] != 1.0:
            raise RuntimeError('trial video: the record does not use one adaptation jump at a 1-ms step')
        n = len(io.csv_load(paths.out(PV + 'trial_cells.csv'), usecols=['cell']))
        j = io.npz_load(folder / 'resolved-dynamics.npz', keys=['adapt_jump'])['adapt_jump'].astype(np.float64)
        j = np.full(n, float(lif['adapt_jump'])) if j.size == 0 else j     # an empty vector: every cell has the scalar jump
        decay = np.exp(-lif['dt'] / lif['adapt_tau'])
        if jump is None:
            jump = j
        elif not np.array_equal(jump, j):
            raise RuntimeError('trial video: the two records differ in their adaptation jumps')
        worst, last, per_phase = 0.0, 0, {}
        for cue, case in ((0, i0), (1, i1)):
            a = np.zeros(len(jump))
            for _, ms, A in _dense_case(folder, meta, case, kinds):
                for k, t in enumerate(ms):
                    s = A[k, 0]
                    a = a * decay + jump * s
                    worst = max(worst, float(np.abs(a - A[k, 3]).max()))
                    on = np.flatnonzero(s > 0)
                    if len(on):
                        rows.append(pd.DataFrame(dict(delay_ms=d, cue=cue, ms=int(t), cell=on)))
                        if d == 160:
                            ph = next(p for p, (lo, hi) in TRIAL_PHASES.items() if lo <= t <= hi)
                            per_phase.setdefault((cue, ph), []).append(on)
                    last = max(last, int(t))
        if worst > REBUILD_TOL:
            raise RuntimeError(f'trial video, {d} ms: the adaptation rebuilt from the spikes differs from the record by {worst:g}')
        if d == 160:
            tc = io.csv_load(paths.out(PV + 'trial_cells.csv'))
            for (cue, ph), lists in per_phase.items():
                got = np.bincount(np.concatenate(lists), minlength=len(jump))
                if not np.array_equal(got, tc[f'spikes_{ph}_cue{cue}'].to_numpy(np.int64)):
                    raise RuntimeError(f'trial video: spikes in the {ph} phase (cue {cue}) disagree with trial_cells')
        delay_end = 40 + d
        phases.append(dict(delay_ms=d, prefix_windows=TRIAL_PREFIX, cue_end_ms=40, delay_end_ms=delay_end, input_end_ms=delay_end + 40 * TRIAL_PREFIX,
                           decision_end_ms=last, adaptation_rebuild_max_error=worst))
    sp = pd.concat(rows, ignore_index=True)
    csvout.write_table(sp, PV + 'video_trial_spikes.csv', title='Videos S2 and S3: every spike of every cell in one memory trial, both cues, at the trained 160-ms and the untrained 640-ms delay (scene 1, readout trained, 4 prefix windows)', describe={
        'delay_ms': 'Delay (ms)', 'cue': 'Cue', 'ms': 'Millisecond of the trial (1 = the first ms of the cue)', 'cell': 'Model cell index'},
        key=['delay_ms', 'cue', 'ms', 'cell'], notes=f'Source: studies/brain-sokoban-v1/mechanisms-v1/runs/record-{TRIAL_MODEL}/dense '
        '(free-after-d160-p4, free-after-d640-p4). Gated: the 160-ms spikes per cell and phase equal trial_cells.')
    cells = pd.DataFrame(dict(cell=np.arange(len(jump)), adapt_jump=jump))
    csvout.write_table(cells, PV + 'video_trial_cells.csv', title="Videos S2 and S3: each cell's adaptation jump per spike (the trace follows from the spikes)", describe={
        'cell': 'Model cell index', 'adapt_jump': 'Adaptation added per spike (the variable decays with 200 ms)'}, key=['cell'],
        notes=f'Source: studies/brain-sokoban-v1/mechanisms-v1/runs/record-{TRIAL_MODEL}/brain.json (lif).')
    csvout.write_table(pd.DataFrame(phases), PV + 'video_trial_phases.csv', title='Videos S2 and S3: the phases of the two trials', describe={
        'delay_ms': 'Delay (ms)', 'prefix_windows': 'Prefix windows', 'cue_end_ms': 'Last ms of the cue', 'delay_end_ms': 'Last ms of the delay',
        'input_end_ms': 'Last ms of the prefix input', 'decision_end_ms': 'Last ms of the first decision window',
        'adaptation_rebuild_max_error': 'Largest difference at any ms between the adaptation rebuilt from the spikes and the recorded one'},
        key=['delay_ms'], notes='Gated: the rebuild error is below 1e-4 at every ms and cell.')


LINK = paths.ROOT / 'studies' / 'brain-pathology-v1' / 'runs' / 'lesion-link'
LESION_COMBOS = {('x1-r1', 'H-H'): 'healthy', ('x1-r1', 'L1-L1'): 'lesion after learning', ('b-d1-r1', 'L1-L1'): 'learning under the lesion',
                 ('c-d1-r1', 'L1-L1'): 'relearned under the lesion', ('b-d1-r1', 'H-H'): 'restored late',
                 ('x1-r1', 'H-L1'): 'written healthy, read lesioned', ('x1-r1', 'L1-H'): 'written lesioned, read healthy'}


def lesion():
    """Video S4: the lesion-link recording (amendment 0015) at the 160-ms delay with 4 prefix windows, dose 1. Gates: every
    combination's statuses equal its readout's in the study table, and its first decisions equal the readout job's."""
    rec = io.json_load(LINK / 'video.json')
    out = io.csv_load(paths.ROOT / 'studies' / 'brain-pathology-v1' / 'results' / 'lesion_link_outcomes.csv').set_index(['readout', 'condition'])
    z = io.npz_load(LINK / 'video.npz')
    spikes, steps = [], []
    for c in rec['combinations']:
        key = (c['readout'], c['condition'])
        name = LESION_COMBOS[key]
        if not c['equals_readout_job']:
            raise RuntimeError(f'lesion video: {key} differs from its readout job')
        st = dict(x.split(':') for x in str(out.loc[key, 'statuses']).split(';'))
        for cue, status in zip((0, 1), c['status']):
            if st[f'{cue}/160/4'] != status:
                raise RuntimeError(f'lesion video: {key} cue {cue} status disagrees with the study table')
        ev = np.asarray(z[f"{c['readout']}|{c['condition']}"], np.int64)
        e = ev[ev[:, 1] == 1]                       # the video draws the cue-1 history's spikes
        spikes.append(pd.DataFrame(dict(combination=name, cue=1, ms=e[:, 0] + 1, cell=e[:, 2])))
        for cue in (0, 1):
            for i, d in enumerate(c['decisions'][cue]):
                steps.append(dict(combination=name, cue=cue, step=i, player=int(d['state'][0]), box=int(d['state'][1][0]), action=int(d['action']),
                                  **{f'logit_{a}': float(d['logits'][a]) for a in range(4)}, status=c['status'][cue]))
    sp = pd.concat(spikes, ignore_index=True)
    csvout.write_table(sp, PV + 'video_lesion_spikes.csv', title='Video S4: every spike of every cell in the cue-1 history, 160-ms delay with 4 prefix windows, dose 1, for the regimes of Figure 7B and the trace moved between brain states (the input phase and the first 30 decisions)', describe={
        'combination': 'Readout and brain condition', 'cue': 'Cue', 'ms': 'Millisecond of the trial (1 = the first ms of the cue; decisions from 361)', 'cell': 'Model cell index'},
        key=['combination', 'cue', 'ms', 'cell'], notes='Source: studies/brain-pathology-v1/runs/lesion-link/video.npz (amendment 0015). Gated: statuses equal '
        'lesion_link_outcomes and decisions equal the readout jobs.')
    csvout.write_table(pd.DataFrame(steps), PV + 'video_lesion_decisions.csv', title='Video S4: the decisions of each combination and cue (board state before each move, the move, the outcome)', describe={
        'combination': 'Readout and brain condition', 'cue': 'Cue', 'step': 'Decision', 'player': 'Agent cell before the move', 'box': 'Box cell before the move',
        'action': 'Move (0 up, 1 down, 2 left, 3 right)', **{f'logit_{a}': f'Readout logit of move {a}' for a in range(4)},
        'status': 'Outcome of the history'}, key=['combination', 'cue', 'step'])


def video_tables():
    planning()
    trial()
    lesion()
