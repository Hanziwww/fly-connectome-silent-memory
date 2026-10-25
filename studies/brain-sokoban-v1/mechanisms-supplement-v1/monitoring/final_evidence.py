"""Reconstruct final tables and figures from immutable, verified replay outputs.

No simulator, checkpoint, GPU or optimizer is imported. This post-completion
descriptive analysis does not alter the registered scientific source files.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[2]
OUT = ROOT / 'analysis'
MODEL_IDS = [
    'remote-dynamic-M0-head_only', 'remote-dynamic-M0-joint',
    'supp-M0-visibility', 'supp-S1-ring-brain', 'supp-S2-context-brain']
SCOPES = [(MODEL_IDS[0], False), (MODEL_IDS[1], False),
          (MODEL_IDS[2], False), (MODEL_IDS[2], True),
          (MODEL_IDS[3], False), (MODEL_IDS[4], False)]
LABELS = ['M0 readout / hidden', 'M0 joint / hidden', 'M0 shared / hidden',
          'M0 shared / visible', 'S1 / hidden', 'S2 / hidden']
BLUE, RED, GRAY, GREEN = '#3C5488', '#E64B35', '#8491B4', '#008B78'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'savefig.dpi': 300, 'axes.titlesize': 11})


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n', encoding='utf-8')


def save_table(name, frame):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')


def save_figure(fig, name):
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / (name + '.' + suffix), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def slice_scope(frame, scope):
    return frame[(frame.model == scope[0]) & (frame.visible == scope[1])]


def verify_outputs():
    plan = read(ROOT / 'plan.json')
    rows, differences = [], []
    total_files = total_bytes = 0
    for job in plan['jobs']:
        path = ROOT / 'runs' / job['id']
        receipt = read(ROOT / 'receipts' / (job['id'] + '.json'))
        manifest = read(path / 'manifest.json')
        result, verification = read(path / 'result.json'), read(path / 'verification.json')
        if not receipt['transport_verified'] or receipt['mismatches']:
            differences.append([job['id'], 'receipt'])
        canonical_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        if canonical_hash != receipt['manifest_sha256']:
            differences.append([job['id'], 'manifest digest'])
        for name, expected in manifest['files'].items():
            p = path / name
            if (not p.is_file() or p.stat().st_size != expected['bytes']
                    or digest(p) != expected['sha256']):
                differences.append([job['id'], name])
        total_files += receipt['files']
        total_bytes += receipt['bytes']
        assert result['status'] == 'complete', job
        assert verification['status'] == 'verified', job
        assert result['frozen_audit']['parameters_unchanged'] and not result['parameters_updated']
        if job['kind'] == 'timing':
            assert verification['control_checks_passed'] is None
            assert result['timing_and_input_schedule_locked']
        else:
            assert verification['control_checks_passed'] is True
        rows.append(dict(job=job['id'], kind=job['kind'], model=job['model'],
                         conditions=result.get('conditions', 0),
                         wall_seconds=result['wall_seconds'],
                         training_history_solved=result.get('training_solved'),
                         training_history_total=result.get('training_total'),
                         source_acceptance_matches=result.get('source_acceptance_matches'),
                         files=receipt['files'], bytes=receipt['bytes'],
                         invalid_norm_matches=result.get('invalid_norm_matches', 0)))
    source_checks = []
    for key, mapping in plan.items():
        if not key.endswith('hashes') or not isinstance(mapping, dict):
            continue
        for name, expected in mapping.items():
            if not isinstance(expected, str) or len(expected) != 64:
                continue
            p = WORK / name
            if not p.exists():
                p = ROOT / name
            if key == 'code_hashes':
                p = ROOT / 'code' / name
            okay = p.is_file() and digest(p) == expected
            source_checks.append(dict(group=key, path=name, sha256_matches=okay))
            if not okay:
                differences.append(['source', name])
    assert not differences, differences
    save_table('final-job-audit.csv', pd.DataFrame(rows))
    save_table('final-source-audit.csv', pd.DataFrame(source_checks))
    summary = dict(checked_utc=datetime.now(timezone.utc).isoformat(),
                   verified_jobs=len(rows), files=total_files, bytes=total_bytes,
                   mismatches=differences, sources_checked=len(source_checks),
                   sources_match=all(x['sha256_matches'] for x in source_checks),
                   parameters_unchanged=True,
                   training_stop_present=(ROOT.parent / 'STOP').is_file(),
                   local_free_gib=shutil.disk_usage(ROOT).free / 2**30,
                   batch_conditions=sum(x['conditions'] for x in rows))
    assert summary['training_stop_present']
    save_json('final-integrity-audit.json', summary)
    return summary


def tables(frame):
    selective = frame[(frame.module == 'selective') & frame.kind.isin(['cue', 'scrambled'])].copy()
    selective['lost'] = (selective.baseline_solved == 1) & (selective.solved == 0)
    selective['gained'] = (selective.baseline_solved == 0) & (selective.solved == 1)
    keys = ['model', 'visible', 'window', 'part', 'fraction', 'kind']
    factorial = selective.groupby(keys).agg(histories=('solved', 'size'),
        baseline_solved=('baseline_solved', 'sum'), solved=('solved', 'sum'),
        first_push_correct=('first_push_correct', 'sum'), lost=('lost', 'sum'),
        gained=('gained', 'sum'), margin_mean_change=('margin_effect', 'mean')).reset_index()
    save_table('final-state-factorial.csv', factorial)
    route = frame[(frame.module == 'routes') & frame.channel.notna()].copy()
    route['lost'] = (route.baseline_solved == 1) & (route.solved == 0)
    route['gained'] = (route.baseline_solved == 0) & (route.solved == 1)
    routes = route.groupby(['model', 'visible', 'side', 'channel', 'window', 'strength']).agg(
        histories=('solved', 'size'), lost=('lost', 'sum'), gained=('gained', 'sum'),
        margin_max_absolute_change=('margin_effect', lambda x: x.abs().max()),
        margin_mean_absolute_change=('margin_effect', lambda x: x.abs().mean())).reset_index()
    save_table('final-route-factorial.csv', routes)
    timing = frame[frame.module == 'timing'].copy()
    baseline = timing[timing.input_kind == 'blank'][['model', 'visible', 'case', 'total_postcue_ms', 'solved']]
    baseline = baseline.rename(columns={'solved': 'matched_blank_solved'})
    timing = timing.merge(baseline, on=['model', 'visible', 'case', 'total_postcue_ms'], validate='many_to_one')
    timing['lost'] = (timing.matched_blank_solved == 1) & (timing.solved == 0)
    timing['gained'] = (timing.matched_blank_solved == 0) & (timing.solved == 1)
    save_table('final-timing-pairs.csv', timing)
    time_summary = timing.groupby(['model', 'visible', 'input_kind', 'prefix', 'total_postcue_ms']).agg(
        histories=('solved', 'size'), blank_solved=('matched_blank_solved', 'sum'),
        solved=('solved', 'sum'), lost=('lost', 'sum'), gained=('gained', 'sum')).reset_index()
    save_table('final-timing-summary.csv', time_summary)
    return factorial, routes, time_summary


def norm_audit():
    audits = []
    for model in MODEL_IDS:
        for path in (ROOT / 'runs' / ('selective-' + model) / 'conditions').glob('*.json'):
            data = read(path)
            spec = data.get('spec', {})
            if spec.get('kind') != 'scrambled':
                continue
            for a in data.get('audit', []):
                audits.append(dict(model=model, condition=data['id'], **a))
    frame = pd.DataFrame(audits)
    save_table('final-norm-audit.csv', frame)
    summary = frame.groupby(['model', 'field']).agg(entries=('norm_matched', 'size'),
        invalid=('norm_matched', lambda x: int((~x).sum())),
        max_relative_norm_error=('relative_norm_error', 'max'),
        max_absolute_direction_cosine=('cue_direction_cosine', lambda x: x.abs().max()),
        max_zero_boundary_fraction=('zero_boundary_fraction', 'max')).reset_index()
    save_table('final-norm-summary.csv', summary)
    return summary


def state_figure(frame, factorial):
    mat, labels = [], []
    for scope in SCOPES:
        s = slice_scope(frame, scope)
        base = s[(s.module == 'selective') & (s.kind == 'baseline')]
        n = len(base)
        values = [int(base.solved.sum())]
        for fraction, kind in [(.5, 'cue'), (.5, 'scrambled'), (1, 'cue'), (1, 'scrambled')]:
            x = s[(s.module == 'selective') & (s.window == 'read') & (s.part == 'continuous')
                  & (s.fraction == fraction) & (s.kind == kind)]
            assert len(x) == n
            values.append(int(x.solved.sum()))
        mat.append(np.array(values) / n)
        labels.append([f'{v}/{n}' for v in values])
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9.6, 8.2), gridspec_kw={'height_ratios': [1.15, 1]})
    cmap = LinearSegmentedColormap.from_list('solved', ['#FFFFFF', '#3C5488'])
    ax.imshow(mat, vmin=0, vmax=1, cmap=cmap, aspect='auto')
    for i in range(6):
        for j in range(5):
            ax.text(j, i, labels[i][j], ha='center', va='center',
                    color='white' if mat[i][j] > .58 else '#202020')
    ax.set_yticks(range(6), LABELS)
    ax.set_xticks(range(5), ['Baseline', 'Cue midpoint', 'Matched noise\n(a = 0.5)',
                            'Cue swap', 'Matched noise\n(a = 1)'])
    ax.tick_params(length=0)
    ax.set_title('A   Full solutions after continuous-state intervention before reading', loc='left', fontweight='bold')
    directional = pd.DataFrame(read(OUT / 'directional-swap-audit.json')['groups'])
    for offset, kind, color, label in [(-.18, 'cue', RED, 'Opposite-cue state'),
                                      (.18, 'scrambled', GRAY, 'Norm-matched noise')]:
        vals, texts = [], []
        for scope in SCOPES:
            row = slice_scope(directional, scope).query('kind == @kind').iloc[0]
            vals.append(row.follows_donor / row.eligible)
            texts.append(f'{row.follows_donor}/{row.eligible}')
        bars = bx.bar(np.arange(6) + offset, vals, width=.34, color=color, label=label)
        bx.bar_label(bars, labels=texts, padding=3, fontsize=9)
    bx.set_xticks(range(6), [x.replace(' / ', '\n') for x in LABELS])
    bx.set_ylim(0, 1.15); bx.set_yticks([0, .5, 1], ['0', '0.5', '1'])
    bx.set_ylabel('First push follows donor')
    bx.set_title('B   Directional transfer where both baseline cue histories solve', loc='left', fontweight='bold')
    bx.legend(loc='upper center', bbox_to_anchor=(.5, -.23), ncol=2, frameon=False)
    fig.tight_layout(h_pad=2.5)
    save_figure(fig, 'final-01-state-content')

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    parts, windows = ['voltage', 'current', 'adaptation', 'continuous'], ['write', 'hold', 'read']
    for ax, scope, label in zip(axes.flat, SCOPES, LABELS):
        s = slice_scope(factorial, scope).query('fraction == 1')
        p = s.pivot(index='window', columns=['part', 'kind'], values='solved')
        n = int(s.histories.iloc[0])
        val = np.array([[p.loc[w, (q, 'cue')] - p.loc[w, (q, 'scrambled')] for q in parts] for w in windows])
        im = ax.imshow(val / n, cmap='RdBu', vmin=-.8, vmax=.8, aspect='auto')
        for i in range(3):
            for j in range(4):
                ax.text(j, i, f'{int(val[i,j]):+d}/{n}', ha='center', va='center', fontsize=9,
                        color='white' if abs(val[i,j] / n) > .38 else '#202020')
        ax.set_xticks(range(4), ['Voltage', 'Current', 'Adapt.', 'Combined'], rotation=25, ha='right')
        ax.set_yticks(range(3), ['Write', 'Hold', 'Read']);ax.tick_params(length=0)
        ax.set_title(label)
    fig.suptitle('Complete solutions: cue-component swap minus matched noise (a = 1)', fontsize=13)
    fig.tight_layout(rect=[0, .09, 1, .97], h_pad=2)
    cax = fig.add_axes([.32, .05, .36, .025])
    fig.colorbar(im, cax=cax, orientation='horizontal', label='Paired difference in fraction solved')
    save_figure(fig, 'final-02-components-windows')


def route_figure(routes):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, side, color, label in [(-.18, 'indices', BLUE, 'Candidate group'),
                                      (.18, 'controls', GRAY, 'Matched group')]:
        values = []
        for scope in SCOPES:
            x = slice_scope(routes, scope).query("channel == 'readout' & side == @side")
            values.append(x.margin_max_absolute_change.max())
        bars = ax.bar(np.arange(6) + offset, values, width=.34, color=color, label=label)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xticks(range(6), [x.replace(' / ', '\n') for x in LABELS])
    ax.set_ylim(0, .53);ax.set_ylabel('Maximum |first-action margin change|')
    ax.set_title('Direct readout contribution changes scores without changing complete solutions', loc='left', pad=15)
    ax.legend(frameon=False, loc='upper left', ncol=2)
    ax.text(.015, .88, 'Candidate network-only score change = 0\nNo lost or gained full solutions in either group',
            transform=ax.transAxes, fontsize=9, va='top')
    fig.tight_layout()
    save_figure(fig, 'final-03-readout-route')


def timing_figure(timing):
    names = ['blank', 'neutral_stationary', 'neutral_walk_0', 'neutral_walk_1']
    times = [80, 120, 160, 240, 320, 480, 640]
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.8))
    cmap = LinearSegmentedColormap.from_list('fraction', ['#FFFFFF', BLUE])
    for ax, scope, label in zip(axes.flat, SCOPES, LABELS):
        s = slice_scope(timing, scope).query("prefix == 2 | input_kind == 'blank'")
        p = s.pivot(index='input_kind', columns='total_postcue_ms', values='solved').loc[names, times]
        n = int(s.histories.iloc[0]);mat = p.to_numpy() / n
        ax.imshow(mat, vmin=0, vmax=1, cmap=cmap, aspect='auto')
        for i in range(4):
            for j in range(7):
                ax.text(j, i, f'{int(p.iloc[i,j])}', ha='center', va='center', fontsize=9,
                        color='white' if mat[i,j] > .58 else '#202020')
        ax.set_xticks(range(7), times, rotation=45, ha='right')
        ax.set_yticks(range(4), ['Blank', 'Static', 'Walk 0', 'Walk 1'])
        ax.set_xlabel('Total time after cue (ms)');ax.tick_params(length=0)
        ax.set_title(f'{label} (n = {n} per cell)')
    fig.suptitle('Full solutions at matched time: blank versus two neutral-input windows', fontsize=13)
    fig.tight_layout(rect=[0, .02, 1, .96], w_pad=2, h_pad=2)
    save_figure(fig, 'final-04-matched-time')


def probe_figure():
    dec = pd.read_csv(OUT / 'held-history-cue-decoding.csv')
    dyn = pd.read_csv(OUT / 'input-conditioned-dynamics.csv')
    tab = dec.groupby(['model', 'visible', 'split']).agg(histories=('correct', 'size'),
        cue_correct=('correct', 'sum'), solved=('solved', 'sum')).reset_index()
    save_table('final-decoder-summary.csv', tab)
    cross = dec.groupby(['model', 'visible', 'split', 'correct', 'solved']).size().reset_index(name='histories')
    save_table('final-decoder-behavior-cross.csv', cross)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1, 1.15]})
    values, counts = [], []
    for scope in SCOPES:
        row = slice_scope(tab, scope).query("split == 'held_time'").iloc[0]
        values.append(row.cue_correct / row.histories);counts.append(f'{row.cue_correct}/{row.histories}')
    bars = ax.bar(range(6), values, color=[BLUE, BLUE, BLUE, GRAY, BLUE, BLUE])
    ax.bar_label(bars, labels=counts, padding=3, fontsize=9)
    ax.axhline(.5, color='#777777', lw=1, ls='--');ax.set_ylim(0, 1)
    ax.set_xticks(range(6), ['M0 R', 'M0 J', 'M0 H', 'M0 V', 'S1', 'S2'])
    ax.set_ylabel('Cue decoder fraction correct')
    ax.set_title('A   Decoder on held-out times', loc='left', fontweight='bold')
    for offset, method, color, label in [(-.24, 'persistence', GRAY, 'Persistence'),
        (0, 'affine', RED, 'Autonomous affine'), (.24, 'input_phase', BLUE, 'Input + phase')]:
        vals = [dyn.query("model == @m & split == 'held_time' & horizon_ms == 40 & method == @method").r2.iloc[0] for m in MODEL_IDS]
        bx.bar(np.arange(5) + offset, vals, width=.23, color=color, label=label)
    bx.axhline(0, color='#777777', lw=.7)
    bx.set_ylim(-1.1, .6);bx.set_xticks(range(5), ['M0 R', 'M0 J', 'M0 H', 'S1', 'S2'])
    bx.set_ylabel('R² for latent population activity')
    bx.set_title('B   Conditional prediction, 40 ms horizon', loc='left', fontweight='bold')
    bx.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.13), fontsize=8)
    fig.text(.05, .015, 'R: readout-trained; J: joint; H/V: shared head, hidden/visible. Same seed and related histories; no independent-replicate error bars.', fontsize=8)
    fig.tight_layout(rect=[0, .06, 1, 1], w_pad=3)
    save_figure(fig, 'final-05-decoding-dynamics')
    return tab, cross


def main():
    integrity = verify_outputs()
    frame = pd.read_csv(OUT / 'condition-outcomes.csv')
    factorial, routes, timing = tables(frame)
    norms = norm_audit()
    state_figure(frame, factorial)
    route_figure(routes)
    timing_figure(timing)
    dec, cross = probe_figure()
    assert int(routes.lost.sum() + routes.gained.sum()) == 0
    summary = dict(integrity=integrity, case_history_rows=len(frame),
        route_intervention_case_histories=int(routes.histories.sum()),
        route_lost=int(routes.lost.sum()), route_gained=int(routes.gained.sum()),
        max_control_relative_norm_error=float(norms.max_relative_norm_error.max()),
        max_control_adaptation_abs_cosine=float(norms[norms.field == 'adapt'].max_absolute_direction_cosine.max()),
        post_completion_descriptive_audit=True, independent_training_replicates=False,
        figures=[f'final-0{i}-{name}' for i, name in enumerate([
            'state-content', 'components-windows', 'readout-route', 'matched-time', 'decoding-dynamics'], 1)])
    save_json('final-summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
