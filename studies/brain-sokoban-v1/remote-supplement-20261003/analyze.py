"""CPU-only analysis of saved results. Never loads or selects a policy checkpoint."""
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TRAIN_D = (0, 40, 160)
TRAIN_P = (0, 1, 4)
INPUTS = {}


def read(path, default=None):
    if not path.exists():
        return default
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(HERE))] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return json.loads(raw.decode('utf-8-sig'))


def lines(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(HERE))] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    # A sync may expose a partially appended last line. Earlier complete rows remain usable.
    result = []
    for line in raw.decode('utf-8').splitlines():
        if line.strip():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return result


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def counts(rows):
    return dict(solved=sum(r.get('solved', r.get('status') == 'solved') for r in rows),
                first_push_correct=sum(r.get('first_push_correct') is True for r in rows),
                total=len(rows), failures=dict(collections.Counter(r.get('status') for r in rows if r.get('status') != 'solved')))


def groups(rows, keys):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in keys)].append(row)
    return [{**dict(zip(keys, key)), **counts(rs)} for key, rs in grouped.items()]


def causal_pairs(observations):
    index = {(o['window'], o['delay_ms'], o['prefix_windows'], o['intervention']): o['records'] for o in observations}
    paired = []
    for item in observations:
        key = (item['window'], item['delay_ms'], item['prefix_windows'])
        baseline_kind = None if item['window'] == 'decision' else 'sham'
        if item['intervention'] == baseline_kind or key + (baseline_kind,) not in index:
            continue
        baseline = {r['case']: r for r in index[key + (baseline_kind,)]}
        for row in item['records']:
            b = baseline[row['case']]
            partner = next((r for r in baseline.values() if r['cue'] != row['cue'] and
                            r['context'] == row['context'] and r['visible'] == row['visible']), None)
            paired.append(dict(window=key[0], delay_ms=key[1], prefix_windows=key[2], case=row['case'],
                cue=row['cue'], context=row['context'], visible=row['visible'], intervention=item['intervention'],
                baseline_kind=baseline_kind, baseline_solved=b['solved'], solved=row['solved'],
                solved_difference=int(row['solved']) - int(b['solved']),
                baseline_dual_cue_success=bool(b['solved'] and partner and partner['solved']),
                first_push_correct=row['first_push_correct'],
                donor_first_push_match=bool(partner and row['first_push_action'] == partner['first_push_action']),
                original_logits_max_abs=float(np.max(np.abs(np.array(row['logits']) - b['logits']))),
                donor_logits_max_abs=float(np.max(np.abs(np.array(row['logits']) - partner['logits']))) if partner else None))
    output = []
    for window in sorted({r['window'] for r in paired}):
        for visibility in (False, True):
            for kind in sorted({r['intervention'] for r in paired}):
                for subset in ('all_pairs', 'baseline_dual_cue_success'):
                    rs = [r for r in paired if r['window'] == window and r['visible'] == visibility and r['intervention'] == kind
                          and (subset == 'all_pairs' or r['baseline_dual_cue_success'])]
                    if rs:
                        output.append(dict(window=window, visible=visibility, intervention=kind, subset=subset,
                            pairs=len(rs), baseline_solved=sum(r['baseline_solved'] for r in rs),
                            solved=sum(r['solved'] for r in rs), paired_difference=float(np.mean([r['solved_difference'] for r in rs])),
                            donor_first_push_matches=sum(r['donor_first_push_match'] for r in rs),
                            original_logits_max_abs=max(r['original_logits_max_abs'] for r in rs)))
    by_visibility = {(r['window'], r['delay_ms'], r['prefix_windows'], r['context'], r['cue'],
                      r['intervention'], r['visible']): r for r in paired}
    contrasts = []
    for key, hidden in by_visibility.items():
        if key[-1] or key[:-1] + (True,) not in by_visibility:
            continue
        visible = by_visibility[key[:-1] + (True,)]
        contrasts.append(dict(window=key[0], delay_ms=key[1], prefix_windows=key[2], context=key[3],
            cue=key[4], intervention=key[5],
            common_dual_cue_baseline=hidden['baseline_dual_cue_success'] and visible['baseline_dual_cue_success'],
            hidden_baseline=hidden['baseline_solved'], visible_baseline=visible['baseline_solved'],
            hidden_solved=hidden['solved'], visible_solved=visible['solved'],
            hidden_minus_visible_effect=hidden['solved_difference'] - visible['solved_difference']))
    contrast_summaries = []
    for window, kind in sorted({(r['window'], r['intervention']) for r in contrasts}):
        for subset in ('all_pairs', 'common_dual_cue_baseline'):
            rs = [r for r in contrasts if r['window'] == window and r['intervention'] == kind
                  and (subset == 'all_pairs' or r['common_dual_cue_baseline'])]
            if rs:
                contrast_summaries.append(dict(window=window, intervention=kind, subset=subset, paired_histories=len(rs),
                    hidden_baseline=sum(r['hidden_baseline'] for r in rs), visible_baseline=sum(r['visible_baseline'] for r in rs),
                    hidden_solved=sum(r['hidden_solved'] for r in rs), visible_solved=sum(r['visible_solved'] for r in rs),
                    paired_difference_of_effects=float(np.mean([r['hidden_minus_visible_effect'] for r in rs]))))
    return dict(pairs=paired, summaries=output, visibility_pairs=contrasts, visibility_summaries=contrast_summaries,
                interpretation='Sham is matched within each intervention window; all pairs are retained. No inferential sample size from repeated histories.')


def decoder(rows):
    rows = [r for r in rows if not r['visible']]
    if not rows:
        return None
    x = np.asarray([r['features'] for r in rows], dtype=float)
    y = np.asarray([r['cue'] for r in rows], dtype=float)
    fit = np.asarray([r['delay_ms'] in TRAIN_D and r['prefix_windows'] in TRAIN_P for r in rows])
    if fit.sum() < 18 or set(y[fit]) != {0., 1.}:
        return dict(status='waiting_for_all_preregistered_training_time_features')
    mean = x[fit].mean(0); scale = np.maximum(x[fit].std(0), .01)
    z = (x - mean) / scale
    # Fixed ridge alpha=1, fitted only on the registered training-time cells.
    target = 2 * y[fit] - 1
    weights = z[fit].T @ np.linalg.solve(z[fit] @ z[fit].T + np.eye(fit.sum()), target)
    prediction = z @ weights >= 0
    records = [dict(delay_ms=r['delay_ms'], prefix_windows=r['prefix_windows'], cue=r['cue'],
                    train_time=bool(fit[i]), decoded_correct=bool(prediction[i] == y[i]),
                    solved=r['solved']) for i, r in enumerate(rows)]
    summaries = []
    for delay in sorted({r['delay_ms'] for r in records}):
        for prefix in sorted({r['prefix_windows'] for r in records}):
            rs = [r for r in records if r['delay_ms'] == delay and r['prefix_windows'] == prefix]
            if rs:
                summaries.append(dict(delay_ms=delay, prefix_windows=prefix, total=len(rs),
                    decoded_correct=sum(r['decoded_correct'] for r in rs), solved=sum(r['solved'] for r in rs)))
    return dict(status='descriptive', algorithm='Ridge alpha=1; feature scale floor=.01; fixed training-time split',
                policy_unchanged=True, within_one_map_family=True, train_rows=int(fit.sum()),
                train_decoded_correct=int((prediction[fit] == y[fit]).sum()),
                heldout_rows=int((~fit).sum()), heldout_decoded_correct=int((prediction[~fit] == y[~fit]).sum()),
                records=records, by_time=summaries,
                limitation='Time generalization of a separately fitted decoder is not cross-map memory generalization or proof of behavioral necessity.')


def collect():
    plan = read(HERE / 'plan.json')
    tasks = {t['id']: t for t in read(HERE / 'tasks.json')}
    snapshot = read(HERE / 'latest-status.json', {})
    live = {j['id']: j for j in snapshot.get('jobs', [])}
    results = []
    for job in plan['jobs']:
        out = HERE / 'results' / job['id']
        result = read(out / 'result.json', {})
        verified = read(out / 'verification.json', {})
        cfg = read(out / 'config.json', {})
        row = dict(id=job['id'], mode=job['mode'], task=job['task'], family=tasks[job['task']]['family_id'],
            status=result.get('status', live.get(job['id'], {}).get('progress', {}).get('status', 'pending')),
            policy_kind=cfg.get('policy_kind'), arm=cfg.get('arm'),
            updates=result.get('updates'), training_seconds=result.get('training_seconds'),
            original_solved=verified.get('original_solved'), verification_status=verified.get('status'),
            training_accepted=verified.get('status') == 'verified' and verified.get('original_solved') is True,
            core_parameter_changes=sum(v['changed'] for v in result.get('parameter_changes', {}).get('core', {}).values()),
            peak_allocated_gib=result.get('resources', {}).get('peak_allocated_gib'))
        if verified.get('status') == 'verified':
            assert verified['pid'] != verified['training_pid'], 'Verification was not independent'
            assert verified['model_sha256'] == result['model_sha256'], 'Checkpoint identity differs'
            row['verification_adapter'] = verified.get('verification_adapter')
            train = verified['training_history_reload']['rollouts']
            row['train'] = counts(train)
            row['train_by_visibility'] = groups(train, ['visible'])
            rolls = verified['results']['rollouts']
            row['splits'] = groups(rolls, ['history_split'])
            row['by_time'] = groups(rolls, ['history_split', 'delay_ms', 'prefix_windows', 'visible', 'context'])
            row['legacy_verification_interventions'] = groups(verified.get('intervention_rollouts', []), ['intervention', 'delay_ms', 'visible'])
        row['history'] = lines(out / 'history.jsonl')
        row['extensions'] = lines(out / 'extensions.jsonl')
        observations = lines(out / 'observations.jsonl')
        if observations:
            # Last complete record wins at an idempotent resume boundary.
            observations = list({r['index']: r for r in observations}.values())
            flat = [{**r, **{k: o[k] for k in ('delay_ms', 'prefix_windows', 'window', 'intervention')}} for o in observations for r in o['records']]
            row['probe_counts'] = groups(flat, ['window', 'intervention', 'delay_ms', 'prefix_windows', 'visible'])
            if cfg.get('probe') == 'causal':
                row['causal'] = causal_pairs(observations)
            if cfg.get('probe') == 'dense_delay':
                row['cue_decoder'] = decoder(flat)
                by_elapsed = collections.defaultdict(list)
                for item in row['probe_counts']:
                    by_elapsed[item['delay_ms'] + 40 * item['prefix_windows']].append(item)
                row['equal_elapsed_time'] = [dict(ms=k, cells=v) for k, v in sorted(by_elapsed.items()) if len(v) > 1]
        results.append(row)
    return dict(generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                code_sha256=plan['code_sha256'], tasks_sha256=plan['tasks_sha256'],
                supervisor=snapshot.get('supervisor'), runs=results,
                units='seed 10 only; S1 and S2 are two new families; repeated histories and orientations are not independent replications')


def figures(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    paths = []
    train_runs = [r for r in data['runs'] if r['mode'] == 'train' and r['history']]
    if train_runs:
        tasks = sorted({r['task'] for r in train_runs})
        fig, axes = plt.subplots(len(tasks), 2, figsize=(11, 3.3 * len(tasks)), squeeze=False, layout='constrained')
        for i, task in enumerate(tasks):
            task_runs = [r for r in train_runs if r['task'] == task]
            for run in task_runs:
                h = run['history']; label = run['id'].replace('supp-', '')
                y = [r['solved'] / r['total'] for r in h]
                style = dict(alpha=.65, linewidth=.8) if len(h) > 2000 else {}
                axes[i, 0].plot([r['elapsed'] / 60 for r in h], y, label=label, **style)
                axes[i, 1].plot([r['updates'] for r in h], y, label=label, **style)
            for j, xlab in enumerate(('Training-loop minutes (includes assessment)', 'Optimizer updates')):
                axes[i, j].set(title=task, xlabel=xlab, ylabel='Training histories solved', ylim=(-.03, 1.03))
                axes[i, j].legend(fontsize=7, loc='best'); axes[i, j].grid(alpha=.15)
            lengths = [max(r['updates'] for r in run['history']) for run in task_runs]
            if max(lengths) > 10 * max(1, float(np.median(lengths))):
                axes[i, 1].set_xscale('symlog', linthresh=100)
                axes[i, 1].set_xlim(left=0)
                axes[i, 1].set_xlabel('Optimizer updates (symlog; linear to 100)')
        for ext in ('png', 'pdf'):
            name = f'learning.{ext}'; fig.savefig(HERE / name, dpi=180); paths.append(name)
        plt.close(fig)
    verified = [r for r in data['runs'] if r.get('verification_status') == 'verified' and r.get('splits')]
    if verified:
        fig, ax = plt.subplots(figsize=(10, max(3, len(verified) * .65)), layout='constrained')
        y = np.arange(len(verified))
        for offset, split, color in [(-.18, 'train', '#27647B'), (.18, 'heldout', '#C4873B')]:
            values = [next(v for v in r['splits'] if v['history_split'] == split) for r in verified]
            ax.barh(y + offset, [v['solved'] / v['total'] for v in values], height=.32, color=color, label=split)
            for pos, v in zip(y + offset, values):
                ax.text(v['solved'] / v['total'] + .015, pos, f"{v['solved']}/{v['total']}", va='center', fontsize=9)
        ax.set(yticks=y, yticklabels=[r['id'].replace('supp-', '') for r in verified], xlim=(0, 1.18),
               xlabel='Full solutions / evaluated cue histories', title='Independent reload of all trained conditions')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False); ax.invert_yaxis()
        for ext in ('png', 'pdf'):
            name = f'acceptance.{ext}'; fig.savefig(HERE / name, dpi=180); paths.append(name)
        plt.close(fig)
    for run in data['runs']:
        decoded = run.get('cue_decoder') or {}
        if decoded.get('status') != 'descriptive':
            continue
        from matplotlib.patches import Rectangle
        cells = decoded['by_time']
        delays = sorted({c['delay_ms'] for c in cells}); prefixes = sorted({c['prefix_windows'] for c in cells})
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), layout='constrained')
        for ax, measure, title in zip(axes, ('solved', 'decoded_correct'), ('Full hidden-goal solutions', 'Separate cue decoder')):
            grid = np.full((len(prefixes), len(delays)), np.nan)
            for cell in cells:
                grid[prefixes.index(cell['prefix_windows']), delays.index(cell['delay_ms'])] = cell[measure] / cell['total']
            plot = ax.imshow(grid, vmin=0, vmax=1, cmap='viridis', aspect='auto', origin='lower')
            ticks = [i for i, d in enumerate(delays) if d in (0, 80, 160, 320, 480, 640)]
            ax.set(title=title, xlabel='Blank delay (ms)', ylabel='Prefix windows', xticks=ticks,
                   xticklabels=[delays[i] for i in ticks], yticks=range(len(prefixes)), yticklabels=prefixes)
            for i, prefix in enumerate(prefixes):
                for j, delay in enumerate(delays):
                    if prefix in TRAIN_P and delay in TRAIN_D:
                        ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, edgecolor='white', linewidth=1))
        fig.colorbar(plot, ax=axes, label='Fraction correct (two paired cues)')
        fig.suptitle('M0: behavior and linearly decodable cue; outlines = registered training times')
        for ext in ('png', 'pdf'):
            name = f'delay-decoding.{ext}'; fig.savefig(HERE / name, dpi=180); paths.append(name)
        plt.close(fig)
    return paths


def report(data):
    rows = data['runs']; accepted = [r for r in rows if r['training_accepted']]
    text = ['# Task acquisition, state memory, and decision mechanisms in a connectome model: remote supplement analysis', '', f"Analysis data updated at {data['generated_utc']}. This reads synchronized saved data only, without loading a model or starting local training.", '',
        '## Research questions', '',
        'This study uses a fly-connectome-constrained neural dynamical model as a trainable, recordable, and intervenable experimental system to study acquisition, the role of persistent neural state in memory and decisions, and how these processes vary with task requirements. Ranking model performance is not the research objective.', '',
        'This supplement asks whether linear readouts can learn to use fixed brain dynamics, whether hidden-goal decisions depend on prior neural state, and how these effects vary with layout, delay, and context. Parameter learning and dynamic-state memory are interpreted separately. See the full [research framing](../research-framing.en.md).', '',
        'Structure, initialization, fixed-trace, and GRU controls distinguish prior experience, temporal retention, and connection organization. All results remain as references for mechanism interpretation.', '',
        '## Experiment overview', '',
        f"The batch locked 19 queue entries, including 13 training conditions. Independent reload accepted all-history solutions for {len(accepted)}/13 conditions.", '',
        '| Condition | Training end status | Updates | Training-loop minutes | Trained histories solved | Untrained timings solved |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['mode'] != 'train' or not r.get('verification_status'):
            continue
        split = {v['history_split']: v for v in r.get('splits', [])}
        def fraction(key):
            v = split.get(key); return f"{v['solved']}/{v['total']}" if v else 'not_yet_verified'
        text.append(f"| {r['id']} | {r['status']} | {r.get('updates')} | {(r.get('training_seconds') or 0)/60:.2f} | {fraction('train')} | {fraction('heldout')} |")
    text += ['', '## Directly supported results', '']
    for r in accepted:
        heldout = next(v for v in r['splits'] if v['history_split'] == 'heldout')
        text.append(f"- {r['id']}: independently reloaded trained histories {r['train']['solved']}/{r['train']['total']}; untrained delay/prefix combinations {heldout['solved']}/{heldout['total']}.")
        if r['task'] == 'M0-visibility':
            text.append('  One checkpoint handles visible and hidden goals at trained timings, enabling paired interventions. Frozen-core changed elements: ' + str(r['core_parameter_changes']) + '.')
    acquired = [r for r in accepted if r['id'] in ('supp-S1-ring-brain', 'supp-S2-context-brain')]
    if acquired:
        text += ['', 'Freezing P05 and training the linear readout solved every registered history on these new layouts: ' +
                 ', '.join(r['task'] for r in acquired) + '. This is acquisition after training on the new layout, with zero-update transfer separate. Additional learning is in the readout; fixed dynamics provide historical features usable by the behavioral readout.']
    trace_solved = [r for r in accepted if r['policy_kind'] == 'leaky']
    if trace_solved:
        text += ['', 'A fixed decaying trace plus linear readout also solved all trained histories on ' + ', '.join(r['task'] for r in trace_solved) +
                 '. Simple temporal traces satisfy part of these tasks memory demands and provide a mechanism reference. Identifying the neural-state carriers still requires within-model paired interventions.']
    stopped_runs = [r for r in rows if r['status'] in ('stagnation', 'segment_limit') and r.get('train')]
    if stopped_runs:
        text += ['', '## Partial solutions and stopping reasons', '']
        for r in stopped_runs:
            detail = ''
            if r['extensions']:
                decision = r['extensions'][-1]
                lost = set(decision.get('required_histories', [])) - set(decision.get('solved_histories', []))
                delta = decision['min_choice_margin'] - decision['phase_margin']
                detail = f"Entry-margin change at the segment decision: {delta:+.3f}. "
                if lost:
                    detail += f"Lost {len(lost)} required histories; extension criteria were not met."
                elif delta < .02:
                    detail += "Required histories were retained, but entry-margin gain did not reach 0.02; extension criteria were not met."
                elif r['status'] == 'segment_limit':
                    detail += "Progress criteria were met, but all three training segments were exhausted."
            text.append(f"- {r['id']}: independent reload {r['train']['solved']}/{r['train']['total']}, status {r['status']}. {detail}")
        text += ['', 'Stagnation is a registered segment-stop status, not cessation of every improvement. Retention concerns histories of the same memory task, not old Sokoban levels. Independent acceptance uses the saved best checkpoint and can exceed the stopping model. Saturated network capacity is not established.']
    transfers = [r for r in rows if r['id'].startswith('supp-transfer-') and r.get('probe_counts')]
    if transfers:
        text += ['', '## Zero-update transfer of the original M0 model', '']
        for r in transfers:
            n = sum(x['total'] for x in r['probe_counts']); solved = sum(x['solved'] for x in r['probe_counts'])
            text.append(f"- {r['task']}: full hidden-goal solutions in {solved}/{n} cue/delay/prefix combinations; execution status {r['status']}.")
        text += ['', 'These denominators are paired histories within a layout, not independent maps. Tests do not select checkpoints.']
    for r in rows:
        if r.get('causal'):
            text += ['', f"### {r['id']} paired interventions", '', '| Window | Visible goal | Intervention | Pairs | Baseline solved | Intervention solved |', '|---|---|---|---:|---:|---:|']
            for v in r['causal']['summaries']:
                if v['subset'] == 'all_pairs':
                    text.append(f"| {v['window']} | {v['visible']} | {v['intervention']} | {v['pairs']} | {v['baseline_solved']} | {v['solved']} |")
            for v in r['causal']['visibility_summaries']:
                if v['window'] == 'decision' and v['subset'] == 'common_dual_cue_baseline' and v['intervention'] == 'neural_clear':
                    n = v['paired_histories']
                    text += ['', f"After matching timings where both cues solve in both visibility baselines, pre-decision clearing changes hidden-goal solutions from {v['hidden_baseline']}/{n} to {v['hidden_solved']}/{n} and visible-goal solutions from {v['visible_baseline']}/{n} to {v['visible_solved']}/{n}. This supports greater hidden-goal dependence on prior state at these timings, without extrapolation to other maps or seeds."]
        if r.get('cue_decoder', {}).get('status') == 'descriptive':
            d = r['cue_decoder']
            text += ['', f"Dense-scan independent linear decoding: cue identification {d['train_decoded_correct']}/{d['train_rows']} at fitted times and {d['heldout_decoded_correct']}/{d['heldout_rows']} at untrained times. Regularization is fixed; fitting uses original training times only, without policy or model-selection roles. Weak cross-time decoding does not prove cue information absent from the network."]
    text += ['', '## Interpretation and limits', '',
        '- Report GRU and brain conditions by layout; GRU success does not substitute for acceptance of a brain model, and success on one layout does not establish another.',
        '- Trained-history solutions and untrained timing performance are distinct. Timing combinations, cues, orientations, and repeated actions are not independent maps or training seeds.',
        '- S2 reveals orientation after a shared cue and tests conditional cue use. This manipulation is not ordinary static-board play and alone does not establish abstract goal understanding.',
        '- Retain all visibility-paired intervention results while separately showing the subset where both baseline cues solve. Use an unintervened baseline for the decision window and same-window shams for other windows.',
        '- Neural clearing that preserves clock, buffer pointer, and RNG is distinct from the earlier full-state reset. The verification adapter handles only duplicate metadata, not training, forward computation, or acceptance rules.',
        '- Dense scans distinguish cue decodability from behavioral use and do not establish cross-map generalization. Equal-duration prefixes can still differ in input history.',
        '- training_seconds measures the training loop, including behavioral evaluation and saving, not pure GPU backpropagation. Different update costs mean update counts are not equal computation.',
        '- Initialization controls address prior experience; rewiring tests sensitivity to connection organization; traces and GRU characterize temporal processing demands. All serve learning/memory interpretation. With different actual training times, final performance differences are not equal-compute causal effects.',
        '- This is a remote supplementary batch, not completion of the entire original paper plan; the local STOP remains.', '',
        '## Figures and provenance', '',
        'Grouped results are in [analysis.json](analysis.json); source files and SHA-256 are in [analysis-inputs.json](analysis-inputs.json).']
    for name in data.get('figures', []):
        if name.endswith('.png'):
            text += ['', f'![{name}]({name})']
    (HERE / 'analysis.en.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


if __name__ == '__main__':
    data = collect()
    data['figures'] = figures(data)
    report(data)
    save('analysis.json', data)
    save('analysis-inputs.json', dict(analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), files=INPUTS))
    print(json.dumps(dict(training_accepted=sum(r['training_accepted'] for r in data['runs']),
        analyzed_runs=len(data['runs']), figures=data['figures'], report=str(HERE / 'analysis.en.md')), ensure_ascii=False))
