"""P0: environment, input inventory checks and the frozen test plan."""
import os
import platform
import sys

import pandas as pd

from msa import SEED, VERSION, csvout, io, paths, stats, testplan

EXPECTED_TABLES = [
    (paths.M1 / 'interpretation' / 'memory-intervention-pairs.csv', 7272),
    (paths.M1 / 'interpretation' / 'four-combination-behavior.csv', 136),
    (paths.M1 / 'interpretation' / 'decomposition-by-model.csv', 34),
    (paths.M1 / 'interpretation' / 'memory-intervention-summary.csv', 505),
    (paths.M1 / 'tables' / 'memory-behavior.csv', 490),
    (paths.M1 / 'tables' / 'memory-geometry.csv', 735),
    (paths.M1 / 'tables' / 'run-index.csv', 103),
    (paths.MS / 'analysis' / 'condition-outcomes.csv', 10640),
    (paths.MS / 'analysis' / 'selective-pairs.csv', 6048),
    (paths.MS / 'analysis' / 'final-state-factorial.csv', 432),
    (paths.MS / 'analysis' / 'directional-swap-audit.csv', 168),
    (paths.MS / 'analysis' / 'final-timing-pairs.csv', 728),
    (paths.MS / 'analysis' / 'held-history-cue-decoding.csv', 518),
    (paths.MS / 'analysis' / 'input-conditioned-dynamics.csv', 120),
]


def _dirs(path):
    return sorted(p.name for p in path.iterdir() if p.is_dir()) if path.exists() else []


def environment():
    import numpy, scipy, pyarrow
    rows = [('analysis_version', VERSION), ('analysis_seed', SEED), ('n_resamples', stats.N_RESAMPLES),
            ('alpha', stats.ALPHA), ('python', sys.version.split()[0]), ('implementation', platform.python_implementation()),
            ('platform', platform.platform()), ('machine', platform.machine()), ('processor', platform.processor()),
            ('logical_cpus', os.cpu_count()), ('numpy', numpy.__version__), ('scipy', scipy.__version__),
            ('pandas', pd.__version__), ('pyarrow', pyarrow.__version__)]
    try:
        import torch
        rows.append(('torch', torch.__version__))
        rows.append(('torch_cuda_visible', str(torch.cuda.is_available()).lower()))
    except Exception as exc:  # torch is only needed to read checkpoints
        rows.append(('torch', f'unavailable: {exc}'))
    rows.append(('interpreter', paths.rel(sys.executable) if str(sys.executable).startswith(str(paths.ROOT)) else sys.executable))
    rows.append(('cuda_visible_devices', os.environ.get('CUDA_VISIBLE_DEVICES', '')))
    df = pd.DataFrame(rows, columns=['item', 'value'])
    df['value'] = df['value'].astype(str)
    csvout.write_table(df, 'meta/environment.csv', title='Analysis runtime environment',
                       describe={'item': 'Environment item', 'value': 'Value'}, key=['item'], sort=[])


def input_checks():
    rows = []

    def add(check, observed, expected, notes=''):
        rows.append(dict(check_id=check, observed=observed, expected=expected,
                         ok=(observed == expected) if expected is not None else True, notes=notes))

    runs = _dirs(paths.RUNS)
    add('static_runs_total', len(runs), 68)
    add('static_runs_difficulty', sum(r.startswith('difficulty-') for r in runs), 24)
    add('static_runs_control', sum(r.startswith('control-') for r in runs), 34)
    add('static_runs_benchmark', sum(r.startswith('benchmark-') for r in runs), 10)
    add('remote_batch1_results', len(_dirs(paths.R0)), 16)
    add('remote_supplement_results', len(_dirs(paths.RS)), 19)
    add('mechanisms_v1_jobs', len(_dirs(paths.M1 / 'runs')), 55)
    add('mechanisms_v1_record_jobs', sum(d.startswith('record-') for d in _dirs(paths.M1 / 'runs')), 39)
    add('mechanisms_supplement_jobs', len(_dirs(paths.MS / 'runs')), 20)
    levels = io.json_load(paths.TASKS / 'levels.json')
    add('task_maps', len(levels), 34)
    add('task_maps_train', sum(l['split'] == 'train' for l in levels), 24)
    add('task_maps_heldout', sum(l['split'] == 'heldout' for l in levels), 8)
    add('task_certificates', len(io.json_load(paths.TASKS / 'certificates.json')), 34)
    add('schedule_entries', len(io.json_load(paths.STUDY / 'schedule.json')), 89)
    for path, n in EXPECTED_TABLES:
        add(f'rows:{paths.rel(path)}', len(io.csv_load(path)), n)
    df = pd.DataFrame(rows)
    csvout.write_table(df, 'qa/input_checks.csv', title='Inventory and schema checks on the saved inputs',
                       describe={'check_id': 'Check identifier', 'observed': 'Observed count',
                                 'expected': 'Expected count from the inventory', 'ok': 'Observed equals expected'},
                       key=['check_id'], sort=[])
    if not df['ok'].all():
        raise RuntimeError('input checks failed: ' + ', '.join(df.loc[~df['ok'], 'check_id']))


def test_plan():
    df = pd.DataFrame(testplan.FAMILIES)
    csvout.write_table(df, 'qa/test_plan.csv', title='Test families declared before any test was computed',
                       describe={'family': 'Test family identifier (multiplicity is controlled within a family)',
                                 'panel': 'Panel or table the family supports',
                                 'level': 'Statistical level: A maps, B histories/cue pairs within one model, C descriptive',
                                 'correction': 'Multiplicity correction within the family (holm, bh or none)',
                                 'primary': 'Primary inference (true) or secondary/companion (false)',
                                 'posthoc': 'Every test of the family was added after the registered analysis (never primary)',
                                 'membership': 'Rule defining which tests belong to the family'},
                       key=['family'])


def main():
    environment()
    input_checks()
    test_plan()
