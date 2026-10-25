"""Phase 3, item 9: comparison between the retained cases (local, CPU; post hoc family F12.comparison).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\phase3_comparison.py

Olfactory phenotype vector of a lesion: held-out decoding accuracy of the 96 trials at 2 odour pairs x 6 delays x 4
read-out sets (runs/phase3-olfactory). Every condition saw the same input noise (the generator is reseeded per
condition), so trials are paired across conditions. Distance between two lesions: Euclidean norm of the difference
of their accuracy vectors. Null: each trial's outcomes (all 48 cells together, keeping their dependence) are swapped
between the two lesions with probability 1/2 (10,000 draws); p = (draws at or above the observed + 1) / 10,001.
Tests: C04 (Abeta42 late, KC loss) vs C16 (tau, KC hypoexcitability) at doses 1 and 2; C01 (Abeta42 early, Kv4 loss)
vs C04 at dose 1 (the early and late stages of one progression).
Descriptive capacity profile per lesion (the dissociations): memory = trained-timing histories solved of the five
models relative to healthy (regime A); movement = giant-fibre looming spikes relative to healthy; olfaction = mean
accuracy change of the MBON decoders over pairs and delays; activity = task rate relative to healthy.
Writes results/phase3_comparison_tests.csv and phase3_profiles.csv; the tests also go to batch2-style collection via
results/phase3_comparison_tests.csv (collect_tests.py).
"""
import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
OLF = STUDY / 'runs' / 'phase3-olfactory'
RA = STUDY / 'runs' / 'phase3-regime-a'
TRAINED = {(d, p) for d in (0, 40, 160) for p in (0, 1, 4)}
N_PERM = 10_000
READOUTS = ('readout512', 'mbon', 'kc', 'pn')


def outcomes(name):
    """(trials, cells) matrix of held-out correctness and the cell labels."""
    rec = json.loads((OLF / f'{name}.json').read_text(encoding='utf-8'))
    cols, labels = [], []
    for pair in sorted(rec['pairs']):
        for row in rec['pairs'][pair]:
            for k in READOUTS:
                cols.append(np.asarray(row[k]['correct'], float))
                labels.append((pair, row['delay_ms'], k))
    return np.stack(cols, 1), labels


def distance_test(a, b, test_id):
    A, la = outcomes(a)
    B, lb = outcomes(b)
    if la != lb:
        raise RuntimeError('cell layouts differ')
    obs = float(np.linalg.norm(A.mean(0) - B.mean(0)))
    rng = np.random.default_rng(np.random.SeedSequence(20261019, spawn_key=(zlib.crc32(test_id.encode()),)))
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        swap = rng.random(A.shape[0]) < 0.5
        A2 = np.where(swap[:, None], B, A)
        B2 = np.where(swap[:, None], A, B)
        null[i] = np.linalg.norm(A2.mean(0) - B2.mean(0))
    return obs, float((np.sum(null >= obs - 1e-12) + 1) / (N_PERM + 1)), float(np.quantile(null, 0.95))


def profile(name):
    olf = json.loads((OLF / f'{name}.json').read_text(encoding='utf-8'))
    healthy_olf = json.loads((OLF / 'healthy.json').read_text(encoding='utf-8'))
    d = [r['mbon']['accuracy'] - h['mbon']['accuracy'] for p in olf['pairs'] for r, h in zip(olf['pairs'][p], healthy_olf['pairs'][p])]
    ra = json.loads((RA / f'{name}.json').read_text(encoding='utf-8'))
    hra = json.loads((RA / 'healthy.json').read_text(encoding='utf-8'))
    solved = lambda rec: sum(r['status'] == 'solved' for v in rec['memory'].values() for r in v['rows'] if (r['delay_ms'], r['prefix_windows']) in TRAINED)  # noqa: E731
    return dict(condition=name, memory_relative=solved(ra) / solved(hra), movement_gf_relative=ra['loom']['gf_spikes'] / hra['loom']['gf_spikes'],
                olfaction_mbon_accuracy_change=float(np.mean(d)), activity_relative=ra['activity']['task_mean_hz'] / hra['activity']['task_mean_hz'])


def write(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    tests = []
    for a, b in (('C04-dose1', 'C16-dose1'), ('C04-dose2', 'C16-dose2'), ('C01-dose1', 'C04-dose1')):
        obs, p, q95 = distance_test(a, b, f'comparison.{a}.{b}')
        tests.append(dict(family='F12.comparison', test=f'olfactory phenotype distance, {a} vs {b}', n=96, estimate=obs, null_q95=q95, p_value=p))
    profiles = [profile(n) for n in ('C01-dose0.5', 'C01-dose1', 'C01-dose1.5', 'C01-dose2', 'C04-dose1', 'C04-dose2', 'C13-dose1', 'C13-dose2',
                                     'C16-dose1', 'C16-dose2', 'C04-random', 'C13-random', 'C16-random')]
    out = STUDY / 'results'
    write(out / 'phase3_comparison_tests.csv', tests)
    write(out / 'phase3_profiles.csv', profiles)
    for t in tests:
        print(t)
    for r in profiles:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
