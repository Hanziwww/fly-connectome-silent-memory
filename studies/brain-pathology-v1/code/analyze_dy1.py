"""T-DY1 analysis (local, after the pulls; F14.DY.sensitivity, descriptive). Written before any T-DY1 result existed.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\analyze_dy1.py

Divergence (runs/dy1-sensitivity, dy1_sensitivity.py), per timing, cue and perturbation:
- growth rate: least-squares slope of ln(L2 distance of v) on time over the growth window, from the first nonzero
  checkpoint to the first checkpoint at half the largest distance any perturbation reaches at that timing and cue (at
  least 3 points, else not estimated); per second, with its R^2. A Lyapunov-like exponent, not a formal one.
- distance at the decision: the mean v distance over the decision frame's checkpoints, and the mean spike Hamming distance
  over the decision frame (cells whose spike differs, per ms), also as a fraction of the reference's spikes per ms.
- final distance: the last checkpoint.
Size dependence (threshold shifts): slope of ln(decision distance) on ln(shift) over the shifts with a nonzero distance.
About 1 means proportional, a linear response (merely sensitive). About 0, after positive growth, means saturated at a
size-independent level (chaotic).
Memory: the trained-history share solved of the five memory models against the static shift, from runs/phase4-dose-micro
(-1e-4 to -3e-3 mV), runs/phase4-dose-low and runs/phase4-dose (C01 rows) and regime A (healthy, 0.5-2x).
Writes results/dy1_divergence.csv (one row per timing x cue x perturbation), dy1_size.csv (size dependence per timing
and cue) and dy1_memory.csv.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
RUNS = STUDY / 'runs'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STUDY / 'code'))
C01_MV = -3.22


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def growth(t, d, ceiling):
    """Slope (per ms) and R^2 of ln d on t from the first nonzero point to the first point at >= ceiling / 2."""
    t, d = np.asarray(t, float), np.asarray(d, float)
    nz = np.flatnonzero(d > 0)
    if not len(nz):
        return None, None, 0
    start = nz[0]
    stop = next((i for i in range(start, len(d)) if d[i] >= ceiling / 2), len(d) - 1)
    idx = np.arange(start, stop + 1)
    idx = idx[d[idx] > 0]
    if len(idx) < 3:
        return None, None, len(idx)
    x, y = t[idx], np.log(d[idx])
    slope, icpt = np.polyfit(x, y, 1)
    pred = slope * x + icpt
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - pred) ** 2).sum()) / ss if ss > 0 else None
    return float(slope), r2, len(idx)


def divergence(out):
    folder = RUNS / 'dy1-sensitivity'
    rows, size = [], []
    for p in sorted(folder.glob('d*-p*-cue*.json')):
        rec = json.loads(p.read_text(encoding='utf-8'))
        every, onset, total, dec = rec['every_ms'], rec['onset_ms'], rec['total_ms'], rec['decision_ms']
        span = total - onset
        ref_rate = rec['reference_spikes_after_onset'] / span
        ceiling = max(max(r['dv']) for r in rec['runs'] if r['dv'])
        n_cp = len(rec['runs'][0]['dv'])
        t_cp = np.arange(n_cp) * every
        dec_cp = [i for i in range(n_cp) if t_cp[i] >= span - dec]
        for r in rec['runs']:
            slope, r2, npts = growth(t_cp, r['dv'], ceiling)
            ham = np.asarray(r['hamming'], float)
            rows.append(dict(delay_ms=rec['delay_ms'], prefix_windows=rec['prefix_windows'], cue=rec['cue'], kind=r['kind'],
                             size=r['arg'], growth_per_s=None if slope is None else slope * 1000, growth_r2=r2, growth_points=npts,
                             v_distance_decision=float(np.mean([r['dv'][i] for i in dec_cp])),
                             adapt_distance_decision=float(np.mean([r['dadapt'][i] for i in dec_cp])),
                             g_distance_decision=float(np.mean([r['dg'][i] for i in dec_cp])),
                             hamming_decision=float(ham[-dec:].mean()), hamming_decision_fraction=float(ham[-dec:].mean() / ref_rate),
                             v_distance_final=r['dv'][-1], first_divergent_ms=int(np.argmax(ham > 0)) if ham.any() else None,
                             refractory_at_onset=r.get('refractory_at_onset')))
        th = [x for x in rows if x['kind'] == 'threshold' and (x['delay_ms'], x['prefix_windows'], x['cue']) ==
              (rec['delay_ms'], rec['prefix_windows'], rec['cue']) and x['v_distance_decision'] > 0]
        if len(th) >= 2:
            sl = float(np.polyfit(np.log([x['size'] for x in th]), np.log([x['v_distance_decision'] for x in th]), 1)[0])
        else:
            sl = None
        size.append(dict(delay_ms=rec['delay_ms'], prefix_windows=rec['prefix_windows'], cue=rec['cue'], shifts_with_distance=len(th),
                         log_distance_on_log_shift=sl, reference_spikes_per_ms=ref_rate))
    write(out / 'dy1_divergence.csv', rows)
    write(out / 'dy1_size.csv', size)
    return rows, size


def memory(out):
    from analyze_phase4_dose import histories, load
    rows = []
    ra = RUNS / 'phase3-regime-a'
    sources = [(0.0, ra / 'healthy.json', 'regime A')] + [(d, ra / f'C01-dose{d:g}.json', 'regime A') for d in (0.5, 1.0, 1.5, 2.0)]
    for folder, label in (('phase4-dose-micro', 'micro (T-DY1)'), ('phase4-dose-low', 'low extension'), ('phase4-dose', 'dose series')):
        for p in sorted((RUNS / folder).glob('C01-dose*.json')):
            sources.append((None, p, label))
    for dose, path, label in sources:
        if not path.exists():
            continue
        rec = load(path)
        if dose is None:
            dose = rec['dose']
        h = histories(rec)
        flags = [f for v in h.values() for f in v]
        rows.append(dict(source=label, condition=rec.get('condition', 'healthy'), dose=dose, shift_mv=C01_MV * dose,
                         solved_share=float(np.mean(flags)), histories=len(flags), task_mean_hz=rec['activity']['task_mean_hz']))
    rows.sort(key=lambda r: (r['dose'], r['source']))
    write(out / 'dy1_memory.csv', rows)
    return rows


def main():
    out = STUDY / 'results'
    if not (RUNS / 'dy1-sensitivity' / 'done.json').exists():
        print('pending: dy1-sensitivity is not finished')
        return 0
    rows, size = divergence(out)
    mem = memory(out) if (RUNS / 'phase4-dose-micro' / 'done.json').exists() else []
    if not mem:
        print('pending: the micro-dose memory set is not finished')
    for s in size:
        print(f"d{s['delay_ms']}-p{s['prefix_windows']} cue {s['cue']}: ln distance on ln shift slope {s['log_distance_on_log_shift']}, "
              f"reference {s['reference_spikes_per_ms']:.1f} spikes/ms")
    for kind in ('threshold', 'spike', 'noise'):
        g = [r['growth_per_s'] for r in rows if r['kind'] == kind and r['growth_per_s'] is not None]
        f = [r['hamming_decision_fraction'] for r in rows if r['kind'] == kind]
        print(f"{kind}: growth per s median {np.median(g) if g else None}, decision spike mismatch fraction median {np.median(f):.3f}")
    for m in mem:
        print(f"{m['source']:15s} shift {m['shift_mv']:+.4f} mV: share solved {m['solved_share']:.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
