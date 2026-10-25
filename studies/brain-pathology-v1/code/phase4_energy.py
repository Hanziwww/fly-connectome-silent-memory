"""Phase 4 (local, CPU): energy per decoded bit of cue memory, healthy and under the retained lesions (descriptive).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\phase4_energy.py

Energy: spikes of the whole brain during one M0 memory trial (cue 40 ms, delay 160 ms, 4 prefix windows, first
decision; 400 ms), from the mean task rate each lesion run measured (regime A, phase3-regime-a/<condition>.json; and the
Phase 4 dose series and its post hoc low extension, amendment 0001, when present). Bits: the mutual information between the cue and the first push at the 9 trained
timings (plug-in, from the cue x choice table: the cue's own target, the other cue's target, or no push), per memory
model (cases with the same cue pooled over contexts and visibility), averaged over the five models. An inverted mapping
carries information too; the share of correct first pushes is reported beside it. Energy per bit = spikes / bits (reported when
bits > 0.05). The maintenance cost itself: the cue is held without differential spiking (Phase 1B fu4: no cell spikes
differently in the second half of the delay), so the healthy memory costs no spikes beyond the baseline.
Writes results/phase4_energy.csv.
"""
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
TRAINED = {(d, p) for d in (0, 40, 160) for p in (0, 1, 4)}
CELLS, TRIAL_S = 167106, 0.4


def h2(p):
    return 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def mutual_information(table):
    n = sum(sum(r) for r in table)
    rows = [sum(r) / n for r in table]
    cols = [sum(r[j] for r in table) / n for j in range(len(table[0]))]
    return sum((x / n) * math.log2((x / n) / (rows[i] * cols[j])) for i, r in enumerate(table) for j, x in enumerate(r) if x)


def bits(rec):
    per_model, correct = [], []
    for v in rec['memory'].values():
        rows = [r for r in v['rows'] if (r['delay_ms'], r['prefix_windows']) in TRAINED]
        if not rows or 'first_push_correct' not in rows[0]:
            return None, None
        table = [[0, 0, 0], [0, 0, 0]]
        for r in rows:
            c = int(r['cue'])
            choice = 2 if r['first_push_correct'] is None else (c if r['first_push_correct'] else 1 - c)
            table[c][choice] += 1
        per_model.append(mutual_information(table))
        correct.append(sum(r['first_push_correct'] is True for r in rows) / len(rows))
    return sum(per_model) / len(per_model), sum(correct) / len(correct)


def main():
    rows = []
    for folder in ('phase3-regime-a', 'phase4-dose', 'phase4-dose-low'):
        for path in sorted((STUDY / 'runs' / folder).glob('*.json')):
            if path.name in ('conditions.json', 'done.json'):
                continue
            rec = json.loads(path.read_text(encoding='utf-8'))
            b, correct = bits(rec)
            spikes = rec['activity']['task_mean_hz'] * CELLS * TRIAL_S
            rows.append(dict(source=folder, condition=rec['condition'], dose=rec.get('dose'), task_mean_hz=rec['activity']['task_mean_hz'],
                             spikes_per_trial=spikes, cue_bits_per_decision=b, first_push_correct_share=correct,
                             spikes_per_bit=spikes / b if b is not None and b > 0.05 else None))
    if not rows:
        print('no lesion runs found')
        return 0
    out = STUDY / 'results' / 'phase4_energy.csv'
    with open(out, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['source']:16s} {r['condition']:18s} rate {r['task_mean_hz']:7.3f} Hz  spikes/trial {r['spikes_per_trial']:10.0f}  "
              f"bits {r['cue_bits_per_decision'] if r['cue_bits_per_decision'] is None else round(r['cue_bits_per_decision'], 3)}  "
              f"spikes/bit {r['spikes_per_bit'] if r['spikes_per_bit'] is None else round(r['spikes_per_bit'])}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
