"""O14, task level (local, after the pulls): the timing lock and the C01 memory effect of M0 readout-only models trained
under each structural variant. Written before any O14 result existed.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\analyze_o14.py

Runs: runs/o14-<variant>-r<r> (replicates 0-2 of the frozen M0 readout-only protocol, replicate_training.py --variant);
the default structure is X1's M0 readout-only replicates 0-2 (studies/brain-supplement-v2/runs/x1-M0-head_only-r<r>).
Rules (declared 2026-10-19, with the variants):
- per run, from verification.json as in X1 (analyze_campaign.run_row): learned = status learned_solved; net memory use
  at the 9 trained and the 26 untrained timings; lock present = learned, trained net use > 0 and untrained net use < 0.10
  (X1's criteria);
- C01 effect present = fewer trained histories solved under C01 at dose 1 than healthy (runs/o14-lesion); a run whose
  healthy re-assessment does not reproduce its verification.json has its lesion result withheld;
- a task-level conclusion is robust to structure when it holds in every run of every variant whose healthy network is
  stable (results/phase2_variants_c4.csv, c4_network; score_variants.py). Unstable variants are reported, not counted.
Descriptive (3 replicates per variant); no test. Writes results/o14_runs.csv.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
SUPP = ROOT / 'studies' / 'brain-supplement-v2'
TRAINED = {(d, p) for d in (0, 40, 160) for p in (0, 1, 4)}
REPLICATES = (0, 1, 2)


def main():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(SUPP / 'code'))
    sys.path.insert(0, str(STUDY / 'code'))
    from analyze_campaign import run_row
    import variants
    c4 = STUDY / 'results' / 'phase2_variants_c4.csv'
    if not c4.exists():
        print('pending: the circuit-level variant results (score_variants.py)')
        return 0
    stable = {r['variant']: r['c4_network'] == 'True' for r in csv.DictReader(open(c4, encoding='utf-8'))}
    rows, pending = [], []
    for v in variants.VARIANTS:
        for r in REPLICATES:
            folder = SUPP / 'runs' / f'x1-M0-head_only-r{r}' if v == 'default' else STUDY / 'runs' / f'o14-{v}-r{r}'
            if not (folder / 'verification.json').exists():
                pending.append(folder.name)
                continue
            row = dict(variant=v, replicate=r, healthy_network_stable=stable[v], **run_row(folder, folder.name))
            row['lock_present'] = (row['status'] == 'learned_solved' and row['trained_net_use'] > 0 and row['untrained_net_use'] < 0.10)
            les = STUDY / 'runs' / 'o14-lesion' / f'{folder.name}.json'
            if v != 'default' and les.exists():
                rec = json.loads(les.read_text(encoding='utf-8'))
                tr = {k: sum(x['status'] == 'solved' for x in rec[k]['result']['rollouts'] if (x['delay_ms'], x['prefix_windows']) in TRAINED)
                      for k in ('healthy', 'C01')}
                row.update(reproduces_verification=rec['healthy_reproduces_verification'], trained_solved_healthy=tr['healthy'],
                           trained_solved_c01=tr['C01'])
                if rec['healthy_reproduces_verification']:
                    row['c01_effect_present'] = tr['C01'] < tr['healthy']
            rows.append(row)
    if rows:
        with open(STUDY / 'results' / 'o14_runs.csv', 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
    for v in variants.VARIANTS:
        sub = [r for r in rows if r['variant'] == v]
        print(f"{v:15s} stable {stable[v]!s:5} runs {len(sub)}: lock {sum(r['lock_present'] for r in sub)}, "
              f"C01 effect {sum(bool(r.get('c01_effect_present')) for r in sub)} of {sum('c01_effect_present' in r for r in sub)}; "
              f"trained net use {[round(r['trained_net_use'], 2) for r in sub]}, untrained {[round(r['untrained_net_use'], 2) for r in sub]}")
    if pending:
        print('pending:', ', '.join(pending))
        return 0
    counted = [r for r in rows if r['healthy_network_stable']]
    lock = all(r['lock_present'] for r in counted)
    c01 = all(r.get('c01_effect_present') for r in counted if r['variant'] != 'default')
    print(f"timing lock robust to structure: {lock}; C01 memory effect robust to structure: {c01} "
          f"(over {len({r['variant'] for r in counted})} stable variants)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
