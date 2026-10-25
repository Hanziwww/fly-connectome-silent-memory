"""O12, circuit level (local, after the pull): the C4-C6 credibility tests on every structural variant, scored with the
frozen G2 rules (score_credibility.py, unchanged). Written before any variant result existed.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\score_variants.py

Reads runs/phase2-variants/<variant>/{variant,c4,c5_c6}.json (phase2_variants.py). Checks first that 'default'
reproduces runs/phase2-credibility/p05 exactly (else nothing is scored). Then per variant: the C4 signatures (mushroom
body, antennal lobe, network stability to -4 mV, looming, monoamine no-op) and, for the four G2 cases (C01, C04, C13,
C16), the case's circuit signature (C4), C5 (the replication at the calibrated dose) and C6 (the dose direction at 0.5x
and 1.5x).
Decision rule (declared with the variants, 2026-10-19): a case's circuit-level replication is robust to structure when
its circuit signature and C5 pass under every variant whose healthy network is stable (C4 network criterion). A variant
whose healthy network is not stable is reported as such; the cases are not assessed on it. Descriptive; no test.
Writes results/phase2_variants.csv (one row per variant x case) and results/phase2_variants_c4.csv (one row per variant).
"""
import csv
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RUNS = STUDY / 'runs' / 'phase2-variants'
FROZEN = STUDY / 'runs' / 'phase2-credibility' / 'p05'
CASES = ('C01', 'C04', 'C13', 'C16')


def write(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty'], lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    sys.path.insert(0, str(STUDY / 'code'))
    import score_credibility as sc
    import variants
    out = STUDY / 'results'
    if not (RUNS / 'done.json').exists():
        print('pending: the variant runs are not finished')
        return 0
    for name in ('c4.json', 'c5_c6.json'):
        if json.loads((RUNS / 'default' / name).read_text(encoding='utf-8')) != json.loads((FROZEN / name).read_text(encoding='utf-8')):
            raise SystemExit(f'default/{name} does not reproduce the frozen P05 credibility run; nothing scored')
    sc.RUNS = RUNS                                   # score_credibility.load reads RUNS / <brain> / <file>
    rows, c4rows = [], []
    for v in variants.VARIANTS:
        h, r = sc.load(v, 'c4.json'), sc.load(v, 'c5_c6.json')
        g = sc.c4_grades(v)
        meta = sc.load(v, 'variant.json')
        healthy = dict(kc_spikes=sc.mean(h['smell']['kc_mean_spikes']), mbon_hz=sc.mean(h['smell']['mbon_mean_hz']),
                       pn_hz=sc.mean(h['smell']['pn_mean_hz']), pn_ls=h['smell']['pn_lifetime_sparseness_median'],
                       task_hz=h['stable'][0]['mean_hz'], kc_drive_m4m6=sc._healthy_m4m6(r))
        c4rows.append(dict(variant=v, overrides=repr(variants.VARIANTS[v]), entries_changed=meta.get('entries_changed'),
                           abs_weight_change=meta['abs_weight_sum'] / meta['abs_weight_sum_healthy'] - 1,
                           task_hz=healthy['task_hz'], rate_at_minus4mv=h['stable'][-1]['mean_hz'],
                           over_100hz_at_minus4mv=h['stable'][-1]['fraction_over_100hz'],
                           **{f'c4_{k}': g[k] for k in ('mb', 'al', 'network', 'loom', 'noop')},
                           kc_fraction=g['kc_fraction'], kc_fraction_apl_removed=g['kc_fraction_apl_removed'], pn_hz=g['pn_hz'],
                           pn_hz_ln_removed=g['pn_hz_ln_removed'], pn_ls=g['pn_ls'], pn_ls_ln_removed=g['pn_ls_ln_removed'],
                           gf_loom_spikes=next(x for x in h['loom'] if x['condition'] == 'loom')['gf_spikes']))
        for cid in CASES:
            ok, grade, mono, evidence = sc.c5_c6(cid, v, healthy)
            rows.append(dict(variant=v, case=cid, circuit=sc.CIRCUIT[cid], healthy_network_stable=g['network'],
                             circuit_signature=g[sc.CIRCUIT[cid]], c5_pass=ok, c6_pass=mono,
                             assessed=g['network'], evidence=evidence))
    write(out / 'phase2_variants_c4.csv', c4rows)
    write(out / 'phase2_variants.csv', rows)
    for cid in CASES:
        sub = [r for r in rows if r['case'] == cid and r['assessed']]
        robust = all(r['circuit_signature'] and r['c5_pass'] for r in sub)
        print(f"{cid}: assessed on {len(sub)} of {len(variants.VARIANTS)} variants; circuit signature and C5 pass on "
              f"{sum(r['circuit_signature'] and r['c5_pass'] for r in sub)}; C6 on {sum(r['c6_pass'] for r in sub)} -> "
              f"{'robust to structure' if robust and sub else 'not robust'}")
    for r in c4rows:
        print(f"{r['variant']:15s} stable {r['c4_network']!s:5} MB {r['c4_mb']!s:5} AL {r['c4_al']!s:5} loom {r['c4_loom']!s:5} "
              f"noop {r['c4_noop']!s:5} rate {r['task_hz']:.2f} Hz, at -4 mV {r['rate_at_minus4mv']:.2f} Hz")
    return 0


if __name__ == '__main__':
    sys.exit(main())
