"""Gate G2: score the Phase 2 credibility tests (C4-C6) with the literature scores (C1-C3) and freeze the shortlist.

    .\\MIGRATION\\python\\python.exe -I studies\\brain-pathology-v1\\code\\score_credibility.py

Reads runs/phase2-credibility/<brain>/{c4,c5_c6}.json (pulled from the GPU host) and literature/candidates.csv.
Writes credibility-table.csv (one row per candidate, every criterion with its evidence) and g2-shortlist.json (the
retained cases with operator and calibrated dose, the excluded cases with the failing criterion, and SHA-256 of every
input). Decision rules, fixed before scoring (plan v2, Phase 2c):
- a case is retained only if C1-C5 pass; C6 grades robustness;
- grades 0-2 per criterion; the total ranks the passing cases; at most 4 are retained;
- ties are broken by (i) reach to the planned capacity assays (learning, memory, movement) and (ii) the pre-specified
  Abeta42-versus-tau comparison, which needs one case of each.
"""
import csv
import hashlib
import json
import statistics as st
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RUNS = STUDY / 'runs' / 'phase2-credibility'
LIT = STUDY / 'literature'
BRAINS = ('p05', 'm0joint')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(b, name):
    return json.loads((RUNS / b / name).read_text(encoding='utf-8'))


def mean(x):
    return float(st.mean(x))


def c4_grades(b):
    c4 = load(b, 'c4.json')
    sm, apl, ln = c4['smell'], c4['apl_removed'], c4['ln_removed']
    kc = mean(sm['kc_fraction_responding'])
    mb = (0.02 <= kc <= 0.25) and mean(apl['kc_fraction_responding']) > 2 * kc and apl['kc_pair_overlap'] > sm['kc_pair_overlap']
    al = mean(ln['pn_mean_hz']) > 1.5 * mean(sm['pn_mean_hz']) and ln['pn_lifetime_sparseness_median'] < sm['pn_lifetime_sparseness_median'] \
        and max(sm['pn_mean_hz']) < 100
    stable = [r for r in c4['stable'] if r['offset_mv'] >= -4.0]
    net = all(r['fraction_over_100hz'] < 0.01 for r in stable)
    loom = next(r for r in c4['loom'] if r['condition'] == 'loom')['gf_spikes'] > 0 and next(r for r in c4['loom'] if r['condition'] == 'blank')['gf_spikes'] == 0
    return dict(mb=mb, al=al, network=net, loom=loom, noop=c4['noop']['other_cells_bit_identical'], kc_fraction=kc,
                kc_fraction_apl_removed=mean(apl['kc_fraction_responding']), kc_overlap=sm['kc_pair_overlap'],
                kc_overlap_apl_removed=apl['kc_pair_overlap'], pn_hz=mean(sm['pn_mean_hz']), pn_hz_ln_removed=mean(ln['pn_mean_hz']),
                pn_ls=sm['pn_lifetime_sparseness_median'], pn_ls_ln_removed=ln['pn_lifetime_sparseness_median'])


CIRCUIT = {'C01': 'network', 'C02': 'mb', 'C16': 'mb', 'C15': 'mb', 'C14': 'mb', 'C04': 'mb', 'C13': 'al', 'C11': 'al', 'C12': 'al', 'C07': 'mb'}
REACH = {'C01': 'learning, memory and movement tasks (pan-neuronal)'}
AXIS = {'C01': 'abeta42', 'C02': 'abeta42', 'C04': 'abeta42', 'C11': 'abeta42', 'C12': 'abeta42', 'C16': 'tau', 'C15': 'tau',
        'C13': 'fragile_x', 'C14': 'aging', 'C07': 'aging'}


def c5_c6(cid, b, healthy):
    """(C5 pass, C5 grade, C6 pass, evidence) for one candidate on one brain."""
    r = load(b, 'c5_c6.json')
    rows = {row['dose']: row for row in r[cid]['rows']}
    one = rows[1.0]
    o = lambda row, k: mean(row['odour'][k])       # noqa: E731
    if cid == 'C01':
        cal = r['calibration_rheobase']
        cur, chg = cal['ramp_current_ratio'], cal['ramp_charge_ratio']
        ok = abs(cur / 0.54 - 1) <= 0.25 and abs(chg / 0.43 - 1) <= 0.5 and one['task_mean_hz'] > healthy['task_hz']
        mono = rows[0.5]['task_mean_hz'] < one['task_mean_hz'] < rows[1.5]['task_mean_hz']
        return ok, 2, mono, f'ramp current ratio {cur:.3f} (fly 0.54), charge ratio {chg:.3f} (fly 0.43); task rate {healthy["task_hz"]:.2f} -> {one["task_mean_hz"]:.2f} Hz'
    if cid == 'C02':
        ok = o(one, 'kc_mean_spikes') > healthy['kc_spikes']
        mono = o(rows[0.5], 'kc_mean_spikes') < o(one, 'kc_mean_spikes') < o(rows[1.5], 'kc_mean_spikes')
        return ok, 1, mono, f'KC odour spikes {healthy["kc_spikes"]:.3f} -> {o(one, "kc_mean_spikes"):.3f} (direction of Kv4 loss)'
    if cid == 'C16':
        ok = one.get('fitted_offset_mv') is not None and 0 < one['fitted_offset_mv'] < 7 and o(one, 'kc_fraction_responding') > 0.01
        mono = rows[0.5]['fitted_offset_mv'] < one['fitted_offset_mv'] < rows[1.5]['fitted_offset_mv']
        return ok, 1, mono, f'+{one["fitted_offset_mv"]:.2f} mV gives the evoked KC response x0.519 (calibration; KC coding not silenced: {o(one, "kc_fraction_responding"):.3f} responding)'
    if cid == 'C15':
        reached = one['kc_drive_m4m6'] / healthy['kc_drive_m4m6']
        ok = reached >= 2.689 * 0.75 and one['fitted_offset_mv'] > -6.4
        return ok, 0, False, f'M4/M6 response x{reached:.2f} at the fit bound {one["fitted_offset_mv"]:.2f} mV (fly x2.69): not reachable'
    if cid == 'C14':
        ad, ad0 = one.get('kcab_sustained_odour', {}), r['kcab_sustained_odour_healthy']
        changed = ad.get('median_index') is not None and ad0.get('median_index') is not None and ad['median_index'] > ad0['median_index'] * 1.1
        return changed, 0, False, (f'KCab sustained-odour adaptation index {ad0.get("median_index")} -> {ad.get("median_index")} at k = 0.1 '
                                   '(no circuit-level change; the fly phenotype is behavioural, with no matched assay)')
    if cid == 'C04':
        ok = o(one, 'mbon_mean_hz') < healthy['mbon_hz']
        mono = o(rows[0.5], 'mbon_mean_hz') > o(one, 'mbon_mean_hz') > o(rows[1.5], 'mbon_mean_hz')
        return ok, 1, mono, f'MBON odour response {healthy["mbon_hz"]:.2f} -> {o(one, "mbon_mean_hz"):.2f} Hz with 17% of KCs removed'
    if cid == 'C13':
        ls = one['odour']['pn_lifetime_sparseness_median']
        ok = ls < healthy['pn_ls']
        mono = rows[0.5]['odour']['pn_lifetime_sparseness_median'] > ls > rows[1.5]['odour']['pn_lifetime_sparseness_median']
        return ok, 2, mono, f'PN lifetime sparseness {healthy["pn_ls"]:.3f} -> {ls:.3f} (fly: reduced, p = 9.8e-14; not fitted)'
    if cid in ('C11', 'C12'):
        up = cid == 'C11'
        pn = o(one, 'pn_mean_hz')
        ok = pn > healthy['pn_hz'] if up else pn < healthy['pn_hz']
        seq = [o(rows[d], 'pn_mean_hz') for d in (0.5, 1.0, 1.5)]
        mono = seq == sorted(seq) if up else seq == sorted(seq, reverse=True)
        return ok, 1, mono, f'PN odour rate {healthy["pn_hz"]:.2f} -> {pn:.2f} Hz (input x{one["value"]:.3f})'
    if cid == 'C07':
        kc = o(one, 'kc_mean_spikes')
        ok = kc <= healthy['kc_spikes'] * 1.1
        return ok, 0, False, f'KC odour spikes {healthy["kc_spikes"]:.3f} -> {kc:.3f} with ALPN release x1.5 (fly: postsynaptic KC Ca2+ flat or lower)'
    raise KeyError(cid)


def main():
    cands = {r['candidate_id'].split('_')[0]: r for r in csv.DictReader(open(LIT / 'candidates.csv', encoding='utf-8'))}
    rows = []
    for cid in CIRCUIT:
        c = cands[cid]
        g1, g2, g3 = int(c['c1_grade']), int(c['c2_grade']), int(c['c3_grade'])
        p13 = c['c1_evidence_in_flies_pass'] == 'pass' and c['c2_mechanism_measured_and_mappable_pass'] == 'pass' and c['c3_representable_with_existing_operators_pass'] == 'pass'
        per = {}
        for b in BRAINS:
            h = load(b, 'c4.json')
            r = load(b, 'c5_c6.json')
            healthy = dict(kc_spikes=mean(h['smell']['kc_mean_spikes']), mbon_hz=mean(h['smell']['mbon_mean_hz']), pn_hz=mean(h['smell']['pn_mean_hz']),
                           pn_ls=h['smell']['pn_lifetime_sparseness_median'], task_hz=h['stable'][0]['mean_hz'],
                           kc_drive_m4m6=_healthy_m4m6(r))
            per[b] = dict(c4=c4_grades(b), c5=c5_c6(cid, b, healthy))
        c4ok = all(per[b]['c4'][CIRCUIT[cid]] for b in BRAINS)
        c5ok = all(per[b]['c5'][0] for b in BRAINS)
        c6ok = all(per[b]['c5'][2] for b in BRAINS)
        g5 = min(per[b]['c5'][1] for b in BRAINS) if c5ok else 0
        total = g1 + g2 + g3 + (2 if c4ok else 0) + g5 + (2 if c6ok else 0)
        rows.append(dict(candidate=cid, condition=c['condition'], axis=AXIS[cid], circuit=CIRCUIT[cid], operator=c['proposed_model_operator'],
                         c1=g1, c2=g2, c3=g3, c1_c3_pass=p13, c4_pass=c4ok, c5_pass=c5ok, c5_grade=g5, c6_pass=c6ok, total=total,
                         c5_evidence_p05=per['p05']['c5'][3], c5_evidence_m0joint=per['m0joint']['c5'][3],
                         reaches_current_tasks=cid in REACH))
    passing = [r for r in rows if r['c1_c3_pass'] and r['c4_pass'] and r['c5_pass']]
    passing.sort(key=lambda r: (-r['total'], not r['reaches_current_tasks']))
    retained = []
    for r in passing:
        if len(retained) == 4:
            break
        retained.append(r)
    if not any(r['axis'] == 'tau' for r in retained):      # the pre-specified Abeta42-versus-tau comparison needs a tau case
        tau = [r for r in passing if r['axis'] == 'tau']
        if tau:
            drop = next((r for r in reversed(retained) if r['axis'] == 'abeta42' and r['candidate'] != 'C01'
                         and sum(x['axis'] == 'abeta42' for x in retained) > 1), None)
            if drop is not None:
                retained[retained.index(drop)] = tau[0]
    keep = {r['candidate'] for r in retained}
    # candidates decided before simulation (literature/review_main.md): no C4-C6 test was run for them
    for cid, decision, crit in (('C03', 'deferred', 'C3 (no published magnitude to calibrate)'),
                                ('C17', 'deferred', 'C3 (magnitude needs the full text)'),
                                ('C18', 'deferred', 'C2/C3 (weak mapping, no magnitude)'),
                                ('C05', 'merged with C16', 'same measurement (Higham 2019, KC Ca2+ -48%)'),
                                ('C22', 'control arm', 'no change to replicate'),
                                ('C23', 'movement only, not tested', 'retina-only expression; the task input bypasses the photoreceptors'),
                                ('C06', 'excluded', 'C2/C3'), ('C08', 'excluded', 'C3 (DPM output has weight 0)'),
                                ('C09', 'excluded', 'C3 (dopamine synapses have weight 0)'), ('C10', 'excluded', 'C3 (serotonin receptor level)'),
                                ('C19', 'excluded', 'C3 (l-LNv serotonergic; change cannot propagate)'), ('C20', 'excluded', 'C2/C3'),
                                ('C24', 'excluded', 'C3 (dopamine release)'), ('C21', 'excluded', 'C1 (single 1984 study, contradicted)')):
        c = cands[cid]
        rows.append(dict(candidate=cid, condition=c['condition'], axis=None, circuit=None, operator=c['proposed_model_operator'],
                         c1=int(c['c1_grade']), c2=int(c['c2_grade']), c3=int(c['c3_grade']), c1_c3_pass=False, c4_pass=None,
                         c5_pass=None, c5_grade=None, c6_pass=None, total=None, c5_evidence_p05=None, c5_evidence_m0joint=None,
                         reaches_current_tasks=False, decision=decision, failing_criterion=crit))
    for r in rows:
        if 'decision' in r:
            continue
        r['decision'] = 'retained' if r['candidate'] in keep else ('passed, not retained (rank)' if r in passing else 'excluded')
        if r['decision'] == 'excluded':
            r['failing_criterion'] = next(k for k, ok in (('C1-C3', r['c1_c3_pass']), ('C4', r['c4_pass']), ('C5', r['c5_pass'])) if not ok)
        else:
            r['failing_criterion'] = None
    with open(STUDY / 'credibility-table.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    c4 = {b: c4_grades(b) for b in BRAINS}
    inputs = sorted([*(LIT.glob('*.csv')), *(LIT.glob('*.json')), *(LIT.glob('*.md')), *(RUNS.glob('*/*.json'))])
    out = dict(gate='G2', date='2026-10-19', decision_rules=__doc__.split('Decision rules')[1].strip(),
               c4=c4, retained=[dict(candidate=r['candidate'], condition=r['condition'], axis=r['axis'], circuit=r['circuit'],
                                     total=r['total']) for r in retained],
               excluded=[dict(candidate=r['candidate'], failing_criterion=r['failing_criterion'], evidence=r['c5_evidence_p05'])
                         for r in rows if r['decision'] == 'excluded'],
               passed_not_retained=[r['candidate'] for r in rows if r['decision'].startswith('passed')],
               inputs={str(p.relative_to(STUDY)).replace('\\', '/'): sha(p) for p in inputs})
    (STUDY / 'g2-shortlist.json').write_text(json.dumps(out, indent=1), encoding='utf-8', newline='\n')
    for r in rows:
        print(f"{r['candidate']} {str(r['axis']):10s} total {str(r['total']):>4} C4 {r['c4_pass']!s:5} C5 {r['c5_pass']!s:5} C6 {r['c6_pass']!s:5} -> {r['decision']}"
              + (f" ({r['failing_criterion']})" if r['failing_criterion'] else ''))


def _healthy_m4m6(r):
    """M4/M6 response to the KC drive in the healthy brain: the C14 row at dose 0.5 is not healthy, so use the C13 rows,
    whose operator does not touch the KC -> MBON path (all three doses give the same value)."""
    return mean([row['kc_drive_m4m6'] for row in r['C13']['rows']])


if __name__ == '__main__':
    main()
