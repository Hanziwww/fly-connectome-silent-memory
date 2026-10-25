"""Mapping v2 and H1 v2 (local), as declared in tasks/sc-mapping-v2-design.json (amendments 0008-0010, 0013).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\sc_trace_v2.py

Inputs: runs/sc-mapping (mapping v1) and runs/sc-lineage-vnc-v3 (the v3 bridge, sc_lineage_vnc_v3.py on the host,
pulled; amendment 0013 replaced runs 1-2 of sc_lineage_vnc.py, which failed gate V1). The lineage tiers
are used only if the gates V1-V3 passed (gates.json); a cell whose v1 tier is class or unmapped and whose trumanHl label
is in label_map.csv takes its unit and tier, lineage or lineage_pair (exact > alias > fuzzy > lineage > lineage_pair >
class; amendment 0010).
Values as amendment 0005 over each source's v2 mapped units, the lineage units being one more source (allen2020vnc).
H1 v2 at the type + lineage level (any tier if fewer than 50% of the M0 carriers map there): mean A and mean SK z of the
mapped carriers vs label permutations within superclass x source strata and the mapping null (values permuted across
mapped types, and hemilineages for lineage-tier cells, within superclass x source); one-sided, 10,000 draws,
default_rng(20261022) per test. F14.SC.trace.v2 (M0, 4 tests), .secondary (M0 joint, S1, S2; 12 tests) and .vnc (M0
within the VNC compartment, superclass vnc_* or ascending_neuron; 4 tests). Descriptive: the same statistics with the
strict lineage tier alone (pairs left out).
Outputs (results/): sc_mapping_v2_coverage.csv, sc_unit_values_v2.csv, sc_trace_v2_tests.csv, sc_trace_v2_strict.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_common import INDEX_GENES, KV4_GENE, RESULTS, RUNS, SOURCE_ORDER  # noqa: E402
from sc_index import MAPPING, cell_map  # noqa: E402
from sc_trace import SEED, N, p_greater, strata_draws  # noqa: E402

LINEAGE = RUNS / 'sc-lineage-vnc-v3'
LIN_SOURCE = 'allen2020vnc'
TYPE_LINEAGE = ('exact', 'alias', 'fuzzy', 'lineage', 'lineage_pair')
ANY = TYPE_LINEAGE + ('class',)
MODELS = ('m0', 'm0_joint', 's1', 's2')
GENES = INDEX_GENES + (KV4_GENE,)


def table(source, kind):
    folder = LINEAGE if source == LIN_SOURCE else MAPPING
    path = folder / f'units_{source}_{kind}.parquet'
    if not path.exists():
        return None
    t = pd.read_parquet(path, columns=list(GENES))
    t.index = t.index.astype(str)
    return t


def unit_values(cells):
    rows = []
    for source in SOURCE_ORDER + (LIN_SOURCE,):
        us = sorted(set(cells.loc[cells['source'] == source, 'unit'].astype(str)))
        if not us:
            continue
        mean = table(source, 'mean').loc[us]
        z = (mean - mean.mean()) / mean.std(ddof=0)
        out = pd.DataFrame({f'z_{g}': z[g] for g in GENES})
        out['A'] = z[list(INDEX_GENES)].mean(axis=1)
        out.insert(0, 'unit', us)
        out.insert(0, 'source', source)
        rows.append(out.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def mapping_draws(values, perm_unit, strata_of_unit, carriers, rng):
    units = np.array(sorted(set(perm_unit)))
    pos = {u: i for i, u in enumerate(units)}
    vals = np.array([values[u] for u in units])
    groups = {}
    for i, u in enumerate(units):
        groups.setdefault(strata_of_unit[u], []).append(i)
    groups = [np.array(v) for _, v in sorted(groups.items())]
    weights = np.bincount([pos[u] for u in perm_unit[carriers]], minlength=len(units)).astype(float)
    out = np.empty(N)
    for i in range(N):
        perm = vals.copy()
        for idx in groups:
            perm[idx] = vals[idx[rng.permutation(len(idx))]]
        out[i] = (weights * perm).sum() / weights.sum()
    return out


def main():
    gates = json.loads((LINEAGE / 'gates.json').read_text(encoding='utf-8'))
    cells = cell_map()
    if gates.get('passed'):
        lm = pd.read_csv(LINEAGE / 'label_map.csv').set_index('malecns_label')
        hl = cells['trumanHl'].astype('object').where(cells['trumanHl'].notna(), None)
        up = cells['tier'].isin(['class', 'unmapped']).to_numpy() & np.array([h in lm.index for h in hl])
        cells['unit'] = cells['unit'].astype('object')
        cells['source'] = cells['source'].astype('object')
        cells['tier'] = cells['tier'].astype('object')
        cells.loc[up, 'tier'] = lm.loc[hl[up], 'tier'].to_numpy()
        cells.loc[up, 'source'] = LIN_SOURCE
        cells.loc[up, 'unit'] = lm.loc[hl[up], 'unit'].astype(str).to_numpy()
    else:
        print('lineage gates not passed: mapping v2 = mapping v1', gates)
    v1 = cell_map()
    carriers = {m: cells[f'delta_adapt_{m}'].fillna(0).ne(0).to_numpy() for m in MODELS}
    sc = cells['superclass'].fillna('').astype(str)
    vnc = (sc.str.startswith('vnc_') | (sc == 'ascending_neuron')).to_numpy()
    cov = []
    for name, mask in (('all', np.ones(len(cells), bool)), ('VNC compartment (vnc_*, ascending)', vnc),
                       ('readout cells', cells['observed_by_policy'].to_numpy(bool)),
                       *((f'trace carriers ({m})', carriers[m]) for m in MODELS),
                       ('trace carriers (m0), VNC compartment', carriers['m0'] & vnc)):
        t = cells['tier'][mask]
        cov.append(dict(cell_set=name, cells=int(mask.sum()), type_or_fuzzy=float(t.isin(TYPE_LINEAGE[:3]).mean()),
                        lineage=float((t == 'lineage').mean()), lineage_pair=float((t == 'lineage_pair').mean()),
                        type_lineage=float(t.isin(TYPE_LINEAGE).mean()), any_tier=float(t.isin(ANY).mean())))
    pd.DataFrame(cov).to_csv(RESULTS / 'sc_mapping_v2_coverage.csv', index=False, lineterminator='\n', float_format='%.6g')
    vals = unit_values(cells)
    vals.to_csv(RESULTS / 'sc_unit_values_v2.csv', index=False, lineterminator='\n', float_format='%.6g')
    tests = h1(cells, vals, carriers, np.ones(len(cells), bool), MODELS,
               lambda m: 'F14.SC.trace.v2' if m == 'm0' else 'F14.SC.trace.v2.secondary', 'all')
    tests += h1(cells, vals, carriers, vnc, ('m0',), lambda m: 'F14.SC.trace.v2.vnc', 'vnc')
    pd.DataFrame(tests).to_csv(RESULTS / 'sc_trace_v2_tests.csv', index=False, lineterminator='\n')
    # descriptive: the strict lineage tier alone (lineage-pair cells back to their v1 tier)
    strict = cells.copy()
    back = (strict['tier'] == 'lineage_pair').to_numpy()
    for col in ('tier', 'source', 'unit'):
        strict.loc[back, col] = v1.loc[back, col].astype('object').to_numpy()
    svals = unit_values(strict)
    desc = h1(strict, svals, carriers, np.ones(len(cells), bool), ('m0',), lambda m: 'descriptive', 'all-strict', p=False)
    desc += h1(strict, svals, carriers, vnc, ('m0',), lambda m: 'descriptive', 'vnc-strict', p=False)
    pd.DataFrame(desc).to_csv(RESULTS / 'sc_trace_v2_strict.csv', index=False, lineterminator='\n')
    print(pd.DataFrame(cov).to_string(index=False))
    print(pd.DataFrame(tests)[['family', 'test', 'n', 'coverage', 'statistic', 'mean_reference', 'p_value']].to_string(index=False))
    print(pd.DataFrame(desc)[['test', 'n', 'coverage', 'statistic', 'mean_reference']].to_string(index=False))
    return 0


def h1(cells, vals, carriers, compartment, models, family_of, scope, p=True):
    """H1 v2 tests in a compartment: level type + lineage if >= 50% of the M0 carriers there map at it, else any tier."""
    cv = cells.copy()
    cv['unit'] = cv['unit'].astype('object').where(cv['unit'].isna(), cv['unit'].astype(str))
    cv = cv.merge(vals, on=['source', 'unit'], how='left')
    car_m0 = carriers['m0'] & compartment
    level_ok = float(cv.loc[car_m0, 'tier'].isin(TYPE_LINEAGE).mean()) >= 0.5
    level, allowed = ('type_lineage', TYPE_LINEAGE) if level_ok else ('any', ANY)
    mapped = cv['tier'].isin(allowed).to_numpy() & compartment
    sub = cv[mapped].reset_index(drop=True)
    strata = (sub['superclass'].fillna('NA').astype(str) + '|' + sub['source'].astype(str)).to_numpy()
    lineage = sub['tier'].isin(['lineage', 'lineage_pair']).to_numpy()
    perm_unit = np.where(lineage, 'HL:' + sub['unit'].astype(str), sub['type'].astype(str))
    unit_stratum = pd.Series(strata).groupby(perm_unit).agg(lambda s: s.value_counts().idxmax()).to_dict()
    out = []
    for model in models:
        car = carriers[model][mapped]
        for stat, col in (('A', 'A'), ('SK', 'z_SK')):
            x = sub[col].to_numpy(float)
            if np.isnan(x).any():
                raise SystemExit(f'{scope} {model}: missing {col} for a mapped cell')
            if (pd.Series(x).groupby(perm_unit).nunique() > 1).any():
                raise SystemExit('a permutation unit carries more than one value')
            t = float(x[car].mean())
            per_unit = pd.Series(x).groupby(perm_unit).first().to_dict()
            nulls = {'matched-sets': strata_draws(x, strata, car, np.random.default_rng(SEED)),
                     'mapping-null': mapping_draws(per_unit, perm_unit, unit_stratum, car, np.random.default_rng(SEED))}
            for kind, null in nulls.items():
                row = dict(family=family_of(model), test=f'{model}:{scope}:{level}:{stat}:{kind}', n=int(car.sum()),
                           n_reference=int(mapped.sum()), carriers_total=int((carriers[model] & compartment).sum()),
                           coverage=float(cv.loc[carriers[model] & compartment, 'tier'].isin(allowed).mean()),
                           alternative='greater', statistic=t, mean=t, mean_reference=float(null.mean()),
                           null_sd=float(null.std()))
                if p:
                    row['p_value'] = p_greater(t, null)
                out.append(row)
    return out


if __name__ == '__main__':
    sys.exit(main())
