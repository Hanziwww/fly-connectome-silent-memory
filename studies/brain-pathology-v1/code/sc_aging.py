"""T-SC5: an aging operator from cell-type-resolved channel expression, through C4-C6 (G2 rules; F14.SC.aging, descriptive).

    local, once:  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\sc_aging.py --declare
    host (CPU):   .venv-sc/bin/python -I studies/brain-pathology-v1/code/sc_aging.py --expression
    host (GPU):   .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/sc_aging.py --run
    local:        ... sc_aging.py --analyze

Design (tasks/sc-aging-design.json, amendment 0007, written before any age change was computed):
- K+ expression: per unit and sample, pseudobulk CPM of the summed K+ channel genes (sc_common.K_CHANNEL_GENES).
  Primary Davie 2018 (one study): young = 1, 3, 6, 9 days, old = 30, 50 days, samples = Age x Genotype x Replicate;
  log2 f = mean log2(CPM_K + 1) of the old samples minus that of the young. Replication: Aging Fly Cell Atlas, ages 30 vs
  70 days within the AFCA dataset (the 5-day cells come from the FCA, a batch confound), per sex, counts reconstructed
  as expm1(X) x total_counts / 1e4. Units need >= 50 cells in young and in old (per sex for AFCA).
  Pan-neuronal: Davie all cells except glia, plasmatocytes, photoreceptors and the Hsp cluster; AFCA 'CNS neuron'.
- Mapping to threshold offsets through the C01 calibration (-3.22 mV for a Kv4 current ratio of 0.54):
  offset = -3.22 mV x (1 - f) / 0.46 x dose. aging-pan: f of Davie's pan-neuronal unit on every cell. aging-class: a cell
  takes the mean log2 f of its Davie candidate units (mapping v1 candidates, any tier); cells without one take the mean of
  their superclass's cells that have one, else the pan-neuronal value.
- C4-C6 on both brains (P05, M0 joint; credibility.Rig): healthy, then each operator at doses 0.5, 1, 1.5; odour panel,
  task activity, KC-drive MBON responses, looming response. C5 (fly: KC postsynaptic odour Ca2+ flat or lower with age,
  doi:10.1371/journal.pbio.1002563): mean KC odour spikes aged / healthy <= 1.05 at dose 1 on both brains, and the network
  stable (<= 0.1% of cells above 100 Hz under the task input). C6: the same at doses 0.5 and 1.5 on both brains. An
  operator passing C5 and C6 becomes a G2 candidate under the G2 rules; otherwise it is reported with the deciding
  criterion. Descriptive: per-gene age changes; KCNQ against the published brain KCNQ mRNA decline
  (doi:10.1371/journal.pone.0062445).
"""
import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_common import DERIVED, EXT, K_CHANNEL_GENES, RESULTS, ROOT, RUNS, STUDY, write_provenance  # noqa: E402

DESIGN = STUDY / 'tasks' / 'sc-aging-design.json'
OUT = RUNS / 'sc-aging'
YOUNG, OLD = (1, 3, 6, 9), (30, 50)
AFCA_AGES = ('30', '70')
MIN_CELLS = 50
DAVIE_NON_NEURONAL = ('Ensheathing_glia', 'Astrocyte-like', 'Perineurial_glia', 'Cortex_glia', 'Subperineurial_glia',
                      'Chiasm_glia', 'Plasmatocytes', 'Photoreceptors', 'Hsp')
PAN = 'pan-neuronal'
C01_MV, C01_CURRENT_RATIO = -3.22, 0.54
DOSES = (0.5, 1.0, 1.5)
KC_RATIO_MAX, RUNAWAY_MAX = 1.05, 0.001


def declare():
    sys.path.insert(0, str(ROOT))
    from studykit.registry import Study
    if DESIGN.exists():
        raise SystemExit(f'{DESIGN.name} exists; task files are written once')
    design = dict(
        task='T-SC5 aging operator from cell-type-resolved K+ channel expression, C4-C6 (G2 rules; F14.SC.aging)',
        declared='2026-10-22, before any age change of expression was computed',
        code=dict(file='code/sc_aging.py', sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        genes=list(K_CHANNEL_GENES), davie=dict(young=list(YOUNG), old=list(OLD), sample='Age x Genotype x Replicate',
                                                non_neuronal=list(DAVIE_NON_NEURONAL)),
        afca=dict(ages=list(AFCA_AGES), by='sex (female, male)', counts='expm1(X) x total_counts / 1e4',
                  pan="afca_annotation_broad 'CNS neuron'", why='the 5-day cells are FCA data (batch)'),
        min_cells=MIN_CELLS, statistic='log2 f = mean log2(CPM_K + 1) old - young',
        mapping=dict(offset=f'{C01_MV} mV x (1 - f) / {1 - C01_CURRENT_RATIO:.2f} x dose',
                     operators=dict(aging_pan="Davie pan-neuronal f on every cell",
                                    aging_class='mean log2 f of the cell\'s Davie candidate units; else its superclass '
                                                'mean; else pan-neuronal')),
        c4_c6=dict(brains=['p05', 'm0joint'], doses=list(DOSES),
                   c5=f'mean KC odour spikes aged / healthy <= {KC_RATIO_MAX} at dose 1 on both brains; <= '
                      f'{RUNAWAY_MAX:.1%} of cells above 100 Hz under the task input',
                   c6='the C5 criteria at doses 0.5 and 1.5 on both brains',
                   reference='KC postsynaptic odour Ca2+ flat or lower with age (doi:10.1371/journal.pbio.1002563)'),
        descriptive=['per-gene age changes', 'KCNQ vs the published decline (doi:10.1371/journal.pone.0062445)',
                     'GF looming response and MBON responses under the operators'])
    text = json.dumps(design, indent=1) + '\n'
    DESIGN.write_text(text, encoding='utf-8', newline='\n')
    study = Study.named('brain-pathology-v1')
    study.lock_tasks(status='draft', note='inputs written before their runs (plan v2 batches); re-locked for amendment '
                                          '0007 (T-SC5 aging operator design)')
    path = study.amend('T-SC5 aging operator design', dict(
        added_task_file=dict(file=f'tasks/{DESIGN.name}', sha256=hashlib.sha256(text.encode('utf-8')).hexdigest()),
        why='T-SC5 (SINGLE_CELL_INTEGRATION_PLAN.md SC3; execution plan approved 2026-10-22): the expression contrast, the '
            'mapping to threshold offsets and the C4-C6 criteria are fixed before any age change is computed.',
        deviations_from_plan=dict(afca='AFCA 30 vs 70 days only (the 5-day cells are FCA data); Davie is primary')))
    print(DESIGN.name, path.name)


def pseudobulk(X, groups, gene_idx):
    """Per group: summed counts of the genes in gene_idx (each, and their total) and of all genes."""
    import scipy.sparse as sp
    labels, inv = np.unique(groups, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(labels), len(inv)))
    total = np.asarray((G @ X).sum(axis=1)).ravel()
    genes = np.asarray((G @ X[:, gene_idx]).todense())
    cells = np.asarray(G.sum(axis=1)).ravel()
    return labels, cells, genes, total


def expression():
    import anndata as ad
    import pandas as pd
    import scipy.io
    import scipy.sparse as sp
    from sc_mapping import AFCA, DAVIE, DAVIE_META
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # Davie 2018
    d = DERIVED / 'davie57k'
    if not d.exists():
        with tarfile.open(DAVIE) as t:
            t.extractall(d, filter='data')
    X = scipy.io.mmread(next(d.rglob('matrix.mtx'))).tocsr().T.tocsr().astype(np.float64)
    genes = pd.read_csv(next(d.rglob('genes.tsv')), sep='\t', header=None).iloc[:, -1].astype(str)
    barcodes = pd.read_csv(next(d.rglob('barcodes.tsv')), sep='\t', header=None).iloc[:, 0].astype(str).to_numpy()
    meta = pd.read_csv(DAVIE_META, sep='\t').set_index('new_barcode').loc[barcodes]
    gi = [int(np.flatnonzero(genes.to_numpy() == g)[0]) for g in K_CHANNEL_GENES]
    age = meta['Age'].to_numpy()
    sample = (meta['Age'].astype(str) + '|' + meta['Genotype'].astype(str) + '|' + meta['Replicate'].astype(str)).to_numpy()
    unit = meta['annotation'].astype(str).to_numpy()
    for u in sorted(set(unit)) + [PAN]:
        cells = ~np.isin(unit, DAVIE_NON_NEURONAL) if u == PAN else unit == u
        m = cells & np.isin(age, YOUNG + OLD)
        if (m & np.isin(age, YOUNG)).sum() < MIN_CELLS or (m & np.isin(age, OLD)).sum() < MIN_CELLS:
            continue
        labels, n, g, tot = pseudobulk(X[m], sample[m], gi)
        old = np.array([int(lab.split('|')[0]) in OLD for lab in labels])
        cpm = 1e6 * g / tot[:, None]
        k = np.log2(cpm.sum(axis=1) + 1)
        row = dict(source='davie2018', unit=u, cells_young=int(n[~old].sum()), cells_old=int(n[old].sum()),
                   samples_young=int((~old).sum()), samples_old=int(old.sum()), log2f=float(k[old].mean() - k[~old].mean()))
        for j, gene in enumerate(K_CHANNEL_GENES):
            v = np.log2(cpm[:, j] + 1)
            row[f'log2f_{gene}'] = float(v[old].mean() - v[~old].mean())
        rows.append(row)
    # AFCA, 30 vs 70 days within the AFCA dataset, per sex
    a = ad.read_h5ad(AFCA, backed='r')
    obs = a.obs
    keep = ((obs['dataset'].astype(str) == 'AFCA') & obs['age'].astype(str).isin(AFCA_AGES)
            & obs['sex'].astype(str).isin(['female', 'male'])).to_numpy()
    names = a.var_names.astype(str)
    gi = [names.get_loc(g) for g in K_CHANNEL_GENES]
    pieces = []
    for s in range(0, a.n_obs, 50000):
        pieces.append(sp.csr_matrix(a.X[s:s + 50000])[keep[s:s + 50000]])
    a.file.close()
    X = sp.vstack(pieces).tocsr().astype(np.float64)
    X.data = np.expm1(X.data)
    tc = obs['total_counts'].to_numpy(float)[keep]
    X = sp.diags(tc / 1e4) @ X                                         # reconstructed counts
    sub = obs[keep]
    ages, sexes = sub['age'].astype(str).to_numpy(), sub['sex'].astype(str).to_numpy()
    units = sub['afca_annotation'].astype(str).to_numpy()
    neuron = (sub['afca_annotation_broad'].astype(str) == 'CNS neuron').to_numpy()
    for u in sorted(set(units)) + [PAN]:
        cells = neuron if u == PAN else units == u
        ok = all(((cells & (ages == ag) & (sexes == sx)).sum() >= MIN_CELLS) for ag in AFCA_AGES for sx in ('female', 'male'))
        if not ok:
            continue
        groups = np.array([f'{ag}|{sx}' for ag, sx in zip(ages[cells], sexes[cells])])
        labels, n, g, tot = pseudobulk(X[cells], groups, gi)
        cpm = 1e6 * g / tot[:, None]
        lab = dict(zip(labels, range(len(labels))))
        k = np.log2(cpm.sum(axis=1) + 1)
        f = [k[lab[f'70|{sx}']] - k[lab[f'30|{sx}']] for sx in ('female', 'male')]
        row = dict(source='afca2023', unit=u, cells_young=int(sum(n[lab[f'30|{sx}']] for sx in ('female', 'male'))),
                   cells_old=int(sum(n[lab[f'70|{sx}']] for sx in ('female', 'male'))), samples_young=2, samples_old=2,
                   log2f=float(np.mean(f)), log2f_female=float(f[0]), log2f_male=float(f[1]))
        for j, gene in enumerate(K_CHANNEL_GENES):
            v = np.log2(cpm[:, j] + 1)
            row[f'log2f_{gene}'] = float(np.mean([v[lab[f'70|{sx}']] - v[lab[f'30|{sx}']] for sx in ('female', 'male')]))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'expression.csv', index=False, lineterminator='\n', float_format='%.6g')
    # per-cell f of the aging-class operator (Davie candidate units of mapping v1)
    cand = pd.read_parquet(RUNS / 'sc-mapping' / 'cell_candidates.parquet')
    cells = pd.read_parquet(RUNS / 'sc-mapping' / 'cell_map.parquet', columns=['bodyId', 'superclass'])
    dv = table[table['source'] == 'davie2018'].set_index('unit')['log2f']
    pan = float(dv[PAN])
    c = cand[(cand['source'] == 'davie2018') & cand['unit'].astype(str).isin(dv.index)]
    per_cell = c.assign(log2f=c['unit'].astype(str).map(dv)).groupby('bodyId')['log2f'].mean()
    t = cells.assign(log2f=cells['bodyId'].map(per_cell))
    sc_mean = t.dropna(subset=['log2f']).groupby(t['superclass'].fillna('NA'))['log2f'].mean()
    t['source_of_f'] = np.where(t['log2f'].notna(), 'davie_unit', None)
    fill = t['superclass'].fillna('NA').map(sc_mean)
    t.loc[t['log2f'].isna() & fill.notna(), 'source_of_f'] = 'superclass_mean'
    t['log2f'] = t['log2f'].fillna(fill)
    t.loc[t['log2f'].isna(), 'source_of_f'] = 'pan_neuronal'
    t['log2f'] = t['log2f'].fillna(pan)
    t.to_parquet(OUT / 'cell_f.parquet', index=False)
    write_provenance(OUT, __file__, [DAVIE, DAVIE_META, AFCA, RUNS / 'sc-mapping' / 'cell_candidates.parquet',
                                     RUNS / 'sc-mapping' / 'cell_map.parquet'])
    print(table[table['unit'] == PAN][['source', 'unit', 'cells_young', 'cells_old', 'log2f', 'log2f_KCNQ']].to_string(index=False))
    print(t['source_of_f'].value_counts().to_dict(), 'pan log2 f', pan)


def offsets(f, dose):
    return C01_MV * (1.0 - f) / (1.0 - C01_CURRENT_RATIO) * dose


def run():
    import pandas as pd
    sys.path.insert(0, str(ROOT / 'studies' / 'brain-sokoban-v1' / 'mechanisms-v1' / 'code'))
    import mechanism_base  # noqa: F401  (path setup)
    import credibility as cr
    from sc_kv4 import rig_checks
    table = pd.read_csv(OUT / 'expression.csv')
    pan = float(table[(table['source'] == 'davie2018') & (table['unit'] == PAN)]['log2f'].iloc[0])
    cell_f = pd.read_parquet(OUT / 'cell_f.parquet')
    for brain, checkpoint in cr.BRAINS.items():
        target = OUT / f'credibility-{brain}.json'
        if target.exists():
            continue
        rig = cr.Rig(checkpoint, OUT / 'meta' / brain)
        bodies = pd.read_parquet(ROOT / 'flybrain' / 'cache' / 'neurons.parquet', columns=['bodyId'])['bodyId'].to_numpy()
        if not (cell_f['bodyId'].to_numpy() == bodies).all():
            raise RuntimeError('cell order differs')
        f_class = np.power(2.0, cell_f['log2f'].to_numpy(float))
        seq = rig.memory_input()
        res = {}

        def measure(specs):
            activity, loom = rig_checks(rig, specs, rig.torch, cr.SETTLE_MS, seq)     # task activity and looming response
            rig.lesion(specs)
            panel = cr.summarize_odours(rig, rig.odour_panel())
            out = dict(kc_mean_spikes=float(np.mean(panel['kc_mean_spikes'])), odour=panel,
                       task_mean_hz=activity['task_mean_hz'], task_fraction_over_100hz=activity['fraction_above_100hz'],
                       kc_drive_mbon=cr._kc_drive_response(rig, rig.sets['mbon']), loom=loom)
            rig.lesion([])
            return out
        res['healthy'] = measure([])
        for dose in DOSES:
            res[f'aging_pan-dose{dose:g}'] = measure([rig.ops.spec('threshold_offset', np.arange(rig.n), offsets(2.0 ** pan, dose),
                                                                   'aging-pan')])
            off = offsets(f_class, dose)
            values, inverse = np.unique(off, return_inverse=True)
            res[f'aging_class-dose{dose:g}'] = measure([rig.ops.spec('threshold_offset', np.flatnonzero(inverse == i), float(v),
                                                                     'aging-class') for i, v in enumerate(values)])
        res['offsets'] = dict(pan_mv_dose1=offsets(2.0 ** pan, 1.0), class_mv_dose1_mean=float(offsets(f_class, 1.0).mean()),
                              class_mv_dose1_min=float(offsets(f_class, 1.0).min()), class_mv_dose1_max=float(offsets(f_class, 1.0).max()))
        target.write_text(json.dumps(res), encoding='utf-8', newline='\n')
        print(brain, {k: round(v['kc_mean_spikes'], 3) for k, v in res.items() if isinstance(v, dict) and 'kc_mean_spikes' in v})
        del rig


def analyze():
    import pandas as pd
    rows = []
    res = {b: json.loads((OUT / f'credibility-{b}.json').read_text(encoding='utf-8')) for b in ('p05', 'm0joint')}
    for op in ('aging_pan', 'aging_class'):
        crit = {}
        for dose in DOSES:
            for b, r in res.items():
                x = r[f'{op}-dose{dose:g}']
                ratio = x['kc_mean_spikes'] / r['healthy']['kc_mean_spikes']
                rows.append(dict(operator=op, brain=b, dose=dose, kc_ratio=ratio, task_mean_hz=x['task_mean_hz'],
                                 fraction_over_100hz=x['task_fraction_over_100hz'], kc_drive_mbon=x['kc_drive_mbon'],
                                 gf_spikes=x['loom']['gf_spikes'], offset_pan_mv_dose1=r['offsets']['pan_mv_dose1'],
                                 offset_class_mean_mv_dose1=r['offsets']['class_mv_dose1_mean']))
                crit[(dose, b)] = ratio <= KC_RATIO_MAX and x['task_fraction_over_100hz'] <= RUNAWAY_MAX
        c5 = crit[(1.0, 'p05')] and crit[(1.0, 'm0joint')]
        c6 = all(crit[(d, b)] for d in (0.5, 1.5) for b in res)
        rows.append(dict(operator=op, brain='both', dose='verdict', c5=c5, c6=c6,
                         outcome='G2 candidate' if c5 and c6 else ('fails C5' if not c5 else 'fails C6')))
    pd.DataFrame(rows).to_csv(RESULTS / 'sc_aging.csv', index=False, lineterminator='\n', float_format='%.6g')
    expr = pd.read_csv(OUT / 'expression.csv')
    expr.to_csv(RESULTS / 'sc_aging_expression.csv', index=False, lineterminator='\n', float_format='%.6g')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    for mode in ('declare', 'expression', 'run', 'analyze'):
        g.add_argument(f'--{mode}', action='store_true')
    a = ap.parse_args()
    {'declare': declare, 'expression': expression, 'run': run, 'analyze': analyze}[next(m for m in ('declare', 'expression', 'run', 'analyze') if getattr(a, m))]()
