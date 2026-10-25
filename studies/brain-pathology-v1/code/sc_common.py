"""Shared paths and helpers of the single-cell study (T-SC1 to T-SC8; manuscript/SINGLE_CELL_INTEGRATION_PLAN.md, plan
approved 2026-10-22). Runs locally (model-side tables) and on rtx5090x4 (expression data, in .venv-sc).

Data (verified against data/external/receipt.csv):
- D1 Davis et al. 2020: GEO GSE116969 dataTable3 (bulk TPM per driver-line sample);
- D2 Ozel et al. 2021: GEO GSE142787 Log_normalized_average_expression.xlsx (Adult sheet: mean per cluster);
- D4 Davie et al. 2018: GEO GSE107451 57k whole brain (10x counts) and metadata (`annotation`);
- D6 Aging Fly Cell Atlas head (Lu et al. 2023): GEO GSE218661 (log-normalised X, `afca_annotation`, ages 5-70 d);
- D8 AD Fly Cell Atlas head (Park et al. 2025): GEO GSE261656 (`umi_counts` layer, `adfca_annotation`, 14 samples);
- D5 Allen et al. 2026: GEO GSE296540 (four new 10x samples) and supplementary Tables S2-S5 (types named by transmitter;
  no hemilineage labels, so no lineage bridge: the plan's Gate SC-A fallback applies).
Expression metric where counts exist: mean over cells of log1p(counts per 10,000) and the detection fraction (flybrain's
expression_central.csv metric); D6 provides log1p(cp10k) values directly (checked by mapping QC).
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
EXT = ROOT / 'data' / 'external'
INPUTS = STUDY / 'inputs'
RESULTS = STUDY / 'results'
RUNS = STUDY / 'runs'
DERIVED = ROOT / 'runtime' / 'sc' / 'derived'            # host-side extracted copies (not part of the record)
FLYDATA = ROOT / 'flybrain' / 'flybrain' / 'data'
MODEL_CELLS = INPUTS / 'sc_model_cells.parquet'
TIERS = ('exact', 'alias', 'fuzzy', 'class')
TIER_RANK = {t: i for i, t in enumerate(TIERS)}
# H1 levels (Gate SC-A, amendment 0004): the tiers a cell's unit may have at each level; class is the primary level
LEVELS = {'type': ('exact', 'alias'), 'type_fuzzy': ('exact', 'alias', 'fuzzy'), 'class': TIERS}

# Gene panels (T-SC2, SINGLE_CELL_INTEGRATION_PLAN.md section 4), declared before any panel value is computed; FlyBase
# IDs are pinned in tasks/sc-panels-v1.json from the sources' own identifiers.
INDEX_GENES = ('SK', 'slo', 'KCNQ', 'Shab')         # adaptation index: Ca2+-activated (SK, BK) and slow K+ (M, Kv2)
KV4_GENE = 'Shal'
DESCRIPTIVE_GENES = ('Shaw', 'Shal', 'Sh', 'Ih', 'cac', 'Ca-alpha1D', 'CngA', 'CngB', 'Cngl', 'Nos')
K_CHANNEL_GENES = ('Sh', 'Shab', 'Shaw', 'Shawl', 'Shal', 'KCNQ', 'SK', 'slo', 'eag', 'sei', 'Elk', 'Irk1', 'Irk2', 'Irk3',
                   'Ork1', 'Task6', 'Task7')      # aging operator (T-SC5): K+ channel pore-forming subunits
AGING_GENES = K_CHANNEL_GENES + ('Hk', 'para', 'NaCP60E', 'cac', 'Ca-alpha1D', 'Ca-alpha1T', 'Ca-beta', 'Ih', 'brp', 'Syt1',
                                 'nSyb', 'Snap25', 'Syx1A', 'unc-13', 'Rim', 'cpx', 'VAChT', 'ChAT', 'VGlut', 'VGAT',
                                 'Gad1')
PANEL_GENES = tuple(dict.fromkeys(INDEX_GENES + (KV4_GENE,) + DESCRIPTIVE_GENES + AGING_GENES))
DROPOUT = 0.10                                       # detection fraction below which a unit-gene value is flagged (M3)
SOURCE_ORDER = ('davis2020', 'ozel2021', 'davie2018', 'afca2023', 'adfca2025')   # preference among equal tiers
SOURCE_LABEL = {'davis2020': 'D1 Davis 2020 (bulk, driver lines)', 'ozel2021': 'D2 Ozel 2021 (optic lobe)',
                'davie2018': 'D4 Davie 2018 (whole brain)', 'afca2023': 'D6 Aging Fly Cell Atlas (head)',
                'adfca2025': 'D8 AD Fly Cell Atlas (head, controls)'}


def sha256(path, block=1 << 24):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(out, script, inputs):
    """provenance.json beside a host run's outputs: code and input files (bytes, SHA-256), Python and package versions.
    Host outputs go under runs/ (the only study outputs a pull brings back); results/ tables are derived locally."""
    import importlib.metadata as md
    import json
    import platform
    import time

    def rel(p):
        return Path(p).resolve().relative_to(ROOT).as_posix()

    record = dict(script=rel(script), code={rel(p): sha256(p) for p in (script, __file__)},
                  inputs={rel(p): dict(bytes=Path(p).stat().st_size, sha256=sha256(p)) for p in inputs},
                  python=platform.python_version(),
                  packages={p: md.version(p) for p in ('numpy', 'scipy', 'pandas', 'anndata', 'h5py', 'pyarrow', 'openpyxl')},
                  finished_unix=time.time())
    (Path(out) / 'provenance.json').write_text(json.dumps(record, indent=1), encoding='utf-8', newline='\n')


def group_mean_frac(X, groups, log1p_cp10k_from_counts):
    """Per group: mean expression and detection fraction over cells. X: cells x genes (scipy CSR). If
    log1p_cp10k_from_counts, X holds counts and is normalised to log1p(counts per 10,000) row by row first."""
    import scipy.sparse as sp
    X = sp.csr_matrix(X, dtype=np.float64)
    if log1p_cp10k_from_counts:
        totals = np.asarray(X.sum(axis=1)).ravel()
        totals[totals == 0] = 1.0
        X = sp.diags(1e4 / totals) @ X
        X.data = np.log1p(X.data)
    labels, inv = np.unique(np.asarray(groups, dtype=object).astype(str), return_inverse=True)
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(labels), len(inv)))
    n = np.asarray(G.sum(axis=1)).ravel()
    mean = np.asarray((G @ X).todense()) / n[:, None]
    det = X.copy()
    det.data = np.ones_like(det.data)
    frac = np.asarray((G @ det).todense()) / n[:, None]
    return labels, n.astype(int), mean, frac
