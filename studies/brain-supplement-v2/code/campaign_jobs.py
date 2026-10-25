"""Write the training campaign (X1 replicates and Phase 3 regime B, learning under lesion) as a job list for campaign.sh.

    .venv-linux/bin/python -B -X utf8 studies/brain-supplement-v2/code/campaign_jobs.py

Writes runtime/campaign/jobs.txt (one job per line: id, run folder, model, replicate, lesion file or '-') and the
lesion spec files under studies/brain-pathology-v1/runs/regime-b-lesions/. Order = priority (plan v2 budget order):
1. replicate 0 of M0 readout-only (must reproduce the frozen model) and X1 replicates 1-9 of M0 readout-only;
2. regime B: M0 readout-only trained under the C01 lesion at doses 0.5, 1, 1.5, 2 (replicates 1-3);
3. X1 replicates 0-9 of M0 joint; regime B with M0 joint (C01 doses, replicates 1-3);
4. controls: the C04, C13 and C16 lesions (dose 1) with M0 readout-only, replicate 1 (circuits the visual task does
   not recruit; an expected null);
5. X1 replicates 0-6 of S1 and S2 (readout-only).
Severity 0 of regime B is the healthy X1 replicate with the same index.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SUPP = ROOT / 'studies' / 'brain-supplement-v2' / 'runs'
PATH = ROOT / 'studies' / 'brain-pathology-v1'
LESIONS = PATH / 'runs' / 'regime-b-lesions'
N_CELLS = 167106
C01_MV = -3.22
REGIME_B_AMENDMENT = ROOT / 'studies' / 'brain-pathology-v1' / 'amendments' / 'regime-b-n.json'


def regime_b_replicates():
    """Regime B replicates 1..n: n = 3 (the pilot) unless the committed power amendment (plan G3d, power_regime_b.py) sets
    more; the healthy partner of replicate r is the X1 replicate r."""
    n = 3
    if REGIME_B_AMENDMENT.exists():
        n = max(n, int(json.loads(REGIME_B_AMENDMENT.read_text(encoding='utf-8'))['proposed_replicates']))
    return tuple(range(1, n + 1))


def lesion_file(name, specs):
    LESIONS.mkdir(parents=True, exist_ok=True)
    path = LESIONS / f'{name}.json'
    path.write_text(json.dumps(specs), encoding='utf-8')
    return path.relative_to(ROOT).as_posix()


def main():
    conds = {c['name']: c for c in json.loads((PATH / 'runs' / 'phase3-regime-a' / 'conditions.json').read_text(encoding='utf-8'))}
    jobs = []
    m0h, m0j, s1, s2 = 'remote-dynamic-M0-head_only', 'remote-dynamic-M0-joint', 'supp-S1-ring-brain', 'supp-S2-context-brain'

    def add(jid, folder, model, rep, lesion='-'):
        jobs.append(f'{jid}\t{folder}\t{model}\t{rep}\t{lesion}')

    for r in range(10):
        add(f'x1-M0-head_only-r{r}', f'studies/brain-supplement-v2/runs/x1-M0-head_only-r{r}', m0h, r)
    c01 = {}
    for dose in (0.5, 1.0, 1.5, 2.0):
        c01[dose] = lesion_file(f'C01-dose{dose:g}', [dict(kind='threshold_offset', cells=list(range(N_CELLS)), value=C01_MV * dose,
                                                       label=f'pan-neuronal Kv4 loss, dose {dose:g}')])
    for dose in (0.5, 1.0, 1.5, 2.0):
        for r in regime_b_replicates():
            add(f'b-C01-dose{dose:g}-M0-head_only-r{r}', f'studies/brain-pathology-v1/runs/b-C01-dose{dose:g}-M0-head_only-r{r}', m0h, r, c01[dose])
    for r in range(10):
        add(f'x1-M0-joint-r{r}', f'studies/brain-supplement-v2/runs/x1-M0-joint-r{r}', m0j, r)
    for dose in (0.5, 1.0, 1.5, 2.0):
        for r in regime_b_replicates():
            add(f'b-C01-dose{dose:g}-M0-joint-r{r}', f'studies/brain-pathology-v1/runs/b-C01-dose{dose:g}-M0-joint-r{r}', m0j, r, c01[dose])
    for case in ('C04', 'C13', 'C16'):
        c = conds[f'{case}-dose1']
        f = lesion_file(f'{case}-dose1', c['operators'])
        add(f'b-{case}-dose1-M0-head_only-r1', f'studies/brain-pathology-v1/runs/b-{case}-dose1-M0-head_only-r1', m0h, 1, f)
    for model, short in ((s1, 'S1'), (s2, 'S2')):
        for r in range(7):
            add(f'x1-{short}-head_only-r{r}', f'studies/brain-supplement-v2/runs/x1-{short}-head_only-r{r}', model, r)
    target = ROOT / 'runtime' / 'campaign' / 'jobs.txt'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(jobs) + '\n', encoding='utf-8')
    print(f'{len(jobs)} jobs -> {target}')


if __name__ == '__main__':
    sys.exit(main())
