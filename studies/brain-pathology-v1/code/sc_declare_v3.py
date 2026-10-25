"""Amendment 0013 (local): the VNC bridge v3 (sc_lineage_vnc_v3.py), declared before it reads any atlas cell. Runs only
with --declare, after the user's go-ahead.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\sc_declare_v3.py --declare

Records the run-2 diagnosis (runs/sc-lineage-vnc-diag/vnc_sink_diag.json, pulled hash-checked), the v3 procedure and
code hash, what stays unchanged and the final-attempt rule. sc_trace_v2.py then reads the bridge from
runs/sc-lineage-vnc-v3 (its LINEAGE constant, changed in the same commit).
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
DIAG = STUDY / 'runs' / 'sc-lineage-vnc-diag' / 'vnc_sink_diag.json'
CODE = STUDY / 'code' / 'sc_lineage_vnc_v3.py'
TRACE = STUDY / 'code' / 'sc_trace_v2.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if '--declare' not in sys.argv:
        raise SystemExit('pass --declare (after the user approves the v3 bridge)')
    d = json.loads(DIAG.read_text(encoding='utf-8'))
    few = sorted((c for c in d['clusters'] if c['n_markers'] <= 3), key=lambda c: -c['ratio'])
    if "sc-lineage-vnc-v3" not in TRACE.read_text(encoding='utf-8'):
        raise SystemExit('sc_trace_v2.py must read the bridge from runs/sc-lineage-vnc-v3 first')
    sys.path.insert(0, str(ROOT))
    from studykit.registry import Study
    study = Study.named('brain-pathology-v1')
    path = study.amend('T-SC1b VNC bridge v3 by cluster correspondence, final attempt', dict(
        diagnosis=dict(
            evidence=dict(file='runs/sc-lineage-vnc-diag/vnc_sink_diag.json', sha256=sha(DIAG),
                          code='code/sc_lineage_vnc_diag.py', code_sha256=sha(STUDY / 'code' / 'sc_lineage_vnc_diag.py')),
            finding='the three most over-filled published clusters in run 2 are exactly the three with <= 3 published '
                    'markers: ' + '; '.join(f"{c['cluster']} ({c['n_markers']} marker{'s' if c['n_markers'] > 1 else ''}, "
                                            f"{c['recovered']} cells vs {c['published_cells']} published)" for c in few),
            mechanism='a cell score is the mean z of a cluster\'s markers, so its spread falls with the number of markers '
                      '(SD 1.0 for one marker against a median 0.31 for >= 12): any cell detecting the single marker wins the '
                      'per-cell argmax; the remaining clusters are under-filled (median recovery 0.42)'),
        change=dict(
            code=dict(file='code/sc_lineage_vnc_v3.py', sha256=sha(CODE), selftest='synthetic data, 14 planted clusters '
                      'incl. two single-marker ones: sizes rho 1.0, 100% of cells correct'),
            procedure='the authors\' Seurat 2.3.4 pipeline (Allen et al. 2020 Methods) in scanpy: their QC and log1p(cp10k) '
                      '(as run 2); variable genes per sex (Seurat v2 dispersion, cut-offs 0.001), intersected; per sex '
                      'replicate, UMI and mitochondrial fraction regressed out, scaled and clipped at 10; 45 principal '
                      'components (for their 45 CCA dimensions); 30-nearest-neighbour graph; Leiden over-clustered at twice '
                      'the published granularity (240 clusters; bisection on log resolution, seed 0)',
            correspondence='run 2\'s cell scores standardised per published cluster over the cells; per new cluster the '
                           'mean standardised score; every published cluster first takes one new cluster (Hungarian, '
                           'maximum summed score), every remaining new cluster joins its best-scoring published cluster; an '
                           'assignment counts if its score is >= 0.25',
            outputs='runs/sc-lineage-vnc-v3 (run 2 stays in runs/sc-lineage-vnc)'),
        unchanged='QC, normalisation, marker sets, units (>= 30 cells), the label map (amendment 0010), gates V1-V3 and '
                  'their thresholds, H1 v2 and its families (amendments 0008-0010); sc_trace_v2.py reads the v3 bridge',
        seen_so_far='runs 1-2 gates and cluster recovery, the diagnosis above; no H1 v2 statistic has been computed',
        rule='final attempt: if V1, V2 or V3 fails the VNC stays unmapped and F14.SC.trace.v2 and its subfamilies are not '
             'computed (as amendment 0011); if all pass, sc_trace_v2.py runs as declared',
        trace_code=dict(file='code/sc_trace_v2.py', sha256=sha(TRACE))))
    print(path.name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
