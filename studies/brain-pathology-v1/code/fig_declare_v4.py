"""Amendment 0014 (local): the two descriptive recordings for the v4 figures (manuscript/NARRATIVE_V4_OUTLINE.md, Phase 4
data needs 2-3), declared before either runs. No tests; both feed figure panels only.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studies\\brain-pathology-v1\\code\\fig_declare_v4.py
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / 'studies' / 'brain-pathology-v1' / 'code'


def sha(name):
    return hashlib.sha256((CODE / name).read_bytes()).hexdigest()


def main():
    sys.path.insert(0, str(ROOT))
    from studykit.registry import Study
    study = Study.named('brain-pathology-v1')
    path = study.amend('Phase 4 figure recordings, descriptive', dict(
        why='the v4 figure plan (user-approved 2026-10-22) needs two per-cell data sets that no run recorded',
        recordings=[
            dict(code='code/fig_divergence_cells.py', sha256=sha('fig_divergence_cells.py'), host='GPU',
                 what='per-cell first spike-divergence time under the T-DY1 -3e-4 mV threshold shift (P05 brain, M0 input, '
                      'timing 160 ms / 4 prefix windows, both cues)',
                 check='the per-ms count of differing cells must equal the recorded T-DY1 Hamming trajectory exactly',
                 use='Fig. 6D dark band'),
            dict(code='code/sc_lineage_vnc_v3_export.py', sha256=sha('sc_lineage_vnc_v3_export.py'), host='CPU',
                 what='the declared v3 VNC bridge re-run with per-cell UMAP coordinates, Leiden cluster, published cluster '
                      'and hemilineage unit',
                 check='cluster recovery, units and gates V1-V4 must equal runs/sc-lineage-vnc-v3 exactly',
                 use='Fig. 4C Rosetta view')],
        tests='none; descriptive'))
    print(path.name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
