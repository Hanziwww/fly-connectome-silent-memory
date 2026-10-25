# Manuscript freeze v3 (Cell figures and videos)

- Code: commit `5c3694e1715e2e9e0969f7d471e9614875174e3b` (clean).
- Environment: Python 3.12.13; numpy 2.5.3, scipy 1.18.1, pandas 3.0.6, matplotlib 3.11.2, numba 0.67.0, pillow 12.3.0, pyarrow 25.0.1.
- Unit tests: manuscript/analysis/tests/run_tests.py, manuscript/figures/tests/run_tests.py passed.
- Analysis: one complete run from a clean state (226 files); the two-run determinism check was not repeated for this freeze (the user asked for a quick freeze).
- Figures: 23 render scripts, the palette check and 14 main and Supplemental figures rebuilt once in fresh processes; all 388 output files byte-identical across passes; 3 differed from the files on disk before the freeze (manuscript/figures/v4/videos/Video_S1.mp4, manuscript/figures/v4/videos/Video_S1_poster.png, manuscript/figures/v4/videos/Video_S3.mp4).
- QC: every figure has 0 issues (sizes below; width 174 mm, height at most 225 mm).
- Not rebuilt: `figures/renders/pilot` (renderer pilot, used by no figure) and the downloaded MaleCNS skeletons (inputs; SHA-256 in `figures/assets/skeletons/receipt.csv`); the design previews of the revision-3 board (`figures/v4/preview`, built against a preview results root); superseded: the Nature-format figures (`figures/main`, `figures/extended`), pinned by the freeze-v1.1 tag.

| Figure | Size (mm) | Tiles | QC issues |
|---|---|---:|---:|
| fig1 | 174 × 221.161 | 6 | 0 |
| fig2 | 174 × 215.419 | 5 | 0 |
| fig3 | 174 × 224.4 | 14 | 0 |
| fig4 | 174 × 220 | 8 | 0 |
| fig5 | 174 × 208.238 | 4 | 0 |
| fig6 | 174 × 224.238 | 5 | 0 |
| fig7 | 174 × 212.8 | 16 | 0 |
| s1 | 174 × 199 | 4 | 0 |
| s2 | 174 × 211.038 | 9 | 0 |
| s3 | 174 × 155.6 | 4 | 0 |
| s4 | 174 × 200.6 | 6 | 0 |
| s5 | 174 × 137 | 7 | 0 |
| s6 | 174 × 208.25 | 20 | 0 |
| s7 | 174 × 223.2 | 4 | 0 |

| Deliverables | Files | MB |
|---|---:|---:|
| QC record | 84 | 112.9 |
| Supplemental figure | 21 | 83.1 |
| analysis table | 209 | 647.9 |
| figure-ready table | 17 | 44.4 |
| main figure | 21 | 116.8 |
| other | 8 | 9.8 |
| render | 228 | 59.7 |
| source data | 14 | 0.8 |
| supplemental video | 12 | 31.2 |

Every file, with its SHA-256, is listed in `FREEZE_MANIFEST.csv`.

## Amendment (2026-10-24, D89)

The analysis reads the declared size of every test family and a recorded generator state of one bootstrap
(`manuscript/analysis/declared`), so that it reproduces its outputs from a deposit that holds only the conditions the
paper reports. Phases P12, P8, P10 and P9 were rerun on the study's complete tables: every result table is
byte-identical, and five bookkeeping tables (`meta/input_manifest.csv`, `meta/output_manifest.csv`,
`meta/data_dictionary.csv`, `qa/readonly_audit.csv`, `qa/schema_report.csv`) record the added input; their hashes in
`FREEZE_MANIFEST.csv` are updated. On the deposit's tables the readers of Figures 6 and S6 and of Table S2 write the
published tables byte for byte.
