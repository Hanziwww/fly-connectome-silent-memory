# Simulation records, trained models and result tables (part 1 of 2)

Data of the article *Brain state locks silent memories in a trained fly connectome* by Han Wang et al. ({{CITATION}}).
The code is archived at Zenodo, {{CODE_DOI}}; part 2 of the records is at Zenodo, {{EXTENDED_DOI}}.

This part holds every file that the paper's analysis, renders, figure builds and writing scripts read; the models,
configurations, logs and results of every training run and evaluation listed in Table S3; and the published result tables,
figures, videos, figure source data and supplemental tables. Part 2 holds the verification replays, the behavior features
of every training update and the dense recordings that no analysis reads.

## Archives

{{ARCHIVE_TABLE}}

The paths inside the archives are relative to the top folder of the code release; unpack them there. `MANIFEST.csv` lists
every file with its archive, size and SHA-256. A file that the record stores once for several identical paths lists, for
each other path, the stored path under `same_as`.

```
python verify_deposit.py . --into <code release folder>
```

checks every file against the manifest while it unpacks, and restores those paths. It needs Python 3 and nothing else.

## Contents

| Path | Contents |
|---|---|
| `studies/brain-sokoban-v1/runs/`, `remote-20261003/`, `remote-supplement-20261003/` | the registered study's training runs and evaluations: the planning ladder, its controls and the memory learners |
| `studies/brain-sokoban-v1/mechanisms-v1/`, `mechanisms-supplement-v1/` | recordings of every state variable of every cell of the trained models (`runs/record-*`) and the mechanism experiments |
| `studies/brain-supplement-v2/runs/` | the supplementary experiments' training runs and evaluations |
| `studies/brain-pathology-v1/runs/` | the lesions, the tests of the candidate pathologies, the join with single-cell atlases and the brain-state experiments |
| `studies/*/results/` | the studies' result tables, as their analyses wrote them |
| `runs/linear-joint/` | the two warm-up models of the formal method |
| `manuscript/results/` | the paper's analysis tables; `meta/` holds the analysis's input and output manifests |
| `manuscript/figures/v4/`, `manuscript/figures/renders/v4/` | the figures (PDF, SVG, JPG), their QC reports and source data, the renders, the videos and the graphical abstract |
| `manuscript/paper/build/supplement/` | Tables S1–S4 (Excel) and Data S1 |
| `flybrain/cache/`, `flybrain/data/morphology/skeletons/`, `manuscript/figures/assets/skeletons/`, `sokoban/artifacts/` | files derived from the MaleCNS v1.0 connectome: the simulator's connection matrix, neuron skeletons and display geometry |
| `archive/` | the catalog and one mapping file of the method's development, which the analysis reads |

## Notes

- Folders and files keep the study's development names: memory scenes `M0`, `S1` and `S2` (scenes 1 to 3), learners
  `head_only` (readout trained) and `joint` (brain and readout trained), planning maps `D<k>-T<j>` (tier k+1, map j+1),
  lesion cases such as `C01` (Kv4 loss), experiments `X1`–`X10`. The code release's README translates every name, and
  Tables S2 and S3 give the record of every test and run.
- Records are deposited as they were written. {{SUBSETS}}
- Models are PyTorch files (`*.pt`); recordings and behavior features are NumPy archives (`*.npz`); tables are CSV or
  Parquet; run records are JSON or JSON Lines.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0). The files derived from the MaleCNS v1.0 connectome (Berg et al.,
male-cns.janelia.org) are redistributed under its CC BY 4.0 license; cite the MaleCNS release when you use them.
