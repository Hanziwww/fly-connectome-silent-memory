# Brain state locks silent memories in a trained fly connectome: code

Code of the article *Brain state locks silent memories in a trained fly connectome* by Han Wang et al. (prepared for submission).

A spiking model of the whole male *Drosophila* central nervous system, 167,106 neurons wired as in the MaleCNS
connectome, is trained to plan moves in a box-pushing task and to remember a visual cue. Spikes write the cue into
neuronal adaptation, which holds it silently until the next input reads it out, and the learned readout recalls the trace
only in the brain state of learning. This repository holds the code that trains and tests the model, the declared protocol
of every experiment, the analysis that turns the records into the paper's tables, the code of every figure, video and
supplemental table, and the LaTeX source of the paper.

- Core records: [Zenodo 10.5281/zenodo.23629827](https://doi.org/10.5281/zenodo.23629827)
- Extended records: [Zenodo 10.5281/zenodo.23629835](https://doi.org/10.5281/zenodo.23629835)
- Code: [GitHub release v1.0.0](https://github.com/Hanziwww/fly-connectome-silent-memory/releases/tag/v1.0.0)
- Data files are restricted during peer review. Contact Chunjie Tian (tianchunjie@gdmu.edu.cn) for reviewer access.
- Version 1.0.0; based on source commit `ad46d06746bd` (2026-10-24), with author and publication metadata updated 2026-10-25.

## Contents

| Path | Contents |
|---|---|
| `sokoban/` | the planning and memory tasks, the model around the simulator (`lib/brain_model.py`), training, evaluation and the interactive viewer |
| `studies/brain-sokoban-v1/` | the registered study of planning and memory: protocol, task lock, code with its snapshots, and the mechanism experiments |
| `studies/brain-supplement-v2/` | the supplementary experiments (replicates, new scenes, jittered timings, ensembles, controls) with their test plans and amendments |
| `studies/brain-pathology-v1/` | the screen of fly pathologies, the lesions, the join with single-cell atlases and the brain-state experiments, with declarations and the literature record |
| `studykit/` | the run registry, test plans and statistics that the studies share |
| `research.py`, `workflow.py`, `tools/remote/` | the study runner and the queue that ran training on remote GPUs (key-based SSH only) |
| `manuscript/analysis/` | the analysis, from the saved records to `manuscript/results` (CPU only, behind a read-only guard) |
| `manuscript/figures/` | the renderer (`render/`), the figure builds (`build/v4/`), shared styles and checks (`lib/`) and the videos |
| `manuscript/writing/` | the number ledger (every number in the text bound to a result table), Tables S1–S4, Data S1 and the text checks |
| `manuscript/paper/` | the LaTeX source of the paper, Document S1 and the cover letter |
| `manuscript/package/` | the builder of this release and of the deposit, the deposit's checker and the input tracer |
| `docs/`, `method.json` | the formal method, the history of the experiments and the data sources |

The top-level `README.md` is the original workspace's README. The analysis reads it (it checks the numbers quoted there),
so it is kept unchanged; this file is the guide to the release.

## Requirements

- Python 3.12 (the analysis ran on 3.12.13). The analysis, figures and writing scripts use the packages of
  `requirements-analysis.txt` (recorded from the Windows environment, with PyTorch 2.14.0 for CUDA 13.2; the analysis
  itself runs on CPU). The Linux GPU hosts used PyTorch 2.12.1 for CUDA 13.0 with the packages of
  `MIGRATION/requirements-linux.txt` (`MIGRATION/setup_linux.py`; `studies/brain-supplement-v2/g0-host-x4.json` records
  one host).
- FlyBrain 0.1.0 (MIT), the whole-CNS simulator, at `./flybrain`:
  `git clone https://github.com/HEREISCB/flybrain flybrain`, then `git -C flybrain checkout c7878b9ee273564186acf8cbdef96289335efa5e`, and its
  own instructions to fetch the MaleCNS v1.0 data into `flybrain/data`. The code imports it from `./flybrain`.
- XeLaTeX with latexmk and the Arial font for the paper (`manuscript/paper/README.md`), and FFmpeg for the videos.

## Data

Unpack the archives of the Zenodo records into this folder: their paths are relative to it. The first record holds every
file that the analysis, the renders, the figure builds and the writing scripts read, every run's models, configurations,
logs and results, and the result tables, figures and videos; the second holds the verification replays, the behavior
features of every training update and the dense recordings that no analysis reads.

```
python manuscript/package/verify_deposit.py <downloaded record folder> --into .
```

checks every member against the record's `MANIFEST.csv` while it unpacks, and restores the files that the deposit stores
once for several paths.

## Reproducing the paper

Run from this folder, with `flybrain` on the module path: `PYTHONPATH=flybrain` (PowerShell: `$env:PYTHONPATH = "flybrain"`).
On Linux, `python3.12 MIGRATION/setup_linux.py` builds the environment once, and `python3 MIGRATION/launch.py --script <script>`
runs a script in it.

1. **Analysis** (CPU): `python -B -X utf8 manuscript/analysis/run_all.py` writes `manuscript/results` from the records;
   `--determinism` runs it twice from a clean state and compares the passes. `manuscript/FREEZE_MANIFEST.csv` holds the
   SHA-256 of every published output.
2. **Figures, videos and the graphical abstract**: `python -B -X utf8 manuscript/package/trace_inputs.py` rebuilds every
   render, video and figure in fresh processes, checks each output against the published file byte for byte, and lists the
   files they read in `manuscript/package/traced_inputs.csv`. One script at a time: `manuscript/figures/render/v4_*.py`
   (in the order of `trace_inputs.py`), then `manuscript/figures/build/v4/*.py`.
3. **Numbers and supplemental tables**: `manuscript/writing/build_ledger.py`, `build_macros.py`, `build_table_s1.py`,
   `build_tables_s2_s4.py` and `build_data_s1.py`.
4. **The paper**: `python -B -X utf8 manuscript/paper/build.py all`, and `manuscript/writing/check_tex.py` for its checks.
5. **Training** (GPUs): each study's `README.md` and `protocol.json`, and `python research.py --study <study> status`.
   Every experiment was declared before it ran (protocols, test plans and hash-chained amendments), and a new GPU host had
   to replay a reference run bit for bit before use.

## Names in the code and in the paper

The code and the records use the study's development names; the paper uses descriptive names.

| In the code | In the paper |
|---|---|
| `M0`, `S1`, `S2` | memory scenes 1, 2 and 3 |
| `M1`, `M2`; `S1-mirror`, `S1-minus45`, `S2-mirror`, `S2-minus48` | the new scenes: scene 1 with one or two more floor cells; scenes 2 and 3 mirrored or with one floor cell removed |
| `head_only`, `joint`, `brain_only` | readout trained; brain and readout trained; brain trained |
| `D<k>-T<j>`, `D<k>-H` | planning map j+1 of tier k+1; the held-out map of tier k+1 |
| `P03`, `P05` | warm-up maps 1 and 2 (the models of the formal method, `method.json`) |
| `C01`, `C04`, `C13`, `C16` | Kv4 loss (Aβ42); Kenyon-cell loss (late Aβ42); weaker antennal-lobe inhibition (fragile X); less excitable Kenyon cells (tau) |
| `C02`–`C24` | the other candidates of the pathology screen (Table S1) |
| regime A, B, C | the lesion after learning; learning under the lesion; relearning after the lesion |
| `X1`–`X10` | the supplementary experiments: replicates (X1), new scenes (X2), jittered timings (X3), state shifts against scrambles (X4), opposite-cue swaps (X5), ensemble silencing (X6) and synapse removal (X7), planning from a neutral brain (X8), checkpoint pairings, stage replays and controls (X9), planning-map replicates (X10) |
| `F11`–`F15` | the declared test families; Table S2 gives each its name |
| `G0`; `G4` | the bit-exact replay check of a GPU host; the linear response of the decision to the trace |
| `DY1`; `DY2` | kicks and threshold shifts; training across threshold states |
| `SC`; `O14` | the join with single-cell atlases; the connection-rule variants |

Declarations, amendments and run records are kept as they were written, because they are hash-chained. AI tools were used solely for language review of the manuscript. The authors are responsible for the final review and take full responsibility for its content.

## License

The code of this study is released under the MIT License (`LICENSE`). Manuscript and supplemental text are licensed under CC BY-NC-ND 4.0 (`manuscript/paper/LICENSE.txt`). Third-party code and data keep their own licenses
(`THIRD_PARTY_NOTICES.md`).

## Contact

Corresponding authors: Huafeng Liu (liuhf@gdmu.edu.cn); Junfeng Hao (ygzhjf85@gmail.com); Chunjie Tian (tianchunjie@gdmu.edu.cn). Lead contact: Chunjie Tian.
