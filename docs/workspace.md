# Workspace layout and archive

```text
brain/
  README.md                   Current entrypoint and result summary
  method.json                 Formal method and selected checkpoint
  workflow.py                 serve / check / verify / train / analyze
  Start Fly Sokoban.bat        Windows viewer launcher
  docs/                       Method, history, dataset and workspace records
  sokoban/                    Active viewer, formal training and shared dependencies
    lib/                      Brain adapter, game and shared runtime helpers
    artifacts/                Fixed mapping and anatomy/morphology display data
    vendor/                   Bundled viewer dependencies and licenses
  runs/linear-joint/          Original P03 and P05 formal result bundles
  archive/
    INDEX.md                  Complete retained collection index
    experiment-catalog.json   Checkpoint and result-receipt catalog
    experiments/              Earlier attempts, diagnostics, research and UI records
    legacy-code/              Historical-only runners, analyses and tests
    snapshots/                Full pre-organization project snapshot
    maintenance/20261002/     Relocation manifest, scripts and verification receipts
  tools/                      Experiment-catalog builder
  runtime/                    Current logs, process record and worker lock
  local/                      Private machine/connection notes
  flybrain/                   Original simulator checkout, data, cache and environment
```

## Preservation

The [pre-organization ZIP](../archive/snapshots/20261002-before-organization.zip) contains the original README, workflow, launcher, download receipt and complete `sokoban/` tree, including source, checkpoints, logs and display assets. Each member's original path, destination, size and SHA-256 are recorded in the [relocation manifest](../archive/maintenance/20261002/relocation-manifest.json). The ZIP was checked for CRC errors before relocation.

The existing `flybrain/` tree and project `.git/` remain in place. The upstream simulator checkout is `c7878b9ee273564186acf8cbdef96289335efa5e`; its own Git metadata and local installation receipts remain available. Large raw datasets, the compiled connectome cache and Python environment were not duplicated into the snapshot. The private `ssh.txt` was moved to `local/ssh.txt` and excluded from the shareable snapshot. A complete offline backup therefore needs the workspace's retained `flybrain/` tree as well as the archive; the ZIP is a project snapshot, not a standalone environment installer.

Historical experiment files are byte-preserved. Source changes for active path resolution are backed by the original source snapshot. The final preservation receipt lists each modified live source and confirms every original snapshot member and every relocated historical artifact. Original source manifests stay unchanged; `workspace.py` verifies P03/P05's historical code against the snapshot and checks a new run's own manifest against the current code.

Earlier method names remain in shared modules because the formal method imports their task, evaluation and optimization helpers. Only code outside the active dependency closure moved to `archive/legacy-code/`. Archived runners are records rather than current entrypoints. Use `workflow.py` for the maintained method.

## Locate or restore an old file

Look up `original_path` in `relocation-manifest.json` and open its `current_path`. To inspect the complete former layout, extract `20261002-before-organization.zip` into a separate directory; do not overwrite active results. Reproducing an old runner also requires the retained simulator/data/environment and checking its historical absolute paths. Local Python virtual environments are machine-specific and should not be treated as portable bundles.

The snapshot and records cover files present on 2026-10-02. They cannot recreate candidates never saved or files removed before this archive. All existing STOP and pause records remain preserved; indexing and playback do not resume training.

## Normal use

Open the viewer through the root launcher. Current logs and HTTP verification go to `runtime/viewer/`. Keep new formal training results in a fresh `runs/linear-joint/<run-id>/` directory. Refresh the index after an explicitly requested experiment with:

```powershell
.\flybrain\.venv\Scripts\python.exe -B -X utf8 tools/catalog_experiments.py
```

Raw data, the local environment, snapshots and model binaries remain on disk. `.gitignore` separates these from ordinary source control and excludes private connection notes and mutable runtime files.

## Storage cleanup on 2026-10-03

The user-requested cleanup removed 2,664 regenerable bytecode/cache and upload-staging files. It also consolidated 28,053 byte-identical copies in closed experiment results with NTFS hard links, recovering about 27.06 GiB of measured disk space in total. Every experimental path and its original content hash remains valid; no scientific recording, checkpoint or historical archive was discarded. The runtime connectome matrix, neuron table and Python environment remain available.

Treat these completed results as immutable. Make an independent copy before editing any linked result in the future. Each shared file has at most 500 paths; normal readers and the original verification manifests require no changes. File-manager totals that count every path can still show the original logical size.

See the [cleanup report](../runtime/cleanup-20261003/report.en.md), [manifest](../runtime/cleanup-20261003/plan.json) and [receipt](../runtime/cleanup-20261003/receipt.json). Automatic approval blocked the optional empty-directory removal; the empty cache directories remain, while their disposable files have been removed.
