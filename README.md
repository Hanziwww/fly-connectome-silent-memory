# FlyLab: joint brain and linear readout

The formal method is **a trainable full-connectome brain plus a trainable linear readout**, optimized together with a reverse suffix curriculum and recovery loss. The [method registry](method.json) selects **P05** as the current model.

| Retained result | Board | Boxes | Autonomous solution | Training |
|---|---|---:|---:|---|
| [P05](runs/linear-joint/20261002-p05-01/report.md) | 10 x 10 | 2 | 34 moves | 384 joint updates, 771.891 s |
| [P03](runs/linear-joint/20261002-p03-01/report.md) | 8 x 8 | 2 | 22 moves | 216 joint updates, 410.156 s |

Both results were independently reloaded and verified. Previous-brain, previous-readout and cleared-plasticity substitutions failed for each saved final pairing. The goal is any autonomous solution; shortest paths and retention of earlier tasks are not required. Evidence covers selected tasks in one inherited seed-10 lineage.

## Open the viewer

Double-click **Start Fly Sokoban.bat**, or run:

```powershell
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py serve --open --port 8774
```

Open http://127.0.0.1:8774/. P05 is selected by default. P03, all three successful B00 repair checkpoints and the original four-level pilot remain available. The viewer includes anatomical geometry, simulated neural activity, model substitutions and playback intervals of 0.5, 0.8 and 1.6 seconds. Playback performs inference only.

## Documentation and retained evidence

- [Formal method](docs/method.md): architecture, loss, optimization, acceptance rules and continuation commands.
- [Experiment history](docs/experiment-history.md): the full progression of successful, unsuccessful and interrupted attempts.
- [Complete archive index](archive/INDEX.md): every retained run, checkpoint collection, preflight, benchmark and viewer record.
- [Data sources](docs/data-sources.md): MaleCNS tables, anatomical meshes, skeletons and synaptic contacts.
- [Workspace and recovery](docs/workspace.md): directory layout, snapshot boundaries and original-to-current paths.

`runs/linear-joint/` contains the current method's original result bundles. `archive/experiments/` preserves earlier attempts. `archive/snapshots/` contains the project before organization. `sokoban/` holds the maintained application and its required research helpers; `flybrain/` retains the third-party simulator, original data and local Python environment. Current logs are in `runtime/` and private connection notes are in `local/`.

## Commands

The registered training, memory and decision-mechanism study lives in
[studies/brain-sokoban-v1](studies/brain-sokoban-v1/README.md). It has a separate
task lock, run registry, checkpoints and scientific activity records. Use
`workflow.py study status` for its current state and `workflow.py study run` for
the registered serial experiment pipeline. Its results do not replace the formal
P03/P05 registry automatically.

```powershell
# CPU method and viewer checks
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py check
# Verify neural actions against saved results through the running viewer
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py verify
# Options for a newly requested continuation; requires --task and an unused --out
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py train --help
```

Use the existing local environment. A similarly named PyPI package is not a substitute for the retained simulator. GPU training workers run serially with the viewer stopped. Historical STOP markers and original evidence files remain preserved; the 2026-10-02 organization started no training.
