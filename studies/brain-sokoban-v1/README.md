# Brain / Sokoban experiment project

This directory contains the registered experiment implementation, task certificates,
isolated model runs, scientific activity records, and generated analyses. The formal
P03/P05 models and the historical experiment archive are input evidence and are not
modified. The project uses training seed 10 throughout.

## Run

From the repository root in PowerShell:

```powershell
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py study prepare
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py study test
.\flybrain\.venv\Scripts\python.exe -B -X utf8 -u workflow.py study run
```

`run` executes calibration, numerical/state validation, the 89 registered static
training/evaluation conditions, post-selection generalization and parameter-memory
tests, 15 dynamic-memory conditions, activity/intervention recording, and analysis.
The eight-stage continuous curriculum reuses the identical first difficulty run.
GPU workers run serially, use an 11 GiB allocator limit, and require 13 GiB free at
startup. A fresh process verifies each trained checkpoint using its saved backend.

Individual stages are available:

```powershell
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py study benchmark
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py study validate
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py study train --module difficulty
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py study train --run-id difficulty-D1-T0
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py study evaluate
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py study train --module dynamic_memory
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py study evaluate --module activity
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py study analyze
.\flybrain\.venv\Scripts\python.exe -B -X utf8 workflow.py study status
```

Create `studies/brain-sokoban-v1/STOP` to stop the study at a safe checkpoint.
Remove that file only when an explicit restart is intended, then use `study run
--resume` (or `study train --run-id ID --resume`). Resumption requires the recorded
code version and restores current parameters, Adam, RNG states, lesson history,
the retained champion and elapsed phase time. Completed, stalled, numerically
failed and implementation-failed attempts are preserved. A method/code repair
requires a new condition/run ID, not an overwrite of failed evidence.

## Files and evidence

| Path | Contents |
| --- | --- |
| `protocol.json` | Seed, stopping, memory-history splits and acceptance rules |
| `schedule.json` | Static experiment conditions, parents and blocked run order |
| `tasks/levels.json` | 24 training, 8 heldout and 2 calibration maps |
| `tasks/certificates.json` | Exact reverse search and independent forward certificates |
| `tasks/memory.json` | Three balanced cue tasks and irreversible-push certificates |
| `tasks/variants.json` | Symmetries and distinct starts with common family IDs |
| `tasks/matched-bank.json` | Matched-state activity bank and discovery/confirmation roles |
| `tasks/lock.json` | Locked task files and counts |
| `registration-amendments.json` | Geometry-only correction before main-task evaluation |
| `tasks/versions/v1/` | Original lock and maps retained before that correction |
| `backend.json`, `benchmark.json` | Backend selection and all calibration outcomes |
| `speed-probe*.json` | Sparse accumulation reproducibility/performance diagnostics |
| `validation*.json` | Full-history gradient, batching, state restoration and resume checks |
| `runs/ID/` | Config, source identity, checkpoints, behavior, exposure and learning histories |
| `code-snapshots/` | Actual code and source manifest attached to each run |
| `activity/` | Float32 full-neuron statistics, type summaries, selected full traces and lesions |
| `analysis/` | CSVs, figures and `analysis-report.en.md` |
| `supervisor.json`, `events.jsonl` | Current worker and completed-run events |

Behavioral decisions include physical state, legal mask, logits and the 512 brain
features (400 board inputs only for the explicit MLP baseline). Behavior
assessments store lossless float32 feature tables in
`behavior-features/`; every decision retains a `feature_row` reference. The helper
`behavior.restore_behavior` reconstructs complete records. GRU controls save their
28-dimensional hidden features. Short probes have zero common-prefix windows;
cross-step probes use 1, 2, 4 and 8 windows, with 2 and 8 held out from training.
This separation is recorded in `memory-protocol-clarification.json`.
Intermediate and final checkpoints preserve both brain and head. An observed successful route is
accepted even if it disagrees with a demonstration or is longer. Initial success
is labeled `already_solved`, separate from acquired learning.

The default run phases are 20, 40 and 80 additional minutes. An extension requires
verified suffix progress or at least 0.02 improvement at a fixed entry while
preserving necessary behavior. Dynamic probes use the minimum paired choice
margin and preserve previously solved histories. No project-wide compute limit
has been imposed. Stops, OOM, code errors, numerical errors, stagnation and phase
limits have distinct outcomes.

## Computation and controls

The base architecture is the unchanged full MaleCNS LIF model with the existing
type-shared parameter bounds, adaptation, synaptic dynamics and sensory/readout
maps. The adapter records resolved dynamics, including type-specific coefficients.
The affine head uses only the existing 512 activity features.

Calibration compares FP64, FP32 (batch 16 and 8), and a mixed candidate with FP64
forward propagation and FP32 gradient propagation. Same-config two-pass feature
recomputation is checked: absolute tolerance 1e-5 for FP64/mixed and 1e-3 for FP32;
every actual discrepancy is saved. A deterministic segmented-FP32 prototype is
retained as a speed diagnostic; its measured runtime excluded it from full
calibration. Precision variants need not share logits or routes. Eligibility
requires both calibration tasks to learn and independently reload to solve.

The completed calibration selected mixed precision: 494.24 seconds total training
versus 912.42 seconds for FP64 (1.85 times faster on these two tasks). Both passed
independent reload verification. Pure FP32 candidates failed same-config feature
recomputation on the second task; their incomplete timings are not successful
training speedups. All failed attempts and their diagnostic artifacts are retained.

Frozen-core conditions reuse features only under an unchanged core hash and reset
state. Persistent-state experiments have no board-only feature cache or repeated-
board termination. The complete cue/prefix is differentiated, with checkpoint
recomputation across time. Variable histories and their free rollouts are batched
independently; padding is erased at each episode's actual start.

Neutral structure controls share zero core modulation, a freshly seeded linear
head, and the same normalization rule fitted to calibration boards. Degree/sign
rewiring records achieved endpoint changes and preserved statistics. It does not
preserve target weighted strengths or spatial locality. Fixed-topology shuffling
preserves global sign-specific weights. The raw-board MLP has 37,264 trainable
parameters versus 37,362 for the main model; the independent GRU has 28 hidden units
and one recurrent update per 20 ms, the smallest delay interval in the probe bank.
The fly model retains its original 1-ms internal simulation step. The GRU is a
sequence-learning control, not a biophysical timing-matched circuit model.

## Interpretation

The independent units are maps and map families. Neural cells, time points,
decisions and checkpoints are not training replicates. Heldout layouts and memory
histories never choose checkpoints. Continuous learning and parameter-memory runs
record their ancestry explicitly. Forgetting requires that the original task was
actually acquired. Group discovery uses T0 families; confirmation uses separate T1
families. Causal lesions attenuate outgoing transmission during predeclared time
windows and include activity/degree-matched cells and restoration.

Activity storage is float32 with lossless compression, separate from viewer data.
Neuron grouping uses available annotations, not mesh colors. PCA is fitted once
to the matched phase matrix; supervised decoders and time-generalization models
fit discovery families only. Direct-board and nuisance baselines accompany brain
decoding. Model-memory and intervention findings describe this computational
model; animal biological memory is not directly measured.

## Procedural and software references

The experimental-design skill informed blocking, paired units and separation of
discovery from confirmation. Source: Kassis T, Agarwal V, He Y, Patel D, Brueckner AM
(2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
The full-history implementation follows [PyTorch checkpoint semantics](https://docs.pytorch.org/docs/2.14/checkpoint.html).
Connectome provenance is retained in the repository's `docs/data-sources.md`.
