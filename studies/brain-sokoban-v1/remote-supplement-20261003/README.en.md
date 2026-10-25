# Remote memory supplement

On 2026-10-03 the user authorized supplementary experiments on four remote RTX 5090 GPUs. The local STOP remained unchanged.

## Locked questions and conditions

1. S1-ring uses a branching/ring structure different from M0 with two hidden goals. S2-context uses another branch structure, revealing the original or 180-degree orientation only after cue offset so the same cue maps to different push directions. Train a frozen P05 brain plus linear readout and a GRU on each layout: four primary training conditions.
2. M0-visibility starts from the successful M0 model, freezes the brain, and trains only the readout jointly on visible and hidden goals. Use one accepted checkpoint for paired visibility interventions. If it does not learn all conditions, retain every result and separate baseline failure from intervention effects.
3. For the original successful M0 model and shared-visibility model, run sham, same-cue/different-history swaps, opposite-cue swaps, neural-state clearing, voltage/current clearing, restoration, and per-window reset controls. Also perform registered interventions at cue offset and delay midpoint. Neural clearing preserves clock, RNG, and buffer pointer, unlike the old all-state intervention.
4. Evaluate the original successful M0 checkpoint without updates over 0–640 ms in 20 ms increments and five prefix lengths, plus zero-update S1/S2 transfer. Do not use these tests for model selection.
5. On each new layout, compare P05 core+new head, neutral real graph+new head, neutral rewired graph+new head, and fixed decaying trace+new head. Each trains only the readout with shared random initialization. Determine branch-specific fixed normalization only from original C0/C1 calibration data. This adds eight training conditions.

There are 13 training conditions; other queue entries are preflights or fixed-model evaluations. Retain learning failures, stagnation, and technical failures.

## Task and inference scope

Maps were selected using geometry and solving certificates before observing model performance. Every wrong first-push direction is certified to make the goal unreachable; paired current observations and legal actions match. S2 orientations share a map family and are not independent maps. Revealing orientation after the cue is an explicit intervention, not an ordinary player action on a static board. It tests conditional cue use; success alone does not establish abstract goal understanding.

New-task prefixes use legal out-and-back walking; one window uses a stationary observation. M0 rechecks retain the original prefix definition to reproduce the old checkpoint action by action.

New tasks require a full solution within 512 actions and allow any route. Choice correctness refers to the first irreversible push. The first-move demonstration margin is used only for segment extension, not shortest-solution acceptance.

Fix training seed 10. Learning uses delays 0/40/160 ms and prefixes 0/1/4 windows; other prespecified timings are test-only. Reload every final model in an independent process under its recorded configuration.

## Execution

The original and supplementary controllers share GPU exclusion locks; supplementary jobs do not preempt cards still used by the original batch. Use one process per GPU, an 11 GiB allocation cap, and at least 13 GiB free before launch. Retain 20/40/80-minute segments and existing progress criteria, saving Adam, RNG, parameters, best parameters, and complete registered-history settings.

New code resides here; original scientific source, task lock, controller, and results remain unchanged during these experiments. Check code/task versions before running. Do not automatically repeat technical errors.

Local monitor entry points are `flybrain\.venv\Scripts\python.exe -B -X utf8 tools\remote_supplement.py status` and `sync`. They connect to the remote machine and retrieve results without local training.

## Execution records and analysis

`compatibility-repair.json` records a verification-metadata repair: `verify_compat.py` removes only the duplicate intervention field before the original interface writes it. Training, forward computation, and acceptance rules stay unchanged. `controller_v2.py` adopted existing workers without restarting training; the original controller and adoption record remain.

`execution-order-amendment.json` records a later scheduling adjustment. `controller_v3.py` allows the eight registered independent structure/initialization conditions to use free GPUs after preflight. They require no primary-training output and no longer wait for it to finish. No condition was added; original plan.json, task, and training-source hashes remained unchanged during this adjustment.

After synchronization, run `flybrain\.venv\Scripts\python.exe -B -X utf8 studies\brain-sokoban-v1\remote-supplement-20261003\analyze.py` to produce the analysis, grouped JSON, learning curves, acceptance comparisons, and dense-delay/decoding figures. It reads saved data without loading a behavioral model. Source hashes are recorded in analysis-inputs.json. Interpretive decoders use fixed regularization and training-time splits and neither generate behavior nor select checkpoints.
