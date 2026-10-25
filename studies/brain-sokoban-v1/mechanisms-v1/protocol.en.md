# Frozen-checkpoint mechanism study protocol

The focus is task acquisition, short-term neural-state memory, and decision reading. All new jobs replay existing parameters with temporary interventions: no optimizer, gradient update, or new training. Original runs, checkpoints, and training STOP remain in place.

## Inputs and discovery/confirmation

- For 24 difficulty models, record all states along each reference solution at initial, middle, and final checkpoints under matched inputs. The middle checkpoint is the existing mid.pt, not necessarily half the training time.
- Add 1 ms cell-level records for D1/D3/D5/D7 T0 and T1 models. T0 is discovery and T1 independently trained-map confirmation. Other T0 data also enter the prespecified candidate ranking.
- Retain successful and failed trajectories for existing D1/D3/D5 brain-only, readout-only, and no-recovery controls, plus remote D7 readout-only.
- Deep memory analysis covers M0 readout-only, M0 joint, M0 shared visible/hidden readout, S1, and S2. Other existing controls remain in the overall registry and interpretation without retraining.
- Select memory groups on M0 trained histories, confirm on S1, and test contextual requirements on S2. Behavioral results are already known; newly recorded mechanism validation is separate and not described as first-time preregistration.

## Recording and behavioral acceptance

Cover 167106 simulated neurons. A shared index retains original bodyId, shared-type parameter group, input/observation membership, and annotations. Record spikes, g, v, adapt, and res in lossless float32 chunks. Soma annotation does not substitute for synaptic region. Reset resources to 1, voltage to model rest, and other cleared components to zero.

Actual readout features pool 14 ms current and window-end voltage into 256+256 dimensions. Millisecond simultaneous current/voltage is separately labeled instantaneous and must not be confused with the mixed-time policy input.

Memory delays are 0/20/40/80/160/320/640 ms, with 0/1/2/4/8 prefix windows. Original trained timings use delays 0/40/160 and prefixes 0/1/4. Dense cell trajectories at 160/640 ms and 4 prefixes cover every cue/context until the first push or 8 decision windows. Full behavior continues to the 512-action cap; repeated boards do not stop persistent-state models. Summaries explicitly retain the first-8-window recording scope.

For static tasks, load old brain/old head, new brain/old head, old brain/new head, and new brain/new head at matched states, retaining normalization, clipping, and bias. The solver supplies all solvable actions without shortest-route preference. Use a fixed competing action for decomposition and separately report full solutions. Final original models must reproduce their complete/partial behavior.

## Groups and interventions

Candidates comprise at least 8 noninput cells from existing parameter-type groups. Learning scores use absolute matched-state pre/post voltage changes on T0, averaged first across maps and then cells. Memory scores use standardized paired-cue g/v/adapt/res differences at the end of M0 trained histories. Select the highest 3 groups per line, breaking ties by type ID. Hash the lists; confirmation data do not select groups.

Matched groups have equal cell counts and exactly equal observed-membership counts. Deterministic nearest-neighbor matching uses joint standardized discovery-set baseline firing and log(1+total degree), without repeats within a group and excluding direct inputs and all candidate cells. Report achieved distances and means.

Component interventions cover voltage, current, adaptation, resources, delay buffers, and all neural state. Clearing, opposite-cue swaps, independently replayed same-cue swaps, sham operations, and restoration preserve clock, ring-buffer pointer, and RNG. Writing interventions occur at cue end, holding interventions at blank-delay midpoint, and reading interventions before the first decision. Same-cue donors replay to exactly the same time.

Group outgoing suppression uses 25/50/100%. Memory writing spans cue presentation, holding spans delay and common prefix, and reading spans the first decision. Static windows are 0–14/14–26/26–40 ms. Component/group histories use delays 40/160/640 × prefixes 0/4. The shared visible/hidden model supplies matched controls. Save full behavior and the first irreversible push separately. Sham operations must be neutral, and intervention removal/restoration must recover baseline.

## Mathematical analyses

1. Report learning curves by map and hardware, separating failures, interruptions, implementation errors, and stagnation. Difficulty metrics covary, so emphasize descriptions and paired effects.
2. Centered readout row space has at most three dimensions. Decompose brain, readout, and interaction terms and check reconstruction residuals; report cue-difference projection, margins, and pre/post principal angles. Null-space membership does not imply irrelevance to later dynamics.
3. Cell/type atlases cover spikes, state distributions, matched-state changes, and push/solvable-action selectivity. Count shared parameters by type, not as separate learning events when copied onto cells.
4. Shared PCA, regularized low-rank marginal decomposition (dPCA form: cue/context/time), and cross-time cue decoding fit only trained histories and evaluate held-out timings. Fix regularization at 1 without test-based selection; each of the three factor subspaces has at most 3 dimensions.
5. Fit and evaluate affine models of at most 8 dimensions using complete-history splits. Report held-out R², effective eigenvalues, stable-component decay, and transient amplification. Retain failed predictions; eigenvalues alone do not establish an attractor for input-driven trajectories.
6. Record signed one-/two-hop propagation capacity on the sparse directed graph. For M0 160 ms/4 windows, also observe actual sparse-operator input after resource and delay processing. Recorder-on/off outputs must match. Interpret structural paths and associations with independent behavioral interventions.

Map families are the statistical units. Use only seed 10; neurons, milliseconds, timing combinations, and checkpoints do not add independent training replicates. Present biological inference as model finding—candidate mechanism—alternative—model assumption—testable prediction.

## Scheduling, resources, and stopping

Use four remote GPUs with one process per card, the existing exclusion lock, an 11 GiB allocation cap, and a 13 GiB free-memory launch gate. Start formal recording only after preflight passes. Save job progress by model/stage/condition. Technical errors stop dispatch, preserving completed data and failures without repeated automatic restarts.

Cap new array data at 120 GiB; reserve shared storage before writing and retain at least 50 GiB free. Retrieve only closed job directories and verify every file hash. This directory has its own STOP; the original training STOP does not authorize renewed training. End the 30-minute monitor after frozen replay, confirmation, and local analysis finish.
