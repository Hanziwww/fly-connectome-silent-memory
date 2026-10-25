# Frozen-model mechanism supplement: locked protocol

Objective: distinguish cue-related continuous state from general computational state, cellular propagation from direct action readout, and elapsed time from subsequent-input interference. Use only existing models; do not train the brain, policy, GRU, or new tasks. Interpretive regressions and decoders fit saved data and do not generate behavior.

Use five existing checkpoints: M0 readout-only, M0 joint, M0 shared visible/hidden readout, S1, and S2. Retain seed 10, original precision and input mapping, and the 512-action cap. Independently load each model and check original trained-history behavior and recorder neutrality. Reproduce known partial solutions rather than classify them as technical failures. Dispatch the fifteen mechanism-recording jobs only after all five preflights pass. Retain the local training STOP.

## A. Selective interventions on cue differences

Fix delays at 40/160/640 ms × prefixes 0/4. Operate on voltage, current, adaptation, or their combination after writing, at the delay midpoint, or before the first read. Pair actual opposite-cue history states s and s' with the same context, visibility, and clock, and apply s + a(s'-s), with a=0.25/0.5/1. The common midpoint is a=0.5 and component swapping is a=1. Change only the listed continuous variables, preserving spikes, delay buffers, refractory state, clock, pointers, RNG, and parameters; this is not erasure of all historical information.

Controls use a prespecified random permutation of the difference with its original-direction projection removed, matching perturbation norms component by component. After nonnegative adaptation projection, match norms again and report actual norms, direction cosines, and boundary fractions. Mark a control invalid if it cannot match within 1%; retain its data but do not use it for matched-effect inference. Synthetic intermediate states need not lie on natural trajectories. Dose, visible goals, same-cue controls, and restoration bound interpretation. Paired donors are used only for registered interventions, never as baseline-policy inputs.

At each timing, save complete behavior, actual 512-dimensional policy features, logits, first irreversible push, action margins, and component audits. Retain the earlier M0 discovery and S1/S2 cross-layout roles. These are supplements after known behavioral results, not a new blinded-map confirmation.

## B. Separating propagation and direct readout

Use the previously locked observed-cell-containing MNad18/MNad27 group and its original matched group, without selecting by these effects. Conditions are none, propagation only, direct readout only, and both. Strengths are 25/50/100%, during the first decision window or all decision windows.

Propagation suppression uses the existing sparse-operator gate. Direct readout replaces only selected observed-cell contributions to the 14 ms current and 40 ms voltage pools, interpolating toward the checkpoint feature mean while keeping the divisor at 16. Then apply the original normalization, clipping, bias, and legal-action mask. Do not alter cell state or zero the entire 16-cell pool. Unobserved cells must have no direct-readout effect. Match windows, strengths, and control groups when testing joint channel effects.

## C. Duration-matched input interference and timing generalization

Fix cue-to-first-decision duration at 80/120/160/240/320/480/640 ms. Compare all-blank histories with 2/4/8 target-free neutral windows of 40 ms inserted within the same duration; their total cannot exceed that duration. Neutral inputs use canonical cue geometry, omit goals, and are identical across cues and contexts. Record stationary input and two prescribed legal closed walks separately, retaining the original physical board and player start. Reveal S2 context orientation only at formal decision, so early context revelation is not mislabeled neutral interference.

Save simultaneous g/v summaries each millisecond, actual policy mixed features at every segment end, cell-level v/g/adapt before the first decision, and complete behavior. Instantaneous g/v is distinct from the policy's 14/40 ms mixed features. With fixed regularization, fit interpretive decoders on 80/160/320 ms blank, stationary, and walk0 histories; walk1 is held-out input at seen times, while other durations are held-out times. Report counts, per-history correctness, and layouts without test-based model selection. Fit fixed-dimensional dynamics with input and phase drivers and compare held-out multistep predictions against persistence and autonomous models. Retain negative R² and decoding failures.

## Execution and acceptance

Store code, plan, model references, condition records, failures, and transfer receipts in a separate directory. Existing scientific source and data remain read-only. Use four cloud GPUs with one process per card, the shared exclusion lock, an 11 GiB allocation cap, and a 13 GiB free-memory launch gate. The 20 GiB supplement budget is part of the original 120 GiB mechanism budget; retain at least 50 GiB free locally. Stop dispatch on anomalies, preserve evidence, and do not retry automatically. Audit unchanged parameters for every job. Acceptance requires sham/restoration checks and per-history reproduction of the reference. Report processes, condition progress, anomalies, and retrieval every 30 minutes. End monitoring after complete file verification, analysis, and interpretation.
