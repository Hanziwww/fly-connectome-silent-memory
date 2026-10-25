# Research focus and interpretation framework

Updated 2026-10-03 following the user's clarification of the research purpose. This framework organizes results and later analyses; the original registration, conditions, acceptance criteria, and raw outcomes remain retained.

## Research positioning

Use a fly-connectome-constrained neural dynamical model as a trainable, recordable, and intervenable experimental system to study acquisition, how persistent state retains and supports use of history, and how these processes vary with task requirements.

The original two questions remain: how training produces and changes task ability, and how neural activity contributes to representation, memory, and decision making. Real connectivity constrains the studied system and is an intervention target; ranking model performance is not the objective.

## Organizing results

1. **Acquisition and training mechanisms.** Explain the relation among board input, neural dynamics, 512-dimensional activity, and linear readout. Use frozen/updated conditions to locate parameter changes underlying new behavior, separating core learning from readout learning on a fixed brain.
2. **Persistent state and short-term memory.** With matched current observations and absent goal cues, test history dependence using clearing, swapping, sham, and restoration. Matched visible goals assess how intervention effects depend on historical-information requirements.
3. **Task requirements and ability scope.** Difficulty tiers, layouts, delays, and common prefixes describe acquisition, transfer, timing generalization, and failure types. Interpret segment stops, optimization difficulty, and capacity limits separately.
4. **Activity and decision mechanisms.** Connect recordings, decoding, and behavioral intervention while separating information present in state from information used by behavior. Report finer cell-group, time-window, and cross-layout mechanisms only when their corresponding data support them.

Parameter memory concerns retention/relearning after parameter change; dynamic memory concerns dependence of current model state on previous input. Success from freezing P05 and training a linear readout shows learned use of existing dynamics, not core-parameter learning in that round.

## Interpretive roles of controls

| Control | Question |
|---|---|
| Separately freeze/train brain and readout | Which plastic component produces new behavior, and what usable information do existing dynamics provide? |
| Reset, carry, clear, swap, and restore state | Which states carry history, and do they affect decisions? |
| Matched visible/hidden-goal interventions | How do effects vary with the task's need for historical information? |
| Fixed decaying trace | Which requirements can a simple temporal trace satisfy, bounding necessary mechanistic complexity? |
| Small GRU | Can another trainable state system acquire the same task, and how does it generalize? |
| P05, new readout, and neutral initialization | How do prior experience and readout initialization affect subsequent acquisition? |
| Real and constrained-rewired graphs | How sensitive is observed behavior to specific connection organization? |

Report success, partial success, and failure for every control. Simple-trace solutions show that simpler memory mechanisms can satisfy some task demands; internal state interventions retain their own paired interpretation. Structural controls assess dependence and sensitivity to connectivity without requiring the fly model to outperform alternatives.

## Organizing current conclusions

- **Task ability:** a connectome model with trainable readout can acquire specific tested Sokoban behaviors; new-layout retraining success is separate from zero-update transfer.
- **State memory:** in matched tested M0 histories, hidden-goal behavior depends on persistent neural state. Clearing, swapping, and restoration supply within-model intervention evidence. Visibility comparisons use the same histories and explicit baseline subsets.
- **Where learning occurs:** the supplementary brain core is frozen; positive results show that a linear readout can learn to use historical features of fixed dynamics.
- **Scope:** timing generalization, context changes, and cross-layout transfer bound the current training/readout method; partial successes and stop reasons jointly inform difficulty.
- **Mechanism levels:** decaying traces, GRU, initialization, and connectivity interventions distinguish generic temporal retention, readout learning, and specific connection organization.

These conclusions concern the implemented model, tested tasks, and seed 10. Histories, orientations, and cell counts are not independent maps or training replicates. Unfinished parameter-memory, deep-activity, and cell-group causal modules remain open questions at this snapshot.

## Reporting convention

Report new acquisition, state-memory, or decision-mechanism evidence first, followed by task scope and stopping reasons, then interpret the controls. Retain all numbers and counterexamples. Reframing does not change success criteria or model-selection rules.

Deliverables remain experiments, data, and interpretation; manuscript prose is separate. Reframing does not resume paused local training or restart completed remote batches.
