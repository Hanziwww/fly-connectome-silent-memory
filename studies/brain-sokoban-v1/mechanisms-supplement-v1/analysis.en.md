# Frozen-model supplementary mechanism analysis

Complete jobs included in analysis: 20/20.

Effects are paired by model, history, context, and visibility. History combinations are not independent training or map replicates.
This supplement follows known earlier results and does not claim real-connectome superiority.

## remote-dynamic-M0-head_only

- Hidden: pre-read continuous-state cue midpoint intervention, full solutions 7/12, baseline 9/12; lost 4, gained 2; valid norm controls 12/12.
- Hidden: pre-read continuous-state scrambled midpoint intervention, full solutions 9/12, baseline 9/12; lost 0, gained 0; valid norm controls 12/12.
- Hidden: first-decision candidate group, both full solutions 9/12, baseline 9/12.
- Hidden: first-decision candidate group, network full solutions 9/12, baseline 9/12.
- Hidden: first-decision candidate group, readout full solutions 9/12, baseline 9/12.
- Hidden: matched-duration inputs, full solutions 57/104; see matched-time-behavior.csv for duration/input strata; not a pooled independent-map success rate.

## remote-dynamic-M0-joint

- Hidden: pre-read continuous-state cue midpoint intervention, full solutions 7/12, baseline 8/12; lost 3, gained 2; valid norm controls 12/12.
- Hidden: pre-read continuous-state scrambled midpoint intervention, full solutions 8/12, baseline 8/12; lost 1, gained 1; valid norm controls 12/12.
- Hidden: first-decision candidate group, both full solutions 8/12, baseline 8/12.
- Hidden: first-decision candidate group, network full solutions 8/12, baseline 8/12.
- Hidden: first-decision candidate group, readout full solutions 8/12, baseline 8/12.
- Hidden: matched-duration inputs, full solutions 41/104; see matched-time-behavior.csv for duration/input strata; not a pooled independent-map success rate.

## supp-M0-visibility

- Hidden: pre-read continuous-state cue midpoint intervention, full solutions 7/12, baseline 9/12; lost 4, gained 2; valid norm controls 12/12.
- Hidden: pre-read continuous-state scrambled midpoint intervention, full solutions 10/12, baseline 9/12; lost 0, gained 1; valid norm controls 12/12.
- Hidden: first-decision candidate group, both full solutions 9/12, baseline 9/12.
- Hidden: first-decision candidate group, network full solutions 9/12, baseline 9/12.
- Hidden: first-decision candidate group, readout full solutions 9/12, baseline 9/12.
- Hidden: matched-duration inputs, full solutions 58/104; see matched-time-behavior.csv for duration/input strata; not a pooled independent-map success rate.
- Visible: pre-read continuous-state cue midpoint intervention, full solutions 8/12, baseline 12/12; lost 4, gained 0; valid norm controls 12/12.
- Visible: pre-read continuous-state scrambled midpoint intervention, full solutions 8/12, baseline 12/12; lost 4, gained 0; valid norm controls 12/12.
- Visible: first-decision candidate group, both full solutions 12/12, baseline 12/12.
- Visible: first-decision candidate group, network full solutions 12/12, baseline 12/12.
- Visible: first-decision candidate group, readout full solutions 12/12, baseline 12/12.
- Visible: matched-duration inputs, full solutions 52/104; see matched-time-behavior.csv for duration/input strata; not a pooled independent-map success rate.

## supp-S1-ring-brain

- Hidden: pre-read continuous-state cue midpoint intervention, full solutions 7/12, baseline 11/12; lost 4, gained 0; valid norm controls 12/12.
- Hidden: pre-read continuous-state scrambled midpoint intervention, full solutions 5/12, baseline 11/12; lost 6, gained 0; valid norm controls 12/12.
- Hidden: first-decision candidate group, both full solutions 11/12, baseline 11/12.
- Hidden: first-decision candidate group, network full solutions 11/12, baseline 11/12.
- Hidden: first-decision candidate group, readout full solutions 11/12, baseline 11/12.
- Hidden: matched-duration inputs, full solutions 54/104; see matched-time-behavior.csv for duration/input strata; not a pooled independent-map success rate.

## supp-S2-context-brain

- Hidden: pre-read continuous-state cue midpoint intervention, full solutions 11/24, baseline 15/24; lost 6, gained 2; valid norm controls 24/24.
- Hidden: pre-read continuous-state scrambled midpoint intervention, full solutions 12/24, baseline 15/24; lost 4, gained 1; valid norm controls 24/24.
- Hidden: first-decision candidate group, both full solutions 15/24, baseline 15/24.
- Hidden: first-decision candidate group, network full solutions 15/24, baseline 15/24.
- Hidden: first-decision candidate group, readout full solutions 15/24, baseline 15/24.
- Hidden: matched-duration inputs, full solutions 112/208; see matched-time-behavior.csv for duration/input strata; not a pooled independent-map success rate.

## Interpretive decoding and dynamics

- remote-dynamic-M0-head_only, visible=False, held_input: cue decoding 8/12; this decoder is not part of the policy.
- remote-dynamic-M0-head_only, visible=False, held_time: cue decoding 26/62; this decoder is not part of the policy.
- remote-dynamic-M0-joint, visible=False, held_input: cue decoding 8/12; this decoder is not part of the policy.
- remote-dynamic-M0-joint, visible=False, held_time: cue decoding 35/62; this decoder is not part of the policy.
- supp-M0-visibility, visible=False, held_input: cue decoding 8/12; this decoder is not part of the policy.
- supp-M0-visibility, visible=False, held_time: cue decoding 26/62; this decoder is not part of the policy.
- supp-M0-visibility, visible=True, held_input: cue decoding 10/12; this decoder is not part of the policy.
- supp-M0-visibility, visible=True, held_time: cue decoding 43/62; this decoder is not part of the policy.
- supp-S1-ring-brain, visible=False, held_input: cue decoding 9/12; this decoder is not part of the policy.
- supp-S1-ring-brain, visible=False, held_time: cue decoding 35/62; this decoder is not part of the policy.
- supp-S2-context-brain, visible=False, held_input: cue decoding 13/24; this decoder is not part of the policy.
- supp-S2-context-brain, visible=False, held_time: cue decoding 65/124; this decoder is not part of the policy.
- remote-dynamic-M0-head_only, held_input, 40 ms conditional prediction: persistence R²=-1.181, affine R²=-0.255, input_phase R²=0.161.
- remote-dynamic-M0-head_only, held_time, 40 ms conditional prediction: persistence R²=-0.836, affine R²=-0.082, input_phase R²=0.375.
- remote-dynamic-M0-joint, held_input, 40 ms conditional prediction: persistence R²=-1.178, affine R²=-0.022, input_phase R²=0.353.
- remote-dynamic-M0-joint, held_time, 40 ms conditional prediction: persistence R²=-0.860, affine R²=0.012, input_phase R²=0.437.
- supp-M0-visibility, held_input, 40 ms conditional prediction: persistence R²=-1.181, affine R²=-0.255, input_phase R²=0.161.
- supp-M0-visibility, held_time, 40 ms conditional prediction: persistence R²=-0.836, affine R²=-0.082, input_phase R²=0.375.
- supp-S1-ring-brain, held_input, 40 ms conditional prediction: persistence R²=-1.046, affine R²=-0.090, input_phase R²=0.331.
- supp-S1-ring-brain, held_time, 40 ms conditional prediction: persistence R²=-0.822, affine R²=-0.001, input_phase R²=0.429.
- supp-S2-context-brain, held_input, 40 ms conditional prediction: persistence R²=-1.240, affine R²=-0.070, input_phase R²=0.130.
- supp-S2-context-brain, held_time, 40 ms conditional prediction: persistence R²=-0.864, affine R²=-0.014, input_phase R²=0.269.

## Interpretation limits

Midpoint interventions remove only the operated continuous-variable differences between these paired histories; untouched delay buffers and other states may still carry information. Interpret synthetic-state effects with dose, equal-norm controls, visibility, and restoration.
Direct-readout interventions test a channel in the artificial observation mapping, not biological neuronal inactivation. Joint channel effects use within-history paired decomposition.
Successful decoding shows that the chosen interpretive model can read the cue; failure does not prove information is absent. Held-out prediction by input-driven low-dimensional models is conditional on known inputs and does not establish an attractor.
All 20 jobs, transfer receipts, final file checks, paired results, and five figure sets have been reviewed. See the [full interpretation](interpretation.en.md) for findings and counterexamples, and [acceptance.json](acceptance.json) for delivery acceptance.
