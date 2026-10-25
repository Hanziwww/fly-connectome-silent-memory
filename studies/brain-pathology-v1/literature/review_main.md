# Main-session review of the literature synthesis (Phase 2a, 2026-10-19)

The literature synthesis (`atlas.csv`, `candidates.csv`, `calibration_draft.json`, `search_log.md`, `summary.md`) was written by
a delegated agent. This file records the main session's checks and corrections. The C1–C3 scores in `candidates.csv` stand except
where a correction below changes them. Nothing here is a capacity result: C4–C6 are run on the healthy study model at the GPU host
before gate G2.

## 1. Verification

- **Scale:** 100 atlas rows from 43 primary papers. Values shown only in figures are recorded as "not reported"; nothing was estimated.
- **Spot check against PubMed:** six key DOIs, all matching the stated title, design and finding.

  | DOI | PMID | Paper | Notes |
  |---|---|---|---|
  | 10.1016/j.cub.2017.02.065 | 28366741 | Franco 2017 | |
  | 10.1016/j.celrep.2018.06.029 | 29996096 | Hahm 2018 | |
  | 10.3389/fncel.2019.00409 | 31551716 | Higham 2019 | |
  | 10.1186/s40478-016-0333-4 | 27338814 | Chouhan 2016 | expression confined to the adult retina |
  | 10.1371/journal.pone.0062445 | 23638087 | Cavaliere 2013 | |
  | 10.1371/journal.pbio.1002563 | 27684064 | Gupta 2016 | |

## 2. Corrections to the C1–C3 scoring

1. **C23 (Aβ42 photoreceptor loss) – demoted to a movement-only candidate.**
   - **Expression:** Chouhan 2016 drove Aβ42 with Rh1-GAL4, i.e. in the adult retina only. Aβ42 caused modest photoreceptor loss *without* loss of the electroretinogram.
   - **No route into the tasks:** the Sokoban and cue-memory tasks inject their input into 3,200 visual projection neurons (LC12, TmY14, LC17, LPLC1, LC10a, ...). The photoreceptors (R1–R6 3,377 cells, R7 1,385, R8 1,329; histaminergic) are upstream and receive no task input. So C23 cannot act on learning or memory in this model, and the synthesis's statement that it sits in "a cell type the current visual tasks drive" is wrong.
   - **Remaining use:** it can act only on the embodied fly's eye-driven assays, such as looming escape. Even there, the fly phenotype to replicate (C5) is "cell loss without functional loss".
2. **C01 / C02 (Aβ42 Kv4 loss) – operator corrected.**
   - Kv4 carries the A-type transient K⁺ current. It sets spike threshold and latency; it is not spike-frequency adaptation.
   - The calibration target is the measured rheobase (−46%, 35.9 → 19.5 pA) with passive properties unchanged. The operator is therefore a threshold offset, fitted by an in-silico ramp protocol on the model neuron. It is not a change in adaptation strength.
3. **C14 (aging KCNQ / M-current loss in α/β Kenyon cells) – operator confirmed as adaptation.**
   - The M-current is a slow adaptation current, so the operator is a lower adaptation jump in the target cells.
   - Phase 1B (`manuscript/PLAN.md` §14, item 4) found that the model's cue memory is carried by spike-frequency adaptation. Conditions on this axis therefore act on the model's memory carrier itself.
   - C2 stays at grade 1: the aged measurement is mRNA, not a current.
4. **C03 (intraneuronal Aβ, release probability ↓) and C17 (tau, giant-fibre failure rate) – deferred.** C3 requires a calibration from a reported effect size. C03 has no published magnitude, and C17's numbers need the full text. Both stay out of the shortlist until a magnitude exists.
5. **C22 (aging spares dopaminergic neurons) – control arm, not a candidate.** It describes no change to replicate.
6. **C13 (Fragile X, weaker lateral inhibition) – target narrowed.** The model's ALLN class is mixed: 156 cholinergic, 151 GABAergic, 112 glutamatergic. The finding concerns GABAergic local interneurons, so the operator acts on GABAergic ALLN → ALPN (and ALLN → ALLN) connections only.
7. **C07 (aging active-zone enlargement at ALPN → KC) – C5 marked high-risk.** The published state combines three things:
   - more release;
   - fewer active zones per bouton;
   - flat-to-lower postsynaptic Kenyon-cell Ca²⁺.

   One weight scale cannot reproduce all three. C5 must test the combined constraint, not release alone.
8. **C11 / C12 (Aβ42 release onto ALPNs, ×2.29 early, ×0.58 late) – staged states only.** Hahm 2018 attributes the late decrease to homeostatic synaptic plasticity, which the model lacks. Both stages can be imposed as static states (C3 passes), but the transition between them cannot be modelled and is not claimed.
9. **C05 / C15 / C16 (tau in Kenyon cells and in M4/M6 MBONs) – C2 grade 1.** The measure is K⁺-evoked GCaMP Ca²⁺ signal, linked to more L-type Ca²⁺ channels (Higham 2019). Mapping it to a threshold offset is an inference, not a direct excitability measurement.

## 3. Model facts that set up C4

- **Monoamine cells are inert.** DPM, PAM, PPL1 and the other dopaminergic, serotonergic and octopaminergic cells have synapse sign 0. This is why C08, C09, C10, C19 and C24 fail C3. C4 includes a no-op test: deleting all of these cells must leave every output bitwise unchanged.
- **The tasks barely touch the olfactory system.**
  - Kenyon cells: 4,064, cholinergic; only 2 are in the readout. In the saved cue states their median |Δv| is 0.
  - The other targets are present: MBON01/03/04 (glutamatergic, sign −1), APL (GABAergic), ALPN (686) and ALLN (420).
- **The flybrain package's own battery is a hint, not a C4 result.** It uses the package's shipped settings, not the study model.
  - Apple odour drives 1,249 Kenyon cells above 1 Hz (31%), less sparse than in flies.
  - Projection neurons fire at 13 Hz.
  - Looming escape through the giant fibre passes.

## 4. C4 tests to run on the healthy study model (GPU host)

| Circuit | Test | Needed for |
|---|---|---|
| Antennal lobe | odour-evoked PN rates bounded; silencing GABAergic ALLN broadens PN tuning (lateral inhibition present) | C11, C12, C13 |
| Mushroom body | KC response sparseness per odour vs the fly reference; APL lesion raises KC activity (feedback inhibition present); MBON responses follow KC drive | C02, C04, C05, C07, C14, C15, C16 |
| Whole network | firing rates and runaway onset as the uniform threshold offset grows (stability margin) | C01 |
| Escape pathway | looming → giant-fibre burst → TTMn/DLMn; whether the looming assay runs through R1–R6 | C17, C18, C23 |
| Neuromodulator cells | deleting all DA, 5-HT and OA cells is a bitwise no-op | the C3 exclusions |

## 5. Provisional ranking after this review (G2 keeps 2–4 after C4–C6)

| Rank | Case | Target and operator | Reaches the current tasks? |
|---:|---|---|---|
| 1 | C01 Aβ42 Kv4 loss, pan-neuronal | threshold offset fitted to rheobase −46%; mapping to all cells is an extrapolation | yes |
| 2 | C14 aging KCNQ loss | α/β Kenyon cells; adaptation ↓ (the model's memory carrier) | only with an olfactory task |
| 3 | C13 Fragile X lateral inhibition | GABAergic ALLN → ALPN release ↓; best mapped | olfactory |
| 4 | C04 Aβ42 late Kenyon-cell loss (−17%) | neuron removal | olfactory |
| 5 | C02 / C16 | the conflicting pair of Kenyon-cell excitability states (Aβ42 up, tau down), run as separate arms | olfactory |
| 6 | C11 / C12 | staged Aβ42 release change onto ALPNs | olfactory |
| 7 | C07 aging active-zone enlargement | high C5 risk | olfactory |
| 8 | C15 tau M4/M6 MBON hyperexcitability | threshold offset | olfactory |

- **Movement-only:** C23.
- **Deferred:** C03, C17 (no magnitude); C18 (weak mapping).
- **Excluded:**
  - C3: C06, C08, C09, C10, C19, C20, C24;
  - C1: C21.

**Consequence for the olfactory assay.** All retained candidates except C01 act in the antennal lobe or the mushroom body. Under plan v2, an olfactory memory assay is added only if such a case reaches G2 and C4 confirms healthy mushroom-body function. For the present tasks, C01 is the only case that can reach learning, memory and movement directly.

Sources: PubMed records cited above, via DOI.
