r"""Builds Tables S2-S4, the supplemental Excel tables of the Cell paper, into manuscript/paper/build/supplement/ with the
dependency-free writer xlsx.py (no spreadsheet package is installed or downloaded).

  Table S2  every declared test the paper reports: the plan-v2 families of manuscript/results/plan_v2/tests.csv, the class
            (c) families of studies/brain-pathology-v1/results/all_tests.csv that STAR Methods records (state shifts against
            scrambles, swaps with the target hidden or shown, learning from a neutral brain) and the follow-up families of
            manuscript/results/stats/statistical_tests.csv that the paper reports (brain change, timing, swaps, equivalence);
            P adjusted within each complete declared family, as recorded.
  Table S3  the run registry: every training run with its configuration, record and outcome (sheet "Training runs"), and
            the evaluation and replay sets (sheet "Evaluations").
  Table S4  the single-cell panels: the genes with their FlyBase identifiers, and their values in every atlas unit.

Every reader-facing cell is written from the records here, with the study's development codes (scenes, learners,
lesion cases, experiment and family codes) translated; the builder fails if a code survives anywhere but in the record
paths and file names, which are the deposit's identifiers. Each workbook opens with a legend sheet, and a rebuild is
byte-identical.

  .\MIGRATION\python\python.exe -B -X utf8 MIGRATION\launch.py --script manuscript\writing\build_tables_s2_s4.py
"""
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "manuscript" / "figures"))
import xlsx  # noqa: E402
from lib.names import DEV_LABELS, INTERNAL, british  # noqa: E402

OUT = ROOT / "manuscript" / "paper" / "build" / "supplement"
RES = ROOT / "manuscript" / "results"
SUP = ROOT / "studies" / "brain-supplement-v2"
PAT = ROOT / "studies" / "brain-pathology-v1"
SOK = ROOT / "studies" / "brain-sokoban-v1"

# ---------------------------------------------------------------- reader-facing names of the study's codes
LEARNER = {
    "M0-head_only": "scene 1, readout trained", "M0": "scene 1, readout trained", "remote-dynamic-M0-head_only": "scene 1, readout trained",
    "m0": "scene 1, readout trained",
    "M0-joint": "scene 1, brain and readout trained", "M0joint": "scene 1, brain and readout trained",
    "remote-dynamic-M0-joint": "scene 1, brain and readout trained", "m0_joint": "scene 1, brain and readout trained",
    "supp-M0-visibility": "scene 1, one readout for hidden and shown targets",
    "M0vis.hidden": "scene 1, one readout for hidden and shown targets, target hidden",
    "M0vis.visible": "scene 1, one readout for hidden and shown targets, target shown",
    "S1-head_only": "scene 2, readout trained", "S1": "scene 2, readout trained", "supp-S1-ring-brain": "scene 2, readout trained",
    "s1": "scene 2, readout trained",
    "S2-head_only": "scene 3, readout trained", "S2": "scene 3, readout trained", "supp-S2-context-brain": "scene 3, readout trained",
    "s2": "scene 3, readout trained",
}
NEW_SCENE = {"M1": "scene 1 with one more floor cell", "M2": "scene 1 with two more floor cells", "S1-mirror": "scene 2, mirror image",
             "S1-minus45": "scene 2 with one floor cell removed", "S2-mirror": "scene 3, mirror image",
             "S2-minus48": "scene 3 with one floor cell removed"}
CASE = {"C01": "Kv4 loss (Aβ42)", "C04": "Kenyon-cell loss (late Aβ42)", "C13": "weaker antennal-lobe inhibition (fragile X)",
        "C16": "less excitable Kenyon cells (tau)"}
READ_SET = {"kc": "Kenyon cells", "mbon": "mushroom-body output neurons", "pn": "projection neurons", "readout512": "readout features"}
ODORS = {"apple-banana": "apple vs banana (dissimilar)", "apple-grape": "apple vs grape (similar)"}
POP = {"kc": "Kenyon cells", "alln_gaba": "GABAergic antennal-lobe local neurons"}
CHECKPOINT = {"0": "initial", "m": "midway", "1": "final"}
TIER_MAP = re.compile(r"\bD(\d)-T(\d)\b")


def planning_map(code):
    m = re.fullmatch(r"D(\d)-(T(\d)|H)", code)
    return f"tier {int(m.group(1)) + 1}, " + ("held-out map" if m.group(2) == "H" else f"map {int(m.group(3)) + 1}")


def condition(code):
    """'C01-dose0.5' -> 'Kv4 loss (Aβ42), dose 0.5'; 'C04-random' -> 'random control of the Kenyon-cell loss (late Aβ42)'."""
    case, _, rest = code.partition("-")
    if rest == "random":
        return f"size-matched random control of the {CASE[case]}"
    return f"{CASE[case]}, dose {rest.removeprefix('dose')}"


def learner_ctx(model, ctx=None, vis=None):
    """A memory model with its context (scene 3 has two orientations) and goal visibility, from the follow-up test ids."""
    out = LEARNER[model]
    if model.startswith(("supp-S2", "S2", "s2")) and ctx is not None:
        out += {"c0": ", first orientation", "c1": ", second orientation", "o0": ", first orientation",
                "o180": ", second orientation"}[ctx]
    if model == "supp-M0-visibility" and vis:
        out += {"hidden": ", target hidden", "visible": ", target shown"}[vis]
    return out


# ---------------------------------------------------------------- Table S2: the declared tests
# family: (reader-facing family, statistic, reported in)
FAMILIES = {
    "F13.X10.fig2": ("Learning cost tracks certified difficulty, per replicate set", "Spearman ρ, permutation test", "Figure 1G"),
    "F13.X9.pairing": ("Final brain with final readout against pairings with a midway checkpoint (maps solved)",
                       "exact McNemar over the 24 maps", "Figure 1H"),
    "F11.brain_change": ("Brain parameter change and training length", "Spearman and partial Spearman ρ, permutation",
                         "Figure S1B"),
    "F13.X1.trained": ("Memory use at the trained timings is positive, per learner", "exact one-sided sign-flip over replicates",
                       "Figure 1I"),
    "F13.X1.untrained": ("No net memory use at the untrained timings (below +0.10), per learner",
                         "exact one-sided sign-flip over replicates (non-superiority)", "STAR Methods"),
    "F13.X1.lock": ("Timing lock: trained minus untrained memory use is positive, per learner",
                    "exact one-sided sign-flip over replicates", "Figures 1I and 5C"),
    "F13.X8.primary": ("From a neutral brain: readout alone against brain and readout, updates to learn",
                       "stratified log-rank over four maps (permutation)", "STAR Methods"),
    "F13.X8.map": ("From a neutral brain, per map: readout alone against brain and readout", "exact Mann–Whitney; Fisher exact",
                   "STAR Methods"),
    "F13.X5.swap": ("Opposite-cue state swap against baseline at the trained timings", "exact McNemar, paired by history",
                    "Figure 2G"),
    "F13.X5.cross": ("Scene 3: swaps with the states of its other orientation", "exact McNemar, paired by history", "Figure S2G"),
    "F13.X5.visibility": ("Swap disruption with the target hidden against shown", "exact McNemar on matched pairs", "STAR Methods"),
    "F13.X4.content": ("Shifting the state toward the other cue against norm-matched scrambled shifts (full shift)",
                       "exact one-sided sign-flip over histories", "STAR Methods"),
    "F13.X4.dose": ("The same difference across shift sizes", "exact Page trend, histories as blocks", "STAR Methods"),
    "F13.X4.rank": ("Rank of the cue shift's effect among 20 scrambled shifts", "rank (descriptive)", "STAR Methods"),
    "F11.swap.combined": ("Follow-up: same-cue against opposite-cue swaps, all hidden-target readouts",
                          "exact McNemar, stratified by model", "STAR Methods"),
    "F11.swap.history": ("Follow-up: same-cue against opposite-cue swaps, per model", "exact McNemar, paired by history",
                         "STAR Methods"),
    "F11.swap.timing": ("Follow-up: same-cue against opposite-cue swaps, per timing", "exact sign test, clustered by cue pair",
                        "STAR Methods"),
    "F11.swap.visibility": ("Follow-up: swap disruption with the target hidden against shown", "exact sign test, matched histories",
                            "STAR Methods"),
    "F13.X6.ensemble": ("Silencing the cue ensemble at the decision against matched sets",
                        "exact one-sided sign-flip over histories", "Figure 3C"),
    "F13.X6.rank": ("Rank of the ensemble's silencing effect among 20 matched sets", "rank (descriptive)", "Figure 3C"),
    "F13.X7.edges": ("Removing the ensemble's internal synapses against matched synapses",
                     "exact one-sided sign-flip over histories", "Figure 3D"),
    "F13.X7.rank": ("Rank of the ensemble's synapse-removal effect among 20 matched sets", "rank (descriptive)", "Figure 3D"),
    "F12.dependency.equivalence": ("Population removal leaves the trained histories as in health",
                                   "Tango score test, two one-sided (±5 percentage points)", "Figure 3E"),
    "F12.dependency.null": ("Population removal against matched random removal",
                            "exact one-sided sign-flip over five memory models", "STAR Methods"),
    "F14.SC.trace.v2": ("Trace carriers against matched sets and the mapping null, hemilineage mapping (scene 1)",
                        "one-sided permutation (10,000 draws)", "Figure 4D"),
    "F14.SC.trace.v2.secondary": ("The same in the other learners", "one-sided permutation (10,000 draws)", "Figure 4D"),
    "F14.SC.trace.v2.vnc": ("The same in the ventral nerve cord, on the hemilineage units only",
                            "one-sided permutation (10,000 draws)", "Figure 4D"),
    "F14.SC.trace": ("First mapping (cell types and classes): trace carriers' adaptation index", "one-sided permutation",
                     "STAR Methods"),
    "F14.SC.trace.secondary": ("First mapping: the same at other levels and in the other learners", "one-sided permutation",
                               "STAR Methods"),
    "F14.SC.trace.sensitivity": ("First mapping: matched sets stratified by atlas source", "one-sided permutation", "STAR Methods"),
    "F14.SC.variant": ("Heterogeneous adaptation, scene 1: memory, lock and adaptation mechanism",
                       "exact one-sided sign test over 10 runs", "Figure 4J"),
    "F14.SC.variant.secondary": ("Heterogeneous adaptation against homogeneous and type-permuted null brains; scene 2",
                                 "exact Mann–Whitney; exact sign test", "Figure 4J; STAR Methods"),
    "F11.timing.differential": ("Push toward the correct target at untrained against trained timings",
                                "timing-label permutation", "Figures 5D and S5C"),
    "F11.timing.common": ("Push shared by both cues at untrained against trained timings", "timing-label permutation",
                          "STAR Methods"),
    "F11.timing.decoder": ("One fixed linear cue readout for all timings (leave one timing out)", "permutation", "STAR Methods"),
    "F11.equivalence.duration": ("Follow-up: input windows against blank waiting of the same duration",
                                 "Tango score test, two one-sided (±10 percentage points)", "STAR Methods"),
    "F11.equivalence.memory_use": ("Follow-up: no memory use at the untrained timings (below +10 percentage points)",
                                   "Tango score non-superiority test", "STAR Methods"),
    "F11.equivalence.transfer": ("Follow-up: a cue decoder fitted in scene 1, applied to other scenes and timings",
                                 "Tango score test, two one-sided (±10 percentage points)", "STAR Methods"),
    "F13.X2.trained": ("Six new scenes: memory use at the trained timings is positive",
                       "exact one-sided sign-flip over replicates", "Figure 5C"),
    "F13.X2.untrained": ("Six new scenes: no net memory use at the untrained timings (below +0.10)",
                         "exact one-sided sign-flip over replicates (non-superiority)", "Figure 5C"),
    "F13.X2.lock": ("Six new scenes: the timing lock is positive", "exact one-sided sign-flip over replicates", "Figure 5C"),
    "F13.X3.lock": ("Jittered against fixed-timing training: memory use at the untrained timings",
                    "exact one-sided sign-flip, paired by replicate", "Figure 5H"),
    "F13.X3.descriptive": ("Jittered against fixed-timing training: memory use and updates",
                           "exact two-sided sign-flip, paired by replicate", "Figure 5H; STAR Methods"),
    "F14.DY.robust": ("Training across threshold states against one state: the access window",
                      "exact one-sided Mann–Whitney", "Figure 6E"),
    "F14.DY.robust.secondary": ("Training across threshold states: memory use, cell-loss tolerance and updates",
                                "exact Mann–Whitney", "Figure 6E; STAR Methods"),
    "F12.C01.primary": ("Memory and learning decline with the dose of the Kv4 loss", "exact Page trend", "Figure 7C"),
    "F12.regimeA.solved": ("Lesion after learning: trained histories solved against health, per lesion and model",
                           "exact McNemar, paired by history", "Figure S7C"),
    "F12.regimeA.equivalence": ("Lesions of circuits the visual tasks do not use leave memory as in health",
                                "Tango score test, two one-sided (±5 percentage points)", "Figures 7F and S7C"),
    "F12.regimeB": ("Learning under the Kv4 loss against health, per dose and learner",
                    "exact two-sided sign-flip over replicate pairs", "STAR Methods"),
    "F12.regimeC": ("Relearning after the Kv4 loss", "exact Page trend; exact two-sided sign-flip", "STAR Methods"),
    "F12.rescue": ("Late restoration of excitability against the state of learning", "exact McNemar, paired by history",
                   "Figure 7E"),
    "F12.comparison": ("Olfactory phenotypes of the retained cases compared", "permutation of paired trials (10,000 draws)",
                       "Figure 7F"),
    "F12.olfactory": ("Olfactory memory after each lesion against health", "exact McNemar, paired by trial", "Figure S7G"),
    "F12.olfactory.primary": ("Targeted against size-matched random lesions, olfactory memory", "exact McNemar, paired by trial",
                              "STAR Methods"),
    "F14.SC.adfca.pan": ("Alzheimer's disease atlas: pan-neuronal Shal and adaptation index under Aβ42 and tau",
                         "ordinary least squares on genotype, age and sex", "Figure 7G"),
    "F14.SC.adfca": ("Alzheimer's disease atlas: the same per neuron type", "ordinary least squares on genotype, age and sex",
                     "STAR Methods"),
    "F14.SC.adfca.sensitivity": ("Alzheimer's disease atlas: per neuron type, all 14 libraries in one model",
                                 "ordinary least squares on genotype, age and sex", "STAR Methods"),
    "F14.SC.kv4": ("Shal-patterned against uniform Kv4 loss: trained histories solved", "exact two-sided McNemar, paired by history",
                   "Figure 7H"),
    "F14.SC.kv4.equivalence": ("Shal-patterned and uniform Kv4 loss equivalent", "Tango score test, two one-sided (±5 percentage points)",
                               "STAR Methods"),
    "F15.link": ("Recall follows the state that reads the trace, not the state that wrote it", "exact one-sided sign-flip",
                 "Figure 7J; STAR Methods"),
    "F15.link.secondary": ("The same per dose, and two-sided checks", "exact sign-flip", "Figure 7J"),
}
EXTRA = ("F13.X4.", "F13.X5.visibility", "F13.X8.")                  # class (c) families STAR Methods records
FOLLOW_UP = ("F11.brain_change", "F11.timing.", "F11.swap.", "F11.equivalence.")   # follow-ups the paper reports
WRITTEN = {"healthy-trained readouts": "readouts trained healthy", "lesion-trained readouts": "readouts trained under the lesion",
           "relearned readouts": "readouts relearned under the lesion"}


def link_state(code):
    """'H-L0.5' -> 'written healthy, read under the lesion at dose 0.5'."""
    w, r = code.split("-")
    word = lambda s: "healthy" if s == "H" else f"under the lesion at dose {s[1:]}"   # noqa: E731
    return f"written {word(w)}, read {word(r)}"


def test_name(fam, test):
    """The reader-facing name of one test of a declared family."""
    if fam in ("F13.X1.trained", "F13.X1.untrained", "F13.X1.lock"):
        return LEARNER[test]
    if fam.startswith("F13.X2."):
        return NEW_SCENE[test]
    if fam == "F13.X10.fig2":
        m = re.match(r"replicate set (\d): Spearman rho of log\(updates\) with optimal moves", test)
        if m:
            return ("original set" if m.group(1) == "0" else f"replicate set {m.group(1)}") + \
                ": Spearman ρ of log updates with the certified optimal moves"
        return "median updates per map over the six sets: Spearman ρ with the certified optimal moves"
    if fam == "F13.X9.pairing":
        pair = re.search(r"pairing (\w)(\w)$", test)
        return f"final brain and readout against the {CHECKPOINT[pair.group(1)]} brain with the {CHECKPOINT[pair.group(2)]} readout"
    if fam.startswith("F13.X3."):
        return {"trained_net_use": "memory use at the trained timings", "untrained_inside_net_use": "memory use at the untrained timings "
                "inside the jittered range", "untrained_outside_net_use": "memory use at the untrained timings outside the jittered range",
                "updates": "updates to learn", "untrained-timing net memory use, jittered minus fixed training":
                "memory use at the untrained timings"}[test.split(":")[0]]
    if fam == "F13.X5.swap":
        unit = test.split(":")[0]
        return f"{LEARNER[unit]}: opposite-cue swap against baseline (histories solved)"
    if fam == "F13.X5.cross":
        return {"S2: cross_opposite_cue swap vs baseline (solved)": "scene 3: the opposite cue's state from the other orientation "
                "against baseline (histories solved)", "S2: cross_same_cue swap vs baseline (solved)": "scene 3: the same cue's state "
                "from the other orientation against baseline (histories solved)", "S2: same-cue vs opposite-cue cross-orientation swap "
                "(solved)": "scene 3: same-cue against opposite-cue swaps from the other orientation (histories solved)"}[test]
    if fam == "F13.X5.visibility":
        return "scene 1, one readout for hidden and shown targets: disruption by the opposite-cue swap, target hidden against shown"
    if fam.startswith(("F13.X6.", "F13.X7.")):
        m = re.match(r"X[67]: ensemble from (m0|s1) tested in (m0|s1): (.*)", test)
        where = {"m0": "scene 1", "s1": "scene 2"}
        what = ("rank among the 20 matched sets" if m.group(3).startswith("rank") else
                "margin change minus the matched sets' mean, < 0")
        return f"ensemble found in {where[m.group(1)]}, tested in {where[m.group(2)]}: {what}"
    if fam.startswith("F13.X4."):
        unit, _, rest = test.partition(": ")
        if fam == "F13.X4.content":
            return f"{LEARNER[unit]}: cue shift minus the mean scrambled shift (margin change), full shift, < 0"
        if fam == "F13.X4.dose":
            return f"{LEARNER[unit]}: that difference decreases over shift sizes 0.125 to 1"
        return f"{LEARNER[unit]}: rank of the cue shift among 20 scrambled shifts, shift size {rest.rsplit(' ', 1)[1]}"
    if fam == "F13.X8.primary":
        return "readout alone against brain and readout from a neutral brain: updates to learn"
    if fam == "F13.X8.map":
        code, _, rest = test.partition(": ")
        what = ("updates to learn, readout alone against brain and readout (unsolved runs ranked last)" if rest.startswith("updates")
                else "runs that learned, readout alone against brain and readout")
        return f"{planning_map(code)}: {what}"
    if fam == "F12.C01.primary":
        m = re.match(r"learning under C01 lesion \((M0-\w+)\)", test)
        if m:
            return f"learning under the lesion, {LEARNER[m.group(1)]}: trained histories solved decline with dose"
        return "lesion after learning: trained-timing histories solved decline with dose (five memory models as blocks)"
    if fam == "F12.comparison":
        a, b = test.split(", ")[1].split(" vs ")
        return f"olfactory phenotype distance: {condition(a)} against {condition(b)}"
    if fam.startswith("F12.dependency."):
        m = re.match(r"(alln_gaba|kc)-(0\.5|1)\b", test)
        name = ("half" if m.group(2) == "0.5" else "all") + f" of the {POP[m.group(1)]}"
        if fam.endswith("equivalence"):
            return f"{name} removed against health: trained histories solved"
        return f"{name} removed: the targeted removal is less harmful than matched random removal"
    if fam == "F12.olfactory":
        c, pair, d, s = test.split("|")
        return f"{condition(c)}; {ODORS[pair]}; {d[1:]}-ms delay; read from the {READ_SET[s]}"
    if fam == "F12.olfactory.primary":
        case = test.split(":")[0]
        return f"{CASE[case]}: targeted against size-matched random lesion (mushroom-body output neurons, apple vs grape, 160 ms)"
    if fam == "F12.regimeA.equivalence":
        m = re.match(r"(C\d\d) dose (\S+) vs healthy", test)
        return f"{CASE[m.group(1)]}, dose {m.group(2)}, against health: trained histories solved"
    if fam == "F12.regimeA.solved":
        c, model = test.split("|")
        return f"{condition(c)}; {LEARNER[model]}"
    if fam == "F12.regimeB":
        m = re.match(r"C01 dose (\S+) vs healthy \((M0-\w+)\): (\w+)", test)
        what = {"log_updates": "log updates to learn", "trained_solved": "trained histories solved",
                "untrained_solved": "untrained histories solved"}[m.group(3)]
        return f"dose {m.group(1)} against health, {LEARNER[m.group(2)]}: {what}"
    if fam == "F12.regimeC":
        return {"recovery fraction decreasing in C01 dose (head_only retraining)": "readout relearned: the recovered share declines "
                "with dose", "recovery fraction decreasing in C01 dose (joint retraining)": "brain and readout relearned: the recovered "
                "share declines with dose", "recovery fraction: joint minus readout-only retraining": "recovered share, brain and "
                "readout relearned minus readout alone"}[test]
    if fam == "F12.rescue":
        m = re.match(r"(M0-\w+) dose (\S+): restored vs as trained", test)
        return f"{LEARNER[m.group(1)]}, dose {m.group(2)}: excitability restored against the state of learning (trained histories solved)"
    if fam.startswith("F14.DY.robust"):
        what = {"ed50-c01": "threshold lowering that halves trained memory use (access window)",
                "own-perturbation": "trained memory use at held-out threshold offsets",
                "removal-ed50": "share of cells whose random removal halves trained memory use (cell-loss tolerance)",
                "trained-net-use": "memory use at the trained timings", "untrained-net-use": "memory use at the untrained timings",
                "updates": "updates to the end of training"}[test.split(":")[1]]
        return f"across threshold states (SD 1 μV) against one state: {what}"
    if fam.startswith("F14.SC.adfca"):
        parts = test.split(":")
        geno = {"Abeta42": "Aβ42", "tau": "tau"}[parts[0]]
        what = {"Shal": "Shal", "adaptation_index": "adaptation index"}[parts[2]]
        tail = " (all 14 libraries)" if len(parts) > 3 else ""
        return f"{geno} against control, {parts[1]}: {what}{tail}"
    if fam.startswith("F14.SC.kv4"):
        m = re.match(r"(class|global) dose (\S+): Kv4-weighted vs uniform", test)
        return f"Shal-patterned ({m.group(1)} imputation) against uniform Kv4 loss, dose {m.group(2)}: trained histories solved"
    if fam.startswith("F14.SC.trace.v2"):
        model, comp, tier, stat, null = test.split(":")
        where = "all cells, any tier" if comp == "all" else "ventral nerve cord, cell type and hemilineage tiers"
        stat = "adaptation index" if stat == "A" else "SK"
        null = "against the mapping null" if null == "mapping-null" else "against matched sets"
        return f"{LEARNER[model]}; {where}; {stat} {null}"
    if fam.startswith("F14.SC.trace"):
        model, level, null = test.split(":")
        level = {"class": "cell class", "type": "cell type", "type_fuzzy": "cell type with fuzzy names"}[level]
        null = {"mapping-null": "against the mapping null", "matched-sets": "against matched sets",
                "source-strata": "against matched sets stratified by source", "davis-only": "units of one source (Davis et al.) only"}[null]
        return f"{LEARNER[model]}; mapping by {level}; {null}"
    if fam.startswith("F14.SC.variant"):
        what = {"adaptation drop > 0": "memory lost when adaptation is cleared, > 0", "lock > 0": "timing lock, > 0",
                "trained net use > 0": "memory use at the trained timings, > 0",
                "ed50_c01_mv": "threshold lowering that halves trained memory use", "lock": "timing lock",
                "trained_net_use": "memory use at the trained timings", "untrained_net_use": "memory use at the untrained timings"}
        head, _, rest = test.partition(": ")
        if head == "variant M0":
            return f"heterogeneous adaptation, scene 1, readout trained: {what[rest]}"
        if head == "variant S1":
            return f"heterogeneous adaptation, scene 2, readout trained: {what[rest]}"
        other = "homogeneous adaptation" if "X1" in head else "the type-permuted null brain"
        return f"heterogeneous adaptation against {other}, scene 1: {what[rest]}"
    if fam == "F15.link":
        head, _, rest = test.partition(": ")
        who, _, dose = head.partition(", ")
        rest = (rest.replace("solved(lesion-written, read healthy)", "solved when written under the lesion and read healthy")
                    .replace("solved(healthy-written, read under the lesion)", "solved when written healthy and read under the lesion"))
        return f"{WRITTEN[who]}, {dose}: {rest.replace(' - ', ' minus ')}"
    if fam == "F15.link.secondary":
        head, _, rest = test.partition(": ")
        who, _, dose = head.partition(", ")
        rest = re.sub(r"solved\(([HL\d.]+-[HL\d.]+)\)", lambda m: f"solved ({link_state(m.group(1))})", rest)
        return f"{WRITTEN[who]}, {dose}: {rest.replace(' - ', ' minus ')}"
    raise SystemExit(f"Table S2: no reader-facing name for {fam}: {test}")


def follow_up_name(test_id):
    """The reader-facing name of a follow-up test (statistical_tests.csv), from its id."""
    m = re.match(r"fu1\.(common|decoder|differential)\.([\w.-]+)\|(c\d)\|(hidden|visible)$", test_id)
    if m:
        what = {"common": "push shared by both cues, 26 untrained against 9 trained timings",
                "decoder": "one fixed linear cue readout for all 35 timings, leave one timing out",
                "differential": "push toward the correct target, 26 untrained against 9 trained timings"}[m.group(1)]
        return f"{learner_ctx(m.group(2), m.group(3), m.group(4))}: {what}"
    if test_id == "fu2.swap_combined.hidden":
        return "hidden-target readouts of scenes 1 to 3: solved only after the same-cue against only after the opposite-cue swap"
    m = re.match(r"fu2\.swap_(history|timing)\.([\w.]+)$", test_id)
    if m:
        what = ("histories solved after a same-cue against an opposite-cue swap" if m.group(1) == "history" else
                "solved per timing after same-cue minus opposite-cue swaps")
        return f"{LEARNER[m.group(2)]}: {what}"
    if test_id == "fu2.swap_visibility.shared_readout":
        return "scene 1, one readout for hidden and shown targets: swap disruption, target hidden minus shown"
    m = re.match(r"fu3\.duration\.([\w.]+)\.(neutral_stationary|neutral_walk_0|neutral_walk_1)$", test_id)
    if m:
        kind = {"neutral_stationary": "input with the player standing still", "neutral_walk_0": "input with the player walking, "
                "first path", "neutral_walk_1": "input with the player walking, second path"}[m.group(2)]
        return f"{LEARNER[m.group(1)]}: solved after two windows of {kind} against blank waiting of the same duration"
    m = re.match(r"fu3\.memory_use\.([\w.-]+)$", test_id)
    if m:
        return f"{LEARNER[m.group(1)]}: memory use at the untrained timings"
    m = re.match(r"fu3\.transfer_decoding\.(S1|S2)\.(train|untrained)$", test_id)
    if m:
        scene = {"S1": "scene 2", "S2": "scene 3"}[m.group(1)]
        when = "trained timings" if m.group(2) == "train" else "untrained timings"
        return f"cue decoder fitted in scene 1, applied to {scene} at its {when}: both cues right minus both wrong"
    if test_id == "fu3.transfer_decoding.dense_untrained":
        return "dense delay scan, untrained timings: both cues decoded right minus both wrong by the fixed linear decoder"
    return {"fu7.spearman.updates": "24 maps: brain parameter change against log updates",
            "fu7.partial.brain_term": "24 maps: brain parameter change against the brain term of the margin change, given log updates",
            "fu7.partial.total_margin_change": "24 maps: brain parameter change against the total margin change, given log updates",
            }[test_id]


def declared():
    """{family: posthoc} from the two test plans (parsed with ast; the code is never imported)."""
    out = {}
    for rel in ("studies/brain-supplement-v2/code/testplan.py", "studies/brain-pathology-v1/code/testplan.py"):
        for node in ast.walk(ast.parse((ROOT / rel).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "family" and node.args:
                kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg == "posthoc"}
                out[ast.literal_eval(node.args[0])] = bool(kw.get("posthoc", False))
    return out


def table_s2():
    t = pd.read_csv(RES / "plan_v2" / "tests.csv", low_memory=False)
    a = pd.read_csv(PAT / "results" / "all_tests.csv", low_memory=False)
    s = pd.read_csv(RES / "stats" / "statistical_tests.csv", low_memory=False)
    if t["family"].astype(str).str.startswith(EXTRA).any():
        raise SystemExit("Table S2: the paper's test table now holds the class (c) families; drop them from EXTRA")
    rows = []
    for r in t.itertuples():
        rows.append(dict(family=r.family, test=test_name(r.family, str(r.test)), n=r.n, estimate=r.estimate, p=r.p_value,
                         p_adj=r.p_adjusted, size=r.family_tests, correction=r.correction, primary=r.primary,
                         source="manuscript/results/plan_v2/tests.csv"))
    size = a.groupby("family")["test"].size()
    for r in a[a["family"].astype(str).str.startswith(EXTRA)].itertuples():
        rows.append(dict(family=r.family, test=test_name(r.family, str(r.test)), n=r.n, estimate=r.estimate, p=r.p_value,
                         p_adj=r.p_adjusted_family, size=size[r.family], correction=r.correction, primary=r.primary,
                         source="studies/brain-pathology-v1/results/all_tests.csv"))
    f = s[s["family"].astype(str).str.startswith(FOLLOW_UP)]
    fsize = s.groupby("family")["test_id"].size()
    for r in f.itertuples():
        rows.append(dict(family=r.family, test=follow_up_name(r.test_id), n=r.n, estimate=r.effect, p=r.p_value, p_adj=r.p_adjusted,
                         size=fsize[r.family], correction=r.family_correction, primary=r.family_primary,
                         source="manuscript/results/stats/statistical_tests.csv", posthoc=r.family_posthoc))
    posthoc = declared()
    order = {fam: i for i, fam in enumerate(FAMILIES)}
    missing = sorted({r["family"] for r in rows} - set(FAMILIES))
    if missing:
        raise SystemExit(f"Table S2: families without a reader-facing name: {missing}")
    rows.sort(key=lambda r: order[r["family"]])                     # stable: each family keeps its record order
    corr = {"holm": "Holm", "bh": "Benjamini–Hochberg", "none": "none"}
    body = []
    for r in rows:
        name, stat, where = FAMILIES[r["family"]]
        ph = r.get("posthoc", posthoc.get(r["family"], False))
        body.append([name, where, r["test"], stat, None if pd.isna(r["n"]) else int(r["n"]),
                     None if pd.isna(r["estimate"]) else float(r["estimate"]), None if pd.isna(r["p"]) else float(r["p"]),
                     None if pd.isna(r["p_adj"]) else float(r["p_adj"]), int(r["size"]), corr[str(r["correction"])],
                     bool(r["primary"]) if not isinstance(r["primary"], str) else r["primary"].lower() == "true", bool(ph),
                     r["source"]])
    header = ["Family", "Reported in", "Test", "Statistic", "n", "Estimate", "P", "P adjusted in the family",
              "Tests in the family", "Correction", "Primary", "Post hoc", "Record"]
    legend = [["Table S2. Every declared test the paper reports, related to Figures 1–7"],
              ["One row per test: its family (the question the family tests), where the paper reports it, the test and its "
               "statistic, n, the estimate where the test records one, the P value, the P value adjusted within the complete "
               "declared family (all of whose tests enter the adjustment), the family's size and correction, whether the family "
               "is primary, and whether it was declared after its data were seen (post hoc; never primary)."],
              ["Every family was declared in a test plan before its tests were computed. Families whose tests the paper does not "
               "report are not listed; their records are in the deposit."],
              ["Estimates: memory use and shares in their own units (net memory use from −1 to +1; shares from 0 to 1); "
               "Spearman ρ for correlations; differences for paired comparisons; empty where the test records none."],
              ["Record: the file in the deposit that holds the test, with its development names."],
              [f"{len(body)} tests in {len({r['family'] for r in rows})} families."]]
    return [xlsx.Sheet("Legend", rows=legend, widths=[150], wrap=True),
            xlsx.Sheet("Tests", header=header, rows=body, widths=[52, 18, 70, 34, 7, 11, 11, 13, 10, 18, 8, 8, 44])]


# ---------------------------------------------------------------- Table S3: the run registry
OUTCOME = {"learned_solved": "learned", "original_solved": "learned", "stagnation": "stopped by stagnation",
           "segment_limit": "stopped at the time limit", "numerical_error": "numerical error", "implementation_error": "implementation error",
           "frozen_baseline": "baseline, not trained", "untrained_baseline": "baseline, not trained", "pending": "registered, not run",
           "stopped": "stopped by hand", "complete": "complete", "completed": "complete", "passed": "passed"}
TRAINED = {"joint": "brain and readout", "head_only": "readout", "brain_only": "brain", "frozen": "nothing",
           "alternating": "brain and readout", "no_recovery": "brain and readout", "entry_ce": "brain and readout",
           "demonstration": "brain and readout", "neutral": "brain and readout", "rewired": "brain and readout",
           "weight_shuffle": "brain and readout", "board_mlp": "control network", "gru": "control network", "carry_untrained": "nothing",
           "reset": "brain and readout", "brain": "readout", "leaky": "readout", "neutral-real": "readout", "neutral-rewired": "readout",
           "p05-fresh": "readout"}
SETTING = {"joint": "", "head_only": "", "brain_only": "", "frozen": "baseline without training",
           "alternating": "brain and readout updated in turn", "no_recovery": "without the recovery term of the loss",
           "entry_ce": "cross-entropy at the new start in place of the margin loss", "demonstration": "cross-entropy on the demonstration",
           "neutral": "from neutral brain parameters and a fresh readout", "rewired": "rewired connectome, degrees and signs kept",
           "weight_shuffle": "weight-shuffled connectome", "board_mlp": "a multilayer perceptron reading the board (control)",
           "gru": "a small recurrent network reading the board (control)", "carry_untrained": "baseline without training, state carried",
           "reset": "state reset before each decision", "brain": "", "leaky": "a fixed decaying trace in place of the brain (control)",
           "neutral-real": "neutral brain parameters, real connectome", "neutral-rewired": "neutral brain parameters, rewired connectome",
           "p05-fresh": "the planning-trained brain with a fresh readout"}
MEMORY_TASK = {"M0": "scene 1", "M0-visibility": "scene 1, target hidden or shown", "S1-ring": "scene 2", "S2-context": "scene 3",
               "M0-audit": "scene 1"}
VARIANT = {"receptor-none": "signs from the presynaptic transmitter alone", "same-type-0.05": "same-type gain 0.05",
           "same-type-0.2": "same-type gain 0.2", "same-type-off": "no same-type damping", "conn-cap-30": "connection cap 30 synapses",
           "conn-cap-120": "connection cap 120 synapses", "conn-cap-off": "no connection cap"}
LEARNER_TRAINED = {"M0-head_only": ("scene 1", "readout"), "M0-joint": ("scene 1", "brain and readout"),
                   "S1-head_only": ("scene 2", "readout"), "S2-head_only": ("scene 3", "readout")}
V1_USE = {"linear_joint": ("Warm-up maps", "STAR Methods"), "benchmark": ("Precision calibration", "STAR Methods"),
          "difficulty": ("Planning ladder, original set", "Figure 1G"), "controls": ("Planning controls", "—"),
          "static": ("Planning controls", "—"), "memory": ("Memory task, original runs", ""), "train": ("Memory task, original runs", ""),
          "curriculum": ("Registered, not run", "—"), "parameter_memory": ("Registered, not run", "—"),
          "transfer": ("Registered, not run", "—")}
USED_ORIGINAL = {"remote-dynamic-M0-head_only": "Figures 1–7", "remote-dynamic-M0-joint": "Figures 1–5",
                 "supp-M0-visibility": "Figures 2 and 3; Figure S2", "supp-S1-ring-brain": "Figures 1–5",
                 "supp-S2-context-brain": "Figures 1–5"}


def v1_task(module, task, run_id):
    if module == "linear_joint":
        return {"P03": "warm-up map 1 (8 by 8 cells, two boxes)", "P05": "warm-up map 2 (10 by 10 cells, two boxes)"}[task]
    if module == "benchmark":
        return f"calibration map {int(task[1]) + 1}"
    if module in ("memory", "train"):
        return MEMORY_TASK[task]
    m = re.fullmatch(r"(D\d-(?:T\d|H))(-S0)?", str(task))
    return planning_map(m.group(1)) + (", a variant start" if m.group(2) else "")


def v1_setting(module, cond, run_id):
    if module == "benchmark":
        m = re.match(r"benchmark(?:-v(\d))?-(fp64|fp32|mixed)-b(\d+)-C\d", run_id)
        prec = {"fp64": "float64", "fp32": "float32", "mixed": "float64 with float32 gradient sums"}[m.group(2)]
        return f"{prec}, batch {m.group(3)}, implementation {m.group(1) or '1'}"
    if module == "curriculum":
        return "one continuous curriculum through the tiers"
    if module == "parameter_memory":
        return "relearning after training on another map (A, then B, then A)"
    if module == "transfer":
        return "retraining on a held-out map" + (", from neutral parameters" if cond == "neutral" else "")
    return SETTING[cond]


def registry_row(group, used, task, trained, setting, replicate, status, updates, seconds, solved, tn, un, verified, sha, record):
    num = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else v   # noqa: E731
    return [group, used, task, trained, setting, replicate, OUTCOME.get(str(status), str(status)), num(updates), num(seconds),
            num(solved), num(tn), num(un), None if verified is None else str(verified) == "verified", sha, record]


def table_s3():
    reg = pd.read_csv(RES / "supplementary" / "st1_run_registry.csv", low_memory=False)
    train_mod = set(V1_USE)
    rows, evals = [], []
    for r in reg.itertuples():
        if r.module in train_mod and not (r.layer == "registered_not_run" and r.module == "controls"):
            group, used = V1_USE[r.module]
            if r.layer == "registered_not_run":
                group, used = "Registered, not run", "—"
            if r.module in ("memory", "train"):
                used = USED_ORIGINAL.get(r.run_id, "—")
            rows.append(registry_row(group, used, v1_task(r.module, r.task, r.run_id), TRAINED.get(r.condition, "brain and readout"),
                                     v1_setting(r.module, r.condition, r.run_id), 0, r.status, r.updates, r.training_seconds, None,
                                     None, None, r.verification_status if isinstance(r.verification_status, str) else None, None,
                                     r.source_file))
        elif r.layer == "registered_not_run":                    # the registered D7 controls, run later as their own runs
            rows.append(registry_row("Registered, not run", "—", v1_task("controls", r.task, r.run_id), TRAINED[r.condition],
                                     SETTING[r.condition] or "standard", 0, r.status, None, None, None, None, None, None, None,
                                     r.source_file))
        else:
            evals.append(r)
    sup, pat = "studies/brain-supplement-v2/runs/", "studies/brain-pathology-v1/runs/"
    x10 = pd.read_csv(SUP / "results" / "x10_maps.csv")
    for r in x10[x10.replicate > 0].itertuples():
        rows.append(registry_row("Planning ladder, replicate sets", "Figure 1G", planning_map(r.map), "brain and readout", "", int(r.replicate),
                                 r.status, r.updates, r.training_seconds, None, None, None, None, None,
                                 f"{sup}difficulty-{r.map}-joint-r{r.replicate}"))
    d7 = pd.read_csv(SUP / "results" / "x9_d7_controls.csv")
    for r in d7[d7.source != "frozen"].itertuples():
        rows.append(registry_row("Planning controls, hardest map", "Figure S1G", planning_map(r.task), TRAINED[r.condition],
                                 SETTING[r.condition] or "standard", 0, r.status, r.updates, r.training_seconds, None, None, None,
                                 None, None, sup + r.run))
    x8 = pd.read_csv(SUP / "results" / "x8_runs.csv")
    for r in x8.itertuples():
        rows.append(registry_row("Planning from a neutral brain", "STAR Methods", planning_map(r.map), TRAINED[r.arm],
                                 "from neutral brain parameters and a fresh readout", int(r.replicate), r.status, r.updates,
                                 r.training_seconds, None, None, None, r.verified, None, sup + r.run))

    def memory(group, used, job, task, trained, setting, replicate, r, record):
        rows.append(registry_row(group, used, task, trained, setting, replicate, r.status, r.updates, r.training_seconds,
                                 r.trained_solved, r.trained_net_use, r.untrained_net_use, r.verified, r.model_sha256, record))

    camp = pd.read_csv(SUP / "results" / "campaign_runs.csv")
    for r in camp.itertuples():
        m = re.fullmatch(r"x1-(\w+-\w+)-r(\d+)", r.job)
        if m:
            task, trained = LEARNER_TRAINED[m.group(1)]
            memory("Memory learners, replicates", "Figures 1I and 5C", r.job, task, trained, "", int(m.group(2)), r, sup + r.job)
            continue
        m = re.fullmatch(r"b-(C\d\d)-dose([\d.]+)-(M0-\w+)-r(\d+)", r.job)
        task, trained = LEARNER_TRAINED[m.group(3)]
        used = "Figures 7B, 7C and S7D" if m.group(1) == "C01" else "STAR Methods"
        memory("Learning under a lesion", used, r.job, task, trained, f"{CASE[m.group(1)]}, dose {m.group(2)}", int(m.group(4)), r,
               pat + r.job)
    for r in pd.read_csv(SUP / "results" / "x2_runs.csv").itertuples():
        m = re.fullmatch(r"x2-([\w-]+)-r(\d+)", r.job)
        memory("New scenes", "Figure 5C", r.job, NEW_SCENE[m.group(1)], "readout", "", int(m.group(2)), r, sup + r.job)
    x3 = pd.read_csv(SUP / "results" / "x3_runs.csv")
    for r in x3[x3.job.str.startswith("x3-")].itertuples():
        memory("Jittered timings", "Figure 5H", r.job, "scene 1", "readout", "timings drawn at random from the jittered range",
               int(r.job.rsplit("-r", 1)[1]), r, sup + r.job)
    o14 = pd.read_csv(PAT / "results" / "o14_runs.csv")
    for r in o14[o14.variant != "default"].itertuples():
        memory("Connection-rule variants", "Figure 3G", r.job, "scene 1", "readout", VARIANT[r.variant], int(r.replicate), r, pat + r.job)
    dy = pd.read_csv(PAT / "results" / "dy2_models.csv")
    for r in dy[dy.condition == "th0.001"].itertuples():
        rows.append(registry_row("Training across threshold states", "Figure 6E", "scene 1", "readout",
                                 "one threshold offset per training episode, SD 1 μV", int(r.replicate), r.status, r.updates,
                                 r.training_seconds, None, r.trained_net_use, r.untrained_net_use, None, None, pat + r.model))
    sv = pd.read_csv(PAT / "results" / "sc_variant_runs.csv")
    for r in sv.itertuples():
        task = {"M0": "scene 1", "S1": "scene 2"}[r.family]
        setting = ("heterogeneous adaptation (atlas-informed)" if r.brain == "variant" else
                   "heterogeneous adaptation, permuted across types (null brain)")
        rows.append(registry_row("Heterogeneous adaptation", "Figure 4J", task, "readout", setting, int(r.replicate), r.status, r.updates,
                                 None, None, r.trained_net_use, r.untrained_net_use, None, None, pat + r.run))
    rc = pd.read_csv(PAT / "results" / "regime_c.csv")
    for r in rc.itertuples():
        trained = {"head_only": "readout", "joint": "brain and readout"}[r.mode]
        rows.append(registry_row("Relearning after the lesion", "Figure 7D" if r.mode == "head_only" else "STAR Methods", "scene 1",
                                 trained, f"{CASE['C01']}, dose {r.dose}; 100 updates after training healthy", int(r.replicate),
                                 "relearned", 100, r.seconds, r.solved_100, None, None, None, None, f"{pat}regime-c/{r.job}.json"))
    header = ["Group", "Used in", "Task", "Trained", "Setting", "Replicate", "Outcome", "Updates", "Training time (s)",
              "Trained histories solved", "Memory use, trained timings", "Memory use, untrained timings", "Verified by reload",
              "Model SHA-256", "Record"]
    ev = evaluation_sets(evals)
    legend = [["Table S3. The run registry, related to Figures 1–7"],
              ["Training runs: training runs of the study, each with its group, where the paper uses it, its task (planning maps "
               "by tier, memory scenes), what was trained (the readout, or the brain's type-level parameters and the readout), its "
               "setting, its replicate (0 for single runs), its outcome under the stopping rule, its updates and training time, "
               "and for memory runs the trained histories solved and the net memory use at the trained and untrained timings "
               "(from −1 to +1). Verified: an independent process reloaded the model and reproduced its outcome. Runs the study "
               "registered but never ran are listed as such."],
              ["Evaluations: the replays and evaluation sets run on trained models, with what they evaluate and their records."],
              ["Record: the run's folder or file in the deposit, with its development names."],
              [f"{len(rows)} training runs; {len(ev)} evaluation entries."]]
    return [xlsx.Sheet("Legend", rows=legend, widths=[150], wrap=True),
            xlsx.Sheet("Training runs", header=header, rows=rows, widths=[30, 20, 34, 18, 44, 9, 22, 9, 11, 11, 11, 11, 9, 20, 60]),
            xlsx.Sheet("Evaluations", header=["Set", "Used in", "What it evaluates", "Records", "Record"], rows=ev,
                       widths=[40, 22, 80, 9, 66])]


V1_EVAL = {"preflight": "a replay check before the mechanism runs", "routes": "how the cue travels from input to readout",
           "selective": "selective interventions on cell groups", "timing": "the dense scan of delays",
           "components": "the decomposition of the decision state", "discover": "the cells that carry learning or memory",
           "groups": "interventions on cell groups", "intervene": "interventions in a planning map", "pathways": "the input-to-readout pathways",
           "record": "dense records of every state variable", "evaluate": "zero-update transfer to the eight held-out maps",
           "import": "the import of the trained model onto the remote host", "smoke": "the memory histories before training",
           "probe": "causal probes and transfer between scenes"}
V2_EVAL = [  # (set, used in, what, record folder, count rule)
    ("Linear response of the decision to the trace", "Figures 2F and 5G", "transplants and injected trace patterns at seven delays",
     "studies/brain-supplement-v2/runs/g4-linear-response", None),
    ("State shifts against scrambles", "STAR Methods", "the state shifted toward the other cue against 20 norm-matched scrambles",
     "studies/brain-supplement-v2/runs/x4-scrambles", None),
    ("Opposite-cue swaps", "Figures 2G and S2G", "state swaps at the nine trained timings, with the target hidden or shown",
     "studies/brain-supplement-v2/runs/x5-swaps", None),
    ("Ensemble silencing and synapse removal", "Figures 3C and 3D", "the cue ensembles of two scenes and their matched sets",
     "studies/brain-supplement-v2/runs/x6-x7", None),
    ("Checkpoint pairings and stage replays", "Figure 1H; STAR Methods", "the planning maps replayed with every pairing of brain and "
     "readout checkpoints, and at every curriculum stage", "studies/brain-supplement-v2/runs/x9-replays", None),
    ("Lesions after learning", "Figures 7B, 7C and S7C", "the retained cases and their random controls in the five memory models",
     "studies/brain-pathology-v1/runs/phase3-regime-a", None),
    ("Population removal", "Figure 3E; STAR Methods", "Kenyon cells and GABAergic antennal-lobe local neurons removed, with matched "
     "random sets and anchors", "studies/brain-pathology-v1/runs/phase3-dependency", None),
    ("Olfactory memory assay", "Figures 7F and S7G", "odor memory read from four read-out sets after each lesion",
     "studies/brain-pathology-v1/runs/phase3-olfactory", None),
    ("Late restoration", "Figure 7E", "the lesion lifted from the models that learned under it",
     "studies/brain-pathology-v1/runs/rescue", None),
    ("Credibility tests of the candidate pathologies", "Figure 7A; Table S1", "healthy circuits, calibration and replication on two brains",
     "studies/brain-pathology-v1/runs/phase2-credibility", None),
    ("Healthy circuits under the connection-rule variants", "Figure 3G", "the healthy-circuit tests repeated under each rule set",
     "studies/brain-pathology-v1/runs/phase2-variants", None),
    ("Lesions under the connection-rule variants", "Figure 3G", "the Kv4 loss in the runs trained under each rule set",
     "studies/brain-pathology-v1/runs/o14-lesion", None),
    ("Dose series", "STAR Methods", "the Kv4 loss from small to full dose, and random cell removal",
     "studies/brain-pathology-v1/runs/phase4-dose, phase4-dose-low, phase4-dose-micro", None),
    ("Shal-patterned Kv4 loss", "Figures 7H, 7I and S7", "the patterned against the uniform lesion, doses, superclasses and the "
     "models that learned under the lesion", "studies/brain-pathology-v1/runs/sc-kv4", None),
    ("Aging operator", "Figure S7K", "the operator's healthy-circuit tests on two brains", "studies/brain-pathology-v1/runs/sc-aging", None),
    ("Kicks and threshold shifts", "Figure 6C; STAR Methods", "single extra spikes and uniform threshold shifts after cue offset, "
     "two trained timings and both cues", "studies/brain-pathology-v1/runs/dy1-sensitivity", 4),
    ("Persistent threshold shift, cell by cell", "STAR Methods", "the cells that diverge under a 0.3-μV lowering",
     "studies/brain-pathology-v1/runs/fig-divergence-cells", None),
    ("Training across threshold states: evaluations", "Figure 6E", "access window, cell-loss tolerance and memory at held-out offsets "
     "for the five runs trained across states and the ten trained in one state", "studies/brain-pathology-v1/runs/dy2-eval", 15),
    ("Heterogeneous adaptation: evaluations", "Figure 4J", "memory, lock and mechanism of the runs on the two adaptation brains",
     "studies/brain-pathology-v1/runs/sc-variant-eval", None),
    ("Writing and reading the trace in different brain states", "Figure 7J; Video S4", "the full state moved between the healthy and "
     "the lesioned brain, read by three kinds of readout", "studies/brain-pathology-v1/runs/lesion-link", None),
    ("Mean-field linearization", "STAR Methods", "a rate linearization of the network (not validated, so not interpreted)",
     "studies/brain-pathology-v1/runs/g4-meanfield", None),
]


def evaluation_sets(v1_rows):
    out = []
    for r in v1_rows:
        what = V1_EVAL[r.module]
        model = r.task if isinstance(r.task, str) else ""
        if model in LEARNER:
            what += f"; {LEARNER[model]}"
        elif model.startswith("difficulty-"):
            what += f"; {planning_map(model.removeprefix('difficulty-'))}"
        elif model.startswith("control-"):
            mm = re.fullmatch(r"control-(D\d)-(\w+)", model)
            what += f"; {planning_map(mm.group(1) + '-T0')}, {TRAINED[mm.group(2)]} trained"
        elif model in MEMORY_TASK:
            what += f"; {MEMORY_TASK[model]}"
        elif model.startswith("remote-D7-"):
            what += f"; {planning_map('D7-T0')}, {TRAINED[model.rsplit('-', 1)[1]]} trained"
        out.append(["Earlier study: " + {"mechanisms_v1": "mechanism replays", "mechanisms_supplement": "supplementary replays",
                                         "remote_batch1": "remote checks", "remote_supplement": "supplementary probes"}[r.layer],
                    "Figures 1–5" if r.layer.startswith("mechanisms") else "—", what, 1, r.source_file])
    for name, used, what, rel, count in V2_EVAL:
        n = count
        if n is None:
            n = sum(len(list((ROOT / part.strip() if i == 0 else ROOT / rel.split(",")[0].rsplit("/", 1)[0] / part.strip()).glob("*.json")))
                    for i, part in enumerate(rel.split(",")))
        out.append([name, used, what, int(n), rel])
    return out


# ---------------------------------------------------------------- Table S4: the single-cell panels
SOURCES = {"davis2020": ("Davis et al. 2020", "mean over samples of log(1 + TPM); bulk, so no detection share"),
           "ozel2021": ("Özel et al. 2021", "mean log-normalized expression per cluster, as published; no detection share"),
           "davie2018": ("Davie et al. 2018", "mean over cells of log(1 + counts per 10,000); detection share"),
           "afca2023": ("Aging Fly Cell Atlas (Lu et al. 2023)", "mean over cells of log(1 + counts per 10,000); detection share"),
           "adfca2025": ("Alzheimer's disease Fly Cell Atlas, controls (Park et al. 2025)",
                         "mean over cells of log(1 + counts per 10,000); detection share"),
           "allen2020vnc": ("Ventral nerve cord atlas (Allen et al. 2020), hemilineage units",
                            "mean over cells of log(1 + counts per 10,000); detection share")}
ROLES = {"index": "adaptation index", "kv4": "Kv4 weight", "descriptive": "channel panel",
         "k_channel": "potassium channels of the aging operator", "aging": "aging panel"}


def table_s4():
    sc = PAT / "runs" / "sc-mapping"
    vnc = PAT / "runs" / "sc-lineage-vnc-v3"
    genes = pd.read_csv(PAT / "results" / "sc_panel_genes.csv")
    gl = genes["gene"].tolist()
    mean = {s: pd.read_parquet(sc / f"units_{s}_mean.parquet") for s in list(SOURCES)[:5]}
    mean["allen2020vnc"] = pd.read_parquet(vnc / "units_allen2020vnc_mean.parquet")
    frac = {s: pd.read_parquet(sc / f"units_{s}_frac.parquet") for s in ("davie2018", "afca2023", "adfca2025")}
    frac["allen2020vnc"] = pd.read_parquet(vnc / "units_allen2020vnc_frac.parquet")
    sizes = pd.read_csv(sc / "unit_sizes.csv", dtype={"unit": str})
    sizes = pd.concat([sizes, pd.read_csv(vnc / "units.csv", dtype={"unit": str})[["unit", "cells"]].assign(source="allen2020vnc")])
    size = {(r.source, str(r.unit)): r.cells for r in sizes.itertuples()}
    v2 = pd.read_csv(PAT / "results" / "sc_unit_values_v2.csv", dtype={"unit": str}).set_index(["source", "unit"])
    v1 = pd.read_csv(PAT / "results" / "sc_unit_values.csv", dtype={"unit": str}).set_index(["source", "unit"])
    gene_rows = []
    for r in genes.itertuples():
        present = [bool(getattr(r, f"in_{s}")) for s in list(SOURCES)[:5]] + [r.gene in mean["allen2020vnc"].columns]
        gene_rows.append([r.gene, r.fbgn, "; ".join(ROLES[x] for x in r.roles.split(";")), bool(r.ids_agree), *present])
    units, expr, det = [], [], []
    for s, (name, _) in SOURCES.items():
        m = mean[s]
        m.index = m.index.astype(str)
        for u in m.index:
            n = size.get((s, u))
            key = (s, u)
            a = v2.loc[key] if key in v2.index else None
            w = v1.loc[key, "shal_r"] if key in v1.index else None
            units.append([name, u, None if n is None or n < 0 else int(n), a is not None,
                          None if a is None else float(a["A"]), None if w is None or pd.isna(w) else float(w),
                          *[None if a is None or pd.isna(a[f"z_{g}"]) else float(a[f"z_{g}"]) for g in ("SK", "slo", "KCNQ", "Shab", "Shal")]])
            expr.append([name, u, *[float(m.at[u, g]) if g in m.columns else None for g in gl]])
            if s in frac:
                f = frac[s]
                f.index = f.index.astype(str)
                det.append([name, u, *[float(f.at[u, g]) if g in f.columns else None for g in gl]])
    legend = [["Table S4. The single-cell panels, related to Figure 4"],
              ["Genes: the 42 genes of the panel declared before any panel value was computed, with their FlyBase identifiers (the "
               "same in every source that names them) and their roles: the adaptation index (mean z score of SK, slo, KCNQ and "
               "Shab), the Kv4 weight (relative Shal), the channel panel, the potassium channels and the wider panel of the aging "
               "operator. Present: whether each atlas measured the gene."],
              ["Atlas units: every unit of the six sources, with its cells (samples for Davis et al.), whether the model's mapping "
               "uses it, and for those units the adaptation index, the Kv4 weight and the z scores of its five genes over the mapped "
               "units of its source. Units keep the names their atlases publish."],
              ["Expression and Detection: each gene's value in every unit, in each source's own units: " +
               "; ".join(f"{n}: {d}" for n, d in SOURCES.values()) + "."],
              ["Values are those the analyses used (manuscript/results and the study's mapping records); see STAR Methods."]]
    src_cols = [f"In {SOURCES[s][0].split(' (')[0]}" for s in SOURCES]
    return [xlsx.Sheet("Legend", rows=legend, widths=[150], wrap=True),
            xlsx.Sheet("Genes", header=["Gene", "FlyBase ID", "Roles", "Same ID in every source", *src_cols], rows=gene_rows,
                       widths=[12, 14, 60, 12] + [14] * len(src_cols)),
            xlsx.Sheet("Atlas units", header=["Source", "Unit", "Cells", "Used by the mapping", "Adaptation index", "Kv4 weight",
                                              "z SK", "z slo", "z KCNQ", "z Shab", "z Shal"], rows=units,
                       widths=[44, 28, 9, 10, 11, 11, 9, 9, 9, 9, 9]),
            xlsx.Sheet("Expression", header=["Source", "Unit", *gl], rows=expr, widths=[44, 28] + [9] * len(gl)),
            xlsx.Sheet("Detection", header=["Source", "Unit", *gl], rows=det, widths=[44, 28] + [9] * len(gl))]


# ---------------------------------------------------------------- checks and output
def check_labels(sheets, keep_cols=("Record", "Unit")):
    """No development code may survive in a reader-facing cell (record paths, the deposit's identifiers, are exempt)."""
    bad = []
    for sh in sheets:
        skip = {i for i, h in enumerate(sh.header or []) if h in keep_cols}
        for row in sh.rows:
            for i, v in enumerate(row):
                if i in skip or not isinstance(v, str):
                    continue
                for rx in (INTERNAL, DEV_LABELS):
                    for m in rx.finditer(v):
                        bad.append(f"{sh.name}: {m.group(0)!r} in {v[:80]!r}")
                for word, us in british(v):
                    bad.append(f"{sh.name}: British spelling {word!r} in {v[:60]!r}")
    if bad:
        raise SystemExit("development labels or spelling in reader-facing cells:\n  " + "\n  ".join(bad[:30]))


def write(name, sheets):
    check_labels(sheets)
    path = OUT / name
    xlsx.write(path, sheets)
    back = xlsx.read(path)
    for sh in sheets:
        want = ([sh.header] if sh.header else []) + sh.rows
        if len(back[sh.name]) != len(want):
            raise SystemExit(f"{name}: sheet {sh.name} read back with {len(back[sh.name])} rows, not {len(want)}")
    print(f"{path.relative_to(ROOT)}: " + ", ".join(f"{sh.name} {len(sh.rows)} rows" for sh in sheets))


def main():
    write("Table_S2.xlsx", table_s2())
    write("Table_S3.xlsx", table_s3())
    write("Table_S4.xlsx", table_s4())


if __name__ == "__main__":
    main()
