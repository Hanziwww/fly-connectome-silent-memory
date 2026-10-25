r"""Builds manuscript/writing/number_ledger.csv: every number used in the Summary, Highlights, In Brief and Results plans of
manuscript/NARRATIVE_V5_CELL.md, each read from manuscript/results (pinned by the freeze manifest) with its file, column and
row, computed the way the figure that shows it computes it. check_text.py --ledger binds the text to these values.

Each number also gets its one written form (`display`, by FORMATS): thousands commas, a true minus sign, percentages, signed
differences, and P values by the figures' rule (figstyle.fmt_p). build_macros.py turns the displays into the manuscript's
\val{ID} macros, so the LaTeX text never types a result by hand.

The S group (star_methods) binds STAR Methods: its design values come from the study's configurations, declarations and
code (module-level literals read with ast, never imported), and the results recorded only in STAR Methods and Table S2
(class (c), the user's rule) from the study's result tables; its file paths are relative to the repository.

  .\MIGRATION\python\python.exe -B -X utf8 MIGRATION\launch.py --script manuscript\writing\build_ledger.py
"""
import ast
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "manuscript" / "results"
OUT = Path(__file__).resolve().parent / "number_ledger.csv"
ROWS = []
MINUS = "\u2212"
# the written form of each number: int (thousands commas), f1-f3 (decimals), sf1-sf2 (signed decimals), pct0 and pct2
# (percent), p (P value as the figures write it)
FORMATS = {
    **dict.fromkeys(["L01", "L03", "L04", "L05", "L06", "L08", "L09", "L10", "L13", "L14", "L15"], "int"),
    "L02": "f1", "L07": "pct0", "L11": "f3", "L12": "f3", "L16": "p", "L17": "p",
    **dict.fromkeys(["H01", "H02", "H03", "H04", "H05", "H07", "H11", "H12"], "int"),
    "H06": "pct0", "H08": "f3", "H09": "p", "H10": "p", "H13": "pct0", "H14": "pct0",
    **dict.fromkeys(["D03", "D06", "D07", "D12", "D13"], "int"),
    "D01": "f1", "D02": "f1", "D04": "p", "D05": "p", "D08": "p", "D09": "pct0", "D10": "f1", "D11": "f1",
    **dict.fromkeys(["M05", "M06", "M13", "M14"], "int"), **dict.fromkeys(["M01", "M02", "M03", "M15"], "p"),
    **dict.fromkeys(["M08", "M09", "M10", "M11", "M12"], "pct0"), "M04": "f2", "M07": "f2",
    **dict.fromkeys(["A01", "A02", "A04", "A06", "A07", "A08", "A11", "A12"], "int"),
    "A03": "f3", "A05": "p", "A09": "f2", "A10": "pct0", "A13": "p", "A14": "sf1", "A15": "sf1", "A16": "sf2", "A17": "sf2",
    **dict.fromkeys(["E02", "E03", "E05", "E06", "E07", "E08"], "int"),
    **dict.fromkeys(["E11", "E12", "E13", "E17", "E18", "E19", "E20"], "f1"),
    "E01": "pct2", "E04": "f2", "E09": "f2", "E10": "f2", "E14": "f2", "E15": "f2", "E16": "p",
    **dict.fromkeys(["Q01", "Q02", "Q03", "Q04", "Q06", "Q07", "Q22"], "int"),
    **dict.fromkeys(["Q08", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "Q20", "Q23", "Q24"], "f2"),
    "Q05": "p", "Q09": "f1", "Q10": "p", "Q11": "sf2", "Q12": "pct0", "Q13": "pct0", "Q21": "f1",
    **dict.fromkeys(["K01", "K02", "K11", "K12", "K13", "K14"], "int"),
    **dict.fromkeys(["K03", "K04", "K05", "K06"], "f1"), **dict.fromkeys(["K07", "K08", "K09", "K10"], "f2"),
    **dict.fromkeys(["K15", "K16", "K17"], "p"),
    "G01": "int", "G02": "pct0", "G03": "int", "G04": "f1", "G05": "int", "G06": "int", "G07": "pct0", "G08": "pct0", "L18": "int", "D14": "int", "D15": "pct0", "G09": "int",
    "G10": "f1", "G11": "pct0", "L19": "int", "L20": "int", "Q25": "int", "Q26": "int", "K18": "int", "K19": "int",
    "E21": "int", "E22": "int", "G12": "int",
}


def fmt_p(p):
    """P as the figures write it (figstyle.fmt_p): two significant figures below 0.1, two decimals above, 'm x 10^k' below
    0.001; here in plain text, with the exponent after a caret."""
    if p < 0.001:
        m, e = f"{p:.1e}".split("e")
        return f"{m} \u00d7 10^{int(e)}".replace("-", MINUS)
    return f"{p:#.2g}" if p < 0.1 else f"{p:.2f}"


def display(value, fmt):
    """The written form of one number (no unit)."""
    v = float(value)
    if fmt == "p":
        return fmt_p(v)
    if fmt == "sci":                                 # 1e-4 -> 10^-4, 3e-4 -> 3 x 10^-4
        m, e = f"{v:.0e}".split("e")
        return (f"10^{int(e)}" if m == "1" else f"{m} × 10^{int(e)}").replace("-", MINUS)
    if fmt == "int":
        out = f"{round(v):,}"
    elif fmt in ("f1", "f2", "f3"):
        out = f"{v:.{fmt[1]}f}"
    elif fmt in ("sf1", "sf2"):
        out = f"{v:+.{fmt[2]}f}"
    elif fmt in ("pct0", "pct2"):
        out = f"{100 * v:.{fmt[3]}f}%"
    else:
        raise ValueError(f"unknown format {fmt}")
    if float(out.replace(",", "").rstrip("%")) == 0:
        out = out.lstrip("+-")                       # no sign on a value that rounds to zero
    return out.replace("-", MINUS)


def T(rel):
    return pd.read_csv(R / rel, low_memory=False)


def add(key, section, statement, rel, column, row, value, unit=""):
    ROWS.append(dict(id=key, section=section, statement=statement, value=float(value), unit=unit, file=f"manuscript/results/{rel}",
                     column=column, row=row))


def truthy(s):
    return s.astype(str).str.lower().eq("true")


def tests(family):
    t = T("plan_v2/tests.csv")
    return t[t["family"] == family]


def holm(p):
    """Holm step-down adjusted P values (as manuscript/analysis/msa/stats.holm)."""
    p = np.asarray(p, float)
    order = np.argsort(p, kind="mergesort")
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def star_methods(spec):
    """The S group: every number STAR Methods writes, with its repository file, field and row."""
    def S(key, statement, rel, column, row, value, unit="", fmt="int"):
        FORMATS[key] = fmt
        ROWS.append(dict(id=key, section="STAR Methods", statement=statement, value=float(value), unit=unit, file=rel, column=column,
                         row=row))

    def J(rel):
        return json.loads((ROOT / rel).read_text(encoding="utf-8"))

    def C(rel):
        return pd.read_csv(ROOT / rel, low_memory=False)

    def lit(node):
        """A literal, allowing dict(key=literal, ...) and dicts of them (the form of the variant and condition tables)."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args:
            return {kw.arg: lit(kw.value) for kw in node.keywords}
        if isinstance(node, ast.Dict):
            return {lit(k): lit(v) for k, v in zip(node.keys, node.values)}
        return ast.literal_eval(node)

    def const(rel, name):
        """A module-level literal of a Python file, read with ast (the file is never imported)."""
        for n in ast.parse((ROOT / rel).read_text(encoding="utf-8")).body:
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id == name:
                        return lit(n.value)
                    if isinstance(t, ast.Tuple) and name in [e.id for e in t.elts if isinstance(e, ast.Name)]:
                        return lit(n.value.elts[[e.id for e in t.elts].index(name)])
        raise SystemExit(f"{name} is not a module-level literal of {rel}")

    def grep(rel, pattern, group=1):
        m = re.search(pattern, (ROOT / rel).read_text(encoding="utf-8"))
        if not m:
            raise SystemExit(f"{pattern!r} not found in {rel}")
        return float(m.group(group))

    # the model and its training (the spec table of Fig. 1 holds the run's resolved settings)
    F = "manuscript/results/main/fig1/fig1a_model_specification.csv"
    for key, (sec, q), stmt, unit, fmt in [
            ("Sdt", ("simulation", "time_step"), "integration step (ms)", "ms", "int"),
            ("Svrest", ("lif", "v_rest"), "resting and reset potential (mV)", "mV", "int"),
            ("Svth", ("lif", "v_th"), "spike threshold (mV)", "mV", "int"),
            ("Staum", ("lif", "tau_m"), "membrane time constant (ms)", "ms", "int"),
            ("Staus", ("lif", "tau_syn"), "synaptic time constant (ms)", "ms", "int"),
            ("Stref", ("lif", "t_ref"), "refractory period (ms)", "ms", "f1"),
            ("Sdelay", ("lif", "delay"), "synaptic delay (ms)", "ms", "f1"),
            ("Swsyn", ("lif", "w_syn"), "weight per synapse (mV)", "mV", "f3"),
            ("Sjump", ("lif", "adapt_jump"), "adaptation increment per spike (mV)", "mV", "f1"),
            ("Stauad", ("lif", "adapt_tau"), "adaptation time constant (ms)", "ms", "int"),
            ("Sstdcells", ("lif", "cells_with_nonzero_depression"), "cells with short-term depression", "", "int"),
            ("Sstdu", ("lif", "max_depression_u"), "resource used per spike at depressing terminals", "", "f1"),
            ("Sstdtau", ("lif", "std_tau"), "recovery time constant of the depression resource (ms)", "ms", "int"),
            ("Scap", ("lif", "conn_cap"), "synapse cap per connection", "", "int"),
            ("Sstg", ("lif", "same_type_gain"), "gain between cells of one type", "", "f1"),
            ("Snormref", ("lif", "input_norm_ref"), "input synapses above which unitary weights are scaled down", "", "int"),
            ("Sdrive", ("simulation", "input_drive"), "input current per unit board value (mV)", "mV", "int"),
            ("Spools", ("readout", "populations"), "readout groups", "", "int"),
            ("Spoolsize", ("readout", "cells_per_population"), "cells per readout group", "", "int"),
            ("Sfcur", ("readout", "feature_snapshot_current_ms"), "time of the synaptic-input feature (ms)", "ms", "int"),
            ("Sfvolt", ("readout", "feature_snapshot_voltage_ms"), "time of the voltage feature (ms)", "ms", "int"),
            ("Sfeat", ("readout", "feature_dim"), "readout features", "", "int"),
            ("Sclip", ("readout", "feature_clip"), "feature clip (standard deviations)", "", "int"),
            ("Sgroups", ("network", "shared_parameter_groups"), "groups of trained parameters", "", "int")]:
        S(key, stmt, F, "value", f"section {sec}, quantity {q}", spec[(sec, q)], unit, fmt)
    brain = "flybrain/flybrain/brain.py"
    path_gain = const(brain, "DEFAULT_PATH_GAIN")
    S("Spathdn", "rule gain, descending neurons onto the ventral nerve cord", brain, "DEFAULT_PATH_GAIN", "entry 1, factor", path_gain[0][2])
    S("Spathvp", "rule gain, visual projection onto descending neurons", brain, "DEFAULT_PATH_GAIN", "entry 2, factor", path_gain[1][2])
    S("Spathgf", "rule gain, LC4 and LPLC2 onto DNp01", brain, "DEFAULT_TYPE_PATH_GAIN", "entry 1, factor",
      const(brain, "DEFAULT_TYPE_PATH_GAIN")[0][2])
    S("Ssurr", "steepness of the surrogate spike derivative (per mV)", brain, "_SurrogateSpike.backward", "1 / (1 + beta |v|)^2",
      grep(brain, r"grad / \(1\. \+ ([\d.]+) \* voltage\.abs\(\)\)\.square\(\)"))
    bm = "sokoban/lib/brain_model.py"
    S("Sinper", "input cells per board channel and cell", bm, "FullBrain.encode", "repeat_interleave factor",
      grep(bm, r"repeat_interleave\((\d+), 1\)"))
    S("Sgainlo", "lower bound of the trained gains", bm, "Core.project", "clamp of the log gains", grep(bm, r"clamp_\(math\.log\(([\d.]+)\)"),
      fmt="f1")
    S("Sgainhi", "upper bound of the trained gains", bm, "Core.project", "clamp of the log gains",
      grep(bm, r"clamp_\(math\.log\([\d.]+\), math\.log\(([\d.]+)\)\)"), fmt="f2")
    S("Sthr", "bound of the trained threshold offsets (mV)", bm, "Core.project", "threshold_mv clamp",
      grep(bm, r"threshold_mv\.clamp_\(-([\d.]+), [\d.]+\)"), "mV")
    H = "manuscript/results/methods/m1_hyperparameters.csv"
    hp = pd.read_csv(ROOT / H).set_index(["scope", "parameter"])["value"]
    for key, (sc_, par), stmt, fmt in [
            ("Slrbrain", ("study_protocol", "core_lr"), "learning rate of the brain parameters", "sci"),
            ("Slrhead", ("study_protocol", "head_lr"), "learning rate of the readout", "sci"),
            ("Sclipbrain", ("optimization", "gradient_clip_brain"), "gradient-norm clip, brain", "int"),
            ("Sclipread", ("optimization", "gradient_clip_readout"), "gradient-norm clip, readout", "int"),
            ("Sbatch", ("optimization", "batch_boards"), "boards per recomputed batch", "int"),
            ("Stol", ("optimization", "recompute_tolerance_mixed_fp64"), "tolerance of the recomputed features", "sci"),
            ("Smaxmoves", ("curriculum", "action_ceiling"), "move limit of a rollout", "int")]:
        S(key, stmt, H, "value", f"scope {sc_}, parameter {par}", float(hp[(sc_, par)]), fmt=fmt)

    # the planning and memory protocols
    P = "studies/brain-sokoban-v1/protocol.json"
    pr = J(P)
    tiers = np.array(pr["tiers"])
    S("Sseed", "training seed", P, "seed", "", pr["seed"])
    S("Stiers", "tiers of the planning ladder", P, "tiers", "rows", len(tiers))
    S("Sboardlo", "smallest board (cells across)", P, "tiers", "column 1 (min)", tiers[:, 0].min())
    S("Sboardhi", "largest board (cells across)", P, "tiers", "column 1 (max)", tiers[:, 0].max())
    S("Sboxlo", "fewest boxes", P, "tiers", "column 2 (min)", tiers[:, 1].min())
    S("Sboxhi", "most boxes", P, "tiers", "column 2 (max)", tiers[:, 1].max())
    S("Soptlo", "shortest certified optimum (moves)", P, "tiers", "column 3 (min)", tiers[:, 2].min())
    S("Ssteps", "updates per behavioral assessment", P, "steps_per_assessment", "", pr["steps_per_assessment"])
    for i, sec in enumerate(pr["segments_seconds"], 1):
        S(f"Sseg{i}", f"wall-clock segment {i} (min)", P, "segments_seconds", f"entry {i} / 60", sec / 60, "min")
    S("Sprog", "margin progress needed to extend a run", P, "minimum_margin_progress", "", pr["minimum_margin_progress"], fmt="f2")
    for i, d in enumerate(pr["delay_ms"], 1):
        S(f"Sdel{i}", f"memory-task delay {i} (ms)", P, "delay_ms", f"entry {i}", d, "ms")
    for i, w in enumerate(pr["prefix_windows"], 1):
        S(f"Spfx{i}", f"input windows, option {i}", P, "prefix_windows", f"entry {i}", w)
    HY = "sokoban/hybrid_tasks.py"
    S("Starget", "target margin at the new start", HY, "TARGET", "", const(HY, "TARGET"), fmt="f2")
    S("Stemp", "softplus temperature", HY, "TEMPERATURE", "", const(HY, "TEMPERATURE"), fmt="f2")
    S("Sfloor", "margin floor of learned decisions", HY, "ANCHOR_FLOOR", "", const(HY, "ANCHOR_FLOOR"), fmt="f2")
    RT = "studies/brain-supplement-v2/code/replicate_training.py"
    S("Sjit", "readout noise of a replicate (fraction of each tensor's SD)", RT, "JITTER_SD", "", const(RT, "JITTER_SD"), fmt="f1")
    X3 = "studies/brain-supplement-v2/code/x3_jittered_training.py"
    S("Sjitwin", "input windows of jittered training, the option not in the protocol", X3, "PREFIXES", "entry 5", const(X3, "PREFIXES")[4])

    # circuits, routes and robustness
    CE = "manuscript/results/plan_v2/fig3_cue_energy.csv"
    ce = C(CE)
    S("Skc", "Kenyon cells", CE, "cells", "circuit Kenyon cells", ce.loc[ce.circuit == "Kenyon cells", "cells"].iloc[0])
    S("Saln", "GABAergic antennal-lobe local neurons", CE, "cells", "circuit GABAergic ALLNs", ce.loc[ce.circuit == "GABAergic ALLNs", "cells"].iloc[0])
    DP = "studies/brain-pathology-v1/code/phase3_dependency_prepare.py"
    S("Snullsets", "null sets per removed population", DP, "N_NULL", "", const(DP, "N_NULL"))
    VA = "studies/brain-pathology-v1/code/variants.py"
    var = const(VA, "VARIANTS")
    S("Sstglo", "same-type gain, low variant", VA, "VARIANTS", "same-type-0.05", var["same-type-0.05"]["same_type_gain"], fmt="f2")
    S("Sstghi", "same-type gain, high variant", VA, "VARIANTS", "same-type-0.2", var["same-type-0.2"]["same_type_gain"], fmt="f1")
    S("Scaplo", "connection cap, low variant", VA, "VARIANTS", "conn-cap-30", var["conn-cap-30"]["conn_cap"])
    S("Scaphi", "connection cap, high variant", VA, "VARIANTS", "conn-cap-120", var["conn-cap-120"]["conn_cap"])
    DY = "studies/brain-pathology-v1/code/dy2_perturbation.py"
    S("Sthsd", "SD of the per-episode threshold offset when training across states (microvolts)", DY, "CONDITIONS",
      "th0.001, threshold_sd_mv x 1000", const(DY, "CONDITIONS")["th0.001"]["threshold_sd_mv"] * 1000, "uV")
    S("Sheldq", "held-out threshold offsets (multiples of the SD)", DY, "QUARTILE", "", const(DY, "QUARTILE"), fmt="f3")
    AD = "studies/brain-pathology-v1/code/analyze_dy2.py"
    S("Sed", "memory use at which the access window and cell-loss tolerance are read", AD, "THRESHOLD", "", const(AD, "THRESHOLD"), fmt="f1")

    # single-cell atlases
    GA = "studies/brain-pathology-v1/results/sc_lineage_vnc_v3_gates.json"
    ga = J(GA)
    S("Shvg", "variable genes of the bridge", GA, "variable_genes", "", ga["variable_genes"])
    V3 = "studies/brain-pathology-v1/code/sc_lineage_vnc_v3.py"
    S("Spcs", "principal components of the bridge", V3, "N_PCS", "", const(V3, "N_PCS"))
    S("Sres", "Leiden resolution of the bridge", GA, "leiden_resolution", "", ga["leiden_resolution"], fmt="f1")
    S("Sclus", "Leiden clusters of the bridge", GA, "leiden_clusters", "", ga["leiden_clusters"])
    PN = "studies/brain-pathology-v1/tasks/sc-panels-v1.json"
    S("Sgenes", "genes of the declared panel", PN, "panels.genes", "entries", len(J(PN)["panels"]["genes"]))
    VL = "studies/brain-pathology-v1/tasks/sc-variant-lesion.json"
    vl = J(VL)
    S("Shetlo", "adaptation scaling, lowest third of the adaptation index", VL, "value", "spec 1", vl[0]["value"], fmt="f1")
    S("Shethi", "adaptation scaling, highest third", VL, "value", "spec 2", vl[1]["value"], fmt="f1")
    VD = "studies/brain-pathology-v1/tasks/sc-variant-design.json"
    vd = J(VD)
    S("Stypes", "mapped cell types ranked by the adaptation index", VD, "tertiles.types", "", vd["tertiles"]["types"])
    S("Shetn", "scene-2 runs per heterogeneous brain", VD, "runs", "brain variant, family S1 (count)",
      sum(r["brain"] == "variant" and r["family"] == "S1" for r in vd["runs"]))

    # pathologies and the lesion link
    CC = "studies/brain-pathology-v1/runs/phase2-credibility/p05/c5_c6.json"
    cc = J(CC)
    cal = cc["calibration_rheobase"]
    S("Scal", "threshold offset of the Kv4 loss at dose 1 (mV)", CC, "calibration_rheobase.offset_mv", "", cal["offset_mv"], "mV", fmt="f2")
    S("Sramp", "ramp-current ratio of the model at dose 1", CC, "calibration_rheobase.ramp_current_ratio", "", cal["ramp_current_ratio"], fmt="f3")
    S("Srampfly", "ramp-current ratio of Abeta42 flies", CC, "calibration_rheobase.reported_current_ratio", "", cal["reported_current_ratio"], fmt="f2")
    dose1 = lambda case: next(r["value"] for r in cc[case]["rows"] if r["dose"] == 1.0)       # noqa: E731
    S("Skcloss", "share of Kenyon cells removed at dose 1 (late Abeta42)", CC, "C04.rows", "dose 1, value", dose1("C04"), fmt="pct0")
    S("Sfxrel", "release factor of GABAergic antennal-lobe local neurons at dose 1 (fragile X)", CC, "C13.rows", "dose 1, value", dose1("C13"), fmt="f1")
    LA = "manuscript/results/plan_v2/fig7_lost_access.csv"
    la = C(LA)
    S("Srepb", "replicates per dose and learner, learning under the lesion", LA, "replicate", "regime learning under the lesion (max)",
      la.loc[la.regime == "learning under the lesion", "replicate"].max())
    RC = "studies/brain-pathology-v1/code/phase3_regime_c.py"
    S("Supd", "updates of relearning", RC, "UPDATES", "", const(RC, "UPDATES"))
    OL = "studies/brain-pathology-v1/code/phase3_olfactory.py"
    S("Sfolds", "cross-validation folds of the olfactory decoder", OL, "FOLDS", "", const(OL, "FOLDS"))
    CO = "studies/brain-pathology-v1/code/phase3_comparison.py"
    S("Snperm", "draws of the case-comparison null", CO, "N_PERM", "", const(CO, "N_PERM"))
    LD = "studies/brain-pathology-v1/tasks/lesion-link-design.json"
    S("Srelearn", "relearned readouts per dose in the lesion link", LD, "readouts", "kind relearned, dose 1 (count)",
      sum(r["kind"] == "relearned" and r["dose"] == 1.0 for r in J(LD)["readouts"]))

    # statistics
    MS = "manuscript/analysis/msa/stats.py"
    S("Sexact", "largest n resolved by exact enumeration", MS, "exact tests", "if n <= N", grep(MS, r"if n <= (\d+):"))
    S("Sdraws", "draws of a resampling test", MS, "N_RESAMPLES", "", const(MS, "N_RESAMPLES"))
    B2 = "studies/brain-pathology-v1/code/analyze_batch2.py"
    S("Smargin", "equivalence margin (percentage points)", B2, "MARGIN", "x 100", const(B2, "MARGIN") * 100, "pp")

    # results recorded only in STAR Methods and Table S2 (class (c))
    SUP, PAT = "studies/brain-supplement-v2/results/", "studies/brain-pathology-v1/results/"
    TS = "manuscript/results/plan_v2/tests.csv"
    ts = C(TS)
    x8 = C(SUP + "x8_runs.csv")
    S("Sx8runs", "runs from a neutral brain that learned", SUP + "x8_runs.csv", "status", "learned_solved (count)", (x8.status == "learned_solved").sum())
    x8t = C(SUP + "x8_tests.csv")
    S("Sx8p", "stratified log-rank P, readout alone vs brain and readout from a neutral brain", SUP + "x8_tests.csv", "p_value",
      "family F13.X8.primary", x8t.loc[x8t.family == "F13.X8.primary", "p_value"].iloc[0], fmt="p")
    cx = C(SUP + "campaign_x1_summary.csv").set_index("learner")
    for key, lr in (("Sstag0", "M0-joint"), ("Sstag1", "S1-head_only"), ("Sstag2", "S2-head_only")):
        S(key, f"runs stopped by stagnation, {lr}", SUP + "campaign_x1_summary.csv", "planned - learned_solved", f"learner {lr}",
          cx.loc[lr, "planned"] - cx.loc[lr, "learned_solved"])
    ZT = "manuscript/results/main/fig2/fig2e_zero_update_transfer.csv"
    zt = C(ZT)
    hs = zt[zt.row_type == "summary_heldout"]
    if hs.numerator.max() != 0 or hs.denominator.nunique() != 1:
        raise SystemExit("zero-update transfer: the summary rows no longer read 0 of the same held-out set")
    S("Sheldout", "held-out planning maps", ZT, "denominator", "row_type summary_heldout (every model)", hs.denominator.iloc[0])
    S("Sstages", "stage checkpoints replayed from the original start", SUP + "x9_stages.csv", "rows", "", len(C(SUP + "x9_stages.csv")))
    G4 = "studies/brain-supplement-v2/runs/g4-linear-response/result.json"
    g4row = next(r for r in J(G4)["rows"] if r["prefix"] == 4 and r["delay"] == 160)
    S("Sadaptonly", "cosine of the adaptation-only transplant with the native decision (160 ms, 4 windows)", G4, "adapt_eps1_cos",
      "prefix 4, delay 160", g4row["adapt_eps1_cos"], fmt="f2")
    x45 = C(SUP + "x4_x5_tests.csv")
    content = x45.loc[x45.family == "F13.X4.content", "p_value"].to_numpy()
    S("Sx4sig", "models whose cue shift beats norm-matched scrambles (Holm)", SUP + "x4_x5_tests.csv", "p_value",
      "family F13.X4.content, Holm-adjusted < 0.05 (count)", int((holm(content) < 0.05).sum()))
    S("Sx4n", "models of the scramble test", SUP + "x4_x5_tests.csv", "p_value", "family F13.X4.content (rows)", len(content))
    S("Svisp", "P, hidden vs visible target under swaps", SUP + "x4_x5_tests.csv", "p_value", "family F13.X5.visibility",
      x45.loc[x45.family == "F13.X5.visibility", "p_value"].iloc[0], fmt="p")
    bt = C(PAT + "batch2_tests.csv")
    dn = bt.loc[bt.family == "F12.dependency.null", "estimate"]
    S("Sdeplo", "extra cost of matched random removal over the populations, lowest (percentage points)", PAT + "batch2_tests.csv",
      "estimate x 100", "family F12.dependency.null (min)", 100 * dn.min(), "pp")
    S("Sdephi", "the same, highest (percentage points)", PAT + "batch2_tests.csv", "estimate x 100", "family F12.dependency.null (max)",
      100 * dn.max(), "pp")
    dc = C(PAT + "dependency_conditions.csv")
    S("Sdnsolved", "trained histories solved with every descending neuron removed", PAT + "dependency_conditions.csv", "solved",
      "condition descending-1", dc.loc[dc.condition == "descending-1", "solved"].iloc[0])
    o14 = C(PAT + "o14_runs.csv")
    u = o14.loc[o14.variant == "same-type-0.05", "untrained_net_use"]
    S("So14lo", "untrained net use at same-type gain 0.05, lowest run", PAT + "o14_runs.csv", "untrained_net_use", "variant same-type-0.05 (min)",
      u.min(), fmt="sf2")
    S("So14hi", "the same, highest run", PAT + "o14_runs.csv", "untrained_net_use", "variant same-type-0.05 (max)", u.max(), fmt="sf2")
    an = ts[ts.family.isin(["F14.SC.trace.v2", "F14.SC.trace.v2.secondary", "F14.SC.trace.v2.vnc"]) & ts.test.str.endswith(":A:mapping-null")]
    S("Smaplo", "adjusted P, adaptation index against the mapping null, lowest", TS, "p_adjusted", "F14.SC.trace.v2*, *:A:mapping-null (min)",
      an.p_adjusted.min(), fmt="p")
    S("Smaphi", "the same, highest", TS, "p_adjusted", "F14.SC.trace.v2*, *:A:mapping-null (max)", an.p_adjusted.max(), fmt="p")
    vs = ts[ts.family == "F14.SC.variant.secondary"]
    vn = vs[vs.test.str.startswith("variant vs null brain") & ~vs.test.str.endswith(": trained_net_use")]
    S("Shetplo", "adjusted P, heterogeneous vs null brain (untrained use, lock, access window), lowest", TS, "p_adjusted",
      "F14.SC.variant.secondary, variant vs null brain (min)", vn.p_adjusted.min(), fmt="p")
    S("Shetphi", "the same, highest", TS, "p_adjusted", "F14.SC.variant.secondary, variant vs null brain (max)", vn.p_adjusted.max(), fmt="p")
    S("Shets1p", "adjusted P, heterogeneous brain in scene 2", TS, "p_adjusted", "F14.SC.variant.secondary, variant S1",
      vs.loc[vs.test.str.startswith("variant S1"), "p_adjusted"].max(), fmt="p")
    x3 = ts[ts.family == "F13.X3.descriptive"].set_index("test")["estimate"]
    S("Sx3cost", "fall in trained-timing use with jittered training (absolute)", TS, "estimate", "F13.X3.descriptive, trained_net_use",
      abs(x3["trained_net_use: jittered minus fixed (two-sided sign-flip)"]), fmt="f2")
    S("Sx3upd", "extra updates of jittered training", TS, "estimate", "F13.X3.descriptive, updates",
      x3["updates: jittered minus fixed (two-sided sign-flip)"])
    xu = ts[(ts.family == "F13.X1.untrained") & (ts.test == "M0-joint")]
    S("Sjointun", "Holm P, untrained use of the scene-1 brain-and-readout learner", TS, "p_adjusted", "F13.X1.untrained, M0-joint",
      xu.p_adjusted.iloc[0], fmt="p")
    dm = C(PAT + "dy2_models.csv")
    x1e = dm.loc[dm.group == "x1", "ed50_c01_mv"] * 1000
    S("Sed50lo", "threshold lowering that halves memory use, lowest replicate (microvolts)", PAT + "dy2_models.csv", "ed50_c01_mv x 1000",
      "group x1 (min)", x1e.min(), "uV", fmt="f2")
    S("Sed50hi", "the same, highest replicate (microvolts)", PAT + "dy2_models.csv", "ed50_c01_mv x 1000", "group x1 (max)", x1e.max(), "uV",
      fmt="f2")
    th = dm[dm.condition == "th0.001"]
    S("Sdy2tr", "trained net use, training across threshold states (mean)", PAT + "dy2_models.csv", "trained_net_use", "condition th0.001 (mean)",
      th.trained_net_use.mean(), fmt="f2")
    S("Sdy2un", "untrained net use, training across threshold states (mean)", PAT + "dy2_models.csv", "untrained_net_use",
      "condition th0.001 (mean)", th.untrained_net_use.mean(), fmt="f2")
    S("Sdy2upd", "updates, training across states over training in one state (ratio of means)", PAT + "dy2_models.csv", "updates",
      "condition th0.001 over group x1 (means)", th.updates.mean() / dm.loc[dm.group == "x1", "updates"].mean(), fmt="f1")
    m1 = C(PAT + "dy1_memory.csv")
    small = m1[m1.shift_mv.abs() <= 0.003 + 1e-12]
    S("Srate0lo", "mean task rate for threshold lowerings up to 3 microvolts, lowest (Hz)", PAT + "dy1_memory.csv", "task_mean_hz",
      "|shift_mv| <= 0.003 (min)", small.task_mean_hz.min(), "Hz", fmt="f2")
    S("Srate0hi", "the same, highest (Hz)", PAT + "dy1_memory.csv", "task_mean_hz", "|shift_mv| <= 0.003 (max)", small.task_mean_hz.max(), "Hz",
      fmt="f2")
    share = lambda mv: m1.loc[np.isclose(m1.shift_mv, mv) & (m1.condition != "healthy"), "solved_share"].iloc[0]   # noqa: E731
    S("Sdy1h", "share of trained histories solved, healthy", PAT + "dy1_memory.csv", "solved_share", "condition healthy",
      m1.loc[m1.condition == "healthy", "solved_share"].iloc[0], fmt="f2")
    S("Sdy1sa", "the smaller threshold lowering of the micro series (mV)", PAT + "dy1_memory.csv", "shift_mv", "micro, first", 1e-4, "mV", fmt="sci")
    S("Sdy1a", "share solved at that lowering", PAT + "dy1_memory.csv", "solved_share", "shift_mv -0.0001", share(-1e-4), fmt="f2")
    S("Sdy1sb", "the next lowering of the micro series (mV)", PAT + "dy1_memory.csv", "shift_mv", "micro, second", 3e-4, "mV", fmt="sci")
    S("Sdy1b", "share solved at that lowering", PAT + "dy1_memory.csv", "solved_share", "shift_mv -0.0003", share(-3e-4), fmt="f2")
    S("Sdosemin", "smallest lowering of the low dose extension (mV)", PAT + "dy1_memory.csv", "shift_mv", "source low extension, dose > 0 (min |shift|)",
      m1.loc[(m1.source == "low extension") & (m1.dose > 0), "shift_mv"].abs().min(), "mV", fmt="f2")
    DV = "studies/brain-pathology-v1/runs/fig-divergence-cells/d160-p4-cue0.json"
    dv = J(DV)
    S("Sdivuv", "persistent threshold lowering of the divergence record (microvolts)", DV, "shift_mv", "x -1000", -1000 * dv["shift_mv"], "uV", fmt="f1")
    S("Sdivcells", "cells diverged by the end of the first decision", DV, "cells_diverged", "", dv["cells_diverged"])
    dd = C(PAT + "dy1_divergence.csv")
    row = dd[(dd.kind == "threshold") & (dd.delay_ms == 160) & (dd.prefix_windows == 4) & (dd.cue == 0) & np.isclose(dd["size"], 3e-4)].iloc[0]
    S("Sdivms", "first divergence after input resumed (ms)", PAT + "dy1_divergence.csv", "first_divergent_ms - delay_ms",
      "threshold 0.0003 mV, 160 ms, 4 windows, cue 0", row.first_divergent_ms - row.delay_ms, "ms")
    df = C(PAT + "phase4_dose_fits.csv")
    S("Sed50rm", "share of cells whose random removal halves behavior (logistic fit)", PAT + "phase4_dose_fits.csv", "ed50",
      "set main + low (post hoc), series random removal",
      df.loc[(df.set == "main + low (post hoc)") & (df.series == "random removal"), "ed50"].iloc[0], fmt="pct2")
    rc = ts[ts.family == "F12.regimeC"]
    S("Sjointrec", "recovered share, relearning with the brain minus the readout alone", TS, "estimate", "F12.regimeC, joint minus readout-only",
      rc.loc[rc.test.str.startswith("recovery fraction: joint"), "estimate"].iloc[0], fmt="f2")
    S("Sregbp", "smallest adjusted P of the per-dose tests of learning under the lesion", TS, "p_adjusted", "F12.regimeB (min)",
      ts.loc[ts.family == "F12.regimeB", "p_adjusted"].min(), fmt="p")


def main():
    # ---- the model (Fig. 1)
    spec = pd.to_numeric(T("main/fig1/fig1a_model_specification.csv").set_index(["section", "quantity"])["value"], errors="coerce")
    S = "main/fig1/fig1a_model_specification.csv"
    add("L01", "Summary; Results 1", "neurons in the model", S, "value", "section network, quantity neurons", spec[("network", "neurons")])
    add("L02", "Results 1", "stored connectome entries, millions (fixed connections)", S, "value", "network, stored_connectome_entries",
        spec[("network", "stored_connectome_entries")] / 1e6, "million")
    add("L03", "Results 1", "visual input cells", S, "value", "network, input_cells", spec[("network", "input_cells")])
    add("G12", "Video legends", "the decision window, one simulation per move (ms)", S, "value", "simulation, decision_window",
        spec[("simulation", "decision_window")], "ms")
    add("L04", "Results 1", "readout cells", S, "value", "network, observed_cells", spec[("network", "observed_cells")])
    b = T("main/fig1/fig1b_parameter_budget.csv")
    B = "main/fig1/fig1b_parameter_budget.csv"
    add("L05", "Results 1", "trained gains and thresholds", B, "n_elements", "model_kind connectome_linear, component brain (sum)",
        b[(b.model_kind == "connectome_linear") & (b.component == "brain")]["n_elements"].sum())
    add("L06", "Results 1", "trained readout weights", B, "n_elements", "connectome_linear, readout, trainable_in_joint (sum)",
        b[(b.model_kind == "connectome_linear") & (b.component == "readout") & truthy(b.trainable_in_joint)]["n_elements"].sum())
    g = T("plan_v2/graph_paths.csv")
    k1 = float(g[(g.pathway == "visual inputs -> readout") & (g.k == 1)]["targets_reached_within_k"].iloc[0])
    add("L07", "Results 1", "share of readout cells one synapse from the visual input", "plan_v2/graph_paths.csv", "targets_reached_within_k",
        "pathway visual inputs -> readout, k 1", k1)
    rt = T("plan_v2/fig6_routes.csv")
    add("L08", "Results 1", "readout cells with a direct input synapse", "plan_v2/fig6_routes.csv", "hops_0", "role readout, hops_0 = 1 (count)",
        int(((rt.role == "readout") & (rt.hops_0 == 1)).sum()))
    x10 = T("plan_v2/x10_maps.csv")
    add("L09", "Results 1", "maps in the planning ladder", "plan_v2/x10_maps.csv", "map", "distinct maps", x10["map"].nunique())
    add("L10", "Results 1", "replicate sets", "plan_v2/x10_maps.csv", "replicate", "distinct replicate sets", x10["replicate"].nunique())
    rho = tests("F13.X10.fig2")
    rho = rho[rho.test.str.startswith("replicate set")]
    add("L11", "Results 1", "Spearman rho, learning cost vs difficulty, lowest set", "plan_v2/tests.csv", "estimate", "family F13.X10.fig2, replicate sets (min)", rho.estimate.min())
    add("L12", "Results 1", "Spearman rho, highest set", "plan_v2/tests.csv", "estimate", "family F13.X10.fig2, replicate sets (max)", rho.estimate.max())
    x9 = T("plan_v2/x9_pairings.csv")
    fin = x9[x9.combination == "11"]
    mid = x9[~x9.combination.isin(["11"])]
    add("L13", "Results 1", "maps solved by the final brain with the final readout", "plan_v2/x9_pairings.csv", "solved", "combination 11 (sum)", truthy(fin.solved).sum())
    add("L14", "Results 1", "most maps solved by any other pairing", "plan_v2/x9_pairings.csv", "solved", "combinations other than 11 (max of sums)",
        mid.assign(s=truthy(mid.solved)).groupby("combination")["s"].sum().max())
    x1 = T("plan_v2/x1_runs.csv")
    add("L15", "Results 1", "replicates of the four memory learners", "plan_v2/x1_runs.csv", "replicate", "rows", len(x1))
    per = x1.groupby(["scene", "trained"]).size()
    add("L19", "Figure legends", "replicates of each scene-1 learner", "plan_v2/x1_runs.csv", "replicate", "rows per scene and trained parameters (max)",
        int(per.max()))
    add("L20", "Figure legends", "replicates of each scene-2 and scene-3 learner", "plan_v2/x1_runs.csv", "replicate",
        "rows per scene and trained parameters (min)", int(per.min()))
    lock = pd.concat([tests("F13.X1.lock"), tests("F13.X1.trained")])
    add("L16", "Results 1", "Holm P, trained use and lock, lowest", "plan_v2/tests.csv", "p_adjusted", "families F13.X1.lock and .trained (min)", lock.p_adjusted.min())
    add("L17", "Results 1", "Holm P, trained use and lock, highest", "plan_v2/tests.csv", "p_adjusted", "families F13.X1.lock and .trained (max)", lock.p_adjusted.max())
    # ---- the trace (Fig. 2)
    de = T("plan_v2/fig2_delay_end_summary.csv")
    add("H01", "Results 2", "last differential spike after cue offset, earliest learner (ms)", "plan_v2/fig2_delay_end_summary.csv",
        "last_differential_spike_ms_after_cue_offset", "four learners (min)", de.last_differential_spike_ms_after_cue_offset.min(), "ms")
    add("H02", "Results 2", "the same, latest learner (ms)", "plan_v2/fig2_delay_end_summary.csv", "last_differential_spike_ms_after_cue_offset",
        "four learners (max)", de.last_differential_spike_ms_after_cue_offset.max(), "ms")
    tt = T("plan_v2/trial_timecourse.csv")
    add("H03", "Results 2", "cells holding the trace at the end of the delay", "plan_v2/trial_timecourse.csv", "trace_cells", "ms 200 (sum over superclasses)",
        tt.loc[tt.ms == 200, "trace_cells"].sum())
    fu4 = T("followup/fu4_state_summary.csv")
    add("H04", "Results 2", "adaptation decay time constant, lowest (ms)", "followup/fu4_state_summary.csv", "tau_eff_adapt_ms", "all models (min)", fu4.tau_eff_adapt_ms.min(), "ms")
    add("H05", "Results 2", "adaptation decay time constant, highest (ms)", "followup/fu4_state_summary.csv", "tau_eff_adapt_ms", "all models (max)", fu4.tau_eff_adapt_ms.max(), "ms")
    dr = T("followup/fu9_drive_split.csv")
    dr = dr[dr.visibility == "hidden"]
    four = ["remote-dynamic-M0-head_only", "remote-dynamic-M0-joint", "supp-S1-ring-brain", "supp-S2-context-brain"]
    dr = dr[dr.model.isin(four)]
    add("H06", "Results 2", "decision-time drive from synapses, lowest share over all timings", "followup/fu9_drive_split.csv", "share_drive_synaptic",
        "four learners, hidden goal (min)", dr.share_drive_synaptic.min())
    add("H07", "Results 2", "timings per learner", "followup/fu9_drive_split.csv", "delay_ms, prefix_windows", "rows per learner and context (max)",
        dr.groupby(["model", "context"]).size().max())
    tr = T("plan_v2/fig2_transplant.csv")
    add("H08", "Results 2", "cosine of the transplanted with the native decision difference (full state, from 20 ms)", "plan_v2/fig2_transplant.csv",
        "full_state_cosine", "delay_ms >= 20 (min)", tr.loc[tr.delay_ms >= 20, "full_state_cosine"].min())
    sw = tests("F13.X5.swap")
    add("H09", "Results 2", "Holm P, opposite-cue swaps, lowest", "plan_v2/tests.csv", "p_adjusted", "family F13.X5.swap (min)", sw.p_adjusted.min())
    add("H10", "Results 2", "Holm P, opposite-cue swaps, highest", "plan_v2/tests.csv", "p_adjusted", "family F13.X5.swap (max)", sw.p_adjusted.max())
    st = T("followup/fu4_state_dynamics.csv")
    delay = int(st.delay_ms.max())
    q = st[st.model.isin(four) & (st.visibility == "hidden") & (st.delay_ms == delay) & (st.prefix_windows == 4)]
    silent = []
    for _, g_ in q.groupby(["model", "context"]):
        dl = g_[g_.segment == "delay"]
        nz = dl.loc[dl.energy_spikes > 0, "ms"]
        silent.append(int(dl.ms.max() - (nz.max() if len(nz) else dl.ms.min())))
    add("H11", "Results 2", "ms without a differential spike in the 640-ms delay, shortest", "followup/fu4_state_dynamics.csv", "energy_spikes",
        f"four learners, delay {delay} ms, prefix 4: delay end minus last nonzero ms (min)", min(silent), "ms")
    add("H12", "Results 2", "the longest delay (ms)", "followup/fu4_state_dynamics.csv", "delay_ms", "max", delay, "ms")
    ty = T("plan_v2/fig2_cue_flow_types.csv").set_index("type")
    e = T("plan_v2/fig2_cue_flow_edges.csv")
    ro = set(ty[ty.role.astype(str).str.contains("readout")].index)
    into = e[e.post.isin(ro) & (e.pre != e.post) & (e.pre != "untyped") & (e.post != "untyped")]
    held = into.pre.map(ty.hold) > 0
    add("H13", "Results 2", "share of the cue drive into readout types from types that held the trace", "plan_v2/fig2_cue_flow_edges.csv", "abs_drive",
        "post a readout type, pre held the trace (fig2_cue_flow_types.hold > 0) over all", into.loc[held, "abs_drive"].sum() / into.abs_drive.sum())
    add("H14", "Results 2", "share of cell types that held the trace", "plan_v2/fig2_cue_flow_types.csv", "hold", "hold > 0 (share of types)", (ty.hold > 0).mean())
    # ---- distributed (Fig. 3)
    w = T("plan_v2/fig3_set_wiring.csv")
    for k_, sc in (("D01", "scene 1"), ("D02", "scene 2")):
        q_ = w[w.ensemble_scene == sc]
        ens = float(q_.loc[q_.set == "ensemble", "internal_connections"].iloc[0])
        add(k_, "Results 3", f"ensemble internal connections over the mean of its 20 matched sets, {sc}", "plan_v2/fig3_set_wiring.csv",
            "internal_connections", f"{sc}: ensemble / mean(matched)", ens / q_.loc[q_.set != "ensemble", "internal_connections"].mean())
    add("D03", "Results 3", "cells in the scene-1 ensemble (top 1%)", "plan_v2/fig3_set_wiring.csv", "cells", "scene 1, ensemble",
        w.loc[(w.ensemble_scene == "scene 1") & (w.set == "ensemble"), "cells"].iloc[0])
    add("D04", "Results 3", "Holm P, silencing the ensemble", "plan_v2/tests.csv", "p_adjusted", "family F13.X6.ensemble (min)", tests("F13.X6.ensemble").p_adjusted.min())
    add("D05", "Results 3", "Holm P, cutting its internal synapses", "plan_v2/tests.csv", "p_adjusted", "family F13.X7.edges (min)", tests("F13.X7.edges").p_adjusted.min())
    rm = T("plan_v2/fig3_removals.csv")
    h_ = rm[rm.kind == "healthy"].iloc[0]
    add("D06", "Results 3", "trained histories solved by the healthy network", "plan_v2/fig3_removals.csv", "solved", "kind healthy", h_.solved)
    add("D07", "Results 3", "trained histories", "plan_v2/fig3_removals.csv", "histories", "kind healthy", h_.histories)
    add("D08", "Results 3", "TOST Holm P, populations removed equal healthy", "plan_v2/tests.csv", "p_adjusted", "family F12.dependency.equivalence (max)",
        tests("F12.dependency.equivalence").p_adjusted.max())
    ce = T("plan_v2/fig3_cue_energy.csv")
    rc_ = ce[ce.circuit == "readout cells"]
    add("D09", "Results 3", "share of the cue difference in the readout cells, scene 1", "plan_v2/fig3_cue_energy.csv", "share_of_cue_energy",
        "circuit readout cells, scene 1", rc_.loc[rc_.scene == "scene 1", "share_of_cue_energy"].iloc[0])
    add("D10", "Results 3", "readout-cell enrichment of the cue, lowest scene", "plan_v2/fig3_cue_energy.csv", "enrichment", "readout cells (min)", rc_.enrichment.min())
    add("D11", "Results 3", "readout-cell enrichment of the cue, highest scene", "plan_v2/fig3_cue_energy.csv", "enrichment", "readout cells (max)", rc_.enrichment.max())
    so = T("plan_v2/fig3_structure.csv")
    add("D12", "Results 3", "runs under the connection-rule variants that learn", "plan_v2/fig3_structure.csv", "learned", "learned true (count)", truthy(so.learned).sum())
    add("D13", "Results 3", "connection-rule variants", "plan_v2/fig3_structure.csv", "variant", "distinct variants", so.variant.nunique())
    # ---- SK (Fig. 4)
    tt4 = T("plan_v2/fig4_trace_tests.csv")
    ms = tt4[tt4.null.astype(str).str.contains("matched")]
    mn = tt4[tt4.null.astype(str).str.contains("mapping") & (tt4.statistic_name.astype(str).str.contains("SK"))]
    add("M01", "Results 4", "adjusted P against matched sets, highest", "plan_v2/fig4_trace_tests.csv", "p_adjusted", "null matched sets (max)", ms.p_adjusted.max())
    add("M02", "Results 4", "adjusted P of SK against the mapping null, lowest", "plan_v2/fig4_trace_tests.csv", "p_adjusted", "null mapping, SK (min)", mn.p_adjusted.min())
    add("M03", "Results 4", "adjusted P of SK against the mapping null, highest", "plan_v2/fig4_trace_tests.csv", "p_adjusted", "null mapping, SK (max)", mn.p_adjusted.max())
    hl = T("plan_v2/fig4_hemilineages.csv")
    add("M04", "Results 4", "Spearman rho, carrier share vs SK across hemilineages", "plan_v2/fig4_hemilineages.csv", "carrier_share, z_SK", "21 units",
        spearmanr(hl.carrier_share, hl.z_SK).statistic)
    add("M05", "Results 4", "VNC hemilineages in the bridge", "plan_v2/fig4_hemilineages.csv", "unit", "rows", len(hl))
    add("M06", "Results 4", "hemilineages whose transcriptome transmitter equals the EM transmitter", "plan_v2/fig4_hemilineages.csv",
        "em_transmitter, transcriptome_transmitter", "equal (count)", (hl.em_transmitter == hl.transcriptome_transmitter).sum())
    cl = T("plan_v2/fig4_bridge_clusters.csv")
    add("M07", "Results 4", "Spearman rho, published vs recovered atlas cluster sizes", "plan_v2/fig4_bridge_clusters.csv", "published_cells, recovered_cells",
        "120 clusters", spearmanr(cl.published_cells, cl.recovered_cells).statistic)
    cov = T("plan_v2/fig4_coverage.csv").set_index("cell_set")
    car, vnc = cov.loc["trace carriers (m0)"], cov.loc["trace carriers (m0), VNC compartment"]
    add("M08", "Results 4", "scene-1 carriers mapped at the cell-type tier", "plan_v2/fig4_coverage.csv", "type_or_fuzzy", "trace carriers (m0)", car.type_or_fuzzy)
    add("M09", "Results 4", "scene-1 carriers mapped with the hemilineage tiers", "plan_v2/fig4_coverage.csv", "type_lineage", "trace carriers (m0)", car.type_lineage)
    add("M10", "Results 4", "VNC carriers mapped", "plan_v2/fig4_coverage.csv", "type_lineage", "trace carriers (m0), VNC compartment", vnc.type_lineage)
    at = T("plan_v2/fig4_access_tiles.csv")
    add("M11", "Results 4", "share of a carrier type's cue drive into readout types, lowest", "plan_v2/fig4_access_tiles.csv", "drive_to_readout_share", "four types (min)", at.drive_to_readout_share.min())
    add("M12", "Results 4", "the same, highest", "plan_v2/fig4_access_tiles.csv", "drive_to_readout_share", "four types (max)", at.drive_to_readout_share.max())
    add("M13", "Results 4", "carrier types with split-GAL4 lines shown", "plan_v2/fig4_access_tiles.csv", "type", "rows", len(at))
    het = T("plan_v2/fig4_heterogeneity.csv")
    add("M14", "Results 4", "heterogeneous runs", "plan_v2/fig4_heterogeneity.csv", "replicate", "rows", len(het))
    add("M15", "Results 4", "Holm P, memory, lock and mechanism in heterogeneous brains", "plan_v2/tests.csv", "p_adjusted", "family F14.SC.variant (max)",
        tests("F14.SC.variant").p_adjusted.max())
    # ---- the timing lock (Fig. 5)
    t5 = T("plan_v2/fig5_timings.csv")
    fx = t5[(t5.arm == "fixed timings")]
    add("A01", "Results 5", "trained timings", "plan_v2/fig5_timings.csv", "trained", "fixed timings, distinct trained timings",
        fx[truthy(fx.trained)][["delay_ms", "prefix_windows"]].drop_duplicates().shape[0])
    add("A02", "Results 5", "net use at the trained timings (every replicate)", "plan_v2/fig5_timings.csv", "net_use", "fixed timings, trained (min)", fx.loc[truthy(fx.trained), "net_use"].min())
    add("A03", "Results 5", "mean net use at the untrained timings", "plan_v2/fig5_timings.csv", "net_use", "fixed timings, untrained (mean)", fx.loc[~truthy(fx.trained), "net_use"].mean())
    add("A04", "Results 5", "timings in the grid", "plan_v2/fig5_timings.csv", "delay_ms, prefix_windows", "fixed timings, distinct timings",
        fx[["delay_ms", "prefix_windows"]].drop_duplicates().shape[0])
    add("A05", "Results 5", "Holm P, the lock (scene 1, readout trained)", "plan_v2/tests.csv", "p_adjusted", "family F13.X1.lock, M0-head_only",
        tests("F13.X1.lock").set_index("test").loc["M0-head_only", "p_adjusted"])
    s5 = T("plan_v2/fig5_scenes.csv")
    add("A06", "Results 5", "scenes", "plan_v2/fig5_scenes.csv", "scene", "distinct scenes", s5.scene.nunique())
    add("A07", "Results 5", "runs with a positive lock", "plan_v2/fig5_scenes.csv", "lock", "lock > 0 (count)", (s5.lock > 0).sum())
    add("A08", "Results 5", "runs", "plan_v2/fig5_scenes.csv", "lock", "rows", len(s5))
    co = T("plan_v2/fig5_decision_cosines.csv").set_index(["timing_a", "timing_b"])["cosine"]
    add("A09", "Results 5", "cosine of the 320-ms with the trained 160-ms decision pattern", "plan_v2/fig5_decision_cosines.csv", "cosine",
        "timing_a dv_d320_p4_mv, timing_b dv_d160_p4_mv", co.loc[("dv_d320_p4_mv", "dv_d160_p4_mv")])
    dc = T("plan_v2/fig5_decision_cells.csv")
    a_, b_ = dc["dv_d160_p4_mv"].to_numpy(float), dc["dv_d320_p4_mv"].to_numpy(float)
    top = np.abs(a_) >= np.quantile(np.abs(a_), 0.99)
    add("A10", "Results 5", "share of the strongest 1% of cells that change sign between 160 and 320 ms", "plan_v2/fig5_decision_cells.csv",
        "dv_d160_p4_mv, dv_d320_p4_mv", "top 1% by |dv at 160 ms|: sign(320) = -sign(160) (share)", np.mean(np.sign(b_[top]) == -np.sign(a_[top])))
    ro5 = T("plan_v2/fig5_record_outcomes.csv")
    r1 = ro5[(ro5.scene == "scene 1") & (ro5.trained_parameters == "readout")]
    add("A11", "Results 5", "median adaptation trace at the decision, untrained timings", "plan_v2/fig5_record_outcomes.csv", "adaptation_energy_decision",
        "scene 1, readout trained, untrained (median)", r1.loc[~truthy(r1.trained), "adaptation_energy_decision"].median())
    add("A12", "Results 5", "the same at trained timings", "plan_v2/fig5_record_outcomes.csv", "adaptation_energy_decision",
        "scene 1, readout trained, trained (median)", r1.loc[truthy(r1.trained), "adaptation_energy_decision"].median())
    s = T("stats/statistical_tests.csv")
    s = s[(s.family == "F11.timing.differential") & s.p_value.notna()]
    p_h = holm(s.p_value.astype(float).to_numpy())
    keep = s.test_id.str.contains("remote-dynamic-M0-head_only|remote-dynamic-M0-joint", regex=True).to_numpy()
    add("A13", "Results 5", "Holm P, the push collapses at untrained timings (both scene-1 learners)", "stats/statistical_tests.csv", "p_value",
        "family F11.timing.differential, Holm within the family; the two scene-1 learners (max)", p_h[keep].max())
    dm = T("plan_v2/fig5_drift_margin.csv")
    d4 = dm[(dm.prefix_windows == 4) & truthy(dm.trained_delay)]
    add("A14", "Results 5", "push of the injected trace at the trained delays, lowest", "plan_v2/fig5_drift_margin.csv", "margin_shift", "prefix 4, trained delays (min)", d4.margin_shift.min())
    add("A15", "Results 5", "the same, highest", "plan_v2/fig5_drift_margin.csv", "margin_shift", "prefix 4, trained delays (max)", d4.margin_shift.max())
    x3 = tests("F13.X3.descriptive").set_index("test")["estimate"]
    add("A16", "Results 5", "jittered minus fixed training, untrained use inside the jitter range", "plan_v2/tests.csv", "estimate",
        "F13.X3.descriptive, untrained_inside_net_use", x3[[i for i in x3.index if i.startswith("untrained_inside_net_use")][0]])
    add("A17", "Results 5", "the same outside the range (640 ms)", "plan_v2/tests.csv", "estimate", "F13.X3.descriptive, untrained_outside_net_use",
        x3[[i for i in x3.index if i.startswith("untrained_outside_net_use")][0]])
    # ---- robust access (Fig. 6)
    rs = T("plan_v2/fig6_routes_summary.csv").set_index("share_removed")
    add("E01", "Results 6", "readout cells reachable with 90% of the other cells removed", "plan_v2/fig6_routes_summary.csv", "readout_reachable", "share_removed 0.9", rs.loc[0.9, "readout_reachable"])
    add("E02", "Results 6", "cells on routes, intact", "plan_v2/fig6_routes_summary.csv", "route_cells", "share_removed 0", rs.loc[0.0, "route_cells"])
    add("E03", "Results 6", "cells on routes, 90% removed", "plan_v2/fig6_routes_summary.csv", "route_cells", "share_removed 0.9", rs.loc[0.9, "route_cells"])
    add("E04", "Results 6", "three-step routes kept at 90% removal", "plan_v2/fig6_routes_summary.csv", "walks3_relative", "share_removed 0.9", rs.loc[0.9, "walks3_relative"])
    ty6 = T("plan_v2/fig6_route_types.csv")
    add("E05", "Results 6", "cell types carrying routes, intact", "plan_v2/fig6_route_types.csv", "traffic_0", "traffic_0 > 0 (count)", (ty6.traffic_0 > 0).sum())
    add("E06", "Results 6", "cell types carrying routes, 90% removed", "plan_v2/fig6_route_types.csv", "traffic_90", "traffic_90 > 0 (count)", (ty6.traffic_90 > 0).sum())
    kk = T("plan_v2/fig6_kicks.csv")
    add("E07", "Results 6", "single extra spikes that leave nothing at the decision", "plan_v2/fig6_kicks.csv", "hamming_decision", "= 0 (count)", (kk.hamming_decision == 0).sum())
    add("E08", "Results 6", "single extra spikes given", "plan_v2/fig6_kicks.csv", "kicked_cell", "rows", len(kk))
    rr = T("plan_v2/fig6_rate.csv")
    rr1 = rr[(rr.source == "memory task, analysed learner") & (rr.shift_mv >= -1.0)]
    add("E09", "Results 6", "mean rate for threshold changes up to 1 mV, lowest (Hz)", "plan_v2/fig6_rate.csv", "mean_rate_hz",
        "source memory task, analysed learner; shift_mv >= -1 (min)", rr1.mean_rate_hz.min(), "Hz")
    add("E10", "Results 6", "the same, highest (Hz)", "plan_v2/fig6_rate.csv", "mean_rate_hz", "source memory task, analysed learner; shift_mv >= -1 (max)",
        rr1.mean_rate_hz.max(), "Hz")
    add("E20", "Results 6", "threshold lowering from which the network runs away (mV)", "plan_v2/fig6_rate.csv", "runaway_from_mv", "first row (sign reversed)",
        -float(rr.runaway_from_mv.iloc[0]), "mV")
    rsum = T("plan_v2/fig6_robust_summary.csv").set_index("quantity")
    aw = rsum.loc["ED50 ratio, across states / one state"]
    add("E11", "Results 6", "access window, training across states over one state (fold)", "plan_v2/fig6_robust_summary.csv", "estimate", f"{aw.name}", aw.estimate)
    add("E12", "Results 6", "its 95% interval, low", "plan_v2/fig6_robust_summary.csv", "ci_low", f"{aw.name}", aw.ci_low)
    add("E13", "Results 6", "its 95% interval, high", "plan_v2/fig6_robust_summary.csv", "ci_high", f"{aw.name}", aw.ci_high)
    rb6 = T("plan_v2/fig6_robust.csv")
    add("E21", "Figure legends", "runs trained at one state", "plan_v2/fig6_robust.csv", "replicate", "training one state (rows)",
        int((rb6.training == "one state").sum()))
    add("E22", "Figure legends", "runs trained across states", "plan_v2/fig6_robust.csv", "replicate", "training across states (rows)",
        int((rb6.training == "across states").sum()))
    add("G11", "Results 6", "the coverage of the bootstrap intervals (E12-E13, E18-E19)", "plan_v2/fig6_robust_summary.csv", "ci_low, ci_high",
        "the documented coverage of ci_low and ci_high (95% bootstrap interval)", 0.95)
    cl_ = rsum.loc["removal ED50 ratio, across states / one state"]
    add("E17", "Results 6", "cell-loss tolerance, training across states over one state (fold; geometric mean)", "plan_v2/fig6_robust_summary.csv", "estimate", cl_.name, cl_.estimate)
    add("E18", "Results 6", "its 95% interval, low", "plan_v2/fig6_robust_summary.csv", "ci_low", cl_.name, cl_.ci_low)
    add("E19", "Results 6", "its 95% interval, high", "plan_v2/fig6_robust_summary.csv", "ci_high", cl_.name, cl_.ci_high)
    rb = T("plan_v2/fig6_robust.csv")
    mem = rb.groupby("training")["memory_under_jitter"].mean()
    for k_, (lab, m_) in zip(("E14", "E15"), sorted(mem.items(), key=lambda kv: -kv[1])):
        add(k_, "Results 6", f"memory under 1-microvolt jitter, {lab} (mean)", "plan_v2/fig6_robust.csv", "memory_under_jitter", f"training {lab} (mean)", m_)
    add("E16", "Results 6", "Holm P, training across states widens the window", "plan_v2/tests.csv", "p_adjusted", "family F14.DY.robust, th0.001:ed50-c01",
        tests("F14.DY.robust").set_index("test").loc["th0.001:ed50-c01", "p_adjusted"])
    # ---- disease (Fig. 7)
    fu = T("plan_v2/fig7_funnel.csv").sort_values("order")
    add("Q01", "Results 7", "fly pathologies screened", "plan_v2/fig7_funnel.csv", "cases", "first stage", fu.cases.iloc[0])
    add("Q02", "Results 7", "credible cases retained", "plan_v2/fig7_funnel.csv", "cases", "last stage", fu.cases.iloc[-1])
    la = T("plan_v2/fig7_lost_access.csv")
    after = la[(la.regime == "lesion after learning") & (la.trained_parameters == "readout") & (la.dose == 1.0)]
    under = la[la.regime == "learning under the lesion"]
    add("Q03", "Results 7", "trained histories solved after the lesion (dose 1, readout trained)", "plan_v2/fig7_lost_access.csv", "trained_solved",
        "regime lesion after learning, readout, dose 1 (mean)", after.trained_solved.mean())
    add("Q04", "Results 7", "trained histories", "plan_v2/fig7_lost_access.csv", "trained_histories", "regime lesion after learning (first)", after.trained_histories.iloc[0])
    add("Q05", "Results 7", "Holm P, memory falls with lesion dose", "plan_v2/tests.csv", "p_adjusted", "family F12.C01.primary, memory dose trend",
        tests("F12.C01.primary").set_index("test").filter(like="memory", axis=0).p_adjusted.iloc[0])
    add("Q06", "Results 7", "runs of learning under the lesion", "plan_v2/fig7_lost_access.csv", "replicate", "regime learning under the lesion (rows)", len(under))
    add("Q07", "Results 7", "trained histories solved when learning under the lesion, readout trained (lowest run)", "plan_v2/fig7_lost_access.csv", "trained_solved",
        "learning under the lesion, readout (min)", under.loc[under.trained_parameters == "readout", "trained_solved"].min())
    rl = T("plan_v2/fig7_relearning.csv")
    add("Q08", "Results 7", "share of the loss recovered by retraining the readout", "plan_v2/fig7_relearning.csv", "recovery_fraction", "readout (mean)",
        rl.loc[rl.trained_parameters == "readout", "recovery_fraction"].mean())
    rs7 = T("plan_v2/fig7_restoration.csv")
    add("Q09", "Results 7", "trained histories solved after late restoration (dose 1, readout trained)", "plan_v2/fig7_restoration.csv", "trained_solved_restored",
        "readout, dose 1 (mean)", rs7.loc[(rs7.trained_parameters == "readout") & (rs7.dose == 1.0), "trained_solved_restored"].mean())
    add("Q10", "Results 7", "BH P, late restoration removes access, highest", "plan_v2/tests.csv", "p_adjusted", "family F12.rescue (max)", tests("F12.rescue").p_adjusted.max())
    add("Q25", "Figure legends", "runs of late restoration", "plan_v2/fig7_restoration.csv", "replicate", "rows", len(rs7))
    add("Q26", "Figure legends", "runs of relearning with the readout retrained", "plan_v2/fig7_relearning.csv", "replicate", "trained_parameters readout (rows)",
        int((rl.trained_parameters == "readout").sum()))
    ad = tests("F14.SC.adfca.pan").set_index("test")
    add("Q11", "Results 7", "Abeta42-induced change of pan-neuronal Shal mRNA (log2)", "plan_v2/tests.csv", "estimate", "F14.SC.adfca.pan, Abeta42 Shal",
        ad.loc["Abeta42:pan-neuronal (CNS neuron):Shal", "estimate"], "log2")
    sr = T("plan_v2/fig7_shal_runaway.csv")
    sh = sr[sr.pattern.astype(str).str.contains("Shal") & sr.dose.isin([1.5, 2.0])]
    add("Q12", "Results 7", "cells above 100 Hz under the Shal-patterned lesion at 1.5-2x, lowest", "plan_v2/fig7_shal_runaway.csv", "share_above_100hz",
        "Shal patterns, dose 1.5-2 (min)", sh.share_above_100hz.min())
    add("Q13", "Results 7", "the same, highest", "plan_v2/fig7_shal_runaway.csv", "share_above_100hz", "Shal patterns, dose 1.5-2 (max)", sh.share_above_100hz.max())
    ag = T("plan_v2/fig7_aging.csv")
    for i, (d_, g_) in enumerate(ag.groupby("dose")):
        for j, (fn, v_) in enumerate((("min", g_.kc_ratio.min()), ("max", g_.kc_ratio.max()))):
            add(f"Q{14 + 2 * i + j}", "Results 7", f"Kenyon-cell odor response under the ageing operator at {d_:g}x, of healthy ({fn} over the two brains)",
                "plan_v2/fig7_aging.csv", "kc_ratio", f"dose {d_:g} ({fn})", v_)
    en = T("plan_v2/fig7_energy.csv")
    hl7 = en[en.code.astype(str).str.lower().isin(["healthy", "none"]) | en.case.astype(str).str.contains("healthy", case=False)]
    c01 = en[en.code.astype(str) == "C01"]
    base = float(hl7.spikes_per_trial.iloc[0])
    add("Q20", "Results 7", "healthy bits of cue per decision", "plan_v2/fig7_energy.csv", "cue_bits_per_decision", "healthy", hl7.cue_bits_per_decision.iloc[0], "bits")
    add("Q21", "Results 7", "spikes per trial under the pan-neuronal lesion, fold of healthy, lowest", "plan_v2/fig7_energy.csv", "spikes_per_trial",
        "pan-neuronal case over healthy (min)", (c01.spikes_per_trial / base).min())
    add("Q22", "Results 7", "the same, highest", "plan_v2/fig7_energy.csv", "spikes_per_trial", "pan-neuronal case over healthy (max)", (c01.spikes_per_trial / base).max())
    add("Q23", "Results 7", "bits per decision under the pan-neuronal lesion, lowest", "plan_v2/fig7_energy.csv", "cue_bits_per_decision", "pan-neuronal case (min)",
        c01.cue_bits_per_decision.min(), "bits")
    add("Q24", "Results 7", "the same, highest", "plan_v2/fig7_energy.csv", "cue_bits_per_decision", "pan-neuronal case (max)", c01.cue_bits_per_decision.max(), "bits")
    # ---- the lesion link (Fig. 7J; amendment 0015): where the trace is written and where it is read
    lo_ = T("plan_v2/link_outcomes.csv")
    hk = lo_[lo_.readout_kind == "trained healthy"]
    lk = lo_[(lo_.readout_kind == "trained under the lesion") & np.isclose(lo_.dose.astype(float), 1.0)]
    d1 = np.isclose(lo_.lesion_dose.astype(float), 1.0)
    add("K18", "Figure legends", "readouts trained healthy in the lesion link", "plan_v2/link_outcomes.csv", "replicate", "trained healthy (distinct)",
        hk.replicate.nunique())
    add("K19", "Figure legends", "readouts trained under the lesion at dose 1", "plan_v2/link_outcomes.csv", "replicate",
        "trained under the lesion, dose 1 (distinct)", lk.replicate.nunique())

    def cell(q, wr, rd, dose_col=True):
        sel = q[(q.trace_written == wr) & (q.trace_read == rd)]
        return sel[np.isclose(sel.lesion_dose.astype(float), 1.0)] if dose_col and not (wr == rd == "healthy") else sel

    learned = ((lo_.readout_kind == "trained healthy") & (lo_.trace_written == "healthy") & (lo_.trace_read == "healthy")) | \
              ((lo_.readout_kind == "trained under the lesion") & (lo_.trace_written == "lesion") & (lo_.trace_read == "lesion"))
    add("K01", "Results 7", "trained histories solved with the learning state at both ends (every readout trained healthy or under the lesion, min)",
        "plan_v2/link_outcomes.csv", "solved", "learning state at both ends (min)", lo_[learned].solved.min())
    mism = pd.concat([cell(hk, w, r) for w, r in (("healthy", "lesion"), ("lesion", "healthy"), ("lesion", "lesion"))]
                     + [cell(lk, w, r, False) for w, r in (("healthy", "lesion"), ("lesion", "healthy"), ("healthy", "healthy"))])
    add("K02", "Results 7", "the most solved with a mismatch at either end (dose 1, max)", "plan_v2/link_outcomes.csv", "solved", "mismatched, dose 1 (max)", mism.solved.max())
    for k_, (q_, w_, r_, lab) in zip(("K03", "K04", "K05", "K06"), ((hk, "healthy", "lesion", "trained healthy: written healthy, read lesioned"),
                                                                    (hk, "lesion", "healthy", "trained healthy: written lesioned, read healthy"),
                                                                    (lk, "healthy", "lesion", "trained under the lesion: written healthy, read lesioned"),
                                                                    (lk, "lesion", "healthy", "trained under the lesion: written lesioned, read healthy"))):
        add(k_, "Results 7", f"trained histories solved, {lab} (dose 1, mean)", "plan_v2/link_outcomes.csv", "solved", f"{lab} (mean)",
            cell(q_, w_, r_, q_ is hk).solved.mean())
    lc = T("plan_v2/link_cosines.csv")
    ls_ = T("plan_v2/link_states.csv")
    doses_ = sorted(float(d) for d in ls_.lesion_dose.dropna().unique())
    tr = [lc[(lc.quantity == "trace") & (lc.condition_code_a == "H-H") & (lc.condition_code_b == f"L{d:g}-L{d:g}")].cosine.median() for d in doses_]
    rd_ = [lc[(lc.quantity == "window_v") & (lc.condition_code_a == "H-H") & (lc.condition_code_b == f"H-L{d:g}")].cosine.median() for d in doses_]
    add("K07", "Results 7", "cosine of the trace written under the lesion with the healthy trace (median over timings; lowest dose value)", "plan_v2/link_cosines.csv",
        "cosine", "trace, H-H vs Ld-Ld (median; min over doses)", min(tr))
    add("K08", "Results 7", "the same, highest", "plan_v2/link_cosines.csv", "cosine", "trace (median; max over doses)", max(tr))
    add("K09", "Results 7", "cosine of the decision-time pattern when the healthy trace is read under the lesion, with its healthy reading (median; min)",
        "plan_v2/link_cosines.csv", "cosine", "window_v, H-H vs H-Ld (median; min over doses)", min(rd_))
    add("K10", "Results 7", "the same, highest", "plan_v2/link_cosines.csv", "cosine", "window_v (median; max over doses)", max(rd_))
    lcells = [ls_[ls_.condition_code == f"L{d:g}-L{d:g}"].trace_cells.median() for d in doses_]
    add("K11", "Results 7", "cells holding the trace under the lesion, lowest dose (median over timings)", "plan_v2/link_states.csv", "trace_cells", "Ld-Ld (median; min)", min(lcells))
    add("K12", "Results 7", "the same, highest dose", "plan_v2/link_states.csv", "trace_cells", "Ld-Ld (median; max)", max(lcells))
    late = ls_[(ls_.delay_ms == 160) & ls_.condition_code.isin([f"L{d:g}-L{d:g}" for d in doses_])].groupby("lesion_dose").delay_differ_cells.max()
    add("K13", "Results 7", "cells spiking differently in the second half of the 160-ms delay at the lowest dose", "plan_v2/link_states.csv", "delay_differ_cells",
        "L0.5-L0.5, delay 160", late.loc[min(doses_)])
    add("K14", "Results 7", "the same at dose 1", "plan_v2/link_states.csv", "delay_differ_cells", "L1-L1, delay 160", late.loc[1.0])
    fl = tests("F15.link").reset_index(drop=True)
    for k_, part in zip(("K15", "K16", "K17"), ("lesion-trained", "healthy-trained", "relearned")):
        row = fl[fl.test.astype(str).str.startswith(part)]
        add(k_, "Results 7", f"Holm P, F15.link, {part} readouts: read in the learning state beats written in it", "plan_v2/tests.csv", "p_adjusted",
            f"F15.link, {part}", row.p_adjusted.iloc[0])
    # ---- design values the text names (delays, removal level, doses, thresholds)
    t5d = T("plan_v2/fig5_decision_cosines.csv")
    add("G01", "Results 5", "the untrained delay compared with the trained 160 ms (ms)", "plan_v2/fig5_decision_cosines.csv", "timing_a", "dv_d320_p4_mv", 320, "ms")
    add("G02", "Results 6", "share of the other cells removed", "plan_v2/fig6_routes_summary.csv", "share_removed", "max", rs.index.max())
    sr2 = T("plan_v2/fig7_shal_runaway.csv")
    add("G03", "Results 7", "rate above which a cell counts as runaway (Hz)", "plan_v2/fig7_shal_runaway.csv", "share_above_100hz", "column definition", 100, "Hz")
    add("G04", "Results 7", "lesion dose, the lower of the runaway range (x calibrated)", "plan_v2/fig7_shal_runaway.csv", "dose", "1.5", sr2.dose[sr2.dose == 1.5].iloc[0])
    add("G05", "Results 7", "lesion dose, the upper of the runaway range (x calibrated)", "plan_v2/fig7_shal_runaway.csv", "dose", "max", sr2.dose.max())
    add("G06", "Figure legends", "the trained delay of scene 1 (ms)", "plan_v2/fig5_decision_cosines.csv", "timing_b", "dv_d160_p4_mv", 160, "ms")
    levels = sorted(x for x in rs.index if 0 < x < rs.index.max())
    add("G07", "Figure legends", "share of the other cells removed, the lowest level", "plan_v2/fig6_routes_summary.csv", "share_removed",
        "the lowest level above 0", levels[0])
    add("G08", "Figure legends", "share of the other cells removed, the middle level", "plan_v2/fig6_routes_summary.csv", "share_removed",
        "the second level above 0", levels[1])
    sc1 = w[w.ensemble_scene == "scene 1"]
    add("D14", "Results 3", "matched sets per ensemble", "plan_v2/fig3_set_wiring.csv", "set", "scene 1, sets other than the ensemble (count)",
        int((sc1.set != "ensemble").sum()))
    add("D15", "Results 3", "the ensemble's share of the model's cells", "plan_v2/fig3_set_wiring.csv", "cells", "scene 1 ensemble over all neurons",
        float(sc1.loc[sc1.set == "ensemble", "cells"].iloc[0]) / float(spec[("network", "neurons")]))
    add("G09", "Results 6", "the bound of the threshold changes that E09-E10 summarize (mV)", "plan_v2/fig6_rate.csv", "shift_mv",
        "memory task, analysed learner; shift_mv >= -1 (the filter of E09-E10)", 1.0, "mV")
    add("G10", "Results 7", "the lowest lesion dose of the lesion link (x calibrated)", "plan_v2/link_states.csv", "condition_code", "Ld-Ld doses (min)",
        min(doses_))
    mv = T("plan_v2/video_planning_moves.csv")
    add("L18", "Video S1", "moves of the final brain and readout on the hardest map (its certified optimum)", "plan_v2/video_planning_moves.csv",
        "move", "rows", len(mv))
    star_methods(spec)
    missing = [r["id"] for r in ROWS if r["id"] not in FORMATS]
    if missing:
        raise SystemExit(f"no display format for {missing}")
    rows = [{**{k: r[k] for k in ("id", "section", "statement", "value", "unit")}, "display": display(r["value"], FORMATS[r["id"]]),
             **{k: r[k] for k in ("file", "column", "row")}} for r in ROWS]
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(f"{r['id']:4s} {r['value']:>16.6g} {r['display']:>14s}  {r['statement']}")
    print(f"{len(ROWS)} numbers -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
