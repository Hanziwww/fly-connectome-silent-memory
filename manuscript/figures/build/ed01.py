"""Extended Data Fig. 1 - Method development and the formal lineage (180 x 126 mm).

a  the 14 recorded method stages (docs/experiment-history.md via ED1a): what was tried
b  every archived trial of those stages: training time and outcome, aligned with a (archive folders mapped to stages by name)
c  formal lineage P03 -> P05 (joint brain + linear readout): suffix lessons solved before and after training
d  substitution tests: the final readout with the source brain, the final brain with the source readout, no plasticity
e  parameters changed (of 11,770 groups per type) and readout entries changed
f  training compute of the two formal runs

  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\ed01.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import boards as bd, figstyle as fs, palette as pal, qc  # noqa: E402

U = (fs.FULL - 3 * fs.GAP) / 4
XS = [k * (U + fs.GAP) for k in range(4)]
ROW1, R1H = 0.0, 74.0
ROW2, R2H = 78.0, 48.0
H = ROW2 + R2H
EDGE = "#ABABAB"
STAGE_OF = [("archive/experiments/campaign", 0), ("archive/experiments/boundary/", 1), ("archive/experiments/diagnostics/", 2),
            ("archive/experiments/repair/", 3), ("archive/experiments/progressive/", 4), ("archive/experiments/outcome/", 5),
            ("archive/experiments/lessons/", 6), ("archive/experiments/joint/", 7), ("archive/experiments/frontier/", 9),
            ("archive/experiments/hybrid/", 10), ("archive/experiments/readouts/", 11), ("runs/linear-joint/20261002-p03", 12),
            ("runs/linear-joint/20261002-p05", 13)]
SOLVED = {"original_solved", "learned", "bridge_solved", "autonomous_success", "complete"}
STOPPED = {"paused_by_user", "stopped"}
ALREADY = {"already_solved", "already_solved_at_stage_start"}


def outcome(r):
    s = r["status"] if isinstance(r["status"], str) else r["stop"]
    if s in ALREADY or r["stop"] in ALREADY:
        return "already solved"
    if s in SOLVED or r["stop"] in SOLVED:
        return "solved"
    if s in STOPPED:
        return "stopped by the user"
    return "budget reached"


MARK = {"solved": dict(marker="o", fc=fs.TEXT, ec=fs.TEXT), "budget reached": dict(marker="o", fc="white", ec=fs.TEXT),
        "already solved": dict(marker="o", fc="#BDBDBD", ec="#BDBDBD"), "stopped by the user": dict(marker="x", fc=fs.TEXT, ec=fs.TEXT)}


def stage_of(unit):
    for prefix, k in STAGE_OF:
        if unit.startswith(prefix):
            return k
    raise KeyError(unit)


# ---------------------------------------------------------------- a, b: stages and archived trials

def panel_ab(cv, sd):
    t = cm.table("extended/ed01/ed01a_method_development.csv")
    stages = t[t["record"] == "method_stage"].sort_values("index")
    trials = t[t["record"] == "archive_result"].copy()
    trials["stage"] = trials["unit"].map(stage_of)
    trials["outcome"] = trials.apply(outcome, axis=1)
    n = len(stages)
    cv.letter(XS[0], ROW1, "a")
    cv.letter(XS[2], ROW1, "b")
    top, rh = 5.6, (R1H - 5.6 - 4.4) / n
    ax = bd.mm_axes(cv, XS[0], ROW1, 2 * U + fs.GAP, R1H)
    ax.text(5.6, top - 1.2, "Stage", ha="left", va="bottom", fontsize=fs.PT_SMALL)
    ax.text(33.0, top - 1.2, "Method", ha="left", va="bottom", fontsize=fs.PT_SMALL)
    for i, r in enumerate(stages.itertuples()):
        yc = top + (i + 0.5) * rh
        if i % 2 == 0:
            ax.add_patch(bd.Rectangle((0, top + i * rh), 2 * U + fs.GAP, rh, facecolor="#F4F4F4", edgecolor="none", linewidth=0.5, zorder=0))
        ax.text(1.6, yc, str(i + 1), ha="center", va="center", fontsize=fs.PT_SMALL)
        ax.text(5.6, yc, r.stage, ha="left", va="center", fontsize=fs.PT_SMALL)
        lines = fs.wrap(r.method, 55.0, fs.PT_SMALL)
        ax.text(33.0, yc, "\n".join(lines[:2]), ha="left", va="center", fontsize=fs.PT_SMALL, linespacing=1.0)
        sd.add("a", "stage", stage=i + 1, name=r.stage, method=r.method, recorded_result=r.recorded_result)
    # b: trials per stage (training seconds, log axis)
    bx = cv.axes(XS[2] + 7.0, ROW1 + top, 2 * U + fs.GAP - 7.4, n * rh)
    for k in range(0, n, 2):
        bx.axhspan(k, k + 1, color="#F4F4F4", lw=0, zorder=0)
    rng = np.random.default_rng(4)
    for tr in trials.itertuples():
        y = tr.stage + 0.5 + (rng.random() - 0.5) * 0.45
        x = 0.5 if not np.isfinite(tr.training_seconds) else max(float(tr.training_seconds), 1.0)    # n.r. at 0.5; 0 s shown at 1 s
        m = MARK[tr.outcome]
        bx.plot([x], [y], ls="none", marker=m["marker"], ms=2.6, mfc=m["fc"], mec=m["ec"], mew=0.5, zorder=3, clip_on=False)
        sd.add("b", "trial", unit=tr.unit, index=int(tr.index), stage=int(tr.stage) + 1, task=tr.task if isinstance(tr.task, str) else None,
               arm=tr.arm if isinstance(tr.arm, str) else None, seed=None if not np.isfinite(tr.seed) else int(tr.seed),
               training_seconds=None if not np.isfinite(tr.training_seconds) else float(tr.training_seconds), outcome=tr.outcome,
               generations=None if not np.isfinite(tr.generations) else int(tr.generations), plotted_seconds=float(x))
    counts = trials.groupby("stage").size()
    for k in range(n):
        if k in counts.index:
            cv.text(XS[2] + 5.6, ROW1 + top + (k + 0.5) * rh, f"{int(counts[k])}", size=fs.PT_SMALL, ha="right", va="center")
    cv.text(XS[2] + 5.6, ROW1 + top - 1.2, "trials", size=fs.PT_SMALL, ha="right", va="bottom")
    bx.set_xscale("log")
    bx.set_xlim(0.36, 2000)
    bx.set_xticks([0.5, 1, 10, 100, 1000], ["n.r.", "≤1", "10", "100", "1,000"])
    fs.minor(bx, y=False)
    bx.set_ylim(n, 0)
    bx.set_yticks([])
    fs.clean(bx, left=False)
    bx.set_xlabel("Training time of the trial (s)", labelpad=1)
    cv.key(XS[2] + 7.0, ROW1 + 2.0, [([dict(kind="mark", s=6, **{"fc": MARK[k]["fc"], "ec": MARK[k]["ec"], "marker": MARK[k]["marker"]})], k)
                                    for k in ("solved", "budget reached", "already solved", "stopped by the user")], gap=1.6)


# ---------------------------------------------------------------- c-f: formal lineage

def lineage():
    b = cm.table("extended/ed01/ed01b_formal_lineage.csv")
    out = {}
    for model, g in b.groupby("model"):
        out[model] = dict(zip(g["item"], g["value"]))
    return out


def parse(v):
    s, _, k = str(v).partition(":")
    return s, (int(k) if k.isdigit() else None)


def panel_c(cv, sd, x0, y0, w, h, lin):
    cv.letter(x0, y0, "c")
    ax = bd.mm_axes(cv, x0, y0, w, h)
    cv.text(x0 + 3.6, y0 + 1.5, "Suffix lessons after training", size=fs.PT_SMALL, va="center")
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="box", fc=fs.TEXT, size=1.3)], "solved"), ([dict(kind="box", fc="white", ec=EDGE, lw=0.3, size=1.3)],
                                                                                      "not solved")], gap=1.6)
    models = [("20261002-p03-01", "P03 (8 × 8, 2 boxes)"), ("20261002-p05-01", "P05 (10 × 10, 2 boxes)")]
    for j, (m, lab) in enumerate(models):
        d = lin[m]
        task = m.split("-")[1].upper()
        depth = int(d["after.original_steps"])
        suffixes = sorted(int(k.split("-")[-1]) for k in d if k.startswith(f"after_suite.{task}-suffix-"))
        yy = 8.5 + j * 15.5
        ax.text(0.0, yy, lab, ha="left", va="bottom", fontsize=fs.PT_SMALL)
        per = 12
        sq, pitch = 1.6, 1.95
        for i, k in enumerate(suffixes):
            s, steps = parse(d[f"after_suite.{task}-suffix-{k}"])
            ok = s == "solved"
            r, c = divmod(i, per)
            ax.add_patch(bd.Rectangle((c * pitch, yy + 0.8 + r * pitch), sq, sq, facecolor=fs.TEXT if ok else "white",
                                      edgecolor="none" if ok else EDGE, linewidth=0.3))
            sd.add("c", "suffix", model=m, suffix=k, status=s, steps=steps)
        before = parse(d[f"before_suite.{task}"])
        after = parse(d[f"after_suite.{task}"])
        rows = int(np.ceil(len(suffixes) / per))
        ax.text(0.0, yy + 1.0 + rows * pitch + 0.4, f"before: {before[0].replace('_', ' ')} at move {before[1]}" + "\n" +
                f"after: solved in {after[1]} moves", ha="left", va="top", fontsize=fs.PT_SMALL, linespacing=1.05)
        sd.add("c", "task", model=m, before_status=before[0], before_steps=before[1], after_status=after[0], after_steps=after[1], depth=depth,
               suffixes=len(suffixes))


def panel_d(cv, sd, x0, y0, w, h, lin):
    cv.letter(x0, y0, "d")
    ax = bd.mm_axes(cv, x0, y0, w, h)
    cv.text(x0 + 3.6, y0 + 1.5, "Swapping trained parts back to the source", size=fs.PT_SMALL, va="center")
    rows = [("substitution.source_brain_final_readout", "source brain, final readout"),
            ("substitution.final_brain_source_readout", "final brain, source readout"),
            ("substitution.zero_plasticity_final_readout", "no plasticity, final readout"), ("after.original_status", "final brain, final readout")]
    cols = [("20261002-p03-01", "P03"), ("20261002-p05-01", "P05")]
    lab_w, top, rh, cw = 23.6, 7.4, 6.6, (w - 23.6 - 0.4) / 2
    for j, (m, lab) in enumerate(cols):
        ax.text(lab_w + (j + 0.5) * cw, top - 0.8, lab, ha="center", va="bottom", fontsize=fs.PT_SMALL)
    for i, (key, lab) in enumerate(rows):
        yy = top + i * rh
        ax.text(lab_w - 0.8, yy + rh / 2, lab, ha="right", va="center", fontsize=fs.PT_SMALL)
        for j, (m, _) in enumerate(cols):
            d = lin[m]
            if key == "after.original_status":
                s, k = d[key], int(d["after.original_steps"])
            else:
                s, k = parse(d[key])
            ok = s == "solved"
            cx = lab_w + j * cw
            ax.add_patch(bd.Rectangle((cx + 0.4, yy + 0.4), cw - 0.8, rh - 0.8, facecolor="#D4D4D4" if ok else "white",
                                      edgecolor=fs.TEXT if ok else EDGE, linewidth=0.5 if ok else 0.35))
            txt = f"solved, {k} moves" if ok else (f"repeat at move {k}" if s == "repeated_state" else f"{s.replace('_', ' ')}, move {k}")
            ax.text(cx + cw / 2, yy + rh / 2, fs.wrap(txt, cw - 1.6, fs.PT_SMALL)[0] if fs.text_width(txt, fs.PT_SMALL) < cw - 1.4 else
                    "\n".join(fs.wrap(txt, cw - 1.6, fs.PT_SMALL)), ha="center", va="center", fontsize=fs.PT_SMALL,
                    linespacing=1.0)
            sd.add("d", "substitution", model=m, test=lab, status=s, steps=k)


def panel_e(cv, sd, x0, y0, w, h, lin):
    cv.letter(x0, y0, "e")
    cv.text(x0 + 3.6, y0 + 1.5, "Parameters changed by training", size=fs.PT_SMALL, va="center")
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="box", fc="#4D4D4D", size=1.3)], "P03"), ([dict(kind="box", fc="#A6A6A6", size=1.3)], "P05")], gap=1.6)
    ax = cv.axes(x0 + 14.0, y0 + 7.4, w - 15.0, h - 18.4)
    labels = [("pre_log_gain", "input gain"), ("post_log_gain", "output gain"), ("threshold_mv", "threshold")]
    for j, (m, col) in enumerate((("20261002-p03-01", "#4D4D4D"), ("20261002-p05-01", "#A6A6A6"))):
        core = json.loads(lin[m]["checks.core_changes"])
        bounds = json.loads(lin[m]["checks.bounds"])
        total = bounds["changed_parameters"]["pre_log_gain"]
        for i, (k, lab) in enumerate(labels):
            v = 100 * core[k] / 11770
            ax.barh(i + (j - 0.5) * 0.36, v, height=0.34, color=col, linewidth=0)
            sd.add("e", "core", model=m, parameter=k, groups_changed=int(core[k]), groups=11770, percent=v)
        head = json.loads(lin[m]["checks.head_changes"])
        sd.add("e", "head", model=m, weights_changed=int(head["proposal.weight"]), biases_changed=int(head["proposal.bias"]),
               pre_gain_range=str(bounds["pre_gain_range"]), post_gain_range=str(bounds["post_gain_range"]),
               threshold_shift_range_mv=str(bounds["threshold_shift_range_mv"]), groups_in_record=int(total))
    ax.set_yticks(range(3), [lab for _, lab in labels])
    ax.set_ylim(2.6, -0.6)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.tick_params(axis="y", length=0)
    fs.clean(ax)
    ax.set_xlabel("Groups changed (% of 11,770)", labelpad=1)
    hd = json.loads(lin["20261002-p03-01"]["checks.head_changes"])
    cv.text(x0 + 14.0, y0 + h - 1.4, f"readout: {hd['proposal.weight']:,} weights and {hd['proposal.bias']} biases changed in both", size=fs.PT_SMALL,
            va="center")


def panel_f(cv, sd, x0, y0, w, h, lin):
    cv.letter(x0, y0, "f")
    cv.text(x0 + 3.6, y0 + 1.5, "Training compute", size=fs.PT_SMALL, va="center")
    items = [("training.joint_updates", "updates"), ("training.training_seconds", "seconds"), ("training.assessments", "assessments"),
             ("training_compute.forward_calls", "forward calls"), ("training_compute.gradient_forward_calls", "gradient calls")]
    ax = bd.mm_axes(cv, x0, y0, w, h)
    lab_w, top, rh = 17.0, 7.4, (h - 7.4 - 2.0) / len(items)
    cw = (w - lab_w - 0.4) / 2
    for j, lab in enumerate(("P03", "P05")):
        ax.text(lab_w + (j + 0.5) * cw, top - 0.8, lab, ha="center", va="bottom", fontsize=fs.PT_SMALL)
    for i, (key, lab) in enumerate(items):
        yy = top + i * rh
        if i % 2 == 0:
            ax.add_patch(bd.Rectangle((0, yy), w, rh, facecolor="#F4F4F4", edgecolor="none", linewidth=0.5))
        ax.text(lab_w - 0.8, yy + rh / 2, lab, ha="right", va="center", fontsize=fs.PT_SMALL)
        for j, m in enumerate(("20261002-p03-01", "20261002-p05-01")):
            v = float(lin[m][key])
            ax.text(lab_w + (j + 0.5) * cw, yy + rh / 2, f"{v:,.0f}", ha="center", va="center", fontsize=fs.PT_SMALL)
            sd.add("f", "compute", model=m, item=lab, value=v)


def main():
    cv = fs.Canvas(fs.FULL, H, ppi_min=300)
    sd = qc.SourceData("ed01")
    for rel in ("extended/ed01/ed01a_method_development.csv", "extended/ed01/ed01b_formal_lineage.csv"):
        sd.source(cm.RESULTS / rel)
    panel_ab(cv, sd)
    lin = lineage()
    panel_c(cv, sd, XS[0], ROW2, U, R2H, lin)
    panel_d(cv, sd, XS[1], ROW2, U, R2H, lin)
    panel_e(cv, sd, XS[2], ROW2, U, R2H, lin)
    panel_f(cv, sd, XS[3], ROW2, U, R2H, lin)
    cm.finish(cv, "ed01", sd, kind="extended")


if __name__ == "__main__":
    main()
