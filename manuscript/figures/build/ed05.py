"""Extended Data Fig. 5 - Memory learners: training, outcomes, transfer across layouts and contexts (180 x 140 mm).

a  training curves of every memory learner: share of training histories solved vs updates (M0 and shared readout, S1, S2)
b  for all 21 learner configurations: histories solved at trained and untrained timings (95% Clopper-Pearson CI) and the share of
   cue pairs in which the first push was the same after both cues (what a policy without memory does)
c  a cue decoder fitted on M0 states applied to M0, S1 and S2 (cue pairs, Holm); the M0 readout applied without updates
d  first-decision margin by delay for the M0 readout-only model, its GRU, the shared readout with the goal visible and the reset
   state model (mean over prefixes and cues; bars: interquartile range)
e  S2: histories solved per orientation context and cue at trained and untrained timings (95% CI)

  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\ed05.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import figstyle as fs, palette as pal, qc  # noqa: E402

U = (fs.FULL - 3 * fs.GAP) / 4
XS = [k * (U + fs.GAP) for k in range(4)]
ROW1, ROW2, ROW3, ROW_H = 0.0, 48.0, 96.0, 44.0
H = ROW3 + ROW_H
AX_TOP = 6.5
EDGE = "#ABABAB"
LEARNER = {"connectome_readout_only": ("readout-only", pal.LEARNER["connectome readout-only"], "-"),
           "connectome_joint": ("joint", pal.LEARNER["connectome joint"], "-"),
           "connectome_shared_visibility_readout": ("shared readout", pal.LEARNER["visible goal"], "-"),
           "connectome_readout_only_reset_state": ("state reset", pal.LEARNER["reset state"], "-"),
           "connectome_untrained_readout": ("untrained readout", pal.LEARNER["untrained readout"], "-"),
           "gru": ("GRU", pal.LEARNER["GRU"], "-"), "fixed_decaying_trace": ("fixed trace", pal.LEARNER["fixed trace"], "-"),
           "neutral_core_real_graph": ("neutral, real graph", pal.LEARNER["neutral, real graph"], "-"),
           "neutral_core_rewired_graph": ("neutral, rewired", pal.LEARNER["neutral, rewired graph"], "-"),
           "connectome_new_readout": ("new readout", "#5E5E5E", (0, (2, 1.2))),
           "zero_update_transfer_from_M0": ("M0 readout, no updates", "#9A9A9A", (0, (1, 1)))}
FAM_ORDER = {"M0": 0, "M0vis": 0, "S1": 1, "S2": 2}


def clopper(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return float(lo), float(hi)


def truthy(s):
    return s.astype(str).str.lower().eq("true")


def split_key(cv, x, y):
    cv.key(x, y, [([dict(kind="mark", s=6, fc=fs.TEXT)], "trained timings"), ([dict(kind="mark", s=6, fc="white", ec=fs.TEXT, lw=0.5)],
                                                                            "untrained timings")], gap=1.6)


# ---------------------------------------------------------------- a: training curves

def panel_a(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "a")
    b = cm.table("extended/ed05/ed05b_memory_training_runs.csv")
    learners = [k for k in LEARNER if k in set(b["learner"])]
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="hline", colour=LEARNER[k][1], lw=0.9, w=2.4, ls=LEARNER[k][2])], LEARNER[k][0]) for k in learners[:5]],
           gap=1.4)
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="hline", colour=LEARNER[k][1], lw=0.9, w=2.4, ls=LEARNER[k][2])], LEARNER[k][0]) for k in learners[5:]],
           gap=1.4)
    aw = (w - 8.4 - 2 * 2.4) / 3
    for k, fam in enumerate(("M0", "S1", "S2")):
        ax = cv.axes(x0 + 8.4 + k * (aw + 2.4), y0 + AX_TOP + 3.0, aw, h - AX_TOP - 3.0 - 7.4)
        d = b[(b["family"] == fam) | ((fam == "M0") & (b["family"] == "M0vis"))]
        for (model, learner), g in d.groupby(["model", "learner"]):
            g = g.sort_values("assessment")
            _, col, ls = LEARNER[learner]
            x = np.maximum(g["updates"].to_numpy(float), 1)
            y = (g["solved"] / g["total"]).to_numpy(float)
            ax.plot(x, y, color=col, lw=0.8, ls=ls, drawstyle="steps-post", zorder=3 if learner == "connectome_readout_only" else 2)
            sd.add("a", "run", family=fam, model=model, learner=learner, assessments=int(len(g)), final_updates=int(g["updates"].max()),
                   final_share_solved=float(y[-1]))
        ax.set_xscale("log")
        ax.set_xlim(1, 2e5)
        ax.set_xticks([1, 100, 10000], ["1", "100", "10$^{4}$"])
        ax.set_ylim(0, 1.04)
        ax.set_yticks([0, 0.5, 1])
        fs.minor(ax, y=False)
        fs.clean(ax)
        ax.set_title("M0 and shared readout" if fam == "M0" else fam, fontsize=fs.PT_SMALL, pad=1.5)
        if k == 0:
            ax.set_ylabel("Training histories solved", labelpad=1)
        else:
            ax.tick_params(axis="y", labelleft=False)
        if k == 1:
            ax.set_xlabel("Updates", labelpad=1)


# ---------------------------------------------------------------- b: outcomes and the memoryless signature

def panel_b(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "b")
    c = cm.table("extended/ed05/ed05c_cue_pair_classes.csv")
    c = c.assign(vis=truthy(c["visible"]))
    rows = c[c["split"] == "train"][["family", "model", "learner", "vis"]].drop_duplicates()
    rows = rows.assign(fo=rows["family"].map(FAM_ORDER), lo=rows["learner"].map(list(LEARNER).index)).sort_values(["fo", "lo", "vis"])
    split_key(cv, x0 + 3.6, y0 + 1.5)
    lab_w = 30.6
    gap = 3.4
    aw1 = (w - lab_w - gap - 0.6) * 0.55
    aw2 = w - lab_w - gap - 2.2 - aw1
    top = y0 + AX_TOP + 2.6
    ah = h - AX_TOP - 2.6 - 7.4
    ax1 = cv.axes(x0 + lab_w, top, aw1, ah)
    ax2 = cv.axes(x0 + lab_w + aw1 + gap, top, aw2, ah)
    labels = []
    for i, r in enumerate(rows.itertuples()):
        lab = LEARNER[r.learner][0]
        if r.learner == "connectome_shared_visibility_readout":
            lab = "shared, goal " + ("visible" if r.vis else "hidden")
        labels.append(f"{'M0' if r.family == 'M0vis' else r.family}  {lab}")
        col = LEARNER[r.learner][1]
        for split, filled, off in (("train", True, -0.18), ("untrained", False, 0.18)):
            q = c[(c["model"] == r.model) & (c["vis"] == r.vis) & (c["split"] == split)].iloc[0]
            k, n = int(q["histories_solved"]), int(q["histories"])
            lo, hi = clopper(k, n)
            ax1.plot([lo, hi], [i + off, i + off], color=col, lw=0.6)
            ax1.plot([k / n], [i + off], marker="o", ms=2.3, mfc=col if filled else "white", mec=col, mew=0.6, ls="none")
            s_, m_ = int(q["same_first_push_both_cues"]), int(q["cue_pairs"])
            ax2.barh(i + off, s_ / m_, height=0.32, color=col if filled else "white", edgecolor=col, linewidth=0.5)
            sd.add("b", split, family=r.family, model=r.model, learner=r.learner, goal_visible=bool(r.vis), solved=k, histories=n, ci_low=lo,
                   ci_high=hi, same_first_push=s_, cue_pairs=m_)
    for ax in (ax1, ax2):
        ax.set_ylim(len(labels) - 0.45, -0.6)
        fs.clean(ax)
        ax.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL)
    ax1.axvline(0.5, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)))
    ax1.set_yticks(range(len(labels)), labels)
    ax1.set_xlim(0, 1)
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_xlabel("Histories solved (fraction)", labelpad=1)
    ax2.set_yticks([])
    ax2.set_xlim(0, 1)
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_xlabel("Same first push", labelpad=1)
    cv.text(x0 + lab_w + aw1 + gap, y0 + AX_TOP + 1.6, "memoryless signature", size=fs.PT_SMALL, va="center")
    cv.text(x0 + lab_w, y0 + AX_TOP + 1.6, "success", size=fs.PT_SMALL, va="center")


# ---------------------------------------------------------------- c: transfer across layouts

def panel_c(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    d = cm.table("main/fig5/fig5d_cross_layout_transfer.csv")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    split_key(cv, x0 + 3.6, y0 + 1.5)
    cv.text(x0 + 3.6, y0 + 3.9, f"numbers: Holm {fs.P_SYM}, cue pairs decoded both-correct vs both-wrong", size=fs.PT_SMALL, va="center")
    aw = (w - 8.4 - 10.0) / 2
    top = y0 + AX_TOP + 4.8
    ah = h - AX_TOP - 4.8 - 7.0
    ax = cv.axes(x0 + 8.4, top, aw, ah)
    dec = d[d["record_type"] == "cross_family_cue_decoding"]
    for i, fam in enumerate(("M0", "S1", "S2")):
        for split, off, filled in (("train", -0.16, True), ("untrained", 0.16, False)):
            g = dec[(dec["family"] == fam) & (dec["split"] == split)]
            k, n = int(truthy(g["decoded_correct"]).sum()), int(len(g))
            lo, hi = clopper(k, n)
            ax.plot([i + off] * 2, [lo, hi], color=fs.TEXT, lw=0.6)
            ax.plot([i + off], [k / n], marker="o", ms=2.6, mfc=fs.TEXT if filled else "white", mec=fs.TEXT, mew=0.6, ls="none")
            t = tests.loc[f"fig5d.decoder_pairs.{fam}.{split}"]
            ax.text(i + off, 1.06, fs.fmt_p(float(t["p_adjusted"])), ha="center", va="bottom", fontsize=fs.PT_SMALL, rotation=90)
            sd.add("c", "decoder from M0", family=fam, split=split, decoded=k, histories=n, ci_low=lo, ci_high=hi, p_holm=float(t["p_adjusted"]),
                   test_id=t.name)
    ax.axhline(0.5, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)))
    ax.set_xticks(range(3), ["M0", "S1", "S2"])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1])
    fs.clean(ax)
    ax.set_ylabel("Cue decoded (fraction)", labelpad=1)
    ax.set_xlabel("M0-fitted cue decoder on", labelpad=1)
    ax2 = cv.axes(x0 + 8.4 + aw + 10.0, top, aw, ah)
    zb = d[d["record_type"] == "zero_update_behaviour"]
    for i, fam in enumerate(("S1", "S2")):
        for split, off, filled in (("train", -0.16, True), ("untrained", 0.16, False)):
            g = zb[(zb["family"] == fam) & (zb["split"] == split)]
            k, n = int(truthy(g["solved"]).sum()), int(len(g))
            lo, hi = clopper(k, n)
            ax2.plot([i + off] * 2, [lo, hi], color="#7A7A7A", lw=0.6)
            ax2.plot([i + off], [k / n], marker="o", ms=2.6, mfc="#7A7A7A" if filled else "white", mec="#7A7A7A", mew=0.6, ls="none")
            sd.add("c", "zero-update transfer", family=fam, split=split, solved=k, histories=n, ci_low=lo, ci_high=hi)
    ax2.set_xticks(range(2), ["S1", "S2"])
    ax2.set_xlim(-0.6, 1.6)
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0, 0.5, 1])
    fs.clean(ax2)
    ax2.set_ylabel("Histories solved", labelpad=1)
    ax2.set_xlabel("M0 readout, no updates, on", labelpad=1)


# ---------------------------------------------------------------- e: S2 contexts

def panel_contexts(cv, sd, x0, y0, w, h, letter):
    cv.letter(x0, y0, letter)
    e = cm.table("main/fig5/fig5e_context_s2.csv")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    learners = [k for k in LEARNER if k in set(e["learner"])]
    split_key(cv, x0 + 3.6, y0 + 1.5)
    t_tr, t_un = tests.loc["fig5e.mcnemar.orientation.train"], tests.loc["fig5e.mcnemar.orientation.untrained"]
    cv.text(x0 + 3.6, y0 + 3.9, f"S2 readout-only, 0° vs 180°: Holm {fs.P_SYM} = {fs.fmt_p(float(t_tr['p_adjusted']))} (trained), "
                                f"{fs.fmt_p(float(t_un['p_adjusted']))} (untrained)", size=fs.PT_SMALL, va="center")
    lab_w = 22.0
    ax = cv.axes(x0 + lab_w, y0 + AX_TOP + 3.6, w - lab_w - 0.6, h - AX_TOP - 3.6 - 7.0)
    cols = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i, learner in enumerate(learners):
        col = LEARNER[learner][1]
        for j, (ctx, cue) in enumerate(cols):
            base = j * 1.25
            for split, off, filled in (("train", -0.18, True), ("untrained", 0.18, False)):
                r = e[(e["learner"] == learner) & (e["context"] == ctx) & (e["cue"] == cue) & (e["split"] == split)].iloc[0]
                ax.plot([base + r["ci_low"], base + r["ci_high"]], [i + off] * 2, color=col, lw=0.6)
                ax.plot([base + r["estimate"]], [i + off], marker="o", ms=2.2, mfc=col if filled else "white", mec=col, mew=0.5, ls="none")
                sd.add(letter, split, learner=learner, context=int(ctx), cue=int(cue), solved=int(r["numerator"]), histories=int(r["denominator"]),
                       estimate=float(r["estimate"]), ci_low=float(r["ci_low"]), ci_high=float(r["ci_high"]))
    for j, (ctx, cue) in enumerate(cols):
        ax.axvline(j * 1.25, color=EDGE, lw=fs.LW_HAIR, zorder=0)
        ax.axvline(j * 1.25 + 1.0, color=EDGE, lw=fs.LW_HAIR, zorder=0)
        ax.text(j * 1.25 + 0.5, -0.75, f"{180 * ctx}°, cue {cue}", ha="center", va="bottom", fontsize=fs.PT_SMALL)
    ax.set_yticks(range(len(learners)), [LEARNER[k][0] for k in learners])
    ax.set_ylim(len(learners) - 0.5, -0.6)
    ax.set_xlim(-0.1, 4 * 1.25 - 0.15)
    ax.set_xticks([j * 1.25 + v for j in range(4) for v in (0, 1)], ["0", "1"] * 4)
    ax.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL)
    fs.clean(ax)
    ax.set_xlabel("Histories solved (fraction, 95% CI)", labelpad=1)


# ---------------------------------------------------------------- d: margin by delay

def panel_margins(cv, sd, x0, y0, w, h, letter):
    cv.letter(x0, y0, letter)
    a = cm.table("extended/ed05/ed05a_memory_behaviour_all.csv")
    a = a.assign(vis=truthy(a["visible"]))
    sel = [("remote-dynamic-M0-head_only", False, "readout-only", pal.LEARNER["connectome readout-only"]),
           ("remote-dynamic-M0-gru", False, "GRU", pal.LEARNER["GRU"]),
           ("supp-M0-visibility", True, "shared, goal visible", pal.LEARNER["visible goal"]),
           ("remote-dynamic-M0-reset", False, "state reset", pal.LEARNER["reset state"])]
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="hline", colour=c, lw=0.9, w=2.4)], lab) for _, _, lab, c in sel], gap=1.4)
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="vline", h=2.0, lw=0.5, colour=EDGE)], "trained delays (prefixes 0, 1 and 4 only)")])
    ax = cv.axes(x0 + 8.4, y0 + AX_TOP, w - 8.8, h - AX_TOP - 9.0)
    delays = (0, 20, 40, 80, 160, 320, 640)
    xs = np.arange(len(delays))
    for k_, (model, vis, lab, col) in enumerate(sel):
        d = a[(a["model"] == model) & (a["vis"] == vis)]
        m = np.array([float(d[d["delay_ms"] == dl]["first_decision_margin"].mean()) for dl in delays])
        lo = np.array([float(d[d["delay_ms"] == dl]["first_decision_margin"].quantile(0.25)) for dl in delays])
        hi = np.array([float(d[d["delay_ms"] == dl]["first_decision_margin"].quantile(0.75)) for dl in delays])
        xo = xs + (k_ - 1.5) * 0.1
        for xx, l_, h_ in zip(xo, lo, hi):
            ax.plot([xx, xx], [l_, h_], color=col, lw=0.5, zorder=2)
        ax.plot(xo, m, color=col, lw=0.9, marker="o", ms=2.0, mew=0, zorder=3)
        for dl, mm, l_, h_ in zip(delays, m, lo, hi):
            sd.add(letter, model, goal_visible=vis, delay_ms=dl, mean_margin=float(mm), q25=float(l_), q75=float(h_))
    for k in (0, 2, 4):
        ax.axvline(k, color=EDGE, lw=fs.LW_HAIR, zorder=0)
    ax.axhline(0, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)))
    ax.set_xticks(xs, [str(dl) for dl in delays])
    ax.set_xlim(-0.4, len(delays) - 0.6)
    fs.clean(ax)
    ax.set_xlabel("Delay after the cue (ms)", labelpad=1)
    ax.set_ylabel("First-decision margin (logit)", labelpad=1)


def main():
    cv = fs.Canvas(fs.FULL, H, ppi_min=300)
    sd = qc.SourceData("ed05")
    for rel in ("extended/ed05/ed05a_memory_behaviour_all.csv", "extended/ed05/ed05b_memory_training_runs.csv",
                "extended/ed05/ed05c_cue_pair_classes.csv", "main/fig5/fig5d_cross_layout_transfer.csv", "main/fig5/fig5e_context_s2.csv",
                "stats/statistical_tests.csv"):
        sd.source(cm.RESULTS / rel)
    panel_a(cv, sd, XS[0], ROW1, 2 * U + fs.GAP, ROW_H)
    panel_b(cv, sd, XS[2], ROW1, 2 * U + fs.GAP, ROW2 + ROW_H)
    panel_c(cv, sd, XS[0], ROW2, 2 * U + fs.GAP, ROW_H)
    panel_margins(cv, sd, XS[0], ROW3, 2 * U + fs.GAP, ROW_H, "d")
    panel_contexts(cv, sd, XS[2], ROW3, 2 * U + fs.GAP, ROW_H, "e")
    cm.finish(cv, "ed05", sd, kind="extended")


if __name__ == "__main__":
    main()
