"""v4 Fig. 7 (Cell) - Disease steals access, not memories (174 mm wide, full page).

A  credible cases only: 24 fly pathologies from the literature to 4 retained by the declared criteria, and a new case from
   the transcriptome (the ageing operator, S7K)
B  four regimes, one rule: the brain state at learning and at test, and the trained histories solved (dose 1, readout
   trained). Access survives when the test state is the learning state
C  lost access, spared learning: the lesion after learning collapses memory with dose; learning under the lesion is spared
   (48 runs)
D  relearning in the lesioned state restores access in part (readout retrained)
E  late restoration of excitability removes access to memories formed in the disease state (24 runs)
F  case dissociation: only the pan-neuronal Kv4 case reaches visual memory and escape; the circuit cases change olfaction
   only, Aβ42-late and tau in opposite directions
G  dark band, "disease painted on the brain": Aβ42 raises Shal mRNA (AD Fly Cell Atlas); the uniform Kv4 loss speeds
   cells across the CNS; the same mean lesion patterned by Shal; the ageing operator with the Kenyon cells outlined
H  cell-type vulnerability: patterned by Shal, the lesion drives the high-Shal cells into runaway at 1.5-2x
I  memories formed under a lesion are locked to its cell-by-cell pattern
J  recall needs the learning state at both ends (the lesion link, amendment 0015): the trace written and held in one state
   and read in another (the full state moved at the end of the delay), trained histories solved at dose 1 for readouts
   trained healthy and trained under the lesion; and across doses the stored trace's and the reading's cosine with the
   healthy brain: the lesion rewrites the trace into other cells and the same healthy trace is read differently
   (the ageing operator and the energy cost moved to S7I and S7K)

Every number in a label is read from manuscript/results (plan_v2 via P12) or from the render metadata. Reader-facing names
only (NARRATIVE_V4_OUTLINE.md section 12). Not shown (RESULTS_INVENTORY_V4.md, class c): the olfactory targeted-vs-random
tests, the relearning dose trend and the joint-learner relearning comparison (P15), the equivalence of the Shal pattern and
the uniform lesion at 0.5-1x (P17) and the late-life atlas direction (P18).

Run after P12, render/v4_fig7_tiles.py and render/v4_icons.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\v4\\fig7.py
"""
import json
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
from lib import figstyle as fs, palette as pal, qc  # noqa: E402

cm.use_v4()
TILES = cm.RENDERS / "v4" / "fig7"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
ICONS = cm.RENDERS / "v4" / "icons"
ICON = json.loads((ICONS / "icons.json").read_text(encoding="utf-8"))   # B, F: flat silhouettes in the MaleCNS grammar
BAND_H, TW, TITLE_MM = META["band_mm"], META["tile_mm"], META["title_mm"]
U = fs.unit(1)
XS = [fs.unit_x(i) for i in range(4)]
R1, R1_H = 0.0, 48.0
R2, R2_H = R1 + R1_H + 3.4, 48.0
BAND_Y = R2 + R2_H + 4.2
R4, R4_H = BAND_Y + BAND_H + 4.2, 45.0
H = R4 + R4_H
WHITE = fs.ON_DARK
NL_ = chr(10)
READOUT, BRAIN = pal.STATE_SET["m0"], pal.STATE_SET["m0_joint"]
LEARNER_COL = {"readout": READOUT, "brain and readout": BRAIN}
LEARNER_LABEL = {"readout": "readout trained", "brain and readout": "brain + readout trained"}
HEALTHY_STATE, LESION_STATE = "#D9D9D9", "#7B3294"
SHAL, SHAL_LIGHT = "#B4009E", "#E07BD3"
DOSE_SHADE = {0.5: "#BDBDBD", 1.0: "#8C8C8C", 1.5: "#525252", 2.0: "#1A1A1A"}


def tile_text(cv, x, y, s, size=None, **kw):
    t = cv.text(x, y, s, size=size or fs.PT_SMALL, color=WHITE, **kw)
    t.set_path_effects(fs.dark_halo(0.9))
    return t


def white_letter(cv, x, y, s):
    cv.letter(x, y, s)
    cv.fig.texts[-1].set_color(WHITE)


def title(cv, x0, y0, letter, text):
    cv.letter(x0, y0, letter)
    cv.text(x0 + 4.0, y0 + 0.5, text, size=fs.PT, va="top", fontweight="bold")


def tests(family):
    t = cm.table("plan_v2/tests.csv")
    return t[t["family"] == family]


def gradient(cv, x, y, w, h, colours):
    """Vertical key bar, first colour at the top."""
    ax = cv.axes(x, y, w, h)
    ax.imshow(np.asarray(colours)[:, None, :3] * np.ones((1, 2, 1)), aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax.set_axis_off()


def icon_img(name):
    return np.asarray(Image.open(ICONS / f"{name}.png").convert("RGB"))


def icon_mm(name):
    """The icon's native size (mm) at the print resolution."""
    h_px, w_px = icon_img(name).shape[:2]
    return w_px / fs.DPI / fs.MM, h_px / fs.DPI / fs.MM


def fmt_n(v):
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"


# ---------------------------------------------------------------- A: credibility funnel

def panel_funnel(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "A", "Credible cases only")
    f = cm.table("plan_v2/fig7_funnel.csv").sort_values("order")
    n0 = float(f["cases"].max())
    bw_max = w - 9.0
    for i, (_, r) in enumerate(f.iterrows()):
        yy = y0 + 5.6 + i * 5.85
        lab = re.sub(r"\s*\(C\d(?:–C\d)?\)", "", str(r["stage"]))
        last = i == len(f) - 1
        cv.text(x0 + 1.0, yy, lab, size=fs.PT_SMALL, va="top", fontweight="bold" if last else "normal")
        bw = bw_max * float(r["cases"]) / n0
        cv.box(x0 + 1.0, yy + 2.5, bw, 1.7, fc=LESION_STATE if last else "#BDBDBD")
        cv.text(x0 + 1.0 + bw + 0.8, yy + 3.35, f"{int(r['cases'])}", size=fs.PT_SMALL, va="center")
        sd.add("A", "stage", stage=lab, cases=int(r["cases"]))
    cv.text(x0 + 1.0, y0 + h - 1.4, "+ aging, from transcriptomes", size=fs.PT_SMALL, va="center", color=fs.TEXT2)


# ---------------------------------------------------------------- B: four regimes, one rule

def panel_regimes(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "B", "Four regimes, one rule")
    la = cm.table("plan_v2/fig7_lost_access.csv")
    rl = cm.table("plan_v2/fig7_relearning.csv")
    rs = cm.table("plan_v2/fig7_restoration.csv")
    dose, lrn = 1.0, "readout"
    a = la[(la["regime"] == "lesion after learning") & (la["trained_parameters"] == lrn) & (la["dose"] == dose)]["trained_solved"].mean()
    b = la[(la["regime"] == "learning under the lesion") & (la["trained_parameters"] == lrn) & (la["dose"] == dose)]["trained_solved"].mean()
    r_ = rl[(rl["trained_parameters"] == lrn) & (rl["dose"] == dose)]
    c_after = r_["solved_100"].mean()
    t_ = rs[(rs["trained_parameters"] == lrn) & (rs["dose"] == dose)]["trained_solved_restored"].mean()
    total = int(r_["total"].iloc[0])
    # the brain state at learning and at test as CNS silhouettes; = or not-equal where the two states match or differ
    rows = [(f"lesion after{NL_}learning", "cns_healthy", "cns_lesion", "≠", a),
            (f"learning under{NL_}the lesion", "cns_lesion", "cns_lesion", "=", b),
            (f"relearning{NL_}in the lesion", "cns_split", "cns_lesion", "", c_after),
            (f"late{NL_}restoration", "cns_lesion", "cns_healthy", "≠", t_)]
    iw, ih = icon_mm("cns_healthy")
    cx_learn, cx_test, bar_x = x0 + 16.4, x0 + 24.0, x0 + 30.6
    bar_w = x0 + w - 5.2 - bar_x
    cv.text(cx_learn + iw / 2, y0 + 6.6, "learned", size=fs.PT_SMALL, ha="center", va="center", color=fs.TEXT2)
    cv.text(cx_test + iw / 2, y0 + 6.6, "tested", size=fs.PT_SMALL, ha="center", va="center", color=fs.TEXT2)
    cv.text(bar_x, y0 + 6.6, f"solved{NL_}(of {total})", size=fs.PT_SMALL, ha="left", va="center", color=fs.TEXT2, linespacing=1.0)
    for i, (name, learned, tested, rel, val) in enumerate(rows):
        yc = y0 + 12.0 + i * 8.0
        cv.text(x0 + 1.0, yc, name, size=fs.PT_SMALL, va="center", linespacing=1.0)
        for xx, which in ((cx_learn, learned), (cx_test, tested)):
            cv.tile(xx, yc - ih / 2, icon_img(which), dark=False, label=f"B {which} {i}")             # native size: 600 ppi
        if rel:
            cv.text((cx_learn + iw + cx_test) / 2, yc, rel, size=fs.PT, ha="center", va="center")
        cv.box(bar_x, yc - 0.9, bar_w * val / total, 1.8, fc=fs.TEXT)
        cv.box(bar_x, yc - 0.9, bar_w, 1.8, fc="none", ec=fs.GRID, lw=0.3)
        cv.text(bar_x + bar_w + 0.6, yc, fmt_n(val), size=fs.PT_SMALL, va="center")
        sd.add("B", "regime", regime=name.replace(NL_, " "), learned_in=learned, tested_in=tested, solved=float(val), of=total, dose=dose,
               trained=lrn)
    cv.key(x0 + 1.0, y0 + h - 1.4, [([dict(kind="box", fc=ICON["healthy"], ec=ICON["line"], lw=0.3)], "healthy"),
                                    ([dict(kind="box", fc=ICON["lesion"])], "Kv4 loss, dose 1")], size=fs.PT_SMALL, gap=1.6)


# ---------------------------------------------------------------- C: lost access, spared learning

def panel_lost(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "C", "Lost access, spared learning")
    d = cm.table("plan_v2/fig7_lost_access.csv")
    ax = cv.axes(x0 + 9.0, y0 + 7.4, 38.0, h - 15.4)
    rng = np.random.default_rng(71)
    doses = sorted(d["dose"].unique())
    off = {"readout": -0.06, "brain and readout": 0.06}
    for lrn in ("readout", "brain and readout"):
        col = LEARNER_COL[lrn]
        b = d[(d["regime"] == "learning under the lesion") & (d["trained_parameters"] == lrn)]
        for dose, q in b.groupby("dose"):
            ax.scatter(dose + off[lrn] + rng.uniform(-0.035, 0.035, len(q)), q["trained_solved"], s=5, facecolor=col, edgecolor="none", alpha=0.8, zorder=3)
        mb = b.groupby("dose")["trained_solved"].mean()
        ax.plot(mb.index + off[lrn], mb.to_numpy(), color=col, lw=0.8, zorder=2)
        a = d[(d["regime"] == "lesion after learning") & (d["trained_parameters"] == lrn)].sort_values("dose")
        ax.plot(a["dose"] + off[lrn], a["trained_solved"], color=col, lw=0.8, ls=(0, (2.2, 1.4)), marker="o", ms=3.2, mfc="white", mec=col, mew=0.7,
                zorder=4)
        for _, r in d[d["trained_parameters"] == lrn].iterrows():
            sd.add("C", "run", regime=r["regime"], trained=lrn, dose=float(r["dose"]), replicate=int(r["replicate"]), trained_solved=int(r["trained_solved"]),
                   of=int(r["trained_histories"]))
    ax.set_xlim(-0.2, max(doses) + 0.2)
    ax.set_ylim(0, 19.5)
    ax.set_yticks([0, 9, 18])
    ax.set_xticks(doses, [f"{x:g}" for x in doses])
    ax.set_xlabel("Kv4-loss dose (× calibrated)", labelpad=0.8)
    ax.set_ylabel("Trained histories solved", labelpad=0.8)
    fs.clean(ax)
    t = tests("F12.C01.primary")
    trend = t[t["test"].astype(str).str.startswith("memory:")]["p_adjusted"].iloc[0]
    learn = t[t["test"].astype(str).str.startswith("learning under")]["p_adjusted"]
    n_runs = int((d["regime"] == "learning under the lesion").sum())
    tx = x0 + 50.0
    cv.key(tx, y0 + 8.6, [([dict(kind="hline", colour=fs.TEXT, lw=0.8, w=3.0)], "learning under the lesion")], size=fs.PT_SMALL)
    cv.text(tx + 3.8, y0 + 10.4, f"Holm {fs.P_SYM} = {fs.fmt_p(float(learn.min()))}–{fs.fmt_p(float(learn.max()))}",
            size=fs.PT_SMALL, va="top", color=fs.TEXT2, linespacing=1.0)
    cv.key(tx, y0 + 18.4, [([dict(kind="hline", colour=fs.TEXT, lw=0.8, w=3.0, ls=(0, (2.2, 1.4)), mark_fc="white", s=7)], "lesion after learning")],
           size=fs.PT_SMALL)
    cv.text(tx + 3.8, y0 + 20.2, f"Holm {fs.P_SYM} = {fs.fmt_p(float(trend))}", size=fs.PT_SMALL, va="top",
            color=fs.TEXT2, linespacing=1.0)
    cv.key(tx, y0 + 31.6, [([dict(kind="mark", s=7, fc=READOUT, ec=READOUT)], LEARNER_LABEL["readout"])], size=fs.PT_SMALL)
    cv.key(tx, y0 + 34.6, [([dict(kind="mark", s=7, fc=BRAIN, ec=BRAIN)], LEARNER_LABEL["brain and readout"])], size=fs.PT_SMALL)
    sd.add("C", "tests", dose_trend_holm_p=float(trend), learning_trend_holm_p=learn.astype(float).tolist(), learning_runs=n_runs)


# ---------------------------------------------------------------- D: relearning

def panel_relearn(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "D", "Relearning restores in part")
    r = cm.table("plan_v2/fig7_relearning.csv")
    ax = cv.axes(x0 + 9.0, y0 + 9.4, w - 11.4, h - 17.4)
    steps = [0, 50, 100]
    for _, q in r.iterrows():
        col = DOSE_SHADE[float(q["dose"])]
        ax.plot(steps, [q["solved_0"], q["solved_50"], q["solved_100"]], color=col, lw=0.6, marker="o", ms=1.8, zorder=3)
        sd.add("D", "run", dose=float(q["dose"]), replicate=int(q["replicate"]), solved=[int(q["solved_0"]), int(q["solved_50"]), int(q["solved_100"])],
               of=int(q["total"]), recovery_fraction=float(q["recovery_fraction"]))
    total = int(r["total"].iloc[0])
    ax.axhline(total, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
    ax.set_xlim(-8, 108)
    ax.set_xticks(steps)
    ax.set_ylim(0, total + 1.5)
    ax.set_yticks([0, 9, 18])
    ax.set_xlabel("Readout updates after the lesion", labelpad=0.8)
    ax.set_ylabel("Trained histories solved", labelpad=0.8)
    fs.clean(ax)
    rec = float(r["recovery_fraction"].mean())
    items = [([dict(kind="hline", colour=c, lw=0.8, w=1.8)], f"{d:g}" if d != 2.0 else f"{d:g}× dose") for d, c in DOSE_SHADE.items()]
    cv.key(x0 + 4.0, y0 + 6.2, items, size=fs.PT_SMALL, gap=0.9)
    sd.add("D", "summary", mean_recovery_fraction=rec, runs=int(len(r)), trained="readout")


# ---------------------------------------------------------------- E: late restoration

def panel_restore(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "E", "Late restoration removes access")
    r = cm.table("plan_v2/fig7_restoration.csv")
    ax = cv.axes(x0 + 9.0, y0 + 9.4, w - 11.4, h - 17.4)
    rng = np.random.default_rng(72)
    for _, q in r.iterrows():
        col = LEARNER_COL[q["trained_parameters"]]
        j = rng.uniform(-0.06, 0.06)
        ax.plot([0 + j, 1 + j], [q["trained_solved_as_trained"], q["trained_solved_restored"]], color=col, lw=0.5, alpha=0.7, zorder=2)
        ax.scatter([0 + j], [q["trained_solved_as_trained"]], s=6, facecolor=col, edgecolor=col, linewidth=0.3, zorder=3)
        ax.scatter([1 + j], [q["trained_solved_restored"]], s=6, facecolor="white", edgecolor=col, linewidth=0.5, zorder=3)
        sd.add("E", "run", trained=q["trained_parameters"], dose=float(q["dose"]), replicate=int(q["replicate"]),
               as_trained=float(q["trained_solved_as_trained"]), restored=float(q["trained_solved_restored"]))
    ax.set_xlim(-0.4, 1.4)
    ax.set_xticks([0, 1], ["in the lesion", "restored"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 19.5)
    ax.set_yticks([0, 9, 18])
    ax.set_ylabel("Trained histories solved", labelpad=0.8)
    fs.clean(ax)
    t = tests("F12.rescue")
    est = t["estimate"].astype(float)
    cv.text(x0 + w - 0.8, y0 + 6.2, f"{len(r)} runs; BH {fs.P_SYM} ≤ {fs.fmt_p(float(t['p_adjusted'].max()))}", size=fs.PT_SMALL, ha="right",
            va="center", color=fs.TEXT2)
    sd.add("E", "tests", runs=int(len(r)), bh_p_max=float(t["p_adjusted"].max()), change_range=[float(est.min()), float(est.max())])


# ---------------------------------------------------------------- F: case dissociation

def panel_cases(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "F", "Only the pan-neuronal case reaches visual memory")
    p = cm.table("plan_v2/fig7_profiles.csv")
    cases = cm.table("plan_v2/fig7_cases.csv").set_index("code")["case"]
    order = list(dict.fromkeys(p["code"]))
    cols = [("memory_relative", f"Visual memory{NL_}(× healthy)", (0, 1.15), [0, 0.5, 1], False),
            ("movement_gf_relative", f"Escape{NL_}(× healthy)", (0.85, 1.55), [1, 1.4], False),
            ("olfaction_mbon_accuracy_change", f"Odor memory{NL_}(Δ accuracy)", (-0.25, 0.09), [-0.2, 0], False),
            ("activity_relative", f"Activity{NL_}(× healthy)", (0.7, 150), [1, 10, 100], True)]
    lab_w, gap = 32.4, 1.6
    bw, bh = icon_mm("brain_C01")
    pw = (w - lab_w - 1.0 - gap * (len(cols) - 1)) / len(cols)
    top, ph = y0 + 12.4, h - 21.6
    rows = {c: i for i, c in enumerate(order)}
    for i, c in enumerate(order):
        name = cases[c]
        head, _, tail = name.partition(" (")
        tail = NL_.join(textwrap.wrap(tail.rstrip(")"), 17, break_on_hyphens=True))
        yc = top + ph * (i + 0.5) / len(order)
        cv.tile(x0 + 1.0, yc - bh / 2, icon_img(f"brain_{c}"), dark=False, label=f"F site {c}")
        cv.text(x0 + bw + 2.2, yc, f"{head}{NL_}{tail}", size=fs.PT_SMALL, va="center", linespacing=1.0)
        sd.add("F", "site", case=name, acts_on=ICON["icons"][f"brain_{c}"]["site"])
    for k, (col, lab, lim, ticks, log) in enumerate(cols):
        ax = cv.axes(x0 + lab_w + k * (pw + gap), top, pw, ph)
        ref = 0.0 if col.startswith("olfaction") else 1.0
        ax.axvline(ref, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
        for _, r in p.iterrows():
            v = float(r[col])
            ax.scatter([v], [rows[r["code"]]], s=9, facecolor=DOSE_SHADE[float(r["dose"])], edgecolor=fs.TEXT, linewidth=0.3, zorder=3)
            sd.add("F", "phenotype", case=cases[r["code"]], dose=float(r["dose"]), measure=col, value=v)
        if log:
            ax.set_xscale("log")
            fs.minor(ax, x=False, y=False)
        ax.set_xlim(*lim)
        ax.set_xticks(ticks, [f"{t:g}".replace("-", fs.MINUS) for t in ticks])
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=fs.PT_SMALL)
        fs.clean(ax, left=False)
        ax.spines["left"].set_visible(False)
        cv.text(x0 + lab_w + k * (pw + gap) + pw / 2, top - 0.6, lab, size=fs.PT_SMALL, ha="center", va="bottom", linespacing=1.0)
    comp = tests("F12.comparison")
    eq = tests("F12.regimeA.equivalence")
    cv.text(x0 + w - 0.8, y0 + h - 3.0, f"odor, Holm {fs.P_SYM} ≤ {fs.fmt_p(float(comp['p_adjusted'].max()))}", size=fs.PT_SMALL,
            ha="right", va="center", color=fs.TEXT2)
    cv.text(x0 + w - 0.8, y0 + h - 0.6, f"visual, TOST Holm {fs.P_SYM} = {fs.fmt_p(float(eq['p_adjusted'].max()))}",
            size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)
    items = [([dict(kind="mark", s=8, fc=c, ec=fs.TEXT, lw=0.3)], f"{d:g}" if d != 2.0 else f"{d:g}× dose") for d, c in DOSE_SHADE.items()]
    xe = cv.key(x0 + lab_w, y0 + 5.6, items, size=fs.PT_SMALL, gap=1.0)
    cv.key(xe + 3.0, y0 + 5.6, [([dict(kind="box", fc=ICON["lesion"])], "where the case acts")], size=fs.PT_SMALL, gap=1.0)
    sd.add("F", "tests", comparison_holm_p_max=float(comp["p_adjusted"].max()), equivalence_holm_p=float(eq["p_adjusted"].max()))


# ---------------------------------------------------------------- G: disease painted on the brain

def band_key(cv, x, lo_lab, hi_lab, label):
    """Diverging key in the black space right of the VNC: high (orange) at the top, low (blue) at the bottom."""
    bx, by, half, gap = x + TW - 2.4, BAND_Y + BAND_H - 21.0, 5.0, 1.0
    u = np.linspace(1, 0, 96)
    gradient(cv, bx, by, 1.2, half, pal.SIGNED_ON_DARK(0.5 + 0.22 + 0.28 * u))
    gradient(cv, bx, by + half + gap, 1.2, half, pal.SIGNED_ON_DARK(0.5 - 0.22 - 0.28 * u[::-1]))
    tile_text(cv, bx - 0.4, by + 0.3, hi_lab, va="center", ha="right")
    tile_text(cv, bx - 0.4, by + 2 * half + gap - 0.3, lo_lab, va="center", ha="right")
    tile_text(cv, bx + 1.2, by - 0.9, label, va="bottom", ha="right", linespacing=1.0)


def panel_band(cv, sd):
    cells = cm.table("plan_v2/fig7_cells.csv")
    types = cm.table("plan_v2/fig7_adfca_types.csv")
    pan = tests("F14.SC.adfca.pan")
    pan = pan[pan["test"].astype(str).str.startswith("Abeta42:pan-neuronal") & pan["test"].astype(str).str.endswith(":Shal")].iloc[0]
    faster = int(((cells["rate_uniform_dose1_hz"] - cells["rate_healthy_hz"]) > 0).sum())
    w = cells["kv4_weight"]
    kc = cells["kenyon_cell"].astype(str).str.lower().eq("true")
    up = int((types["shal_log2fc"] > 0).sum())
    lines = {
        "shal": ["Aβ42 raises Shal mRNA", f"up in {up} of {len(types)} cell types",
                 f"pan-neuronal +{float(pan['estimate']):.2f}, Holm {fs.P_SYM} = {fs.fmt_p(float(pan['p_adjusted']))}"],
        "uniform": ["Uniform Kv4 loss, dose 1", f"rate {cells['rate_healthy_hz'].mean():.2f} → {cells['rate_uniform_dose1_hz'].mean():.2f} Hz",
                    f"{faster:,} cells fire faster"],
        "pattern": ["Kv4 loss patterned by Shal", f"same mean, weight {w.min():.0f}–{w.max():.1f}", "high-Shal cells lose most Kv4"],
        "ageing": ["Aging operator", "old vs young K$^+$ channels", f"{int(kc.sum()):,} Kenyon cells outlined"],
    }
    keys = {"shal": (f"{fs.MINUS}{META['shal_max']:g}", f"+{META['shal_max']:g}", f"log2 fold{NL_}change"),
            "uniform": (f"{fs.MINUS}{META['rate_max_hz']:g}", f"+{META['rate_max_hz']:g} Hz", f"rate{NL_}change"),
            "pattern": (f"{1 - META['weight_max']:g}", f"{1 + META['weight_max']:g}", f"lesion{NL_}weight"),
            "ageing": (f"{fs.MINUS}{META['age_max']:g}", f"+{META['age_max']:g}", f"log2 K$^+${NL_}factor")}
    for k, name in enumerate(("shal", "uniform", "pattern", "ageing")):
        x = k * (TW + fs.GUTTER)
        meta = json.loads((TILES / f"{name}.json").read_text(encoding="utf-8"))
        img = np.asarray(Image.open(TILES / f"{name}.png").convert("RGB"))
        cv.tile(x, BAND_Y, img, width=TW, height=BAND_H, group="band", label=name)
        tx = x + (4.4 if k == 0 else 1.2)
        for j, (s, size) in enumerate(zip(lines[name], (fs.PT, fs.PT_SMALL, fs.PT_SMALL))):
            tile_text(cv, tx, BAND_Y + 1.1 + (0 if j == 0 else 1.0 + 2.35 * j), s, va="top", size=size)
        band_key(cv, x, *keys[name])
        if name == "ageing":
            ppm = img.shape[1] / TW
            for hull in meta["kenyon_hulls_px"]:
                pts = np.asarray(hull + [hull[0]], float) / ppm
                cv.line(x + pts[:, 0], BAND_Y + pts[:, 1], WHITE, lw=0.5, ls=(0, (1.6, 1.0)))
        sd.add("G", "tile", tile=name, lines=lines[name], **{k_: v for k_, v in meta.items() if k_ != "kenyon_hulls_px"})
    white_letter(cv, 1.2, BAND_Y + 0.9, "G")
    meta = json.loads((TILES / "ageing.json").read_text(encoding="utf-8"))
    img = np.asarray(Image.open(TILES / "ageing.png").convert("RGB"))
    length = 100 / meta["scale_um_per_px"] / (img.shape[1] / TW)
    x = 3 * (TW + fs.GUTTER)
    cv.line([x + TW - 1.4 - length, x + TW - 1.4], [BAND_Y + BAND_H - 1.8, BAND_Y + BAND_H - 1.8], WHITE, lw=1.0, solid_capstyle="butt")
    sd.add("G", "summary", shal_types_up=up, shal_types=int(len(types)), pan_shal_log2fc=float(pan["estimate"]), pan_holm_p=float(pan["p_adjusted"]),
           cells_faster_uniform_dose1=faster, kenyon_cells=int(kc.sum()), scale_bar_um=100)


# ---------------------------------------------------------------- H: high-Shal runaway

def panel_runaway(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "H", "High-Shal cells run away")
    r = cm.table("plan_v2/fig7_shal_runaway.csv")
    ax = cv.axes(x0 + 9.0, y0 + 9.4, w - 11.4, h - 17.4)
    styles = [("uniform", fs.TEXT, "-"), ("Shal-patterned (type, class means)", SHAL, "-")]     # the global-mean variant: Figure S7
    for pat, col, ls in styles:
        q = r[r["pattern"] == pat].sort_values("dose")
        ax.plot(q["dose"], q["task_mean_hz"], color=col, lw=0.8, ls=ls, marker="o", ms=1.8, zorder=3)
        for _, rr in q.iterrows():
            sd.add("H", "condition", pattern=pat, dose=float(rr["dose"]), task_mean_hz=float(rr["task_mean_hz"]),
                   share_above_100hz=float(rr["share_above_100hz"]), trained_solved=int(rr["trained_solved"]))
    cls = r[(r["pattern"] == styles[1][0]) & (r["dose"] >= 1.5)].sort_values("dose")
    for k, (_, rr) in enumerate(cls.iterrows()):
        lab = f"{100 * rr['share_above_100hz']:.0f}%" + (" > 100 Hz" if k == 0 else "")
        ax.text(rr["dose"] - 0.06, rr["task_mean_hz"] * 1.25, lab, fontsize=fs.PT_SMALL, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.2, 1000)
    ax.set_yticks([1, 10, 100], ["1", "10", "100"])
    fs.minor(ax, x=False, y=False)
    ax.set_xlim(-0.15, 2.15)
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Kv4-loss dose", labelpad=0.8)
    ax.set_ylabel("Task rate (Hz)", labelpad=0.8)
    fs.clean(ax)
    t = tests("F14.SC.kv4")
    hi = t[t["test"].astype(str).str.contains(r"^class dose (?:1\.5|2):", regex=True)]
    if len(hi) != 2:
        raise RuntimeError("H: expected the two class-pattern tests at 1.5 and 2x")
    cv.key(x0 + 4.0, y0 + 6.2, [([dict(kind="hline", colour=fs.TEXT, lw=0.8, w=1.8)], "uniform"),
                                ([dict(kind="hline", colour=SHAL, lw=0.8, w=1.8)], "patterned by Shal")], size=fs.PT_SMALL, gap=1.2)
    sd.add("H", "tests", memory_holm_p_max_at_high_dose=float(hi["p_adjusted"].max()), tests=int(len(hi)), family="F14.SC.kv4 (class pattern)")


# ---------------------------------------------------------------- I: pattern lock

def panel_lock(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "I", "Locked to the lesion's pattern")
    r = cm.table("plan_v2/fig7_pattern_lock.csv")
    ax = cv.axes(x0 + 9.0, y0 + 9.4, w - 11.4, h - 17.4)
    cols = ["trained_as_trained", "trained_kv4", "trained_restored"]
    rng = np.random.default_rng(73)
    for _, q in r.iterrows():
        col = LEARNER_COL[q["trained_parameters"]]
        j = rng.uniform(-0.05, 0.05, 3)
        ax.plot(np.arange(3) + j, [q[c] for c in cols], color=col, lw=0.6, marker="o", ms=2.0, zorder=3)
        sd.add("I", "run", trained=q["trained_parameters"], replicate=int(q["replicate"]), as_trained=int(q[cols[0]]), shal_pattern=int(q[cols[1]]),
               restored=int(q[cols[2]]))
    ax.set_xlim(-0.4, 2.4)
    ax.set_xticks([0, 1, 2], [f"as{NL_}learned", f"Shal{NL_}pattern", "restored"], linespacing=1.0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 19.5)
    ax.set_yticks([0, 9, 18])
    ax.set_ylabel("Trained histories solved", labelpad=0.8)
    fs.clean(ax)
    cv.key(x0 + 4.0, y0 + 6.2, [([dict(kind="mark", s=6, fc=READOUT, ec=READOUT)], "readout"),
                                ([dict(kind="mark", s=6, fc=BRAIN, ec=BRAIN)], "brain + readout")], size=fs.PT_SMALL, gap=1.2)


# ---------------------------------------------------------------- J: both ends (where the trace is written and where it is read)

LINK_DOSE = 1.0
STATES = ("healthy", "lesion")
STATE_LABEL = {"healthy": "healthy", "lesion": "lesion"}
MATRIX_LO, MATRIX_HI = np.array([0.97, 0.97, 0.97]), np.array([0.62, 0.62, 0.62])   # dark numbers read on every fill


def link_matrix(o, kind):
    """Mean trained histories solved (of 18) per (written, read) state at LINK_DOSE, and the readouts' count."""
    if kind == "trained healthy":
        q = o[o["readout_kind"] == kind]
    else:
        q = o[(o["readout_kind"] == kind) & np.isclose(o["dose"].astype(float), LINK_DOSE)]
    m, n, hi = {}, set(), 0
    for wr in STATES:
        for rd in STATES:
            sel = q[(q["trace_written"] == wr) & (q["trace_read"] == rd)]
            if wr != "healthy" or rd != "healthy":
                sel = sel[np.isclose(sel["lesion_dose"].astype(float), LINK_DOSE)] if kind == "trained healthy" else sel
            m[(wr, rd)] = float(sel["solved"].mean())
            n.add(len(sel))
    if len(n) != 1:
        raise RuntimeError(f"link: unequal readouts per condition for {kind}")
    return m, n.pop(), q


def panel_link(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "J", "Recall needs the learning state at both ends")
    o = cm.table("plan_v2/link_outcomes.csv")
    c = cm.table("plan_v2/link_cosines.csv")
    st = cm.table("plan_v2/link_states.csv")
    cv.text(x0 + 4.0, y0 + 6.0, f"lesion dose {fmt_n(LINK_DOSE)}", size=fs.PT_SMALL, va="center", color=fs.TEXT2)
    cell, my = 8.2, y0 + 15.4
    worst = 0
    for k, (kind, learned, label) in enumerate((("trained healthy", "healthy", f"trained healthy"),
                                                ("trained under the lesion", "lesion", f"trained under{NL_}the lesion"))):
        m, n, q = link_matrix(o, kind)
        mx = x0 + 12.4 + k * 20.6
        ax = cv.axes(mx, my, 2 * cell, 2 * cell)
        ax.set_xlim(0, 2)
        ax.set_ylim(2, 0)
        ax.set_axis_off()
        for i, wr in enumerate(STATES):
            for j, rd in enumerate(STATES):
                v = m[(wr, rd)]
                col = MATRIX_LO + (MATRIX_HI - MATRIX_LO) * v / 18.0
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=col, edgecolor="white", lw=0.6, zorder=2))
                match = wr == learned and rd == learned
                if match:
                    ax.add_patch(Rectangle((j + 0.05, i + 0.05), 0.9, 0.9, facecolor="none", edgecolor=fs.TEXT, lw=0.9, zorder=3))
                ax.text(j + 0.5, i + 0.5, fmt_n(round(v, 1)), fontsize=fs.PT_SMALL, ha="center", va="center", zorder=4, color=fs.TEXT,
                        fontweight="bold" if match else "normal")
                sd.add("J", "matrix", readout=kind, readouts=n, written=wr, read=rd, dose=LINK_DOSE, mean_solved=v, learning_state=match)
        miss = q[~((q["trace_written"] == learned) & (q["trace_read"] == learned))]
        if kind == "trained healthy":
            miss = miss[np.isclose(miss["lesion_dose"].astype(float), LINK_DOSE)]
        worst = max(worst, int(miss["solved"].max()))
        for j, rd in enumerate(STATES):
            cv.text(mx + (j + 0.5) * cell, my - 0.8, STATE_LABEL[rd], size=fs.PT_SMALL, ha="center", va="bottom")
        cv.text(mx + cell, my - 3.6, "read", size=fs.PT_SMALL, ha="center", va="bottom", color=fs.TEXT2)
        cv.text(mx + cell, my + 2 * cell + 1.0, f"{label}{NL_}({n} readouts)", size=fs.PT_SMALL, ha="center", va="top", linespacing=1.0)
    for i, wr in enumerate(STATES):
        cv.text(x0 + 11.6, my + (i + 0.5) * cell, STATE_LABEL[wr], size=fs.PT_SMALL, ha="right", va="center")
    cv.text(x0 + 11.6, my - 0.8, "written", size=fs.PT_SMALL, ha="right", va="bottom", color=fs.TEXT2)
    sd.add("J", "summary", matched=18, best_mismatched=worst)
    # the trace is rewritten and the reading drifts: cosines with the healthy brain across doses (median and range over the
    # nine trained timings)
    xl = (-0.35, 2.3)
    ax = cv.axes(x0 + 62.4, y0 + 10.4, w - 63.8, 14.6)
    st = st.copy()
    doses = sorted(float(d) for d in st["lesion_dose"].dropna().unique())
    for q_name, b_fmt, colour, marker, dx in (("trace", "L{d}-L{d}", LESION_STATE, "o", -0.07), ("window_v", "H-L{d}", fs.TEXT2, "s", 0.07)):
        med, lo, hi = [], [], []
        for d in doses:
            v = c[(c["quantity"] == q_name) & (c["condition_code_a"] == "H-H") & (c["condition_code_b"] == b_fmt.format(d=f"{d:g}"))]["cosine"].astype(float)
            if len(v) != 9:
                raise RuntimeError(f"link: {q_name} at dose {d:g} has {len(v)} timings")
            med.append(float(v.median()))
            lo.append(float(v.min()))
            hi.append(float(v.max()))
            sd.add("J", q_name, dose=d, median=med[-1], minimum=lo[-1], maximum=hi[-1], timings=9)
        xs = [d + dx for d in doses]                                  # the two series side by side
        ax.vlines(xs, lo, hi, color=colour, lw=0.6, zorder=2)
        ax.plot([dx] + xs, [1.0] + med, color=colour, lw=0.8, zorder=3)
        ax.scatter([dx], [1.0], s=7, marker=marker, facecolor="white", edgecolor=colour, linewidth=0.6, zorder=4)   # healthy: itself
        ax.scatter(xs, med, s=7, marker=marker, facecolor=colour, edgecolor=colour, linewidth=0.5, zorder=4)
    ax.axhline(0.0, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
    ax.set_xlim(*xl)
    ax.set_xticks([0] + doses, [])
    ax.set_ylim(-0.65, 1.12)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_ylabel(f"Cosine with{NL_}healthy", labelpad=0.8, linespacing=1.0)
    fs.clean(ax)
    bx = cv.axes(x0 + 62.4, y0 + 27.0, w - 63.8, 8.8)
    h_cells = int(st[st["condition_code"] == "H-H"]["trace_cells"].median())
    l_cells = [int(st[st["condition_code"] == f"L{d:g}-L{d:g}"]["trace_cells"].median()) for d in doses]
    bx.bar([0] + doses, [h_cells] + l_cells, width=0.32, color=[HEALTHY_STATE] + [LESION_STATE] * len(doses), edgecolor="none", zorder=2)
    bx.set_yscale("log")
    bx.set_ylim(1e3, 1e5)
    bx.set_yticks([1e3, 1e4, 1e5], ["10$^3$", "10$^4$", "10$^5$"])
    bx.minorticks_off()
    bx.set_xlim(*xl)
    bx.set_xticks([0] + doses, ["0"] + [fmt_n(d) for d in doses])
    bx.set_xlabel("Lesion dose", labelpad=0.8)
    bx.set_ylabel(f"Trace{NL_}cells", labelpad=0.8, linespacing=1.0)
    fs.clean(bx)
    sd.add("J", "trace cells", healthy=h_cells, lesioned=dict(zip([fmt_n(d) for d in doses], l_cells)), statistic="median over the 9 trained timings")
    cv.key(x0 + 62.4, y0 + 6.0, [([dict(kind="mark", s=6, fc=LESION_STATE, ec=LESION_STATE)], "trace"),
                                 ([dict(kind="mark", s=6, fc=fs.TEXT2, ec=fs.TEXT2, marker="s")], "reading")], size=fs.PT_SMALL, gap=1.2)


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("fig7")
    for rel in ("plan_v2/fig7_funnel.csv", "plan_v2/fig7_cases.csv", "plan_v2/fig7_lost_access.csv", "plan_v2/fig7_relearning.csv",
                "plan_v2/fig7_restoration.csv", "plan_v2/fig7_profiles.csv", "plan_v2/fig7_shal_runaway.csv", "plan_v2/fig7_pattern_lock.csv",
                "plan_v2/link_outcomes.csv", "plan_v2/link_cosines.csv", "plan_v2/link_states.csv", "plan_v2/fig7_adfca_types.csv",
                "plan_v2/fig7_cells.csv", "plan_v2/tests.csv"):
        sd.source(cm.RESULTS / rel)
    for f in sorted(TILES.glob("*.png")) + sorted(TILES.glob("*.json")):
        sd.source(f)
    panel_funnel(cv, sd, XS[0], R1, U, R1_H)
    panel_regimes(cv, sd, XS[1], R1, U, R1_H)
    panel_lost(cv, sd, XS[2], R1, fs.unit(2), R1_H)
    panel_relearn(cv, sd, XS[0], R2, U, R2_H)
    panel_restore(cv, sd, XS[1], R2, U, R2_H)
    panel_cases(cv, sd, XS[2], R2, fs.unit(2), R2_H)
    panel_band(cv, sd)
    panel_runaway(cv, sd, XS[0], R4, U, R4_H)
    panel_lock(cv, sd, XS[1], R4, U, R4_H)
    panel_link(cv, sd, XS[2], R4, fs.unit(2), R4_H)
    cm.finish(cv, "fig7", sd)


if __name__ == "__main__":
    main()
