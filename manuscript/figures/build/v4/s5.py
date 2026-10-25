"""v4 Figure S5 (Cell), related to Figure 5 - Access locked to the timing of learning, at every delay (174 mm wide).

A  dark gallery: the decision-time cue difference at all seven delays (prefix 4; scene 1, readout trained), each with its
   outcome and its cosine with the trained 160-ms pattern (Fig. 5A shows four of them)
B  the lock at 20-ms resolution: cues solved at 33 delays x 5 prefixes (the dense scan), trained timings framed
C  at untrained timings the push collapses and turns away, in each of the three units where the collapse holds after Holm
D  every cell group turns: the cosine with the 160-ms pattern within each cell group, by delay
E  operator drift at both prefixes: the response direction of one trace pattern injected at seven delays, and its push
F  jittered training widens use inside its range, timing by timing (untrained timings, mean of 10 replicates per arm)

Every number in a label is read from manuscript/results (plan_v2 via P12, followup, extended, stats) or from the render
metadata. Reader-facing names only. Not shown (class c, RESULTS_INVENTORY_V4.md A7): the collapse in the scene-2 and scene-3
units (not significant after Holm), the held-out cue decoder, the cost of jittered training at the trained timings (the
jittered arm is drawn at untrained timings only, as Fig. 5H), and the X2 scenes per replicate (the partial generalisation
of one scene).

Run after P12 and render/v4_s5_tiles.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\v4\\s5.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
from lib import figstyle as fs, palette as pal, qc  # noqa: E402

cm.use_v4()
TILES = cm.RENDERS / "v4" / "s5"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
GW, GH = META["tile_mm"]
U = fs.unit(1)
XS = [fs.unit_x(i) for i in range(4)]
ROW2, R2_H = GH + 8.6, 46.0
ROW3, R3_H = ROW2 + R2_H + 3.4, 41.0
H = ROW3 + R3_H
WHITE = fs.ON_DARK
NL_ = chr(10)
TRAINED_DELAYS = {0, 40, 160}
UNITS = [("remote-dynamic-M0-head_only", "readout", pal.STATE_SET["m0"]),
         ("remote-dynamic-M0-joint", f"brain +{NL_}readout", pal.STATE_SET["m0_joint"]),
         ("supp-M0-visibility", f"shared{NL_}readout", pal.on_white(pal.STATE_SET["m0vis_visible"]))]
USE = LinearSegmentedColormap.from_list("net_use", [pal.MEMORY_CLASS["none"], pal.MEMORY_CLASS["one"], pal.MEMORY_CLASS["both"]])
COS = LinearSegmentedColormap.from_list("cos", ["#40004B", "#C2A5CF", "#F7F7F7", "#A6DBA0", "#00441B"])   # -1 purple, +1 green
TRAIN_TINT = "#FCEDF2"                  # the accent at 8% on white, as Fig. 5
CELL_EDGE = "#D9D9D9"
DOT = 20.0                              # points^2 per unit of net-use difference
PUSH_TICKS = {0: [0, 1], 4: [-8, 0]}


def truthy(s):
    return s.astype(str).str.lower().eq("true")


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


def holm(test_id):
    t = cm.table("stats/statistical_tests.csv").set_index("test_id")
    return float(t.loc[test_id, "p_adjusted"])


def colour_key(cv, x, y, w, cmap, lo, hi, head):
    """Horizontal colour key on one line centred on y: head, lo, a bar w mm long, hi. Returns the right end (mm)."""
    cv.text(x, y, head, size=fs.PT_SMALL, va="center", color=fs.TEXT2)
    x1 = x + fs.text_width(head, fs.PT_SMALL) + 1.2
    cv.text(x1, y, lo, size=fs.PT_SMALL, va="center")
    x2 = x1 + fs.text_width(lo, fs.PT_SMALL) + 0.6
    ax = cv.axes(x2, y - 0.7, w, 1.4)
    ax.imshow(cmap(np.linspace(0, 1, 128))[None, :, :3], aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax.set_axis_off()
    cv.text(x2 + w + 0.6, y, hi, size=fs.PT_SMALL, va="center")
    return x2 + w + 0.6 + fs.text_width(hi, fs.PT_SMALL)


def no_spines(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)


# ---------------------------------------------------------------- A: one cue, read at every delay

def panel_gallery(cv, sd):
    cos = cm.table("plan_v2/s5_decision_cosines.csv").set_index("delay_ms")
    out = cm.table("plan_v2/fig5_record_outcomes.csv")
    out = out[(out["trained_parameters"] == "readout") & (out["prefix_windows"] == 4)].set_index("delay_ms")
    words = {2: "both cues", 1: "one cue", 0: "no cue"}
    for k, d in enumerate(META["delays"]):
        x = k * (GW + fs.GUTTER)
        cv.tile(x, 0, np.asarray(Image.open(TILES / f"decision_d{d}.png").convert("RGB")), width=GW, height=GH, group="gallery", label=f"d{d}")
        r = out.loc[d]
        solved = int(str(r["cue0_solved"]).lower() == "true") + int(str(r["cue1_solved"]).lower() == "true")
        tx = x + (4.4 if k == 0 else 1.0)
        tile_text(cv, tx, 1.0, f"{d} ms", va="top", size=fs.PT)
        tile_text(cv, tx, 3.9, ("trained · " if d in TRAINED_DELAYS else "") + words[solved], va="top")
        c = float(cos.loc[d, "cosine_with_160"])
        tile_text(cv, x + GW - 1.0, 1.2, "reference" if d == 160 else f"cos {c:.2f}".replace("-", fs.MINUS), ha="right", va="top")
        sd.add("A", "delay", delay_ms=int(d), trained=d in TRAINED_DELAYS, cues_solved=solved, cosine_with_160=c, cells_above=int(cos.loc[d, "cells_above_0p3_mv"]))
    white_letter(cv, 1.2, 0.9, "A")
    lo, hi = META["dv_mv"]
    u = np.linspace(0, 1, 64)
    bar = np.vstack([pal.SIGNED_ON_DARK(0.5 - 0.26 - 0.24 * u[::-1])[:, :3], np.zeros((10, 3)), pal.SIGNED_ON_DARK(0.5 + 0.26 + 0.24 * u)[:, :3]])
    ax = cv.axes(1.0, GH + 2.4, 16.0, 1.4)
    ax.imshow(bar[None], aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax.set_axis_off()
    cv.text(18.0, GH + 3.1, f"cue 1 {fs.MINUS} cue 0 (mV, log): {fs.MINUS}{hi:g} to +{hi:g}", size=fs.PT_SMALL, va="center", color=fs.TEXT2)


# ---------------------------------------------------------------- B: the lock at 20-ms resolution

def panel_dense(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "B", "The lock, every 20 ms")
    e = cm.table("extended/ed05/ed05d_dense_delay_scan.csv")
    delays = sorted(e["delay_ms"].unique())
    prefixes = sorted(e["prefix_windows"].unique())
    m = np.zeros((len(delays), len(prefixes)), int)
    trained = np.zeros_like(m, dtype=bool)
    for (d, p), q in e.groupby(["delay_ms", "prefix_windows"]):
        i, j = delays.index(d), prefixes.index(p)
        m[i, j] = int(truthy(q["solved"]).sum())
        trained[i, j] = bool(truthy(q["train_time"]).any())
        sd.add("B", "timing", delay_ms=int(d), prefix_windows=int(p), cues_solved=int(m[i, j]), trained=bool(trained[i, j]))
    ax = cv.axes(x0 + 9.0, y0 + 9.4, w - 22.0, h - 17.4)
    ax.imshow(USE(m / 2.0), aspect="auto", interpolation="nearest", extent=(-0.5, len(prefixes) - 0.5, len(delays) - 0.5, -0.5))
    for i, j in zip(*np.nonzero(trained)):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor=pal.ACCENT, linewidth=0.6, zorder=3))
    ax.set_xticks(range(len(prefixes)), [str(p) for p in prefixes])
    yt = [i for i, d in enumerate(delays) if d % 160 == 0]
    ax.set_yticks(yt, [str(delays[i]) for i in yt])
    ax.tick_params(length=0, labelsize=fs.PT_SMALL)
    ax.set_xlabel("Prefix windows", labelpad=0.8)
    ax.set_ylabel("Delay (ms)", labelpad=0.8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
        sp.set_color(CELL_EDGE)
    kx = x0 + w - 12.0
    cv.text(kx, y0 + 8.2, "cues solved", size=fs.PT_SMALL, va="center", color=fs.TEXT2)
    for i, (v, lab) in enumerate(((2, "both"), (1, "one"), (0, "none"))):
        cv.box(kx, y0 + 11.0 + 3.0 * i, 1.8, 1.8, fc=USE(v / 2.0), ec=fs.AXIS, lw=0.3)
        cv.text(kx + 2.4, y0 + 11.9 + 3.0 * i, lab, size=fs.PT_SMALL, va="center")
    cv.box(kx, y0 + 20.0, 1.8, 1.8, fc="none", ec=pal.ACCENT, lw=0.6)
    cv.text(kx + 2.4, y0 + 20.9, "trained", size=fs.PT_SMALL, va="center")


# ---------------------------------------------------------------- C: collapse and turn per unit

def panel_units(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "C", "At untrained timings the push collapses and turns away")
    f = cm.table("followup/fu1_timing_lock.csv")
    f = f[(f["visibility"] == "hidden") & (f["context"] == 0)]
    gap = 9.0
    hw = (w - 9.0 - gap - 1.0) / 2
    rng = np.random.default_rng(501)
    cols = [("differential", "Push toward the correct target", (-6.0, 6.0), ([-4, 0, 4], [f"{fs.MINUS}4", "0", "4"])),
            ("cosine_with_trained_difference", "Cosine with the trained push", (-0.6, 0.8), ([-0.5, 0, 0.5], [f"{fs.MINUS}0.5", "0", "0.5"]))]
    counts = {}
    for j, (col, lab, lim, (ticks, tl)) in enumerate(cols):
        ax_x = x0 + 9.0 + j * (hw + gap)
        ax = cv.axes(ax_x, y0 + 10.4, hw, h - 18.4)
        for i, (code, name, colour) in enumerate(UNITS):
            q = f[f["model"] == code]
            for tr, off in ((True, -0.17), (False, 0.17)):
                v = q[truthy(q["trained"]) == tr][col].to_numpy(float)
                counts[tr] = len(v)
                ax.scatter(i + off + rng.uniform(-0.07, 0.07, len(v)), v, s=5, facecolor=colour if tr else "white", edgecolor=colour, linewidth=0.45, zorder=3)
                ax.plot([i + off - 0.12, i + off + 0.12], [np.median(v)] * 2, color=fs.TEXT, lw=0.9, zorder=4)
                sd.add("C", col, unit=name.replace(NL_, " "), trained=tr, n=int(len(v)), median=float(np.median(v)), values=v.tolist())
            if j == 0:
                p = holm(f"fu1.differential.{code}|c0|hidden")
                ax.text(i, lim[1] - 0.15, fs.p_text(p), fontsize=fs.PT_SMALL, ha="center", va="top")
                sd.add("C", "test", unit=name.replace(NL_, " "), family="F11.timing.differential", holm_p=p)
        ax.axhline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
        ax.set_xlim(-0.6, len(UNITS) - 0.4)
        ax.set_ylim(*lim)
        ax.set_yticks(ticks, tl)
        ax.set_xticks(range(len(UNITS)), [n for _, n, _ in UNITS], linespacing=1.0)
        ax.tick_params(axis="x", length=0, labelsize=fs.PT_SMALL)
        fs.clean(ax)
        cv.text(ax_x + hw / 2, y0 + 9.6, lab, size=fs.PT_SMALL, ha="center", va="bottom")
    cv.key(x0 + 9.0, y0 + 6.0, [([dict(kind="mark", s=7, fc=fs.TEXT2, ec=fs.TEXT2)], f"trained timings ({counts[True]})"),
                                ([dict(kind="mark", s=7, fc="white", ec=fs.TEXT2)], f"untrained ({counts[False]})")], size=fs.PT_SMALL, gap=1.4)
    cv.text(x0 + w - 0.8, y0 + 6.0, f"scene 1, goal hidden; {fs.P_SYM} Holm-adjusted", size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)


# ---------------------------------------------------------------- D: the turn in every cell group

def panel_groups(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "D", "Every cell group turns")
    s = cm.table("plan_v2/s5_decision_superclass.csv")
    s["group"] = s["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other")
    delays = sorted(int(d) for d in s["delay_ms"].unique())
    rows = [("whole CNS", s)] + [(pal.SHORT_GROUP[g], s[s["group"] == g]) for g in pal.ATLAS_ON_DARK if (s["group"] == g).any()]
    m = np.zeros((len(rows), len(delays)))
    for i, (name, q) in enumerate(rows):
        t = q.groupby("delay_ms")[["dot_with_160_mv2", "sum_sq_mv2"]].sum()
        m[i] = [t.loc[d, "dot_with_160_mv2"] / np.sqrt(t.loc[d, "sum_sq_mv2"] * t.loc[160, "sum_sq_mv2"]) for d in delays]
        sd.add("D", "cell group", group=name, cells=int(q.loc[q["delay_ms"] == 160, "cells"].sum()),
               cosine_with_160={str(d): float(v) for d, v in zip(delays, m[i])})
    colour_key(cv, x0 + 4.0, y0 + 6.0, 8.0, COS, f"{fs.MINUS}1", "1", "cosine with 160 ms")
    lab_w = 13.2
    ax = cv.axes(x0 + lab_w, y0 + 9.4, w - lab_w - 0.4, h - 17.0)
    ax.imshow(COS((m + 1) / 2), aspect="auto", interpolation="nearest")
    ax.axhline(0.5, color="white", lw=1.0)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xticks(range(len(delays)), [str(d) for d in delays])
    ax.tick_params(length=0, labelsize=fs.PT_SMALL, pad=1.0)
    ax.set_xlabel("Delay (ms)", labelpad=0.8)
    no_spines(ax)


# ---------------------------------------------------------------- E: operator drift at both prefixes

def panel_drift(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "E", "One trace pattern injected at each delay")
    dr = cm.table("plan_v2/fig5_drift.csv")
    mg = cm.table("plan_v2/fig5_drift_margin.csv")
    end = colour_key(cv, x0 + 4.0, y0 + 6.0, 8.0, COS, f"{fs.MINUS}1", "1", "response direction (cosine)")
    cv.box(end + 3.0, y0 + 5.1, 1.8, 1.8, fc=TRAIN_TINT, ec=pal.ACCENT, lw=0.6)
    cv.text(end + 5.4, y0 + 6.0, "trained delays", size=fs.PT_SMALL, va="center")
    cell, pw, top = 3.2, 12.0, y0 + 11.4
    for b, (p, xm) in enumerate(((0, x0 + 9.0), (4, x0 + 48.6))):
        g = mg[mg["prefix_windows"] == p].sort_values("delay_ms")
        delays = [int(d) for d in g["delay_ms"]]
        n, mw = len(delays), len(delays) * cell
        m = np.eye(n)
        for _, r in dr[dr["prefix_windows"] == p].iterrows():
            i, j = delays.index(int(r["delay_a_ms"])), delays.index(int(r["delay_b_ms"]))
            m[i, j] = m[j, i] = r["cosine"]
            sd.add("E", "direction", prefix_windows=p, delay_a_ms=delays[i], delay_b_ms=delays[j], cosine=float(r["cosine"]))
        cv.text(xm + (mw + 1.2 + pw) / 2, top - 1.0, f"prefix {p}", size=fs.PT_SMALL, ha="center", va="bottom", fontweight="bold")
        ax = cv.axes(xm, top, mw, mw)
        ax.imshow(COS((m + 1) / 2), aspect="auto", interpolation="nearest")
        for i, d in enumerate(delays):
            if d in TRAINED_DELAYS:
                ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor=pal.ACCENT, linewidth=0.6))
        ax.set_yticks(range(n), [str(d) for d in delays] if b == 0 else [])
        sel = [0, delays.index(160), n - 1]
        ax.set_xticks(sel, [str(delays[i]) for i in sel])
        ax.tick_params(length=0, labelsize=fs.PT_SMALL, pad=1.0)
        if b == 0:
            ax.set_ylabel("Delay (ms)", labelpad=0.8)
        ax.set_xlabel("Delay (ms)", labelpad=0.8)
        no_spines(ax)
        ax2 = cv.axes(xm + mw + 1.2, top, pw, mw)
        for i, r in enumerate(g.itertuples()):
            tr = int(r.delay_ms) in TRAINED_DELAYS
            if tr:
                ax2.axhspan(i - 0.5, i + 0.5, color=TRAIN_TINT, lw=0, zorder=0)
            ax2.plot([0, r.margin_shift], [i, i], color=fs.TEXT, lw=0.8, zorder=2)
            ax2.scatter([r.margin_shift], [i], s=9, facecolor=fs.TEXT if r.margin_shift > 0 else "white", edgecolor=fs.TEXT, linewidth=0.6, zorder=3)
            sd.add("E", "push", prefix_windows=p, delay_ms=int(r.delay_ms), trained=tr, margin_shift=float(r.margin_shift))
        ax2.axvline(0, color=fs.AXIS, lw=0.5, zorder=1)
        lo, hi = min(float(g["margin_shift"].min()), 0.0), max(float(g["margin_shift"].max()), 0.0)
        ax2.set_xlim(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo))
        ax2.set_ylim(n - 0.5, -0.5)
        ax2.set_yticks([])
        ax2.set_xticks(PUSH_TICKS[p], [str(t).replace("-", fs.MINUS) for t in PUSH_TICKS[p]])
        ax2.tick_params(axis="x", labelsize=fs.PT_SMALL, pad=1.0)
        ax2.set_xlabel("Push", labelpad=0.8)
        for side in ("left", "top", "right"):
            ax2.spines[side].set_visible(False)


# ---------------------------------------------------------------- F: fixed vs jittered timings, timing by timing

def panel_x3(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "F", "Jittered training widens use inside its range")
    t = cm.table("plan_v2/fig5_timings.csv")
    delays = sorted(int(d) for d in t["delay_ms"].unique())
    prefixes = sorted(int(p) for p in t["prefix_windows"].unique())
    fixed, jitter = "fixed timings", "jittered timings"
    trained = {(int(r.delay_ms), int(r.prefix_windows)) for r in t[(t["arm"] == fixed) & truthy(t["trained"])].itertuples()}
    un = t[~truthy(t["trained"])]
    mean = {arm: un[un["arm"] == arm].groupby(["delay_ms", "prefix_windows"])["net_use"].mean() for arm in (fixed, jitter)}
    reps = {arm: int(un.loc[un["arm"] == arm, "replicate"].nunique()) for arm in (fixed, jitter)}
    jmax = int(un.loc[truthy(un["inside_jitter_range"]), "delay_ms"].max())
    cv.text(x0 + 4.0, y0 + 6.0, f"untrained timings, mean of {min(reps.values())} replicates; jittered training range 0–{jmax} ms",
            size=fs.PT_SMALL, va="center", color=fs.TEXT2)
    cw, ch, top = 3.0, 3.2, y0 + 11.4
    mw, mh = len(prefixes) * cw, len(delays) * ch
    heads = ["fixed", "jittered", f"jittered {fs.MINUS} fixed"]
    for k in range(3):
        xm = x0 + 9.0 + k * (mw + 1.6)
        ax = cv.axes(xm, top, mw, mh)
        ax.set_xlim(-0.5, len(prefixes) - 0.5)
        ax.set_ylim(len(delays) - 0.5, -0.5)
        for i, d in enumerate(delays):
            for j, p in enumerate(prefixes):
                if (d, p) in trained:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=TRAIN_TINT, edgecolor=pal.ACCENT, linewidth=0.6, zorder=4))
                    continue
                if k < 2:
                    arm = (fixed, jitter)[k]
                    v = float(mean[arm].loc[(d, p)])
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=USE((v + 1) / 2), edgecolor=CELL_EDGE, linewidth=0.3, zorder=2))
                    sd.add("F", arm, delay_ms=d, prefix_windows=p, mean_net_use=v, replicates=reps[arm], inside_jitter_range=d <= jmax)
                else:
                    diff = float(mean[jitter].loc[(d, p)] - mean[fixed].loc[(d, p)])
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F5F5F5", edgecolor="white", linewidth=0.3, zorder=2))
                    if abs(diff) >= 0.05:
                        ax.scatter([j], [i], s=DOT * abs(diff), facecolor=fs.TEXT if diff > 0 else "white", edgecolor=fs.TEXT, linewidth=0.5, zorder=3)
                    sd.add("F", "difference", delay_ms=d, prefix_windows=p, jittered_minus_fixed=diff, inside_jitter_range=d <= jmax)
        ax.axhline(delays.index(jmax) + 0.5, color=fs.TEXT, lw=0.6, ls=(0, (2.0, 1.4)), zorder=5)
        ax.set_xticks(range(len(prefixes)), [str(p) for p in prefixes])
        ax.set_yticks(range(len(delays)), [str(d) for d in delays] if k == 0 else [])
        ax.tick_params(length=0, labelsize=fs.PT_SMALL, pad=1.0)
        if k == 0:
            ax.set_ylabel("Delay (ms)", labelpad=0.8)
        no_spines(ax)
        cv.text(xm + mw / 2, top - 1.0, heads[k], size=fs.PT_SMALL, ha="center", va="bottom")
    cv.text(x0 + 9.0 + (3 * mw + 3.2) / 2, top + mh + 5.0, "Prefix windows", size=fs.PT, ha="center", va="center")
    # inside / outside the jittered range, beside the difference map
    bx = x0 + 9.0 + 3 * mw + 3.2 + 0.8
    ri = delays.index(jmax)
    cv.box(bx, top + 0.15 * ch, 0.35, (ri + 0.7) * ch, fc=fs.TEXT, ec="none", lw=0)
    cv.box(bx, top + (ri + 1.15) * ch, 0.35, (len(delays) - ri - 1.3) * ch, fc=fs.TEXT2, ec="none", lw=0)
    cv.text(bx + 1.0, top + (ri + 1) * ch / 2, "inside", size=fs.PT_SMALL, va="center")
    cv.text(bx + 1.0, top + (ri + 1 + len(delays)) * ch / 2, "outside", size=fs.PT_SMALL, va="center", color=fs.TEXT2)
    # keys
    kx = x0 + 68.0
    cv.text(kx, top, "net use", size=fs.PT_SMALL, va="top", color=fs.TEXT2)
    ax = cv.axes(kx, top + 2.6, 1.6, 7.0)
    ax.imshow(USE(np.linspace(1, 0, 64))[:, None, :3], aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
        sp.set_color(CELL_EDGE)
    ax.set_xticks([])
    ax.set_yticks([])
    cv.text(kx + 2.2, top + 3.4, "1", size=fs.PT_SMALL, va="center")
    cv.text(kx + 2.2, top + 8.8, f"{fs.MINUS}1", size=fs.PT_SMALL, va="center")
    cv.text(kx, top + 12.0, f"jittered {fs.MINUS} fixed", size=fs.PT_SMALL, va="center", color=fs.TEXT2)
    cv.key(kx, top + 14.8, [([dict(kind="mark", s=DOT, fc=fs.TEXT, ec=fs.TEXT)], "+1"),
                            ([dict(kind="mark", s=DOT, fc="white", ec=fs.TEXT)], f"{fs.MINUS}1")], size=fs.PT_SMALL, gap=1.6)
    cv.box(kx, top + 17.6, 1.8, 1.8, fc=TRAIN_TINT, ec=pal.ACCENT, lw=0.6)
    cv.text(kx + 2.4, top + 17.4, f"fixed training{NL_}timings", size=fs.PT_SMALL, va="top", linespacing=1.0)


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("s5")
    for rel in ("plan_v2/s5_decision_cells.csv", "plan_v2/s5_decision_cosines.csv", "plan_v2/s5_decision_superclass.csv", "plan_v2/fig5_record_outcomes.csv",
                "extended/ed05/ed05d_dense_delay_scan.csv", "followup/fu1_timing_lock.csv", "stats/statistical_tests.csv", "plan_v2/fig5_drift.csv",
                "plan_v2/fig5_drift_margin.csv", "plan_v2/fig5_timings.csv"):
        sd.source(cm.RESULTS / rel)
    for f in sorted(TILES.glob("*.png")) + sorted(TILES.glob("*.json")):
        sd.source(f)
    panel_gallery(cv, sd)
    panel_dense(cv, sd, XS[0], ROW2, U, R2_H)
    panel_units(cv, sd, XS[1], ROW2, fs.unit(2), R2_H)
    panel_groups(cv, sd, XS[3], ROW2, U, R2_H)
    panel_drift(cv, sd, XS[0], ROW3, fs.unit(2), R3_H)
    panel_x3(cv, sd, XS[2], ROW3, fs.unit(2), R3_H)
    cm.finish(cv, "s5", sd, kind="supplement")


if __name__ == "__main__":
    main()
