"""v4 Fig. 5 (Cell) - Access to the memory is locked to the timing of learning (174 mm wide).

A  dark band: one cue, read out at four times. Per-cell cue difference at the first decision after 40, 160 (trained),
   320 and 640 ms (untrained), scene 1, readout trained; after the untrained delays the brain-wide pattern is reversed.
   Compass on each tile (operator drift, in the form of MaleCNS's direction glyphs): gray, the 160-ms decision pattern; white,
   this delay's pattern, turned by the angle between the two (arccos of the cosine printed below it)
B  the delay x prefix grid of net memory use (10 replicates): used at the 9 trained timings only
C  the lock in nine scenes, per replicate (the three trained scenes and six new ones)
D  at untrained timings the trace persists and reaches the readout, but its push toward the correct target is lost
E  cell by cell, the decision-time code after 320 ms is the mirror image of the code after 160 ms
F  decision geometry: every trained timing pushes both cues the right way; untrained timings fall elsewhere
G  operator drift: one trace pattern injected at seven delays evokes turning responses that push the right way only at the
   trained delays
H  training sets the range: jittered-timing training raises use at untrained timings inside the jitter only

Every number in a label is read from manuscript/results (plan_v2 via P12) or from the band's render metadata. Reader-facing
names only (NARRATIVE_V4_OUTLINE.md section 12). Not shown (class c, RESULTS_INVENTORY_V4.md): the trained-timing cost of
jittered training, untrained use in the scene with one floor square removed from scene 2, and the cue decoder.

Run after P12 and render/v4_fig5_tiles.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\v4\\fig5.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, Polygon, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
from lib import circuit as ci, figstyle as fs, palette as pal, qc  # noqa: E402

cm.use_v4()
TILES = cm.RENDERS / "v4" / "fig5"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
TW, TH = META["tile_mm"]
TITLE_MM = META["title_mm"]
U = fs.unit(1)
XS = [fs.unit_x(i) for i in range(4)]
ROW2 = TH + 4.2
R2_H = 46.0
ROW3 = ROW2 + R2_H + 3.4
R3_H = 46.0
ROW4 = ROW3 + R3_H + 3.4
R4_H = 44.0
H = ROW4 + R4_H
WHITE = fs.ON_DARK
NL_ = chr(10)
TRAINED_DELAYS = (0, 40, 160)
LEARNERS = [("readout", pal.STATE_SET["m0"], "Scene 1, readout trained"), ("brain and readout", pal.STATE_SET["m0_joint"], "Scene 1, brain + readout trained")]
SCENE_FAMILIES = [("scene 1", ("scene 1", "scene 4", "scene 5"), pal.STATE_SET["m0"]), ("scene 2", ("scene 2", "scene 6", "scene 8"), pal.STATE_SET["s1"]),
                  ("scene 3", ("scene 3", "scene 7", "scene 9"), pal.STATE_SET["s2_o0"])]
NEW_SCENE_MARK = {"scene 4": "o", "scene 5": "o", "scene 6": "D", "scene 7": "D", "scene 8": "s", "scene 9": "s"}
USE_CMAP = LinearSegmentedColormap.from_list("net_use", [pal.MEMORY_CLASS["none"], pal.MEMORY_CLASS["one"], pal.MEMORY_CLASS["both"]])
COS_CMAP = colormaps["PRGn"]
FIXED_ARM, JITTER_ARM = "#9E9E9E", fs.TEXT
TRAIN_TINT = "#FCEDF2"                       # the accent (trained timings) at 8% on white, for training ranges
COMPASS_REF = "#8C8C8C"                      # A: the 160-ms pattern's arrow


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


def test_rows(family):
    t = cm.table("plan_v2/tests.csv")
    return t[t["family"] == family]


def record(trained="readout"):
    r = cm.table("plan_v2/fig5_record_outcomes.csv")
    return r[(r["scene"] == "scene 1") & (r["trained_parameters"] == trained)]


def tile_img(name):
    return np.asarray(Image.open(TILES / name).convert("RGB"))


# ---------------------------------------------------------------- A: dark band

def panel_band(cv, sd):
    xs = [k * (TW + fs.GUTTER) for k in range(4)]
    rec = record().set_index(["delay_ms", "prefix_windows"])
    cos = cm.table("plan_v2/fig5_decision_cosines.csv").set_index(["timing_a", "timing_b"])["cosine"]
    outcome = {1: "both cues solved", 0: "one cue solved", -1: "no cue solved"}
    ref = "dv_d160_p4_mv"
    for k, ((d, p), x) in enumerate(zip(META["timings"], xs)):
        name = f"decision_d{d}_p{p}"
        meta = json.loads((TILES / f"{name}.json").read_text(encoding="utf-8"))
        cv.tile(x, 0, tile_img(f"{name}.png"), width=TW, height=TH, group="band", label=name)
        r = rec.loc[(d, p)]
        trained = bool(r["trained"])
        tx = x + (4.6 if k == 0 else 1.4)
        tile_text(cv, tx, 1.2, f"{d}-ms delay", va="top", size=fs.PT)
        tile_text(cv, tx, 4.3, ("trained" if trained else "untrained") + " · " + outcome[int(r["net_use"])], va="top")
        key = f"dv_d{d}_p{p}_mv"
        lines = ["the reference", "pattern"] if key == ref else ["vs 160 ms:", f"cosine {fs.num(cos.loc[(key, ref)], 2)}"]
        for i, line in enumerate(lines):
            tile_text(cv, x + 1.0, TH - 9.4 + 2.4 * i, line, va="center")
        c = float(cos.loc[(key, ref)])
        angle = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
        compass(cv, x + 4.4, TH - 15.6, 3.0, c)
        if key != ref:                                     # the gray arrow named where it points; the turn's angle beside it
            cv.text(x + 4.4, TH - 19.2, "160 ms", size=fs.PT_SMALL, ha="center", va="bottom", color=COMPASS_REF)
            tile_text(cv, x + 8.6, TH - 15.6, f"{angle:.0f}°", va="center", size=fs.PT)
        sd.add("A", "tile", delay_ms=d, prefix_windows=p, trained=trained, net_use=int(r["net_use"]), cosine_with_160_ms=c, angle_deg=angle,
               cells_above=int(meta["cells_located"]), positive=int(meta["positive"]), negative=int(meta["negative"]), threshold_mv=meta["dv_lo"])
    white_letter(cv, 1.2, 0.9, "A")
    # colour key (black space right of the VNC in the first tile): log |dv|, both signs
    lo, hi = META["dv_mv"]
    bx, by, half, gap = xs[0] + TW - 2.6, 33.0, 6.0, 3.0
    u = np.linspace(1, 0, 96)
    gradient(cv, bx, by, 1.4, half, pal.SIGNED_ON_DARK(0.5 + 0.26 + 0.24 * u))
    gradient(cv, bx, by + half + gap, 1.4, half, pal.SIGNED_ON_DARK(0.5 - 0.26 - 0.24 * u[::-1]))
    tile_text(cv, bx + 1.4, by - 0.9, "mV", va="bottom", ha="right")
    tile_text(cv, bx - 0.5, by, f"+{hi:g}", va="center", ha="right")
    tile_text(cv, bx - 0.5, by + half, f"+{lo:g}", va="center", ha="right")
    tile_text(cv, bx - 0.5, by + half + gap, f"{fs.MINUS}{lo:g}", va="center", ha="right")
    tile_text(cv, bx - 0.5, by + 2 * half + gap, f"{fs.MINUS}{hi:g}", va="center", ha="right")
    tile_text(cv, bx + 1.4, by + 2 * half + gap + 1.4, f"cue 1{NL_}{fs.MINUS} cue 0", va="top", ha="right", linespacing=1.0)
    # scale bar and orientation (last tile)
    last = f"decision_d{META['timings'][-1][0]}_p{META['timings'][-1][1]}"
    meta = json.loads((TILES / f"{last}.json").read_text(encoding="utf-8"))
    px_per_mm = tile_img(f"{last}.png").shape[1] / TW
    length = 100 / meta["scale_um_per_px"] / px_per_mm
    cv.line([xs[3] + TW - 1.6 - length, xs[3] + TW - 1.6], [TH - 2.0, TH - 2.0], WHITE, lw=1.0, solid_capstyle="butt")
    ox, oy = xs[3] + TW - 12.0, TH - 6.4
    cv.line([ox, ox + 2.8], [oy, oy], WHITE, lw=0.5)
    tile_text(cv, ox + 3.2, oy, "L", va="center")
    cv.line([ox, ox], [oy, oy + 2.8], WHITE, lw=0.5)
    tile_text(cv, ox, oy + 3.2, "V", ha="center", va="top")
    sd.add("A", "view", scale_bar_um=100, um_per_px=meta["scale_um_per_px"], colour_range_mv=[lo, hi], learner="Scene 1, readout trained",
           prefix_windows=int(META["timings"][0][1]))


def compass(cv, cx, cy, r, cosine):
    """Gray arrow: the 160-ms pattern (up). White: this tile's pattern, turned clockwise by arccos(cosine), with the turn as an arc."""
    cv.fig.patches.append(Ellipse((cx / cv.w, 1 - cy / cv.h), 2 * r / cv.w, 2 * r / cv.h, transform=cv.fig.transFigure,
                                  facecolor="none", edgecolor="#6E6E6E", linewidth=0.4, zorder=3))
    ci.arrow(cv, [(cx, cy), (cx, cy - r)], 0.35, COMPASS_REF, "arrow", gap=0.0, zorder=4)
    if cosine < 1 - 1e-9:
        th = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
        ci.ribbon_curve(cv, (cx, cy), (cx + r * np.sin(th), cy - r * np.cos(th)), 0.45, WHITE, bend=0.0, zorder=5)
        a = np.linspace(0, th, 30)
        cv.line(cx + 0.55 * r * np.sin(a), cy - 0.55 * r * np.cos(a), WHITE, lw=0.4)
    else:
        ci.arrow(cv, [(cx, cy), (cx, cy - r)], 0.45, WHITE, "arrow", gap=0.0, zorder=5)


def gradient(cv, x, y, w, h, colours):
    ax = cv.axes(x, y, w, h)
    ax.imshow(np.asarray(colours)[:, None, :3] * np.ones((1, 2, 1)), aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax.set_axis_off()


# ---------------------------------------------------------------- B: the timing grid

def panel_grid(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "B", "Memory use is locked to the trained timings")
    t = cm.table("plan_v2/fig5_timings.csv")
    f = t[t["arm"] == "fixed timings"]
    g = f.groupby(["delay_ms", "prefix_windows"])["net_use"]
    mean = g.mean().unstack()
    agree = int((g.nunique() == 1).sum())
    reps = int(f["replicate"].nunique())
    tr_all = f[f["trained"].astype(str).str.lower().eq("true")]["net_use"]
    un_mean = float(f[~f["trained"].astype(str).str.lower().eq("true")]["net_use"].mean())
    delays, prefixes = list(mean.index), list(mean.columns)
    gw = 42.0
    ax = cv.axes(x0 + 11.0, y0 + 7.6, gw, h - 15.8)
    ax.imshow(mean.to_numpy(float), cmap=USE_CMAP, vmin=-1, vmax=1, aspect="auto", interpolation="nearest",
              extent=(-0.5, len(prefixes) - 0.5, len(delays) - 0.5, -0.5))
    for i in range(len(delays) + 1):
        ax.plot([-0.5, len(prefixes) - 0.5], [i - 0.5, i - 0.5], color=fs.GRID, lw=fs.LW_HAIR, zorder=2)
    for j in range(len(prefixes) + 1):
        ax.plot([j - 0.5, j - 0.5], [-0.5, len(delays) - 0.5], color=fs.GRID, lw=fs.LW_HAIR, zorder=2)
    band = {tuple(x) for x in META["timings"]}
    for i, d in enumerate(delays):
        for j, p in enumerate(prefixes):
            v = float(mean.loc[d, p])
            trained = bool(f[(f["delay_ms"] == d) & (f["prefix_windows"] == p)]["trained"].iloc[0])
            if trained:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor=pal.ACCENT, linewidth=0.9, zorder=4))
            if (d, p) in band:
                ring = WHITE if v > 0.4 else fs.TEXT
                ax.scatter([j], [i], s=7, facecolor="none", edgecolor=ring, linewidth=0.6, zorder=5)
            sd.add("B", "timing", delay_ms=int(d), prefix_windows=int(p), trained=trained, mean_net_use=v, replicates=reps,
                   replicates_agree=bool(g.nunique().loc[(d, p)] == 1))
    ax.set_xlim(-0.5, len(prefixes) - 0.5)
    ax.set_ylim(len(delays) - 0.5, -0.5)
    ax.set_xticks(range(len(prefixes)), [str(int(p)) for p in prefixes])
    ax.set_yticks(range(len(delays)), [str(int(d)) for d in delays])
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Prefix windows", labelpad=0.8)
    ax.set_ylabel("Delay (ms)", labelpad=0.8)
    # key and facts (right of the grid)
    kx = x0 + 11.0 + gw + 5.0
    cv.text(kx, y0 + 8.6, f"Net memory use, mean of {reps}", size=fs.PT_SMALL, va="top")
    for i, (v, lab) in enumerate(((1, "+1  both cues solved"), (0, "0  one cue"), (-1, f"{fs.MINUS}1  no cue"))):
        yy = y0 + 12.6 + 3.0 * i
        cv.box(kx, yy - 1.0, 2.6, 2.0, fc=USE_CMAP((v + 1) / 2), ec=fs.GRID, lw=fs.LW_HAIR)
        cv.text(kx + 3.6, yy, lab, size=fs.PT_SMALL, va="center")
    yy = y0 + 22.2
    cv.box(kx, yy - 1.0, 2.6, 2.0, fc="white", ec=pal.ACCENT, lw=0.9)
    cv.text(kx + 3.6, yy, "trained timing", size=fs.PT_SMALL, va="center")
    cv.mark(kx + 1.3, yy + 3.0, s=7, fc="none", ec=fs.TEXT, lw=0.6)
    cv.text(kx + 3.6, yy + 3.0, "pattern in A", size=fs.PT_SMALL, va="center")
    p_lock = float(test_rows("F13.X1.lock").set_index("test").loc["M0-head_only", "p_adjusted"])
    tr_txt = "+1 in every replicate" if (tr_all == 1).all() else f"mean {fs.num(float(tr_all.mean()), 2, sign=True)}"
    facts = [f"trained: {tr_txt}", f"untrained: mean {fs.num(un_mean, 3, sign=True)}", f"replicates agree at {agree} of {int(mean.size)}",
             f"lock: Holm {fs.P_SYM} = {fs.fmt_p(p_lock)}"]
    for i, s_ in enumerate(facts):
        cv.text(kx, y0 + 31.0 + 2.7 * i, s_, size=fs.PT_SMALL, va="center")
    sd.add("B", "summary", replicates=reps, timings_agree=agree, timings=int(mean.size), trained_all_plus_one=bool((tr_all == 1).all()),
           untrained_mean=un_mean, lock_holm_p=p_lock, learner="Scene 1, readout trained")


# ---------------------------------------------------------------- C: nine scenes

def panel_scenes(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "C", "The lock holds in nine scenes")
    s = cm.table("plan_v2/fig5_scenes.csv")
    x1 = test_rows("F13.X1.lock").set_index("test")["p_adjusted"]
    x2 = test_rows("F13.X2.lock").set_index("test")["p_adjusted"]
    ax = cv.axes(x0 + 9.0, y0 + 11.6, w - 10.6, h - 19.6)
    rng = np.random.default_rng(5)
    pos, ticks, labels, pmax = 0.0, [], [], 0.0
    for fam, scenes, col in SCENE_FAMILIES:
        for scn in scenes:
            q = s[s["scene"] == scn]
            lock = q["lock"].to_numpy(float)
            original = scn == fam
            code = q["code"].iloc[0]
            p = float(x1.loc[code] if original else x2.loc[code])
            pmax = max(pmax, p)
            ax.scatter(pos + rng.uniform(-0.16, 0.16, len(lock)), lock, s=6 if original else 7, marker="o" if original else NEW_SCENE_MARK[scn],
                       facecolor=col if original else "white", edgecolor=col, linewidth=0.5, zorder=3)
            ax.plot([pos - 0.28, pos + 0.28], [np.median(lock)] * 2, color=col, lw=0.9, zorder=4)
            ticks.append(pos)
            labels.append(scn.split()[1])
            for _, r in q.iterrows():
                sd.add("C", scn, origin=r["origin"], replicate=int(r["replicate"]), trained_net_use=float(r["trained_net_use"]), lock=float(r["lock"]))
            sd.add("C", "test", scene=scn, replicates=int(len(q)), median_lock=float(np.median(lock)), holm_p=p)
            pos += 1.0
        pos += 0.5
    ax.axhline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
    ax.set_xlim(-0.6, pos - 0.9)
    ax.set_ylim(-0.1, 1.35)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks(ticks, labels)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Scene", labelpad=0.8)
    ax.set_ylabel(f"Lock (trained {fs.MINUS} untrained use)", labelpad=0.8)
    fs.clean(ax)
    n_pos, n_all = int((s["lock"] > 0).sum()), int(len(s))
    ax.text(pos - 0.95, 1.33, f"Holm {fs.P_SYM} ≤ {fs.fmt_p(pmax)}, each scene", fontsize=fs.PT_SMALL,
            ha="right", va="top", linespacing=1.0)
    items = [([dict(kind="mark", s=6, fc=fs.TEXT2, ec=fs.TEXT2)], "trained scene"),
             ([dict(kind="mark", s=7, marker="o", fc="white", ec=fs.TEXT2)], "+ floor squares"),
             ([dict(kind="mark", s=7, marker="D", fc="white", ec=fs.TEXT2)], "mirrored"),
             ([dict(kind="mark", s=7, marker="s", fc="white", ec=fs.TEXT2)], f"{fs.MINUS} one floor square")]
    cv.key(x0 + 9.0, y0 + 6.4, items, size=fs.PT_SMALL, gap=2.0)
    sd.add("C", "summary", runs=n_all, runs_lock_positive=n_pos, holm_p_max=pmax)


# ---------------------------------------------------------------- D: the cue arrives, its push is lost

def panel_persist(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "D", "The cue still arrives, but its push collapses")
    code = cm.table("plan_v2/fig5_code.csv")
    measures = [("adaptation_energy_decision", "Adaptation trace", True, None),
                ("readout_input_cue_difference", f"Cue difference in{NL_}the readout input", True, None),
                ("differential", f"Push toward the{NL_}correct target", False, code)]
    aw = (w - 9.0 - 2 * 8.6 - 1.4) / 3
    holm = None
    for k, (col, lab, log, src) in enumerate(measures):
        ax = cv.axes(x0 + 9.0 + k * (aw + 8.6), y0 + 13.4, aw, h - 21.4)
        rng = np.random.default_rng(11 + k)
        for i, (trn, colour, name) in enumerate(LEARNERS):
            d = record(trn) if src is None else src[(src["scene"] == "scene 1") & (src["trained_parameters"] == trn)]
            for g, tr in enumerate((True, False)):
                v = d[d["trained"].astype(str).str.lower().eq(str(tr).lower())][col].to_numpy(float)
                xx = g + (i - 0.5) * 0.36
                ax.scatter(xx + rng.uniform(-0.07, 0.07, len(v)), v, s=4, facecolor=colour if tr else "white", edgecolor=colour, linewidth=0.45, zorder=3)
                ax.plot([xx - 0.13, xx + 0.13], [np.median(v)] * 2, color=fs.TEXT if i == 0 else colour, lw=1.0, zorder=4)
                sd.add("D", col, learner=name, trained=tr, n=int(len(v)), median=float(np.median(v)), values=v.tolist())
            if src is not None:
                holm = max(holm or 0.0, float(d["holm_p_differential"].iloc[0]))
        if log:
            ax.set_yscale("log")
            fs.minor(ax, x=False, y=False)
        else:
            ax.axhline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
            ax.set_ylim(-6.6, 5.2)
            ax.set_yticks([-4, 0, 4], [f"{fs.MINUS}4", "0", "4"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1], ["trained", "untrained"])
        ax.tick_params(axis="x", length=0)
        ax.set_title(lab, fontsize=fs.PT_SMALL, pad=1.5, linespacing=1.0)
        fs.clean(ax)
    ax.text(-0.55, ax.get_ylim()[0], f"Holm {fs.P_SYM} = {fs.fmt_p(holm)}", fontsize=fs.PT_SMALL, ha="left", va="bottom")
    items = [([dict(kind="mark", s=6, fc=c, ec=c)], n.replace("Scene 1, ", "")) for _, c, n in LEARNERS]
    cv.key(x0 + 9.0, y0 + 6.2, items, size=fs.PT_SMALL, gap=2.0)
    sd.add("D", "test", family="F11.timing.differential", holm_p_max=holm)


# ---------------------------------------------------------------- E: the code flips cell by cell

def panel_flip(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "E", "The code flips cell by cell")
    c = cm.table("plan_v2/fig5_decision_cells.csv")
    cos = cm.table("plan_v2/fig5_decision_cosines.csv").set_index(["timing_a", "timing_b"])["cosine"]
    a, b = c["dv_d160_p4_mv"].to_numpy(float), c["dv_d320_p4_mv"].to_numpy(float)
    top = np.argsort(-np.abs(a))[: len(a) // 100]
    flip = float(np.mean(np.sign(b[top]) == -np.sign(a[top])))
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 15.6)
    lim = 30.0
    ax.add_patch(Rectangle((-lim, 0), lim, lim, facecolor="#F2F2F2", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0, -lim), lim, lim, facecolor="#F2F2F2", edgecolor="none", zorder=0))
    rest = np.setdiff1d(np.arange(len(a)), top)
    ax.scatter(np.clip(a[rest], -lim, lim), np.clip(b[rest], -lim, lim), s=0.5, facecolor="#BDBDBD", edgecolor="none", rasterized=True, zorder=2)
    ax.scatter(np.clip(a[top], -lim, lim), np.clip(b[top], -lim, lim), s=0.8, facecolor=fs.TEXT, edgecolor="none", rasterized=True, zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ticks = [-20, 0, 20]
    tl = [f"{fs.MINUS}20", "0", "20"]
    ax.set_xticks(ticks, tl)
    ax.set_yticks(ticks, tl)
    ax.set_xlabel("Cue difference, 160 ms (mV)", labelpad=0.8)
    ax.set_ylabel("Cue difference, 320 ms (mV)", labelpad=0.8)
    fs.clean(ax)
    cv_ = float(cos.loc[("dv_d160_p4_mv", "dv_d320_p4_mv")])
    ax.text(lim * 0.95, lim * 0.95, f"cosine {fs.num(cv_, 2)}", fontsize=fs.PT_SMALL, ha="right", va="top")
    ax.text(-lim * 0.95, -lim * 0.95, f"strongest 1%{NL_}(black)", fontsize=fs.PT_SMALL, ha="left", va="bottom",
            linespacing=1.0)
    sd.add("E", "view", axis_limit_mv=lim, cells_beyond_limit=int(((np.abs(a) > lim) | (np.abs(b) > lim)).sum()))
    sd.add("E", "cells", cells=int(len(a)), cosine=cv_, strongest_cells=int(len(top)), strongest_flip_share=flip, learner="Scene 1, readout trained",
           prefix_windows=4)


# ---------------------------------------------------------------- F: decision geometry

def panel_geometry(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "F", "Trained timings push right")
    code = cm.table("plan_v2/fig5_code.csv")
    d = code[(code["scene"] == "scene 1") & (code["trained_parameters"] == "readout")]
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 15.6)
    L = 7.0
    ax.add_patch(Polygon([(0, 0), (L, L), (L, -L)], closed=True, facecolor="#EDEDED", edgecolor="none", zorder=0))
    ax.add_patch(Polygon([(0, 0), (-L, L), (-L, -L)], closed=True, facecolor="#F7F7F7", edgecolor="none", zorder=0))
    ax.plot([-L, L], [-L, L], color=fs.GRID, lw=fs.LW_HAIR, zorder=1)
    ax.plot([-L, L], [L, -L], color=fs.GRID, lw=fs.LW_HAIR, zorder=1)
    tr = d["trained"].astype(str).str.lower().eq("true")
    ax.scatter(d.loc[~tr, "differential"], d.loc[~tr, "common"], s=6, facecolor="white", edgecolor=fs.TEXT, linewidth=0.5, zorder=3)
    ax.scatter(d.loc[tr, "differential"], d.loc[tr, "common"], s=6, facecolor=fs.TEXT, edgecolor=fs.TEXT, linewidth=0.5, zorder=4)
    offsets = {40: ((2.5, 1.5), "left", "bottom"), 160: ((2.5, -1.5), "left", "top"), 320: ((2.5, 1.5), "left", "bottom"), 640: ((-2.5, 1.5), "right", "bottom")}
    for dl, pw in META["timings"]:
        r = d[(d["delay_ms"] == dl) & (d["prefix_windows"] == pw)].iloc[0]
        xy, ha, va = offsets[dl]
        ax.annotate(f"{dl} ms", (r["differential"], r["common"]), xytext=xy, textcoords="offset points", fontsize=fs.PT_SMALL, ha=ha, va=va)
    right = d["differential"] > d["common"].abs()
    rev = d["differential"] < -d["common"].abs()
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_xticks([-5, 0, 5], [f"{fs.MINUS}5", "0", "5"])
    ax.set_yticks([-5, 0, 5], [f"{fs.MINUS}5", "0", "5"])
    ax.set_xlabel("Push toward the correct target", labelpad=0.8)
    ax.set_ylabel("Push shared by both cues", labelpad=0.8)
    fs.clean(ax)
    ax.text(L * 0.95, -L * 0.62, f"both{NL_}right", fontsize=fs.PT_SMALL, ha="right", va="center", linespacing=1.0)
    ax.text(-L * 0.95, -L * 0.62, f"both{NL_}reversed", fontsize=fs.PT_SMALL, ha="left", va="center", linespacing=1.0)
    ax.text(0, L * 0.95, "same push", fontsize=fs.PT_SMALL, ha="center", va="top")
    n_tr = int(tr.sum())
    cv.key(x0 + 9.0, y0 + 6.2, [([dict(kind="mark", s=6, fc=fs.TEXT, ec=fs.TEXT)], f"trained ({int((right & tr).sum())}/{n_tr} right)"),
                                ([dict(kind="mark", s=6, fc="white", ec=fs.TEXT)], "untrained")], size=fs.PT_SMALL, gap=1.6)
    for _, r in d.iterrows():
        sd.add("F", "timing", delay_ms=int(r["delay_ms"]), prefix_windows=int(r["prefix_windows"]), trained=bool(str(r["trained"]).lower() == "true"),
               differential=float(r["differential"]), common=float(r["common"]))
    sd.add("F", "summary", trained_right=int((right & tr).sum()), trained=n_tr, untrained_right=int((right & ~tr).sum()),
           untrained_reversed=int((rev & ~tr).sum()), untrained=int((~tr).sum()), learner="Scene 1, readout trained")


# ---------------------------------------------------------------- G: operator drift

def panel_drift(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "G", "One trace pattern, read at seven delays")
    dr = cm.table("plan_v2/fig5_drift.csv")
    dr = dr[dr["prefix_windows"] == 4]
    mg = cm.table("plan_v2/fig5_drift_margin.csv")
    mg = mg[mg["prefix_windows"] == 4].sort_values("delay_ms")
    delays = sorted(mg["delay_ms"].unique())
    n = len(delays)
    m = np.eye(n)
    idx = {d: i for i, d in enumerate(delays)}
    for _, r in dr.iterrows():
        m[idx[r["delay_a_ms"]], idx[r["delay_b_ms"]]] = r["cosine"]
    side = h - 18.6
    ax = cv.axes(x0 + 11.0, y0 + 10.6, side, side)
    ax.imshow(m, cmap=COS_CMAP, vmin=-1, vmax=1, interpolation="nearest", extent=(-0.5, n - 0.5, n - 0.5, -0.5))
    labels = [str(int(d)) for d in delays]
    ax.set_xticks(range(n), labels, rotation=90)
    ax.set_yticks(range(n), labels)
    ax.tick_params(length=0, pad=1.2)
    for s in ax.spines.values():
        s.set_visible(False)
    for d in TRAINED_DELAYS:
        i = idx[d]
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor=pal.ACCENT, linewidth=0.8, zorder=4))
    ax.set_xlabel("Delay (ms)", labelpad=0.8)
    ax.set_title("Response direction (cosine)", fontsize=fs.PT_SMALL, pad=1.5)
    off = m[~np.eye(n, dtype=bool)]
    for i in range(n):
        for j in range(n):
            sd.add("G", "cosine", delay_a_ms=int(delays[i]), delay_b_ms=int(delays[j]), cosine=float(m[i, j]))
    # colour bar right of the matrix
    bx = x0 + 11.0 + side + 1.8
    bar = cv.axes(bx, y0 + 10.6 + side * 0.2, 1.4, side * 0.6)
    bar.imshow(np.linspace(1, -1, 128)[:, None], cmap=COS_CMAP, vmin=-1, vmax=1, aspect="auto", extent=(0, 1, -1, 1))
    bar.set_xticks([])
    bar.yaxis.tick_right()
    bar.set_yticks([-1, 0, 1], [f"{fs.MINUS}1", "0", "1"])
    bar.tick_params(length=0, pad=1.2)
    for s in bar.spines.values():
        s.set_linewidth(fs.LW_HAIR)
    # the push each injection gives: toward the correct target at the trained delays only
    left2 = 11.0 + side + 15.0
    ax2 = cv.axes(x0 + left2, y0 + 10.6, w - left2 - 1.4, side)
    for i, (_, r) in enumerate(mg.iterrows()):
        tr = bool(r["trained_delay"])
        if tr:
            ax2.axvspan(i - 0.42, i + 0.42, color=TRAIN_TINT, lw=0, zorder=0)
        ax2.plot([i, i], [0, r["margin_shift"]], color=fs.TEXT, lw=0.8, zorder=2)
        ax2.scatter([i], [r["margin_shift"]], s=9, facecolor=fs.TEXT if tr else "white", edgecolor=fs.TEXT, linewidth=0.6, zorder=3)
        sd.add("G", "push", delay_ms=int(r["delay_ms"]), trained_delay=tr, margin_shift=float(r["margin_shift"]))
    ax2.axhline(0, color=fs.TEXT2, lw=fs.LW_THIN, zorder=1)
    ax2.set_xlim(-0.6, n - 0.4)
    ax2.set_ylim(-10.5, 5.0)
    ax2.set_yticks([-8, -4, 0, 4], [f"{fs.MINUS}8", f"{fs.MINUS}4", "0", "4"])
    ax2.set_xticks(range(n), labels, rotation=90)
    ax2.tick_params(axis="x", length=0)
    ax2.set_xlabel("Delay (ms)", labelpad=0.8)
    ax2.set_title("Push toward the correct target", fontsize=fs.PT_SMALL, pad=1.5)
    fs.clean(ax2)
    pos = mg[mg["margin_shift"] > 0]["delay_ms"].tolist()
    if sorted(pos) == sorted(TRAINED_DELAYS):
        cv.text(x0 + w - 0.8, y0 + 6.2, "shaded: trained delays", size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)
    sd.add("G", "summary", off_diagonal_cosine_range=[float(off.min()), float(off.max())], positive_push_delays=pos, trained_delays=list(TRAINED_DELAYS),
           pattern="unit-norm adaptation difference at the end of the 160-ms delay", prefix_windows=4)


# ---------------------------------------------------------------- H: training sets the range

def panel_range(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "H", "Training sets the range")
    t = cm.table("plan_v2/fig5_timings.csv")
    un = t[~t["trained"].astype(str).str.lower().eq("true")]
    desc = test_rows("F13.X3.descriptive").set_index("test")
    lock = test_rows("F13.X3.lock").iloc[0]
    per = un.groupby(["arm", "replicate", "delay_ms"])["net_use"].mean().reset_index()
    delays = sorted(per["delay_ms"].unique())
    jmax = int(un.loc[un["inside_jitter_range"].astype(str).str.lower().eq("true"), "delay_ms"].max())
    aw = 40.0
    ax = cv.axes(x0 + 9.0, y0 + 9.6, aw, h - 17.6)
    xi = {d: i for i, d in enumerate(delays)}
    ax.axvspan(-0.5, xi[jmax] + 0.5, color=TRAIN_TINT, lw=0, zorder=0)
    ax.text(xi[jmax] / 2, 1.42, f"jittered training range, 0–{jmax} ms", fontsize=fs.PT_SMALL, ha="center", va="top")
    rng = np.random.default_rng(23)
    for arm, colour, off in (("fixed timings", FIXED_ARM, -0.14), ("jittered timings", JITTER_ARM, 0.14)):
        q = per[per["arm"] == arm]
        mean = q.groupby("delay_ms")["net_use"].mean()
        ax.scatter([xi[d] + off + rng.uniform(-0.05, 0.05) for d in q["delay_ms"]], q["net_use"], s=3, facecolor=colour, edgecolor="none",
                   alpha=0.7, zorder=2)
        ax.plot([xi[d] + off for d in mean.index], mean.to_numpy(), color=colour, lw=1.0, marker="o", ms=2.4, zorder=3)
        for d, v in mean.items():
            sd.add("H", "delay profile", arm=arm, delay_ms=int(d), mean_untrained_net_use=float(v), replicates=int(q["replicate"].nunique()))
    ax.axhline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
    ax.set_xlim(-0.6, len(delays) - 0.4)
    ax.set_ylim(-1.15, 1.45)
    ax.set_xticks(range(len(delays)), [str(int(d)) for d in delays])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([-1, 0, 1], [f"{fs.MINUS}1", "0", "1"])
    ax.set_xlabel("Delay (ms)", labelpad=0.8)
    ax.set_ylabel("Net use, untrained timings", labelpad=0.8)
    fs.clean(ax)
    # per replicate: inside and outside the jitter range, fixed vs jittered
    x3 = t.copy()
    x3 = x3[~x3["trained"].astype(str).str.lower().eq("true")]
    inside = x3["inside_jitter_range"].astype(str).str.lower().eq("true")
    summ = x3.assign(where=np.where(inside, "inside", "outside")).groupby(["arm", "replicate", "where"])["net_use"].mean().unstack()
    ax2 = cv.axes(x0 + 9.0 + aw + 7.0, y0 + 9.6, w - (9.0 + aw + 7.0) - 1.4, h - 17.6)
    for g, (where, test) in enumerate((("inside", "untrained_inside_net_use: jittered minus fixed (two-sided sign-flip)"),
                                       ("outside", "untrained_outside_net_use: jittered minus fixed (two-sided sign-flip)"))):
        fx = summ.loc["fixed timings", where]
        jt = summ.loc["jittered timings", where]
        for r in fx.index:
            ax2.plot([g - 0.17, g + 0.17], [fx.loc[r], jt.loc[r]], color="#CFCFCF", lw=0.4, zorder=1)
        ax2.scatter(np.full(len(fx), g - 0.17), fx, s=6, facecolor=FIXED_ARM, edgecolor="none", zorder=2)
        ax2.scatter(np.full(len(jt), g + 0.17), jt, s=6, facecolor=JITTER_ARM, edgecolor="none", zorder=3)
        est, p = float(desc.loc[test, "estimate"]), float(desc.loc[test, "p_value"])
        ax2.text(g, -1.10, f"{fs.num(est, 2, sign=True)}{NL_}{fs.P_SYM} = {fs.fmt_p(p)}", fontsize=fs.PT_SMALL, ha="center", va="bottom", linespacing=1.0)
        for r in fx.index:
            sd.add("H", where, replicate=int(r), fixed=float(fx.loc[r]), jittered=float(jt.loc[r]))
        sd.add("H", f"{where} test", estimate=est, p_value=p)
    ax2.axhline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=0)
    ax2.set_xlim(-0.55, 1.55)
    ax2.set_ylim(-1.15, 1.45)
    ax2.set_yticks([-1, 0, 1], [f"{fs.MINUS}1", "0", "1"])
    ax2.set_xticks([0, 1], [f"≤ {jmax} ms", f"{int(max(delays))} ms"])
    ax2.tick_params(axis="x", length=0)
    ax2.set_title("Per replicate", fontsize=fs.PT_SMALL, pad=1.5)
    fs.clean(ax2)
    items = [([dict(kind="mark", s=6, fc=FIXED_ARM, ec=FIXED_ARM)], "fixed timings"), ([dict(kind="mark", s=6, fc=JITTER_ARM, ec=JITTER_ARM)], "jittered")]
    cv.key(x0 + 9.0, y0 + 6.2, items, size=fs.PT_SMALL, gap=2.0)
    cv.text(x0 + w - 0.8, y0 + 1.7, f"Holm {fs.P_SYM} = {fs.fmt_p(float(lock['p_adjusted']))}",
            size=fs.PT_SMALL, ha="right", va="center")
    sd.add("H", "primary", estimate=float(lock["estimate"]), holm_p=float(lock["p_adjusted"]), pairs=int(lock["n"]), jitter_max_delay_ms=jmax)


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("fig5")
    for rel in ("plan_v2/fig5_timings.csv", "plan_v2/fig5_scenes.csv", "plan_v2/fig5_code.csv", "plan_v2/fig5_drift.csv", "plan_v2/fig5_drift_margin.csv",
                "plan_v2/fig5_decision_cells.csv", "plan_v2/fig5_decision_cosines.csv", "plan_v2/fig5_record_outcomes.csv", "plan_v2/tests.csv"):
        sd.source(cm.RESULTS / rel)
    for f in sorted(TILES.glob("*.png")) + sorted(TILES.glob("*.json")):
        sd.source(f)
    panel_band(cv, sd)
    panel_grid(cv, sd, XS[0], ROW2, fs.unit(2), R2_H)
    panel_scenes(cv, sd, XS[2], ROW2, fs.unit(2), R2_H)
    panel_persist(cv, sd, XS[0], ROW3, fs.unit(2), R3_H)
    panel_flip(cv, sd, XS[2], ROW3, U, R3_H)
    panel_geometry(cv, sd, XS[3], ROW3, U, R3_H)
    panel_drift(cv, sd, XS[0], ROW4, fs.unit(2), R4_H)
    panel_range(cv, sd, XS[2], ROW4, fs.unit(2), R4_H)
    cm.finish(cv, "fig5", sd)


if __name__ == "__main__":
    main()
