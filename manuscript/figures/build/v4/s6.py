"""v4 Figure S6 (Cell), related to Figure 6 - Many routes, fading kicks and a wider window, in detail (174 mm wide).

A  dark film strip: the route network at every 10% of removal, random (the first of three seeded draws) and hubs first
   (cells in descending order of connections), each strip from the intact network; numbers: cells on routes
B  the readout stays reachable in every draw and order: cells on routes, and the other readout cells by shortest route
C  single extra spikes fade at both timings and for both cues: per kicked cell, the growth of the difference and the cells
   still changed at the decision, with the cell's direct partners
D  the wider window holds in every bootstrap resample: access window and cell-loss tolerance, across states over one state
E  memory under each held-out jitter draw, replicate by replicate
F  the task's route bypasses the mushroom body: flow capacity with Kenyon cells removed
G  typed routes from the visual input to the readout, intact and with 60% of the other cells removed, in the network grammar of
   MaleCNS Fig. 4H (moved from Fig. 6): the relay types with the most route traffic, the input and readout types that feed and
   read them most, and every connection among them above the threshold; edge width = input-to-readout walks of 2-3 synapses
   that use the connection (the same bins in both networks), edge colour and head = the sender's transmitter

Every number in a label is read from manuscript/results (plan_v2 via P12) or from the render metadata. Reader-facing names
only. Not shown (class c, RESULTS_INVENTORY_V4.md E1, E3, E4b, E6, E8-E11): susceptibility to perturbations (the shift that
halves memory, the behavioural dose response to cell removal, divergence after persistent shifts and its cells) and the
cost of jittered training. The window and the cell-loss tolerance appear only as ratios to training at one
state, as in Fig. 6E.

Run after P12 and render/v4_s6_tiles.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\v4\\s6.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
from lib import circuit as ci, figstyle as fs, palette as pal, qc  # noqa: E402

cm.use_v4()
TILES = cm.RENDERS / "v4" / "s6"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
TW, TH = META["tile_mm"]
LAB = META["label_mm"]
XS = [fs.unit_x(i) for i in range(4)]
STRIP = 2 * TH + fs.GUTTER
ROW2, R2_H = STRIP + 7.6, 46.0
ROW3, R3_H = ROW2 + R2_H + 3.4, 44.0
ROW4, R4_H = ROW3 + R3_H + 3.4, 52.0
H = ROW4 + R4_H
WHITE = fs.ON_DARK
NL_ = chr(10)
ONE, ACROSS = "one state", "across states"
ARM = {ONE: "#9E9E9E", ACROSS: fs.TEXT}                 # as Fig. 6E
HUB_LS = (0, (2.0, 1.2))
READOUT = "#7FD3FF"                                     # as Fig. 6A
ROUTES = [("readout_direct", "1", "#CFCFCF"), ("readout_two", "2", "#9E9E9E"), ("readout_three", "3", "#5E5E5E"),
          ("readout_four_plus", "≥ 4", fs.TEXT), ("readout_unreachable", "none", "#C9A227")]
GROWTH = LinearSegmentedColormap.from_list("growth", ["#3E8E84", "#A9D6CF", "#F5F5F5", "#E9CF8E", "#C08A1E"])   # fades teal, grows ochre
GROWTH_LIM = 30.0
Y_LO, Y_HI = -0.3, 2.0                                  # D: two rows of histograms


def tile_text(cv, x, y, s, size=None, **kw):
    t = cv.text(x, y, s, size=size or fs.PT_SMALL, color=WHITE, **kw)
    t.set_path_effects(fs.dark_halo(0.9))
    return t


def title(cv, x0, y0, letter, text):
    cv.letter(x0, y0, letter)
    cv.text(x0 + 4.0, y0 + 0.5, text, size=fs.PT, va="top", fontweight="bold")


def levels(lv, order, col, draw=None):
    """x (% removed, intact first) and the column for one order (and draw)."""
    q = lv[lv["order"] == order]
    if draw is not None:
        q = q[q["draw"] == draw]
    q = q.sort_values("share_removed")
    intact = lv[lv["order"] == "intact"].iloc[0]
    return np.r_[0.0, 100 * q["share_removed"].to_numpy(float)], np.r_[float(intact[col]), q[col].to_numpy(float)]


# ---------------------------------------------------------------- A: the route network at every level

def panel_strip(cv, sd):
    lv = cm.table("plan_v2/s6_routes_levels.csv")

    def cells(order, pct):
        if pct == 0:
            r = lv[lv["order"] == "intact"]
        else:
            r = lv[(lv["order"] == order) & np.isclose(lv["share_removed"], pct / 100)]
            r = r[r["draw"] == 0] if order == "random" else r
        return int(r["route_cells"].iloc[0])

    cv.letter(0.0, 0.0, "A")
    for s_, (order, prefix, name) in enumerate((("random", "random", f"Random{NL_}removal"), ("hubs first", "hubs", f"Hubs{NL_}first"))):
        y = s_ * (TH + fs.GUTTER)
        cv.text(0.0, y + TH / 2 + 1.5, name, size=fs.PT, va="center", fontweight="bold", linespacing=1.1)
        for k, pct in enumerate([0] + META["levels"]):
            x = LAB + fs.GUTTER + k * (TW + fs.GUTTER)
            key = "intact" if pct == 0 else f"{prefix}_{pct}"
            cv.tile(x, y, np.asarray(Image.open(TILES / f"routes_{key}.png").convert("RGB")), width=TW, height=TH, group="strip", label=key)
            n = cells(order, pct)
            tile_text(cv, x + 1.0, y + 0.8, "intact" if pct == 0 else f"{pct}%", size=fs.PT, va="top")
            tile_text(cv, x + 1.0, y + 3.5, f"{n:,}", va="top")
            sd.add("A", "tile", order=order, removed_percent=pct, route_cells=n, located=META["tiles"][key]["route_cells_located"])
    ky = STRIP + 2.8
    lo, hi = META["magma"]
    ax = cv.axes(LAB + fs.GUTTER, ky - 0.7, 12.0, 1.4)
    ax.imshow(pal.MAGNITUDE_ON_DARK(np.linspace(lo, hi, 128))[None, :, :3], aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax.set_axis_off()
    lab = "route traffic: walks of 2–3 synapses from the visual input cells to the readout cells, per cell (log)"
    cv.text(LAB + fs.GUTTER + 12.6, ky, lab, size=fs.PT_SMALL, va="center")
    end = cv.key(LAB + fs.GUTTER + 12.6 + fs.text_width(lab, fs.PT_SMALL) + 5.0, ky, [([dict(kind="mark", s=7, fc=READOUT, ec=READOUT)], "readout cells")],
                 size=fs.PT_SMALL)
    cv.text(end + 4.0, ky, "numbers: cells on routes", size=fs.PT_SMALL, va="center", color=fs.TEXT2)


# ---------------------------------------------------------------- B: every draw and order

def panel_routes(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "B", "The readout stays reachable in every draw and order")
    lv = cm.table("plan_v2/s6_routes_levels.csv")
    draws = sorted(int(d) for d in lv.loc[lv["order"] == "random", "draw"].unique())
    cv.key(x0 + 4.0, y0 + 6.0, [([dict(kind="hline", w=3.0, colour=fs.TEXT, lw=0.7)], f"random ({len(draws)} draws)"),
                                ([dict(kind="hline", w=3.0, colour=fs.TEXT, lw=0.8, ls=HUB_LS)], "hubs first")], size=fs.PT_SMALL, gap=2.4)
    ro = lv[lv["order"] == "intact"].iloc[0]
    total = int(ro[[c for c, _, _ in ROUTES]].sum())
    direct = lv["readout_direct"].astype(int)
    if direct.nunique() != 1:
        raise RuntimeError("direct readout routes changed with removal")
    reach = float(lv["readout_reachable"].min())
    top, ah = y0 + 10.4, h - 18.4
    # B1: cells on routes
    a1w = 22.0
    ax = cv.axes(x0 + 9.0, top, a1w, ah)
    for d in draws:
        xx, yy = levels(lv, "random", "route_cells", d)
        ax.plot(xx, yy, color=fs.TEXT, lw=0.6, zorder=3)
        sd.add("B", "route cells", order="random", draw=d, removed_percent=xx.tolist(), route_cells=yy.tolist())
    xx, yy = levels(lv, "hubs first", "route_cells")
    ax.plot(xx, yy, color=fs.TEXT, lw=0.8, ls=HUB_LS, zorder=3)
    sd.add("B", "route cells", order="hubs first", removed_percent=xx.tolist(), route_cells=yy.tolist())
    ax.set_yscale("log")
    ax.set_ylim(300, 3e5)
    ax.set_yticks([1e3, 1e4, 1e5], ["1,000", "10,000", "100,000"])
    ax.minorticks_off()
    ax.set_xlim(-4, 94)
    ax.set_xticks([0, 30, 60, 90])
    ax.set_xlabel("Cells removed (%)", labelpad=0.8)
    fs.clean(ax)
    cv.text(x0 + 9.0 + a1w / 2, top - 0.8, "Cells on routes", size=fs.PT_SMALL, ha="center", va="bottom")
    sd.add("B", "summary", readout_reachable_min=reach, readout_direct=int(direct.iloc[0]), readout_cells=total)
    # B2, B3: the other readout cells by shortest route
    bx, bw, gap = x0 + 9.0 + a1w + 10.0, 19.0, 7.0
    for j, order in enumerate(("random", "hubs first")):
        ax = cv.axes(bx + j * (bw + gap), top, bw, ah)
        for col, lab, colour in ROUTES:
            for d in (draws if order == "random" else [None]):
                xx, yy = levels(lv, order, col, d)
                ax.plot(xx, np.maximum(yy, 0.6), color=colour, lw=0.7, ls="-" if order == "random" else HUB_LS, zorder=3)
                sd.add("B", "readout by shortest route", order=order, draw=-1 if d is None else d, synapses=lab, removed_percent=xx.tolist(), readout_cells=yy.tolist())
            if j == 0:
                ax.text(96, max(float(levels(lv, order, col, 0)[1][-1]), 0.6), lab, fontsize=fs.PT_SMALL, ha="left", va="center", color=fs.TEXT2, clip_on=False)
        ax.set_yscale("log")
        ax.set_ylim(0.45, 9000)
        ax.set_yticks([0.6, 10, 100, 1000], ["0", "10", "100", "1,000"] if j == 0 else [""] * 4)
        ax.minorticks_off()
        ax.set_xlim(-4, 94)
        ax.set_xticks([0, 30, 60, 90])
        ax.set_xlabel("Cells removed (%)", labelpad=0.8)
        fs.clean(ax)
        ax.text(2, 1500, order, fontsize=fs.PT_SMALL, ha="left", va="top")
    cv.text(bx + (2 * bw + gap) / 2, top - 0.8, f"The {total:,} readout cells by shortest route (synapses)", size=fs.PT_SMALL, ha="center", va="bottom")


# ---------------------------------------------------------------- C: single extra spikes fade

def panel_kicks(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "C", "Single extra spikes fade at both timings and for both cues")
    k = cm.table("plan_v2/fig6_kicks.csv")
    kp = cm.table("plan_v2/fig6_kick_partners.csv")
    fd = cm.table("figure_data/fd_cells.csv").set_index("cell")
    cells = sorted(k["kicked_cell"].unique(), key=lambda c: -int((kp["kicked_cell"] == c).sum()))   # as Fig. 6C
    conds = sorted({(int(r.delay_ms), int(r.prefix_windows), int(r.cue)) for r in k.itertuples()})
    n0 = int((k["hamming_decision"] == 0).sum())
    med = float(np.median(k["growth_per_s"].to_numpy(float)))
    lab_w, cw, ch = 17.0, 7.4, 4.3
    mx, my = x0 + lab_w, y0 + 14.6
    ax = cv.axes(mx, my, cw * len(conds), ch * len(cells))
    for i, c in enumerate(cells):
        for j, (d, p, cue) in enumerate(conds):
            r = k[(k["kicked_cell"] == c) & (k["delay_ms"] == d) & (k["prefix_windows"] == p) & (k["cue"] == cue)].iloc[0]
            g, n = float(r["growth_per_s"]), float(r["hamming_decision"])
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=GROWTH(np.clip((g + GROWTH_LIM) / (2 * GROWTH_LIM), 0, 1)), edgecolor="white", linewidth=0.8))
            ax.text(j, i, f"{n:.0f}", ha="center", va="center", fontsize=fs.PT_SMALL, color=fs.TEXT)
            sd.add("C", "kick", kicked_cell=int(c), type=str(fd.loc[c, "type"]).replace("_", " "), delay_ms=d, prefix_windows=p, cue=cue,
                   growth_per_s=g, cells_changed_at_decision=n)
    ax.set_xlim(-0.5, len(conds) - 0.5)
    ax.set_ylim(len(cells) - 0.5, -0.5)
    ax.set_axis_off()
    for i, c in enumerate(cells):
        grp = pal.SUPERCLASS_GROUP.get(str(fd.loc[c, "superclass"]), "other")
        cv.text(mx - 0.8, my + (i + 0.5) * ch - 0.95, str(fd.loc[c, "type"]).replace("_", " "), size=fs.PT_SMALL, ha="right", va="center")
        cv.text(mx - 0.8, my + (i + 0.5) * ch + 1.05, pal.SHORT_GROUP[grp], size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)
    for t, d in enumerate(sorted({c[0] for c in conds})):
        js = [j for j, c in enumerate(conds) if c[0] == d]
        cv.text(mx + (min(js) + max(js) + 1) / 2 * cw, my - 3.8, f"{d}-ms delay", size=fs.PT_SMALL, ha="center", va="center")
        cv.line([mx + min(js) * cw + 0.6, mx + (max(js) + 1) * cw - 0.6], [my - 2.8, my - 2.8], fs.TEXT2, lw=0.4)
    for j, (_, _, cue) in enumerate(conds):
        cv.text(mx + (j + 0.5) * cw, my - 1.4, f"cue {cue}", size=fs.PT_SMALL, ha="center", va="center")
    # direct partners per kicked cell
    px0 = mx + cw * len(conds) + 3.0
    ax2 = cv.axes(px0, my, x0 + w - 1.4 - px0, ch * len(cells))
    for i, c in enumerate(cells):
        n = int((kp["kicked_cell"] == c).sum())
        ax2.barh([i], [n], height=0.55, color="#BDBDBD", linewidth=0, zorder=3)
        sd.add("C", "partners", kicked_cell=int(c), partners=n, synapses=int(kp.loc[kp["kicked_cell"] == c, "synapses"].sum()))
    ax2.set_ylim(len(cells) - 0.5, -0.5)
    ax2.set_xlim(0, 620)
    ax2.set_xticks([0, 250, 500])
    ax2.set_yticks([])
    ax2.tick_params(axis="x", labelsize=fs.PT_SMALL, pad=1.0)
    ax2.set_xlabel("Direct partners", labelpad=0.8)
    fs.clean(ax2, left=False)
    # key: growth colour and the numbers
    ky = my + ch * len(cells) + 4.0
    cv.text(mx - 0.8, ky, "fades", size=fs.PT_SMALL, ha="right", va="center")
    ax3 = cv.axes(mx, ky - 0.7, 16.0, 1.4)
    ax3.imshow(GROWTH(np.linspace(0, 1, 128))[None, :, :3], aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax3.set_axis_off()
    cv.text(mx + 16.6, ky, "grows", size=fs.PT_SMALL, va="center")
    kx = mx - 0.8 - fs.text_width("fades", fs.PT_SMALL)
    cv.text(kx, ky + 2.6, f"color: growth after the spike, {fs.MINUS}{GROWTH_LIM:g} to +{GROWTH_LIM:g} s$^{{-1}}$",
            size=fs.PT_SMALL, ha="left", va="center", color=fs.TEXT2)
    cv.text(kx, ky + 5.0, "numbers: cells still changed at the decision", size=fs.PT_SMALL, ha="left", va="center", color=fs.TEXT2)


# ---------------------------------------------------------------- D: every bootstrap resample

def panel_boot(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "D", "The wider window holds in every resample")
    bt = cm.table("plan_v2/s6_bootstrap.csv")
    summ = cm.table("plan_v2/fig6_robust_summary.csv").set_index("quantity")
    rows = [("access window", "ED50 ratio, across states / one state", "Access window"),
            ("cell-loss tolerance", "removal ED50 ratio, across states / one state", "Cell-loss tolerance")]
    n_res = int(bt.groupby("quantity").size().min())
    mins = {q: float(bt.loc[bt["quantity"] == q, "ratio"].min()) for q, _, _ in rows}
    lab_w = 27.0
    ax = cv.axes(x0 + lab_w, y0 + 10.4, w - lab_w - 1.4, h - 18.4)
    bins = np.logspace(np.log10(0.7), np.log10(20), 100)
    for i, (q, sq, lab) in enumerate(rows):
        base = 1.0 - i
        v = bt.loc[bt["quantity"] == q, "ratio"].to_numpy(float)
        cnt, edges = np.histogram(v, bins)
        ax.bar(edges[:-1], 0.6 * cnt / cnt.max(), width=np.diff(edges), bottom=base, align="edge", color=ARM[ACROSS], linewidth=0, zorder=3)
        r = summ.loc[sq]
        ax.plot([r["ci_low"], r["ci_high"]], [base - 0.1] * 2, color=fs.TEXT, lw=0.8, zorder=4)
        ax.scatter([r["estimate"]], [base - 0.1], s=14, marker="^", facecolor="white", edgecolor=fs.TEXT, linewidth=0.6, zorder=5)
        ymm = y0 + 10.4 + (Y_HI - (base + 0.3)) / (Y_HI - Y_LO) * (h - 18.4)
        cv.text(x0 + lab_w - 1.0, ymm - 1.2, lab, size=fs.PT_SMALL, ha="right", va="center")
        cv.text(x0 + lab_w - 1.0, ymm + 1.2, f"×{r['estimate']:.1f} (95% {r['ci_low']:.1f}–{r['ci_high']:.1f})", size=fs.PT_SMALL, ha="right", va="center",
                color=fs.TEXT2)
        sd.add("D", q, resamples=int(len(v)), estimate=float(r["estimate"]), ci=[float(r["ci_low"]), float(r["ci_high"])], minimum=float(v.min()),
               histogram=dict(edges=edges.tolist(), counts=cnt.tolist()))
    ax.axvline(1.0, color=fs.TEXT2, lw=0.6, ls=(0, (2.0, 1.4)), zorder=2)
    ax.text(1.08, 1.95, "one state", fontsize=fs.PT_SMALL, ha="left", va="top", color=fs.TEXT2)
    ax.set_xscale("log")
    ax.set_xlim(0.7, 20)
    ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_yticks([])
    ax.set_xlabel("Across states / one state (×)", labelpad=0.8)
    fs.clean(ax)
    ax.spines["left"].set_visible(False)
    cv.key(x0 + w - 33.0, y0 + 10.4 + 1.6, [([dict(kind="mark", s=12, marker="^", fc="white", ec=fs.TEXT)], "estimate"),
                                            ([dict(kind="hline", w=2.4, colour=fs.TEXT, lw=0.8)], "95% interval")], size=fs.PT_SMALL, gap=1.6)


# ---------------------------------------------------------------- E: held-out jitter draws

def panel_heldout(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "E", "Memory under held-out jitter")
    hd = cm.table("plan_v2/s6_heldout_draws.csv")
    ax = cv.axes(x0 + 9.0, y0 + 9.4, w - 10.6, h - 19.4)
    rng = np.random.default_rng(606)
    for g, tr in enumerate((ONE, ACROSS)):
        q = hd[hd["training"] == tr]
        for rep, qq in q.groupby("replicate"):
            v = qq.sort_values("heldout_draw")["memory_under_jitter"].to_numpy(float)
            xs = g * 2.6 + np.arange(len(v)) + rng.uniform(-0.12, 0.12)
            ax.plot(xs, v, color=ARM[tr], lw=0.5, zorder=2)
            ax.scatter(xs, v, s=6, facecolor=ARM[tr], edgecolor="none", zorder=3)
            sd.add("E", tr, replicate=int(rep), draws=v.tolist())
        for j, (_, dq) in enumerate(q.groupby("heldout_draw")):
            ax.plot([g * 2.6 + j - 0.3, g * 2.6 + j + 0.3], [dq["memory_under_jitter"].mean()] * 2, color=fs.TEXT, lw=0.9, zorder=4)
    ax.axhline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=0)
    ax.set_xlim(-0.6, 4.2)
    ax.set_ylim(-0.25, 1.0)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xticks([0, 1, 2.6, 3.6], ["1", "2", "1", "2"])
    ax.tick_params(axis="x", length=0, pad=1.0)
    ax.set_ylabel("Memory (net use)", labelpad=0.8)
    fs.clean(ax)
    for g, tr in enumerate((ONE, ACROSS)):
        cv.text(x0 + 9.0 + (w - 10.6) * (g * 2.6 + 0.5 + 0.6) / 4.8, y0 + h - 5.4, tr, size=fs.PT_SMALL, ha="center", va="center", color=ARM[tr] if tr == ACROSS else fs.TEXT2)
    cv.text(x0 + 9.0 + (w - 10.6) / 2, y0 + h - 2.6, "held-out draw, by training", size=fs.PT_SMALL, ha="center", va="center", color=fs.TEXT2)


# ---------------------------------------------------------------- F: the visual flow bypasses the mushroom body

def panel_flow(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "F", "Kenyon cells are off-route")
    flow = cm.table("plan_v2/fig6_wiring_flow.csv")
    total = int(flow.loc[flow["lesion"] == "all Kenyon cells", "cells_removed"].iloc[0])
    ax = cv.axes(x0 + 9.0, y0 + 9.4, w - 10.6, h - 17.4)
    for pw, colour, lab, ly in (("visual inputs -> readout", fs.TEXT, "visual input → readout", 1.04),
                                ("ORNs -> MBONs", "#9E9E9E", f"odor input →{NL_}mushroom-body output", 0.06)):
        q = flow[flow["pathway"] == pw].assign(pct=lambda d: 100 * d["cells_removed"] / total).sort_values("pct")
        ax.plot(q["pct"], q["relative"], color=colour, lw=0.9, marker="o", ms=2.2, zorder=3)
        ax.text(3, ly, lab, fontsize=fs.PT_SMALL, ha="left", va="bottom", color=colour if colour == fs.TEXT else fs.TEXT2, linespacing=1.0)
        for r in q.itertuples():
            sd.add("F", pw, kenyon_cells_removed=int(r.cells_removed), percent=float(r.pct), relative_flow=float(r.relative))
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 1.18)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel("Kenyon cells removed (%)", labelpad=0.8)
    ax.set_ylabel("Flow, relative to intact", labelpad=0.8)
    fs.clean(ax)


# ---------------------------------------------------------------- G: typed routes, intact and after removal

N_IN, N_RELAY, N_OUT, PITCH = 6, 7, 6, 4.1                # types per column; chip pitch (mm)
TYPED_STATES = (("intact", "intact", "traffic_0"), ("60% removed", "60% of the other cells removed", "traffic_60"))


def panel_typed(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "G", "The leading route types keep their routes after removal")
    e = cm.table("plan_v2/s6_route_pairs.csv")
    traffic = cm.table("plan_v2/fig6_route_types.csv").set_index("type")
    trn = cm.table("plan_v2/fig2_cue_flow_types.csv").set_index("type")["transmitter"]
    e = e[(e["pre"] != "untyped") & (e["post"] != "untyped")]
    intact = e[e["state"] == "intact"]
    relays = list(traffic["traffic_0"].sort_values(ascending=False).index[:N_RELAY])
    first = intact[(intact["hop"] == "first") & intact["post"].isin(relays)]
    last = intact[(intact["hop"] == "last") & intact["pre"].isin(relays)]
    inputs = list(first.groupby("pre")["walks"].sum().sort_values(ascending=False).index[:N_IN])
    readouts = list(last.groupby("post")["walks"].sum().sort_values(ascending=False).index[:N_OUT])
    sel = pd.concat([first[first["pre"].isin(inputs)], last[last["post"].isin(readouts)]])
    thresh = float(sel["walks"].quantile(0.45))
    sel = sel[sel["walks"] >= thresh]
    shown = e.merge(sel[["pre", "post", "hop"]], on=["pre", "post", "hop"])
    bins = list(np.geomspace(thresh / 3, float(shown["walks"].max()), 5))
    bins[-1] += 1.0
    row = {r: i for i, r in enumerate(relays)}

    def bary(n, hop):                                     # inputs and readouts sit at their partners' mean row
        x = sel[(sel["pre"] == n) & (sel["hop"] == "first")] if hop == "first" else sel[(sel["post"] == n) & (sel["hop"] == "last")]
        other = x["post"] if hop == "first" else x["pre"]
        return float(np.average([row[o] for o in other], weights=x["walks"])) if len(x) else 1e3
    inputs.sort(key=lambda n: bary(n, "first"))
    readouts.sort(key=lambda n: bary(n, "last"))
    half = (w - 2.0) / 2
    top = y0 + 12.6
    for k, (state, heading, col) in enumerate(TYPED_STATES):
        hx = x0 + k * (half + 2.0)
        cols = (hx + 10.0, hx + half / 2, hx + half - 11.0)
        ci.module(cv, hx, y0 + 5.0, half, 37.5, None)
        cv.text(hx + half / 2, y0 + 6.5, heading, size=fs.PT, ha="center", va="center", fontweight="bold")
        for c, lab in zip(cols, ("visual input", "relay", "readout")):
            ci.column_header(cv, c - 8.0, c + 8.0, y0 + 10.4, lab, size=fs.PT_SMALL)
        net = ci.Network(cv)
        for c, names in enumerate((inputs, relays, readouts)):
            for i, n in enumerate(names):
                y = top + PITCH * (i + 0.5) + (N_RELAY - len(names)) * PITCH / 2
                net.add(f"{c}:{n}", n, cols[c], y, c, h=3.0, size=fs.PT_SMALL, fill="#FFFFFF", edge=fs.TEXT if c == 2 else "none")
        drawn = 0
        for r in shown[shown["state"] == state].itertuples():
            a, b = (f"0:{r.pre}", f"1:{r.post}") if r.hop == "first" else (f"1:{r.pre}", f"2:{r.post}")
            if r.walks < bins[0]:
                continue
            bi = ci.bin_of(r.walks, bins)
            t_ = str(trn.get(r.pre, "unknown"))
            net.link(a, b, ci.WIDTHS[bi], ci.colour_for(t_, bi), ci.head_for(t_))
            sd.add("G", "connection", state=state, pre=r.pre, post=r.post, hop=r.hop, walks=float(r.walks), width_bin=bi + 1,
                   transmitter=t_)
            drawn += 1
        net.draw()
        sd.add("G", "network", state=state, connections_drawn=drawn, connections_selected=int(len(sel)),
               relay_traffic={n: float(traffic.loc[n, col]) for n in relays})
    ky = y0 + 46.0
    fmt = lambda v: f"{v / 1e6:.1f}" if v < 1e7 else f"{v / 1e6:.0f}"  # noqa: E731
    labs = [f"{fmt(bins[i])}–{fmt(bins[i + 1])} M" for i in range(4)]
    cv.text(x0 + 1.0, ky + 1.5, "walks through" + NL_ + "the connection", size=fs.PT_SMALL, va="center", color=fs.TEXT2, linespacing=1.0)
    xe = ci.width_key(cv, x0 + 19.0, ky, labs, length=3.6, col_w=fs.text_width("27–150 M", fs.PT_SMALL) + 6.4)
    xe = ci.transmitter_key(cv, xe + 6.0, ky + 1.5)
    cv.key(xe + 6.0, ky + 1.5, [([dict(kind="box", size=2.0, fc="#FFFFFF", ec=fs.TEXT, lw=0.6)], "readout type")], size=fs.PT_SMALL, gap=1.6)
    sd.add("G", "edge bins (walks)", edges=[float(b) for b in bins], threshold=thresh)


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("s6")
    for rel in ("plan_v2/s6_routes.csv", "plan_v2/s6_routes_levels.csv", "plan_v2/fig6_kicks.csv", "plan_v2/fig6_kick_partners.csv",
                "figure_data/fd_cells.csv", "plan_v2/s6_bootstrap.csv", "plan_v2/fig6_robust_summary.csv", "plan_v2/s6_heldout_draws.csv",
                "plan_v2/fig6_wiring_flow.csv", "plan_v2/s6_route_pairs.csv", "plan_v2/fig6_route_types.csv",
                "plan_v2/fig2_cue_flow_types.csv"):
        sd.source(cm.RESULTS / rel)
    for f in sorted(TILES.glob("*.png")) + sorted(TILES.glob("*.json")):
        sd.source(f)
    panel_strip(cv, sd)
    panel_routes(cv, sd, XS[0], ROW2, fs.unit(2), R2_H)
    panel_kicks(cv, sd, XS[2], ROW2, fs.unit(2), R2_H)
    panel_boot(cv, sd, XS[0], ROW3, fs.unit(2), R3_H)
    panel_heldout(cv, sd, XS[2], ROW3, fs.unit(1), R3_H)
    panel_flow(cv, sd, XS[3], ROW3, fs.unit(1), R3_H)
    panel_typed(cv, sd, XS[0], ROW4, fs.FULL, R4_H)
    cm.finish(cv, "s6", sd, kind="supplement")


if __name__ == "__main__":
    main()
