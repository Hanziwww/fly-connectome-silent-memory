"""v4 Fig. 6 (Cell) - The wiring is robust; access follows the trained states (174 mm wide). Rebuilt after the user's correction of
2026-10-22: susceptibility to perturbations is an unfavourable outcome (class c), so this figure shows only the network's
robustness and how training across states widens access.

A  dark band, "many routes, one readout": the cells on routes from the visual input cells to the readout cells, intact and
   after random removal of 30, 60 and 90% of the other cells; the route network thins tenfold, the readout stays reachable
B  reachability and three-step routes against removal (random and hubs first), and the cell classes that carry the routes
C  a single extra spike is absorbed: the kicked cells and their direct partners (dark inset), the outcome of every kick,
   and the partners each kicked cell reaches
D  the network's firing is stable across four decades of threshold change
E  training across states (threshold jitter, sigma 1 uV) widens the access window tenfold, keeps memory under jitter and
   doubles the tolerance to cell loss
F  with the same mechanism
G  routes by cell class, in the form of MaleCNS Fig. 2B: visual input -> intermediate classes -> readout by class, ribbon
   width = input-to-readout walks of 2-3 synapses, strands by presynaptic transmitter, darker core = the walks left after
   90% removal (the band's draw)
H  the share of the routes carried across cell types, and the types that carry them at each level, by class

Every number in a label is read from manuscript/results (plan_v2 via P12) or from the render metadata. Reader-facing names
only. Not shown (class c, RESULTS_INVENTORY_V4.md): the susceptibility results (E1, E3, E4 behaviour, E6, E9-E11) and the
cost of training across states (E8).

Run after P12 and render/v4_fig6_tiles.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\v4\\fig6.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.ticker import NullFormatter
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
from lib import circuit as ci, figstyle as fs, palette as pal, qc  # noqa: E402

cm.use_v4()
TILES = cm.RENDERS / "v4" / "fig6"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
TW, TH = META["tile_mm"]
KW, KH = META["kick_mm"]
U = fs.unit(1)
XS = [fs.unit_x(i) for i in range(4)]
ROW2 = TH + 4.2
R2_H = 48.0
ROW3 = ROW2 + R2_H + 3.4
R3_H = 46.0
ROW4 = ROW3 + R3_H + 3.4
R4_H = 58.0
H = ROW4 + R4_H
LEVELS = ("0", "30", "60", "90")
LEVEL_SHADE = {"0": "#DCD7D0", "30": "#B9B3AB", "60": "#8E8880", "90": "#4F4A44"}   # intact, then what remains after removal
MIN_FLOW, MIN_BACK = 0.002, 0.01                       # route map: forward flows below 0.2% of the walks, flows back across or within a
                                                       # class below 1%, are not drawn (listed in the source data)
WHITE = fs.ON_DARK
NL_ = chr(10)
ONE, ACROSS = "one state", "across states"
ARM = {ONE: "#9E9E9E", ACROSS: fs.TEXT}                 # as Fig. 5H: one training condition grey, a range of them black


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
    return t[t["family"] == family].set_index("test")


def log_axis(ax, which="x"):
    getattr(ax, f"{which}axis").set_minor_formatter(NullFormatter())


def pct(x):
    return f"{100 * x:.2f}".rstrip("0").rstrip(".") + "%"


def gradient(cv, x, y, w, h, colours):
    ax = cv.axes(x, y, w, h)
    ax.imshow(np.asarray(colours)[:, None, :3] * np.ones((1, 2, 1)), aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax.set_axis_off()


# ---------------------------------------------------------------- A: dark band, the route network

def panel_band(cv, sd):
    xs = [k * (TW + fs.GUTTER) for k in range(4)]
    summ = cm.table("plan_v2/fig6_routes_summary.csv").set_index("share_removed")
    for k, (lev, x) in enumerate(zip(META["levels"], xs)):
        name = f"routes_{lev}"
        meta = json.loads((TILES / f"{name}.json").read_text(encoding="utf-8"))
        cv.tile(x, 0, np.asarray(Image.open(TILES / f"{name}.png").convert("RGB")), width=TW, height=TH, group="band", label=name)
        r = summ.loc[int(lev) / 100]
        tx = x + (4.6 if k == 0 else 1.4)
        tile_text(cv, tx, 1.2, "Intact" if lev == "0" else f"{lev}% of cells removed", va="top", size=fs.PT)
        tile_text(cv, tx, 4.3, f"{int(r['route_cells']):,} route cells · readout {pct(r['readout_reachable'])}", va="top")
        sd.add("A", "level", removed_percent=int(lev), cells_removed=int(r["cells_removed"]), route_cells=int(r["route_cells"]),
               readout_reachable=float(r["readout_reachable"]), walks3_relative=float(r["walks3_relative"]), route_cells_drawn=int(meta["route_cells_located"]))
    white_letter(cv, 1.2, 0.9, "A")
    # key, black space right of the VNC in the first tile: route traffic (magma, log) and the readout cells
    lo, hi = META["magma"]
    bx, by, bh = xs[0] + TW - 2.6, 32.0, 13.0
    gradient(cv, bx, by, 1.4, bh, pal.MAGNITUDE_ON_DARK(np.linspace(hi, lo, 128)))
    tile_text(cv, bx - 0.5, by, "many", va="center", ha="right")
    tile_text(cv, bx - 0.5, by + bh, "few", va="center", ha="right")
    tile_text(cv, bx + 1.4, by - 1.0, f"routes through{NL_}the cell", va="bottom", ha="right", linespacing=1.0)
    cv.mark(bx + 0.7, by + bh + 3.0, s=6, fc="#7FD3FF", ec="#7FD3FF")
    tile_text(cv, bx - 0.3, by + bh + 3.0, "readout", va="center", ha="right")
    for i, line in enumerate(["routes: walks of 2–3", "synapses from the", "visual input cells", "to the readout"]):
        tile_text(cv, xs[0] + 1.0, TH - 12.4 + 2.4 * i, line, va="center")
    # scale bar and orientation (last tile)
    meta = json.loads((TILES / f"routes_{META['levels'][-1]}.json").read_text(encoding="utf-8"))
    img = np.asarray(Image.open(TILES / f"routes_{META['levels'][-1]}.png").convert("RGB"))
    length = 100 / meta["scale_um_per_px"] / (img.shape[1] / TW)
    cv.line([xs[3] + TW - 1.6 - length, xs[3] + TW - 1.6], [TH - 2.0, TH - 2.0], WHITE, lw=1.0, solid_capstyle="butt")
    ox, oy = xs[3] + TW - 12.0, TH - 6.4
    cv.line([ox, ox + 2.8], [oy, oy], WHITE, lw=0.5)
    tile_text(cv, ox + 3.2, oy, "L", va="center")
    cv.line([ox, ox], [oy, oy + 2.8], WHITE, lw=0.5)
    tile_text(cv, ox, oy + 3.2, "V", ha="center", va="top")
    sd.add("A", "view", scale_bar_um=100, um_per_px=meta["scale_um_per_px"], traffic_scale="log10, one scale for all four tiles",
           log10_traffic_range=META["log10_traffic_range"])


# ---------------------------------------------------------------- B: reachability and the classes that carry the routes

def panel_routes(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "B", "The readout stays reachable")
    g = cm.table("plan_v2/fig6_wiring_graph.csv")
    flow = cm.table("plan_v2/fig6_wiring_flow.csv")
    aw = 38.0
    ax = cv.axes(x0 + 9.0, y0 + 9.6, aw, h - 17.6)
    ran = g[g["order"] == "random cells"].sort_values("share_removed")
    hub = g[g["order"] == "hubs first"].sort_values("share_removed")
    ax.plot(100 * ran["share_removed"], ran["readout_reachable"], color=fs.TEXT, lw=1.0, marker="o", ms=1.8, zorder=3)
    ax.plot(100 * ran["share_removed"], ran["walks3_relative"], color=fs.TEXT2, lw=0.7, ls=(0, (2, 1)), zorder=3)
    ax.plot(100 * hub["share_removed"], hub["walks3_relative"], color="#BDBDBD", lw=0.7, ls=(0, (2, 1)), zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 100)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([1, 10, 100], ["1", "10", "100"])
    log_axis(ax, "x")
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_xlabel("Cells removed (%)", labelpad=0.8)
    ax.set_ylabel("Relative to intact", labelpad=0.8)
    fs.clean(ax)
    reach90 = float(ran[np.isclose(ran["share_removed"], 0.9)]["readout_reachable_min"].iloc[0])
    ax.text(1.0, 1.02, "readout reachable", fontsize=fs.PT_SMALL, ha="left", va="bottom")
    ax.text(14, 0.78, f"routes,{NL_}random", fontsize=fs.PT_SMALL, ha="right", va="top", color=fs.TEXT2, linespacing=1.0)
    ax.text(2.2, 0.62, f"routes,{NL_}hubs first", fontsize=fs.PT_SMALL, ha="left", va="top", color="#8C8C8C", linespacing=1.0)
    vis = float(flow[(flow["pathway"] == "visual inputs -> readout") & (flow["lesion"] == "all Kenyon cells")]["relative"].iloc[0])
    for _, q in g.iterrows():
        sd.add("B", "graph", order=q["order"], share_removed=float(q["share_removed"]), readout_reachable=float(q["readout_reachable"]),
               walks3_relative=float(q["walks3_relative"]))
    # why: the readout cells by their shortest route from the inputs at each removal level (routes lengthen, not break)
    rt = cm.table("plan_v2/fig6_routes.csv")
    ro = rt[rt["role"] == "readout"]
    levels = META["levels"]
    direct = int((ro["hops_0"] == 1).sum())
    if not all(int((ro[f"hops_{lev}"] == 1).sum()) == direct for lev in levels):
        raise RuntimeError("direct readout routes changed with removal")
    ax2 = cv.axes(x0 + 9.0 + aw + 11.0, y0 + 9.6, w - (9.0 + aw + 11.0) - 1.4, h - 17.6)
    groups = [("2", lambda hh: hh == 2, "#9E9E9E"), ("3", lambda hh: hh == 3, fs.TEXT2), ("≥ 4", lambda hh: hh >= 4, fs.TEXT),
              ("none", lambda hh: hh < 0, "#DADADA")]
    wbar = 0.19
    for j, (lab, sel, colour) in enumerate(groups):
        vals = np.array([int(sel(ro[f"hops_{lev}"]).sum()) for lev in levels])
        xx = np.arange(len(levels)) + (j - 1.5) * wbar
        ax2.bar(xx[vals > 0], vals[vals > 0], width=wbar * 0.92, color=colour, linewidth=0, zorder=3)
        sd.add("B", "readout by shortest route", synapses=lab, counts_by_level=dict(zip(levels, vals.tolist())))
    ax2.set_yscale("log")
    ax2.set_ylim(0.5, 2000)
    ax2.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    log_axis(ax2, "y")
    ax2.set_xlim(-0.6, len(levels) - 0.4)
    ax2.set_xticks(range(len(levels)), levels)
    ax2.tick_params(axis="x", length=0)
    ax2.set_xlabel("Cells removed (%)", labelpad=0.8)
    ax2.set_ylabel("Readout cells", labelpad=0.8)
    fs.clean(ax2)
    ax2.set_title("Shortest route (synapses)", fontsize=fs.PT_SMALL, pad=1.5)
    items = [([dict(kind="mark", marker="s", s=7, fc=c, ec=c)], lab) for lab, _, c in groups]
    cv.key(x0 + 9.0 + aw + 12.0, y0 + 11.6, items, size=fs.PT_SMALL, gap=1.0)
    sd.add("B", "direct", readout_cells=int(len(ro)), direct_input=direct)


# ---------------------------------------------------------------- C: a single extra spike is absorbed

def panel_kicks(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "C", "A single extra spike is absorbed")
    k = cm.table("plan_v2/fig6_kicks.csv")
    kp = cm.table("plan_v2/fig6_kick_partners.csv")
    fd = cm.table("figure_data/fd_cells.csv").set_index("cell")
    meta = json.loads((TILES / "kicks.json").read_text(encoding="utf-8"))
    img = np.asarray(Image.open(TILES / "kicks.png").convert("RGB"))
    kh = h - 6.2
    kw = kh * KW / KH
    cv.tile(x0, y0 + 5.6, img, width=kw, height=kh, group="kicks", label="kicks")
    ppm = img.shape[1] / kw
    name = lambda c: str(fd.loc[c, "type"]).replace("_", " ")  # noqa: E731   published type names, shown with spaces
    left_y = y0 + 5.6 + 20.5
    for c, (px_, py_) in sorted(zip(meta["kicked"], meta["kicked_px"]), key=lambda e: e[1][1] if np.isfinite(e[1][1]) else 1e9):
        if not np.isfinite(px_):
            continue
        sx, sy = x0 + px_ / ppm, y0 + 5.6 + py_ / ppm
        if px_ / ppm > kw / 2:                                    # right half: the label sits beside the soma, on black
            tile_text(cv, sx + 1.2, sy - 1.2, name(c), va="bottom")
        else:                                                     # left half: the label goes to the black margin, with a leader
            tile_text(cv, x0 + 1.0, left_y, name(c), va="center")
            cv.line([x0 + 1.0 + fs.text_width(name(c), fs.PT_SMALL) + 0.6, sx - 0.6], [left_y, sy], WHITE, lw=0.3)
            left_y += 4.0
    located = int(np.isfinite(np.asarray(meta["kicked_px"], float)[:, 0]).sum())
    tile_text(cv, x0 + 1.0, y0 + 5.6 + kh - 4.6, f"{located} of {len(meta['kicked'])} kicked cells", va="center")
    tile_text(cv, x0 + 1.0, y0 + 5.6 + kh - 2.2, "and their partners", va="center")
    # every kick: cells still changed at the decision, per kicked cell (two timings x two cues)
    cells = sorted(k["kicked_cell"].unique(), key=lambda c: -kp[kp["kicked_cell"] == c].shape[0])
    ax = cv.axes(x0 + kw + 9.0, y0 + 9.6, w - kw - 10.4, (h - 17.6) * 0.55)
    rng = np.random.default_rng(66)
    for i, c in enumerate(cells):
        q = k[k["kicked_cell"] == c]
        for _, r in q.iterrows():
            v = float(r["hamming_decision"])
            mk = "o" if int(r["delay_ms"]) == 160 else "s"
            ax.scatter([i + rng.uniform(-0.18, 0.18)], [max(v, 0.6)], s=8, marker=mk, facecolor=fs.TEXT if v == 0 else "white", edgecolor=fs.TEXT,
                       linewidth=0.5, zorder=3)
            sd.add("C", "kick", kicked_cell=int(c), type=str(fd.loc[c, "type"]).replace("_", " "), delay_ms=int(r["delay_ms"]), cue=int(r["cue"]),
                   cells_changed_at_decision=v)
    ax.set_yscale("log")
    ax.set_ylim(0.4, 400)
    ax.set_yticks([0.6, 10, 100], ["0", "10", "100"])
    log_axis(ax, "y")
    ax.set_xlim(-0.6, len(cells) - 0.4)
    ax.set_xticks([])
    ax.set_ylabel(f"Changed at{NL_}decision", labelpad=0.8, linespacing=1.0)
    fs.clean(ax, bottom=False)
    ax.spines["bottom"].set_visible(False)
    n0 = int((k["hamming_decision"] == 0).sum())
    # partners each kicked cell reaches
    ax2 = cv.axes(x0 + kw + 9.0, y0 + 9.6 + (h - 17.6) * 0.55 + 1.6, w - kw - 10.4, (h - 17.6) * 0.45 - 1.6)
    for i, c in enumerate(cells):
        n = int((kp["kicked_cell"] == c).sum())
        ax2.bar([i], [n], width=0.6, color="#BDBDBD", linewidth=0, zorder=3)
        sd.add("C", "partners", kicked_cell=int(c), partners=n, synapses=int(kp[kp["kicked_cell"] == c]["synapses"].sum()))
    ax2.set_xlim(-0.6, len(cells) - 0.4)
    ax2.set_ylim(0, 650)
    ax2.set_yticks([0, 500])
    ax2.set_xticks(range(len(cells)), [str(fd.loc[c, "type"]).replace("_", " ") for c in cells], rotation=30, ha="right", rotation_mode="anchor")
    ax2.tick_params(axis="x", length=0, pad=1.0)
    ax2.set_ylabel("Partners", labelpad=0.8)
    fs.clean(ax2)
    cv.key(x0 + kw + 9.0, y0 + 6.0, [([dict(kind="mark", s=7, marker="o", fc=fs.TEXT2, ec=fs.TEXT2)], "160-ms delay"),
                                     ([dict(kind="mark", s=7, marker="s", fc=fs.TEXT2, ec=fs.TEXT2)], "40 ms")], size=fs.PT_SMALL, gap=1.4)
    sd.add("C", "summary", kicks=int(len(k)), nothing_left=n0, kicked_located=located, fibres_drawn=int(meta["fibres"]))


# ---------------------------------------------------------------- D: stable firing

def panel_stability(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "D", "Firing is stable")
    rate = cm.table("plan_v2/fig6_rate.csv")
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 15.6)
    run_lo, run_hi = -float(rate["last_stable_mv"].iloc[0]), -float(rate["runaway_from_mv"].iloc[0])
    ax.axvspan(run_lo, 12.0, color="#EDEDED", lw=0, zorder=0)
    x_zero = 3e-5
    for src, colour, ls in (("memory task, analysed learner", fs.TEXT, "-"), ("uniform-offset sweep", fs.TEXT2, (0, (2, 1)))):
        q = rate[rate["source"] == src].sort_values("shift_mv", ascending=False)
        xx = np.where(q["shift_mv"] == 0, x_zero, -q["shift_mv"])
        ax.plot(xx, q["mean_rate_hz"], color=colour, lw=0.8, ls=ls, marker="o", ms=1.4, zorder=3)
        for _, row in q.iterrows():
            sd.add("D", "rate", source=src, shift_mv=float(row["shift_mv"]), mean_rate_hz=float(row["mean_rate_hz"]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2e-5, 12.0)
    ax.set_ylim(0.1, 2000)
    ax.set_xticks([1e-4, 1e-2, 1], ["0.1 μV", "10 μV", "1 mV"])
    ax.set_yticks([0.1, 10, 1000], ["0.1", "10", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("Threshold lowered by", labelpad=0.8)
    ax.set_ylabel("Mean rate (Hz)", labelpad=0.8)
    fs.clean(ax)
    task = rate[rate["source"] == "memory task, analysed learner"]
    flat = task[task["shift_mv"] >= -1.0]["mean_rate_hz"]
    ax.text(np.sqrt(run_lo * 12.0), 0.13, "runaway", fontsize=fs.PT_SMALL, ha="center", va="bottom", linespacing=1.0)
    sd.add("D", "summary", rate_up_to_1mv_hz=[float(flat.min()), float(flat.max())], runaway_from_mv=run_hi, last_stable_mv=run_lo)


# ---------------------------------------------------------------- E: training across states widens access

def panel_robust(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "E", "Training across states widens access")
    rb = cm.table("plan_v2/fig6_robust.csv")
    summ = cm.table("plan_v2/fig6_robust_summary.csv").set_index("quantity")
    prim = tests("F14.DY.robust")
    sec = tests("F14.DY.robust.secondary")
    measures = [("access_window_fold", f"Access window{NL_}(× one state)", True, float(prim.loc["th0.001:ed50-c01", "p_adjusted"]), "Holm", "ed50_kind"),
                ("memory_under_jitter", f"Memory under{NL_}±1-μV jitter", False, float(sec.loc["th0.001:own-perturbation", "p_adjusted"]), "BH", None),
                ("cell_loss_tolerance_fold", f"Cell-loss tolerance{NL_}(× one state)", False, float(sec.loc["th0.001:removal-ed50", "p_adjusted"]), "BH",
                 "ed50_removal_kind")]
    aw = (w - 9.0 - 2 * 9.0 - 1.4) / 3
    rng = np.random.default_rng(61)
    for k, (col, lab, log, p, corr, kind) in enumerate(measures):
        ax = cv.axes(x0 + 9.0 + k * (aw + 9.0), y0 + 13.0, aw, h - 21.0)
        for i, tr in enumerate((ONE, ACROSS)):
            q = rb[rb["training"] == tr]
            v = q[col].to_numpy(float)
            censored = q[kind].isin(["right", "left"]).to_numpy() if kind else np.zeros(len(q), bool)
            jit = rng.uniform(-0.12, 0.12, len(v))
            ax.scatter(i + jit[~censored], v[~censored], s=7, facecolor=ARM[tr], edgecolor="none", zorder=3)
            ax.scatter(i + jit[censored], v[censored], s=9, marker="^" if tr == ACROSS else "v", facecolor="white", edgecolor=ARM[tr], linewidth=0.5, zorder=3)
            ax.plot([i - 0.22, i + 0.22], [np.exp(np.mean(np.log(v))) if log else np.mean(v)] * 2, color=fs.TEXT, lw=0.9, zorder=4)
            for r_, val, cen in zip(q["replicate"], v, censored):
                sd.add("E", col, training=tr, replicate=int(r_), value=float(val), censored=bool(cen))
        if log:
            ax.set_yscale("log")
            ax.set_ylim(0.5, 30)
            ax.set_yticks([1, 10], ["1", "10"])
            log_axis(ax, "y")
        elif col == "memory_under_jitter":
            ax.set_ylim(-0.05, 1.0)
            ax.set_yticks([0, 0.5])
            ax.axhline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=0)
        else:
            ax.set_ylim(0, 2.6)
            ax.set_yticks([0, 1, 2])
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1], [f"one{NL_}state", f"across{NL_}states"])
        ax.tick_params(axis="x", length=0)
        ax.set_title(lab, fontsize=fs.PT_SMALL, pad=1.5, linespacing=1.0)
        fs.clean(ax)
        top = ax.get_ylim()[1]
        ax.text(0.5, top * (0.93 if not log else 0.80), f"{corr} {fs.P_SYM} = {fs.fmt_p(p)}", fontsize=fs.PT_SMALL, ha="center", va="top")
        sd.add("E", f"{col} test", correction=corr, p_adjusted=p)
    r = summ.loc["ED50 ratio, across states / one state"]
    cv.text(x0 + w - 0.8, y0 + 6.2, "open: beyond the tested range", size=fs.PT_SMALL, ha="right", va="center", color=fs.TEXT2)
    sd.add("E", "ratio", estimate=float(r["estimate"]), ci=[float(r["ci_low"]), float(r["ci_high"])])


# ---------------------------------------------------------------- F: same mechanism

def panel_mechanism(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "F", "Same mechanism")
    rb = cm.table("plan_v2/fig6_robust.csv")
    ax = cv.axes(x0 + 9.0, y0 + 9.0, w - 10.6, h - 17.0)
    for g, col in enumerate(("adaptation_drop", "swap_drop")):
        for i, tr in enumerate((ONE, ACROSS)):
            v = rb[rb["training"] == tr][col].to_numpy(float)
            ax.scatter(np.full(len(v), g + (i - 0.5) * 0.36), -v, s=9, facecolor=ARM[tr], edgecolor="none", zorder=3)
            sd.add("F", col, training=tr, values=v.tolist())
    ax.axhline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-2.4, 0.3)
    ax.set_yticks([-2, -1, 0], [f"{fs.MINUS}2", f"{fs.MINUS}1", "0"])
    ax.set_xticks([0, 1], [f"adaptation{NL_}cleared", f"state{NL_}swapped"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Change in net use", labelpad=0.8)
    fs.clean(ax)
    same = all(rb.groupby("training")[c].nunique().eq(1).all() for c in ("adaptation_drop", "swap_drop")) and \
        rb.groupby("training")[["adaptation_drop", "swap_drop"]].first().nunique().eq(1).all()
    if same:
        ax.text(1.55, -0.12, f"identical in{NL_}all {len(rb)} runs", fontsize=fs.PT_SMALL, ha="right", va="top", linespacing=1.0)
    cv.key(x0 + 9.0, y0 + 5.6, [([dict(kind="mark", s=7, fc=ARM[ONE], ec=ARM[ONE])], "one state"), ([dict(kind="mark", s=7, fc=ARM[ACROSS], ec=ARM[ACROSS])], "across")],
           size=fs.PT_SMALL, gap=1.2)
    sd.add("F", "mechanism", identical_in_all_runs=bool(same), runs=int(len(rb)))


# ---------------------------------------------------------------- G: routes by class (MaleCNS Fig. 2B); H: the types carrying them

# The route map in the form of MaleCNS Fig. 2B (lib/malecns.py), two units wide. Node places (mm from the drawing's top-left):
# the visual input cells on the left strip, the readout cells by class on the right strip, relay classes between them by
# anatomy (brain above the neck band, VNC below), falling from top left to bottom right as in Fig. 2B.
ZONES = {"brain": (-1.0, 30.6), "neck": (30.6, 37.8), "VNC": (37.8, 46.4)}
RELAY_AT = {"optic lobe": (16.0, 0.8), "visual projection": (27.0, 4.8), "central brain": (42.0, 13.2), "descending": (53.0, 32.4),
            "ascending": (20.0, 32.8), "VNC": (58.0, 41.4), "other": (34.0, 26.0)}
RELAY_LABEL = {"optic lobe": ("optic lobe", "above"), "visual projection": ("visual" + NL_ + "projection", "above"),
               "central brain": ("central brain", "above"), "descending": ("descending" + NL_ + "neurons", "above"),
               "ascending": ("ascending" + NL_ + "neurons", "below"), "VNC": ("VNC", "below"), "other": ("other", "above")}
READOUT_AT = {"optic lobe": 0.4, "visual projection": 3.6, "central brain": 9.0, "descending": 32.2, "ascending": 35.4, "motor": 42.6,
              "other": 38.0}
READOUT_LABEL = {"optic lobe": "optic lobe", "visual projection": "visual proj.", "central brain": "central" + NL_ + "brain",
                 "descending": "descending", "ascending": "ascending", "motor": "motor", "other": "other"}
INPUT_X, INPUT_Y, READOUT_X, SCALE = 6.6, 9.0, 68.6, 0.95e-9   # SCALE: mm per walk


def panel_sankey(cv, sd, x0, y0, w, h):
    from lib import malecns as mc
    title(cv, x0, y0, "G", "Routes thin through every class, and none breaks")
    rc = cm.table("plan_v2/fig6_route_classes.csv")
    rc = rc.assign(level=rc["level"].astype(str))
    summ = cm.table("plan_v2/fig6_routes_summary.csv").set_index("share_removed")["walks3_relative"]
    w3 = rc[(rc["stage_a"] == "relay 2") & (rc["stage_b"] == "readout")].groupby("level")["walks"].sum()
    for lev in LEVELS:                                       # gate: 3-step walks relative to intact = the graph analysis
        if not np.isclose(w3[lev] / w3["0"], float(summ.loc[int(lev) / 100]), rtol=1e-9):
            raise RuntimeError(f"route map: 3-step walks at {lev}% disagree with fig6_routes_summary")
    rc["a"] = np.where(rc["stage_a"] == "input", "input", rc["class_a"])
    rc["b"] = np.where(rc["stage_b"] == "readout", "readout:" + rc["class_b"], rc["class_b"])
    piv = rc.pivot_table(index=["a", "b", "transmitter"], columns="level", values="walks", aggfunc="sum").fillna(0.0)
    tot = piv.groupby(level=["a", "b"])["0"].transform("sum")

    def xpos(k):
        return INPUT_X if k == "input" else (READOUT_X if k.startswith("readout:") else RELAY_AT.get(k, (50.0, 0.0))[0])
    back = np.array([xpos(b) <= xpos(a) for a, b, _ in piv.index])
    small = tot.to_numpy() < np.where(back, MIN_BACK, MIN_FLOW) * float(piv["0"].sum()) / 3
    for idx, r in piv[small].iterrows():
        sd.add("G", "flow not drawn (forward < 0.2%, back or within a class < 1% of the walks)", source=idx[0], target=idx[1],
               transmitter=idx[2], **{f"walks_{lev}": float(r.get(lev, 0.0)) for lev in LEVELS})
    piv = piv[~small]
    D = y0 + 10.8                                            # the drawing's top
    zones = {k: (D + a, D + b) for k, (a, b) in ZONES.items()}
    top, bottom = zones["brain"][0], zones["VNC"][1]
    xl, xr = x0 + INPUT_X - 1.5, x0 + READOUT_X + 1.9        # the strips' outer edges
    cv.box(xl, zones["neck"][0], xr - xl, zones["neck"][1] - zones["neck"][0], mc.STYLE_2B["neck"])   # the neck connective
    mc.strip(cv, xl, top, 3.4, bottom - top)
    mc.strip(cv, x0 + READOUT_X - 1.5, top, 3.4, bottom - top)
    mc.zone_labels(cv, x0 + 1.4, zones)
    fm = mc.FlowMap2B(cv, SCALE)
    keys = set(piv.index.get_level_values(0)) | set(piv.index.get_level_values(1))
    fm.node("input", x0 + INPUT_X, D + INPUT_Y)
    for c, (rx, ry) in RELAY_AT.items():
        if c in keys:
            lab, at = RELAY_LABEL[c]
            fm.node(c, x0 + rx, D + ry, lab, at)
    for c, ry in READOUT_AT.items():
        if "readout:" + c in keys:
            fm.node("readout:" + c, x0 + READOUT_X, D + ry, READOUT_LABEL[c], "right", label_dx=1.2)
    for (a, b), g in piv.groupby(level=["a", "b"]):
        strands = {idx[2]: (float(r.get("0", 0.0)), float(r.get("90", 0.0))) for idx, r in g.iterrows()}
        fm.flow(a, b, strands)
        for t_, (v0, v9) in strands.items():
            sd.add("G", "flow", source=a, target=b, transmitter=t_, walks_intact=v0, walks_left_at_90=v9)
    fm.draw()
    fm.labels()
    mc.axis_arrow(cv, xl, xr, y0 + 6.2, "visual input", "readout")
    kx, ky = x0 + 9.6, D + 41.0                               # key, in the free corner of the VNC zone
    xx = kx
    for t_, lab in (("acetylcholine", "acetylcholine"), ("gaba", "GABA"), ("glutamate", "glutamate")):
        cv.box(xx, ky - 0.9, 1.8, 1.8, mc.STYLE_2B[t_])
        cv.text(xx + 2.4, ky, lab, size=fs.PT_SMALL, va="center")
        xx += 2.4 + fs.text_width(lab, fs.PT_SMALL) + 2.2
    cv.box(kx, ky + 2.0, 3.4, 1.8, mc.STYLE_2B["acetylcholine"])
    cv.box(kx + 0.85, ky + 2.0, 1.7, 1.8, mc.STYLE_2B["core"]["acetylcholine"])
    cv.text(kx + 4.0, ky + 2.9, "core: left after 90% removal", size=fs.PT_SMALL, va="center")
    sd.add("G", "three-step walks relative", **{f"removed_{lev}": float(w3[lev] / w3["0"]) for lev in LEVELS})


# H's bars split the route-carrying types by class (the classes of the route map; the small ones together)
TYPE_CLASS = (("central brain", ("central brain intrinsic",)), ("VNC", ("VNC intrinsic",)),
              ("other classes", ("descending & ascending", "visual projection & centrifugal", "optic lobe intrinsic", "sensory",
                                 "motor & efferent", "other")))


def panel_types(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "H", "Thousands of types carry them")
    t = cm.table("plan_v2/fig6_route_types.csv")
    t["group"] = t["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other")
    known = {g for _, gs in TYPE_CLASS for g in gs}
    if not set(t["group"]) <= known:
        raise RuntimeError(f"route types: classes without a bar segment: {sorted(set(t['group']) - known)}")
    cv.key(x0 + 9.4, y0 + 6.0, [([dict(kind="hline", colour=LEVEL_SHADE[lev], lw=0.9)], "intact" if lev == "0" else f"{lev}% removed")
                                 for lev in LEVELS], size=fs.PT_SMALL, gap=2.0)
    ax = cv.axes(x0 + 9.4, y0 + 9.4, w - 10.8, 18.6)
    for lev in LEVELS:
        v = np.sort(t[f"traffic_{lev}"].to_numpy(float))[::-1]
        v = v[v > 0]
        c = np.cumsum(v) / v.sum()
        ax.plot(np.arange(1, len(c) + 1), 100 * c, color=LEVEL_SHADE[lev], lw=0.9, zorder=2 + LEVELS.index(lev))
        sd.add("H", "cumulative share", removed=lev, types=int(len(v)), types_for_half=int(np.searchsorted(c, 0.5)) + 1)
    ax.set_xscale("log")
    ax.set_xlim(1, 2e4)
    ax.set_ylim(0, 104)
    ax.set_xticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1k", "10k"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Cell types, by route traffic", labelpad=0.8)
    ax.set_ylabel("Routes (cum. %)", labelpad=0.8)
    fs.clean(ax)
    cv.text(x0 + 1.0, y0 + 37.6, "Cell types carrying routes", size=fs.PT_SMALL, va="center", color=fs.TEXT2)
    bx = cv.axes(x0 + 9.4, y0 + 42.4, w - 10.8, h - 42.4)
    split = {lev: [int(((t[f"traffic_{lev}"] > 0) & t["group"].isin(gs)).sum()) for _, gs in TYPE_CLASS] for lev in LEVELS}
    total = {lev: int((t[f"traffic_{lev}"] > 0).sum()) for lev in LEVELS}
    span = max(total.values()) * 1.13
    bx.set_xlim(0, span)
    bx.set_ylim(-0.55, len(LEVELS) - 0.45)
    hb = 0.72
    for j, lev in enumerate(LEVELS):
        y = len(LEVELS) - 1 - j
        ci.stacked_hbar(cv, bx, y, split[lev], [LEVEL_SHADE[lev]] * len(TYPE_CLASS), height=hb, min_frac=0.04)
        edges = np.cumsum(split[lev])[:-1]
        for e in edges:                                       # white hairlines between the class segments
            bx.plot([e, e], [y - hb / 2, y + hb / 2], color="#FFFFFF", lw=0.6, zorder=5)
        bx.text(total[lev] + 0.012 * span, y, f"{total[lev]:,}", fontsize=fs.PT_SMALL, ha="left", va="center")
        if j == 0:                                            # the classes, over the intact bar
            mids = np.cumsum(split[lev]) - np.array(split[lev]) / 2
            for (name, _), m in zip(TYPE_CLASS, mids):
                bx.text(m, y + hb / 2 + 0.12, name, fontsize=fs.PT_SMALL, ha="center", va="bottom", color=fs.TEXT2)
        sd.add("H", "types carrying routes", removed=lev, types=total[lev], **{name: n for (name, _), n in zip(TYPE_CLASS, split[lev])})
    bx.set_yticks(range(len(LEVELS))[::-1], ["intact" if lev == "0" else f"{lev}%" for lev in LEVELS])
    bx.set_xticks([])
    fs.clean(bx, bottom=False)
    bx.tick_params(axis="y", length=0)


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("fig6")
    for rel in ("plan_v2/fig6_routes.csv", "plan_v2/fig6_routes_summary.csv", "plan_v2/fig6_wiring_graph.csv", "plan_v2/fig6_wiring_flow.csv",
                "plan_v2/fig6_kicks.csv", "plan_v2/fig6_kick_partners.csv", "plan_v2/fig6_rate.csv", "plan_v2/fig6_robust.csv",
                "plan_v2/fig6_robust_summary.csv", "plan_v2/tests.csv", "plan_v2/fig6_route_classes.csv", "plan_v2/fig6_route_types.csv",
                "figure_data/fd_cells.csv"):
        sd.source(cm.RESULTS / rel)
    for f in sorted(TILES.glob("*.png")) + sorted(TILES.glob("*.json")):
        sd.source(f)
    panel_band(cv, sd)
    panel_routes(cv, sd, XS[0], ROW2, fs.unit(2), R2_H)
    panel_kicks(cv, sd, XS[2], ROW2, fs.unit(2), R2_H)
    panel_stability(cv, sd, XS[0], ROW3, U, R3_H)
    panel_robust(cv, sd, XS[1], ROW3, fs.unit(2), R3_H)
    panel_mechanism(cv, sd, XS[3], ROW3, U, R3_H)
    panel_sankey(cv, sd, XS[0], ROW4, fs.unit(2), R4_H)
    panel_types(cv, sd, XS[2], ROW4, fs.unit(2), R4_H)
    cm.finish(cv, "fig6", sd)


if __name__ == "__main__":
    main()
