"""Fig. 4 - Persistent state carries cue history (180 x 140 mm).

a  unfolded CNS (M0 readout-only, last window at 160-ms delay and 4 prefix windows): signed cue difference of the
   membrane potential (cells > 1 mV), maximum projection of |difference|, spiking cells (dark tiles)
b  task and cue-pair logic (a memoryless policy can solve only one cue of a pair)
c  carried state vs reset state vs untrained readout: cue pairs at the 9 trained timings
d  graded coverage: share of cells above each threshold, four state variables, six distinct state sets
e  whole-state interventions: every history, margin change and lost solutions; Holm-adjusted tests at the right

Run after render/fig4_tiles.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\fig4.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import boards as bd, figstyle as fs, palette as pal, qc  # noqa: E402

TILES = cm.RENDERS / "fig4"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
TW, TH = META["tile_mm"]
WHITE = fs.ON_DARK
COL_X = 3 * TW + 2 * fs.GUTTER + 2.2          # right column next to the band
ROW2 = TH + 4.2


def tile_text(cv, x, y, s, size=fs.PT_SMALL, **kw):
    t = cv.text(x, y, s, size=size, color=WHITE, **kw)
    t.set_path_effects(fs.dark_halo(0.8))
    return t


def arrow_mm(ax, x0, y0, x1, y1, colour=fs.TEXT, lw=0.5, head=2.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=f"-|>,head_length={head * 0.4},head_width={head * 0.25}",
                                 mutation_scale=4, color=colour, linewidth=lw, shrinkA=0, shrinkB=0, zorder=5))


def split_square(ax, x, y, s, solved0, solved1, edge="#9E9E9E"):
    """A cue pair: left half lit in the cue-0 colour if cue 0 was solved, right half in the cue-1 colour if cue 1 was."""
    ax.add_patch(Rectangle((x, y), s / 2, s, facecolor=pal.CUE[0] if solved0 else "white", edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((x + s / 2, y), s / 2, s, facecolor=pal.CUE[1] if solved1 else "white", edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((x, y), s, s, facecolor="none", edgecolor=edge, linewidth=0.35, zorder=3))


def gradient(cv, x, y, w, h, colours):
    """A vertical colour strip (top to bottom = colours[0] to colours[-1]) drawn as an image."""
    ax = cv.axes(x, y, w, h)
    ax.imshow(np.asarray(colours)[:, None, :3] * np.ones((1, 2, 1)), aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax.set_axis_off()


# ---------------------------------------------------------------- a: dark band

def panel_a(cv, sd):
    imgs = {k: np.asarray(Image.open(TILES / f"{k}.png").convert("RGB")) for k in ("signed", "magnitude", "spiking")}
    meta = {k: json.loads((TILES / f"{k}.json").read_text(encoding="utf-8")) for k in imgs}
    xs = [k * (TW + fs.GUTTER) for k in range(3)]
    for x, k in zip(xs, imgs):
        cv.tile(x, 0, imgs[k], width=TW, height=TH, group="band4", label=k)
    cv.letter(1.2, 0.9, "a")
    cv.fig.texts[-1].set_color(WHITE)
    titles = ["Cue 1 minus cue 0, cells over 1 mV", "|Cue difference|, maximum projection", "Spiking cells"]
    for x, ttl in zip(xs, titles):
        tile_text(cv, x + (4.6 if x == 0 else 1.4), 1.3, ttl, va="top")
    cov = cm.table("figure_data/fd_coverage_graded.csv")
    m0v = cov[(cov.state_key == "m0") & (cov.variable == "v") & (cov.population == "all")].set_index("threshold")["fraction_above"]

    def frac_above(t):
        return float(m0v.loc[m0v.index[np.argmin(np.abs(m0v.index.to_numpy() - t))]])

    # (i) signed difference: two half bars (positive above, negative below), counts per sign
    s = meta["signed"]
    bx, by, half, gap = xs[0] + 1.6, 31.0, 10.0, 3.0
    u = np.linspace(1, 0, 128)
    gradient(cv, bx, by, 1.5, half, pal.SIGNED_ON_DARK(0.5 + 0.26 + 0.24 * u))
    gradient(cv, bx, by + half + gap, 1.5, half, pal.SIGNED_ON_DARK(0.5 - 0.26 - 0.24 * u[::-1]))
    tile_text(cv, bx, by - 0.8, "mV", va="bottom")
    tile_text(cv, bx + 2.0, by, "+20", va="center")
    tile_text(cv, bx + 2.0, by + half, "+1", va="center")
    tile_text(cv, bx + 2.0, by + half + gap, fs.MINUS + "1", va="center")
    tile_text(cv, bx + 2.0, by + 2 * half + gap, fs.MINUS + "20", va="center")
    tile_text(cv, bx, by + 2 * half + gap + 2.4, f"{s['positive']:,} cells +", va="center")
    tile_text(cv, bx, by + 2 * half + gap + 4.6, f"{s['negative']:,} cells {fs.MINUS}", va="center")
    sd.add("a", "signed", cells_over_1mv=s["cells_above"], positive=s["positive"], negative=s["negative"], located=140024,
           colour_range_mv="1 to 20 (log), dark-centred cue map", model="M0 readout-only, 160-ms delay, 4 prefix windows")
    # (ii) magnitude: log colour bar; ticks carry the share of all cells above each threshold
    bx, by, bh = xs[1] + 1.6, 30.0, 27.0
    gradient(cv, bx, by, 1.5, bh, pal.MAGNITUDE_ON_DARK(np.linspace(1, 0, 256)))
    tile_text(cv, bx, by - 0.8, "mV, % of cells above", va="bottom")
    for v, lab in ((10.0, "10"), (1.0, "1"), (0.1, "0.1"), (0.01, "0.01")):
        yv = by + (1 - (np.log10(v) + 2) / 3) * bh
        cv.line([bx + 1.5, bx + 2.2], [yv, yv], WHITE, lw=0.4)
        text = lab if v == 10 else f"{lab}  {100 * frac_above(v):.0f}%"
        tile_text(cv, bx + 2.6, yv, text, va="center")
        if v < 10:
            sd.add("a", "ladder", threshold_mv=v, fraction_above=frac_above(v), cells=META["cells_all"])
    # (iii) spiking
    sp = meta["spiking"]
    kx, ky = xs[2] + 1.6, 40.0
    cv.mark(kx + 0.8, ky, s=7.0, fc="#FFE7A3", ec="#FFE7A3", lw=0.3)
    tile_text(cv, kx + 2.0, ky, f"{sp['spiking']:,} spike", va="center")
    tile_text(cv, kx + 2.0, ky + 2.4, f"({100 * META['spiking_all'] / META['cells_all']:.1f}% of cells)", va="center")
    px_per_mm = imgs["spiking"].shape[1] / TW
    length = 100 / sp["scale_um_per_px"] / px_per_mm
    cv.line([xs[2] + TW - 1.6 - length, xs[2] + TW - 1.6], [TH - 2.0, TH - 2.0], WHITE, lw=1.0, solid_capstyle="butt")
    ox, oy = xs[2] + TW - 12.0, TH - 6.4
    cv.line([ox, ox + 2.8], [oy, oy], WHITE, lw=0.5)
    tile_text(cv, ox + 3.2, oy, "L", va="center")
    cv.line([ox, ox], [oy, oy + 2.8], WHITE, lw=0.5)
    tile_text(cv, ox, oy + 3.2, "V", ha="center", va="top")
    sd.add("a", "spiking", spiking_located=sp["spiking"], spiking_all=META["spiking_all"], cells_all=META["cells_all"])
    sd.add("a", "view", scale_bar_um=100, um_per_px=sp["scale_um_per_px"])


# ---------------------------------------------------------------- b: task logic

def panel_b(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "b")
    ax = bd.mm_axes(cv, x0, y0, w, h)
    boards_t = cm.table("figure_data/fd_boards.csv")
    cell = 0.66
    cues = {c: bd.Board.from_table(boards_t, f"M0:c0-cue{c}") for c in (0, 1)}
    bw, bh = cues[0].size_mm(cell)
    ys = [3.2, 3.2 + bh + 3.2]
    for y, c in zip(ys, (0, 1)):
        bd.draw(ax, cues[c], 0.4, y, cell, goal_colour=pal.CUE[c])
        ax.text(0.4 + bw / 2, y + bh + 0.3, f"cue {c}", ha="center", va="top", fontsize=fs.PT_SMALL)
    dx = 0.4 + bw + 4.4
    dy = (ys[0] + ys[1]) / 2
    for y in ys:
        arrow_mm(ax, 0.4 + bw + 0.4, y + bh / 2, dx - 0.4, dy + bh / 2)
    bd.draw(ax, cues[0], dx, dy, cell, show_goals=False)
    bd.arrow(ax, cues[0], 34, 2, dx, dy, cell, colour=pal.CUE[0], lw=0.7, length=2.4, head=1.0)
    bd.arrow(ax, cues[0], 34, 3, dx, dy, cell, colour=pal.CUE[1], lw=0.7, length=2.4, head=1.0)
    ax.text(dx + bw / 2, dy + bh + 0.3, "same board,\ngoal hidden", ha="center", va="top", fontsize=fs.PT_SMALL, linespacing=1.0)
    lx, s = dx + bw + 3.6, 3.0
    split_square(ax, lx, 3.2, s, True, False)
    ax.text(lx + s + 1.0, 3.2 + s / 2, "memoryless:\none cue per pair", va="center", fontsize=fs.PT_SMALL, linespacing=1.0)
    split_square(ax, lx, 11.0, s, True, True)
    ax.text(lx + s + 1.0, 11.0 + s / 2, "carried state:\nboth cues", va="center", fontsize=fs.PT_SMALL, linespacing=1.0)
    ax.text(lx, 18.4, "lit half = that cue\nsolved", va="top", fontsize=fs.PT_SMALL, linespacing=1.0)
    sd.add("b", "logic", decision_board_identical=True, memoryless_max_per_pair=1)


# ---------------------------------------------------------------- c: carried vs reset vs untrained

def panel_c(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    ax = bd.mm_axes(cv, x0, y0, w, h)
    b = cm.table("main/fig4/fig4b_carry_vs_reset.csv")
    b = b[b["split"] == "train"]
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    models = [("remote-dynamic-M0-head_only", "carried\nstate"), ("remote-dynamic-M0-reset", "state reset\neach window"),
              ("remote-dynamic-M0-carry_untrained", "untrained\nreadout")]
    delays = sorted(b["delay_ms"].unique())
    prefixes = sorted(b["prefix_windows"].unique())
    s, g = 3.0, 0.5
    colw = w / 3
    top = 6.6
    for k, (m, lab) in enumerate(models):
        x = k * colw + (colw - 3 * s - 2 * g) / 2
        ax.text(k * colw + colw / 2, top - 1.0, lab, ha="center", va="bottom", fontsize=fs.PT_SMALL, linespacing=1.0)
        d = b[b["model"] == m]
        both = 0
        for (dly, pre), pair in d.groupby(["delay_ms", "prefix_windows"]):
            sv = {int(r["cue"]): str(r["solved"]).lower() == "true" for _, r in pair.iterrows()}
            i, j = prefixes.index(pre), delays.index(dly)
            split_square(ax, x + i * (s + g), top + j * (s + g), s, sv.get(0, False), sv.get(1, False))
            both += int(sv.get(0, False) and sv.get(1, False))
            sd.add("c", "cue pair", model=m, delay_ms=int(dly), prefix_windows=int(pre), cue0_solved=sv.get(0), cue1_solved=sv.get(1))
        n = d.groupby(["delay_ms", "prefix_windows"]).ngroups
        ax.text(k * colw + colw / 2, top + 3 * (s + g) + 0.4, f"both: {both} of {n}", ha="center", va="top", fontsize=fs.PT_SMALL)
    yb = top + 3 * (s + g) + 4.4
    for k, tid in ((1, "fig4b.mcnemar_pairs.head_only_vs_reset.train"), (2, "fig4b.mcnemar_pairs.head_only_vs_carry_untrained.train")):
        x1, x2 = colw / 2, k * colw + colw / 2
        yy = yb + (k - 1) * 2.9
        ax.plot([x1, x1, x2, x2], [yy - 0.6, yy, yy, yy - 0.6], color=fs.TEXT, lw=0.4)
        p = float(tests.loc[tid, "p_adjusted"])
        ax.text((x1 + x2) / 2, yy - 0.1, fs.p_text(p), ha="center", va="bottom", fontsize=fs.PT_SMALL)
        sd.add("c", "test", test_id=tid, p_adjusted=p)


# ---------------------------------------------------------------- d: graded coverage

def panel_d(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "d")
    cov_all = cm.table("figure_data/fd_coverage_graded.csv")
    cov = cov_all[cov_all["population"] == "all"]
    names = {"v": "Membrane potential (mV)", "g": "Synaptic current (mV)", "adapt": "Adaptation (mV)"}
    order = ["m0", "m0_joint", "m0vis_visible", "s1", "s2_o0", "s2_o180"]
    labels = {"m0": "M0", "m0_joint": "M0 joint", "m0vis_visible": "M0, goal visible", "s1": "S1", "s2_o0": "S2, 0°", "s2_o180": "S2, 180°"}
    ylim = {"v": 100, "g": 100, "adapt": 6}
    aw, ah = (w - 16.0) / 2, (h - 25.0) / 2
    pos = {"v": (0, 0), "g": (0, 1), "adapt": (1, 0), "pop": (1, 1)}
    axes = {}
    for key_, (r, c) in pos.items():
        axes[key_] = cv.axes(x0 + 9.0 + c * (aw + 6.5), y0 + 4.4 + r * (ah + 9.0), aw, ah)
    for var in ("v", "g", "adapt"):
        ax = axes[var]
        d = cov[cov["variable"] == var]
        for key in order[::-1]:
            gk = d[d["state_key"] == key].sort_values("threshold")
            ls = (0, (2.2, 1.2)) if key == "m0_joint" else "-"
            ax.plot(gk["threshold"], 100 * gk["fraction_above"], color=pal.STATE_SET[key], lw=1.0 if key == "m0" else 0.6, ls=ls,
                    zorder=3 if key == "m0" else 2)
            for _, rr in gk.iterrows():
                sd.add("d", "coverage", state_key=key, variable=var, population="all", threshold=float(rr["threshold"]),
                       fraction_above=float(rr["fraction_above"]))
        ax.set_ylim(0, ylim[var])
        ax.set_yticks([0, ylim[var] / 2, ylim[var]], ["0", f"{ylim[var] / 2:g}", f"{ylim[var]:g}"])
        ax.set_xlabel(names[var], labelpad=1.0)
        if var == "v":
            m0 = d[d["state_key"] == "m0"].set_index("threshold")["fraction_above"]
            for t in (0.01, 0.1, 1.0):
                f = float(m0.loc[m0.index[np.argmin(np.abs(m0.index.to_numpy() - t))]])
                ax.plot([t], [100 * f], marker="o", ms=2.6, mfc="white", mec=fs.TEXT, mew=0.5, zorder=4)
                ax.text(t * 1.6, 100 * f + 3, f"{100 * f:.0f}%", fontsize=fs.PT_SMALL, va="bottom", ha="left")
    # membrane potential by population (M0)
    ax = axes["pop"]
    pops = [("readout", "readout", "#7B1FA2"), ("input", "input", pal.on_white(pal.ATLAS_ON_DARK["visual projection & centrifugal"])),
            ("other", "other", "#8C8C8C")]
    m0 = cov_all[(cov_all["state_key"] == "m0") & (cov_all["variable"] == "v")]
    for pkey, plab, col in pops:
        gk = m0[m0["population"] == pkey].sort_values("threshold")
        ax.plot(gk["threshold"], 100 * gk["fraction_above"], color=col, lw=0.9)
        f1 = float(gk.set_index("threshold")["fraction_above"].iloc[np.argmin(np.abs(gk["threshold"].to_numpy() - 1.0))])
        for _, rr in gk.iterrows():
            sd.add("d", "coverage by population", state_key="m0", variable="v", population=pkey, threshold=float(rr["threshold"]),
                   fraction_above=float(rr["fraction_above"]))
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Membrane potential (mV)", labelpad=1.0)
    ax.set_title("M0, by cell population", fontsize=fs.PT_SMALL, loc="left", pad=1.5)
    yk = 33.0
    for pkey, plab, col in pops:
        n = int(m0[(m0["population"] == pkey)]["cells"].iloc[0])
        ax.plot([2e-4, 6e-4], [yk, yk], color=col, lw=0.9)
        ax.text(8e-4, yk, plab, va="center", fontsize=fs.PT_SMALL)
        sd.add("d", "population size", population=pkey, cells=n)
        yk -= 11.0
    for key_, ax in axes.items():
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 30)
        fs.minor(ax, y=False)
        fs.clean(ax)
        if pos[key_][1] == 0:
            ax.set_ylabel("Cells above (%)")
    res = cov_all[(cov_all["variable"] == "res") & (cov_all["population"] == "all") & (cov_all["threshold"] == 1e-6)]
    n_res = int(res["cells_above"].max())
    sd.add("d", "synaptic resources", max_cells_above_threshold=n_res, threshold=1e-6, state_sets=int(res["state_key"].nunique()))
    items = [([{"kind": "hline", "colour": pal.STATE_SET[k], "lw": 1.0 if k == "m0" else 0.6,
                "ls": (0, (2.2, 1.2)) if k == "m0_joint" else "-", "w": 3.0}], labels[k]) for k in order]
    cv.key(x0 + 9.0, y0 + h - 4.0, items[:3], gap=2.0)
    cv.key(x0 + 9.0, y0 + h - 1.4, items[3:], gap=2.0)
    axes["adapt"].text(30, 5.9, f"synaptic resources:" + chr(10) + f"{n_res} cells differ", ha="right", va="top", fontsize=fs.PT_SMALL,
                       linespacing=1.0)


# ---------------------------------------------------------------- e: intervention barcode

def panel_e(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "e")
    c = cm.table("main/fig4/fig4c_state_interventions.csv")
    c = c[c["window"].isin(["hold", "read", "write"])].copy()
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    units = [("M0", "M0"), ("M0vis.hidden", "M0, shared\n(hidden)"), ("M0vis.visible", "M0, shared\n(visible)"), ("S1", "S1"), ("S2", "S2")]
    rows = [("baseline", "baseline"), ("sham", "sham"), ("restore", "restore"), ("same_cue_swap", "same-cue swap"),
            ("clear_all_state", "clear state"), ("swap_all_state", "swap state")]
    windows = ["hold", "read", "write"]
    left, stats_w = 25.0, 22.0
    gw = w - left - stats_w
    blk = (gw - 2 * 1.4) / 3
    top = y0 + 8.0
    unit_gap = 1.6
    pitch = (y0 + h - top - 4 * unit_gap - 6.6) / (len(units) * len(rows))
    norm = Normalize(-6, 6)
    cmap = colormaps[pal.MARGIN_CMAP]
    ax = bd.mm_axes(cv, x0, y0, w, h)
    for j, wdw in enumerate(windows):
        ax.text(left + j * (blk + 1.4) + blk / 2, top - y0 - 1.2, f"{wdw} window", ha="center", va="bottom", fontsize=fs.PT_SMALL)
    ax.text(w - stats_w + 1.0, top - y0 - 1.2, "lost/gained", ha="left", va="bottom", fontsize=fs.PT_SMALL)
    ax.text(w - 0.2, top - y0 - 1.2, "Holm " + fs.P_SYM, ha="right", va="bottom", fontsize=fs.PT_SMALL)
    y = top - y0
    for u, ulab in units:
        d = c[c["unit"] == u].copy()
        d["hist"] = (d["context"].fillna(0).astype(int).astype(str) + "|" + d["case"].astype(str) + "|" +
                     d["delay_ms"].astype(int).map("{:04d}".format) + "|" + d["prefix_windows"].astype(int).astype(str))
        hists = sorted(set(d["hist"]))
        cwid = blk / len(hists)
        ax.text(0.2, y + len(rows) * pitch / 2, ulab, ha="left", va="center", fontsize=fs.PT_SMALL, linespacing=1.0)
        for ri, (iv, rlab) in enumerate(rows):
            yy = y + ri * pitch
            ax.text(left - 1.0, yy + pitch / 2, rlab, ha="right", va="center", fontsize=fs.PT_SMALL)
            for j, wdw in enumerate(windows):
                dd = d[(d["window"] == wdw) & (d["intervention"] == ("sham" if iv == "baseline" else iv))].set_index("hist")
                for k, hk in enumerate(hists):
                    r = dd.loc[hk]
                    xx = left + j * (blk + 1.4) + k * cwid
                    was = str(r["baseline_solved"]).lower() == "true"
                    if iv == "baseline":
                        fc, ec = ("#5E5E5E" if was else "white"), "#D0D0D0"
                    else:
                        fc, ec = cmap(norm(float(r["margin_change"]))), "none"
                    ax.add_patch(Rectangle((xx + 0.08, yy + 0.08), cwid - 0.16, pitch - 0.16, facecolor=fc, edgecolor=ec, linewidth=0.25))
                    if iv != "baseline":
                        now = str(r["solved"]).lower() == "true"
                        rad = min(cwid, pitch) * 0.2
                        if was and not now:
                            ax.add_patch(Circle((xx + cwid / 2, yy + pitch / 2), rad, facecolor=fs.TEXT, edgecolor="none"))
                        elif now and not was:
                            ax.add_patch(Circle((xx + cwid / 2, yy + pitch / 2), rad, facecolor="white", edgecolor=fs.TEXT, linewidth=0.3))
                        sd.add("e", "history", unit=u, intervention=iv, window=wdw, history=str(hk), margin_change=float(r["margin_change"]),
                               baseline_solved=was, solved=now)
            if iv in ("clear_all_state", "swap_all_state"):
                lost = [int(tests.loc[f"fig4c.mcnemar.{u}.{wdw}.{iv}", "b_only"]) for wdw in windows]
                gained = [int(tests.loc[f"fig4c.mcnemar.{u}.{wdw}.{iv}", "a_only"]) for wdw in windows]
                pm = [float(tests.loc[f"fig4c.mcnemar.{u}.{wdw}.{iv}", "p_adjusted"]) for wdw in windows]
                pw = [float(tests.loc[f"fig4c.wilcoxon_margin.{u}.{wdw}.{iv}", "p_adjusted"]) for wdw in windows]

                def rng(v):
                    return f"{min(v)}" if min(v) == max(v) else f"{min(v)}–{max(v)}"
                ax.text(w - stats_w + 1.0, yy + pitch / 2, f"{rng(lost)}/{rng(gained)}", ha="left", va="center", fontsize=fs.PT_SMALL)
                ax.text(w - 0.2, yy + pitch / 2, f"{fs.fmt_p(min(pm))} | {fs.fmt_p(min(pw))}", ha="right", va="center", fontsize=fs.PT_SMALL)
                for wdw, l_, g_, a_, b_ in zip(windows, lost, gained, pm, pw):
                    sd.add("e", "tests", unit=u, intervention=iv, window=wdw, lost=l_, gained=g_, mcnemar_p_holm=a_, margin_wilcoxon_p_holm=b_)
        y += len(rows) * pitch + unit_gap
    # key: colour bar for the margin change, lost/gained glyphs, test columns
    ky = y0 + h - 4.4
    kax = cv.axes(x0 + left, ky - 0.8, 16.0, 1.6)
    kax.imshow(cmap(norm(np.linspace(-6, 6, 128)))[None, :, :3], aspect="auto", extent=(-6, 6, 0, 1))
    kax.set_yticks([])
    kax.set_xticks([-6, 0, 6], [fs.MINUS + "6", "0", "+6"])
    kax.tick_params(axis="x", length=1.2, pad=0.5, labelsize=fs.PT_SMALL)
    for side in ("top", "right", "left"):
        kax.spines[side].set_visible(False)
    cv.text(x0 + left + 17.2, ky, "change in first-decision margin", size=fs.PT_SMALL, va="center")
    xk = x0 + left + 50.0
    cv.mark(xk, ky, s=4.0, fc=fs.TEXT, ec=fs.TEXT, lw=0.3)
    cv.text(xk + 1.2, ky, "solution lost", size=fs.PT_SMALL, va="center")
    cv.mark(xk + 14.0, ky, s=4.0, fc="white", ec=fs.TEXT, lw=0.3)
    cv.text(xk + 15.2, ky, "gained", size=fs.PT_SMALL, va="center")
    cv.text(x0 + w - 0.2, ky, "solved | margin", size=fs.PT_SMALL, va="center", ha="right")


def main():
    cv = fs.Canvas(fs.FULL, 140.0)
    sd = qc.SourceData("fig4")
    for rel in ("figure_data/fd_soma_state.csv", "figure_data/fd_coverage_graded.csv", "figure_data/fd_boards.csv",
                "main/fig4/fig4b_carry_vs_reset.csv", "main/fig4/fig4c_state_interventions.csv", "stats/statistical_tests.csv"):
        sd.source(cm.RESULTS / rel)
    for f in ("signed.png", "magnitude.png", "spiking.png", "signed.json", "magnitude.json", "spiking.json"):
        sd.source(TILES / f)
    panel_a(cv, sd)
    panel_b(cv, sd, COL_X, 0.0, fs.FULL - COL_X, 27.6)
    panel_c(cv, sd, COL_X, 29.6, fs.FULL - COL_X, TH - 29.6)
    panel_d(cv, sd, 0.0, ROW2, fs.span(2), 140.0 - ROW2)
    panel_e(cv, sd, fs.col_x(2), ROW2, fs.FULL - fs.col_x(2), 140.0 - ROW2)
    cm.finish(cv, "fig4", sd)


if __name__ == "__main__":
    main()
