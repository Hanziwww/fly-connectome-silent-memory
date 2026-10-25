"""Fig. 5 - Cue content in the carried state steers the choice (180 x 140 mm), dense layout.

a  the two cue states on the unfolded CNS: cells more depolarised after cue 0 / after cue 1 (dark tiles)
b  every eligible history in six models: the first push after a whole-state swap from an opposite-cue history and
   after a same-size scrambled change, as direction glyphs coloured by the cue whose goal they serve; counts, Holm P
c  first-decision margin for every history and dose of state interpolation towards the other cue (and scrambled
   controls), seven blocks; median trends with Page tests (Holm); inset: the interpolation geometry
d  hidden vs visible goal (shared readout): solutions lost under clearing or swapping each state component, per window
e  which cells carry the cue: share of each superclass group over 1 mV, by sign (M0)

Run after render/fig5_tiles.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\fig5.py
"""
import json
import re
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

TILES = cm.RENDERS / "fig5"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
TW, TH = META["tile_mm"]
WHITE = fs.ON_DARK
H = 140.0
ROW2 = TH + 4.6
ROW3 = 99.0
SCRAMBLE = "#9E9E9E"
DIRS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}          # action -> screen direction (up, down, left, right)
TIMINGS = [(40, 0), (40, 4), (160, 0), (160, 4), (640, 0), (640, 4)]
MARGIN = Normalize(-8, 8)


def tile_text(cv, x, y, s, size=fs.PT_SMALL, **kw):
    t = cv.text(x, y, s, size=size, color=WHITE, **kw)
    t.set_path_effects(fs.dark_halo(0.8))
    return t


def arrow_mm(ax, x0, y0, x1, y1, colour=fs.TEXT, lw=0.5, head=2.0, style="-|>", alpha=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=f"{style},head_length={head * 0.4},head_width={head * 0.25}",
                                 mutation_scale=4, color=colour, linewidth=lw, shrinkA=0, shrinkB=0, zorder=5, alpha=alpha))


def cue_of(case):
    return int(re.search(r"cue(\d)", case).group(1))


# ---------------------------------------------------------------- a: the two cue states

def panel_a(cv, sd):
    for k in (0, 1):
        img = np.asarray(Image.open(TILES / f"cue{k}.png").convert("RGB"))
        x = k * (TW + fs.GUTTER)
        cv.tile(x, 0, img, width=TW, height=TH, group="band5", label=f"cue{k}")
        tile_text(cv, x + (4.0 if k == 0 else 1.2), 1.1, f"After cue {k}", va="top")
        n = META[f"cue{k}"]
        tile_text(cv, x + 1.2, 27.0, f"{n:,}", va="center")
        tile_text(cv, x + 1.2, 29.2, "cells", va="center")
        tile_text(cv, x + 1.2, 31.4, "> 1 mV", va="center")
        sd.add("a", "cells", cue=k, cells=n, threshold_mv=1.0, state="M0 readout-only, 160-ms delay, 4 prefix windows")
    cv.letter(1.0, 0.8, "a")
    cv.fig.texts[-1].set_color(WHITE)
    tile_text(cv, TW - 1.0, 36.6, "swap", ha="right", va="bottom")
    ax = cv.axes(TW - 6.0, 37.2, 12.6, 2.6)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(2.6, 0)
    ax.set_axis_off()
    arrow_mm(ax, 0.4, 1.3, 12.2, 1.3, colour=WHITE, lw=0.6, head=1.6, style="<|-|>")
    meta = json.loads((TILES / "cue0.json").read_text(encoding="utf-8"))
    px_per_mm = Image.open(TILES / "cue0.png").size[0] / TW
    length = 100 / meta["scale_um_per_px"] / px_per_mm
    x1 = 2 * TW + fs.GUTTER - 1.2
    cv.line([x1 - length, x1], [TH - 1.6, TH - 1.6], WHITE, lw=1.0, solid_capstyle="butt")


# ---------------------------------------------------------------- b: every history, swap vs scramble

def panel_b(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "b")
    e = cm.table("main/fig4/fig4e_directional_transfer.csv")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    blocks = [("M0", "M0", [0]), ("M0joint", "M0 joint", [0]), ("M0vis.hidden", "shared, hidden", [0]), ("S1", "S1", [0]),
              ("S2", "S2", [0, 1]), ("M0vis.visible", "shared, visible", [0])]
    ax = bd.mm_axes(cv, x0, y0, w, h)
    lab_w, gap = 10.0, 1.5
    ncol = sum(4 * len(c) for _, _, c in blocks)
    cw = (w - lab_w - gap * (len(blocks) - 1)) / ncol
    top, rh = 11.4, 3.55
    ax.text(lab_w - 0.8, top - 1.0, "delay/prefix", ha="right", va="bottom", fontsize=fs.PT_SMALL)
    for i, (dly, pre) in enumerate(TIMINGS):
        ax.text(lab_w - 0.8, top + (i + 0.5) * rh, f"{dly}/{pre}", ha="right", va="center", fontsize=fs.PT_SMALL)
    x = lab_w
    for u, ulab, ctxs in blocks:
        bw = 4 * len(ctxs) * cw
        ax.text(x + bw / 2, 1.0, ulab, ha="center", va="top", fontsize=fs.PT_SMALL)
        d = e[e["unit"] == u]
        for ci, ctx in enumerate(ctxs):
            for cue in (0, 1):
                cx0 = x + (ci * 4 + cue * 2) * cw
                ax.text(cx0 + cw, top - 3.4, f"cue {cue}" if len(ctxs) == 1 else f"{'0' if ctx == 0 else '180'}°/{cue}", ha="center",
                        va="bottom", fontsize=fs.PT_SMALL)
                for i, (dly, pre) in enumerate(TIMINGS):
                    rows = d[(d["delay"] == dly) & (d["prefix"] == pre) & (d["context"] == ctx) & (d["case"].map(cue_of) == cue)]
                    if rows.empty:
                        continue
                    cy = top + (i + 0.5) * rh
                    eligible = bool(rows["eligible_both_baselines_solved"].iloc[0])
                    if not eligible:
                        ax.add_patch(Rectangle((cx0 + 0.15, cy - rh / 2 + 0.25), 2 * cw - 0.3, rh - 0.5, facecolor="#EFEFEF", edgecolor="none", zorder=1))
                    for j, kind in enumerate(("cue", "scrambled")):
                        r = rows[rows["kind"] == kind].iloc[0]
                        own, donor, act = int(r["own_push"]), int(r["donor_push"]), int(r["actual_push"])
                        if act == own:
                            col = pal.CUE[cue]
                        elif act == donor:
                            col = pal.CUE[1 - cue]
                        else:
                            col = "#5E5E5E"
                        dx, dy = DIRS[act]
                        gx = cx0 + (j + 0.5) * cw
                        L = min(cw, rh) * 0.36
                        arrow_mm(ax, gx - dx * L, cy - dy * L, gx + dx * L, cy + dy * L, colour=col, lw=0.8, head=1.1,
                                 alpha=1.0 if eligible else 0.35)
                        sd.add("b", kind, unit=u, context=int(ctx), cue=cue, delay_ms=dly, prefix_windows=pre, own_push=own,
                               donor_push=donor, actual_push=act, eligible=eligible)
        t = tests.loc[f"fig4e.mcnemar.{u}"]
        yb = top + 6 * rh + 2.6
        ax.text(x + bw / 2, yb, f"{int(t['a_only'])} | {int(t['b_only'])}", ha="center", va="center", fontsize=fs.PT_SMALL)
        ax.text(x + bw / 2, yb + 2.5, fs.fmt_p(float(t["p_adjusted"])), ha="center", va="center", fontsize=fs.PT_SMALL)
        sd.add("b", "test", unit=u, swap_only=int(t["a_only"]), scramble_only=int(t["b_only"]), p=float(t["p_adjusted"]),
               test_id=f"fig4e.mcnemar.{u}", note="Holm within the hidden-goal family; the visible unit is its own family")
        x += bw + gap
    yb = top + 6 * rh + 2.6
    ax.text(lab_w - 0.8, yb, "donor only", ha="right", va="center", fontsize=fs.PT_SMALL)
    ax.text(lab_w - 0.8, yb + 2.5, fs.P_SYM, ha="right", va="center", fontsize=fs.PT_SMALL)
    comb = tests.loc["fig4e.stratified.distinct_states"]
    note = ("Per cue, left arrow: after the whole-state swap; right arrow: after a same-size scrambled change. Colour: the cue whose "
            "goal the push serves; faded: not both baselines solved. Counts: histories that follow the donor only after the swap | "
            f"only after scrambling; combined over distinct hidden-goal states {int(comb['a_only'])} | {int(comb['b_only'])}, "
            f"{fs.P_SYM} = {fs.fmt_p(float(comb['p_value']))} (secondary).")
    lines = fs.wrap(note, w - lab_w - 0.5)
    for k, line in enumerate(lines):
        ax.text(lab_w, h - 0.4 - (len(lines) - 1 - k) * 2.3, line, ha="left", va="bottom", fontsize=fs.PT_SMALL)
    sd.add("b", "combined", swap_only=int(comb["a_only"]), scramble_only=int(comb["b_only"]), p=float(comb["p_value"]),
           test_id="fig4e.stratified.distinct_states", note="secondary")


# ---------------------------------------------------------------- c: dose heatmaps

def panel_c(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    f = cm.table("main/fig4/fig4f_dose_response.csv").copy()
    f["cue"] = f["case"].map(cue_of)
    f["ctx"] = f["case"].str.extract(r"c(\d)")[0].astype(int)
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    cmap = colormaps[pal.MARGIN_CMAP]
    blocks = [("M0", 0, "M0"), ("M0joint", 0, "M0 joint"), ("M0vis.hidden", 0, "shared, hidden"), ("M0vis.visible", 0, "shared, visible"),
              ("S1", 0, "S1"), ("S2", 0, "S2, 0°"), ("S2", 1, "S2, 180°")]
    cols = [("baseline", 0.0), ("cue", 0.25), ("cue", 0.5), ("cue", 1.0), ("scrambled", 0.25), ("scrambled", 0.5), ("scrambled", 1.0)]
    inset_w = 11.0
    gap = 1.6
    bw = (w - inset_w - gap * (len(blocks) - 1)) / len(blocks)
    cwid = bw / len(cols)
    top, rh = 4.4, 1.42
    ax = bd.mm_axes(cv, x0, y0, w, h)
    for b_i, (u, ctx, lab) in enumerate(blocks):
        bx = inset_w + b_i * (bw + gap)
        ax.text(bx + bw / 2, top - 0.6, lab, ha="center", va="bottom", fontsize=fs.PT_SMALL)
        d = f[(f["unit"] == u) & (f["ctx"] == ctx)]
        hist = sorted({(int(r.cue), int(r.delay_ms), int(r.prefix_windows)) for r in d.itertuples()})
        for i, (cue, dly, pre) in enumerate(hist):
            g = d[(d["cue"] == cue) & (d["delay_ms"] == dly) & (d["prefix_windows"] == pre)]
            for j, (kind, fr) in enumerate(cols):
                m = g[(g["kind"] == kind) & (np.isclose(g["fraction"], fr))]["margin"]
                if m.empty:
                    continue
                val = float(m.iloc[0])
                ax.add_patch(Rectangle((bx + j * cwid + (0.35 if j >= 4 else 0) + (0.35 if j >= 1 else 0), top + i * rh), cwid - 0.1, rh - 0.1,
                                       facecolor=cmap(MARGIN(val)), edgecolor="none"))
                sd.add("c", kind, unit=u, context=ctx, cue=cue, delay_ms=dly, prefix_windows=pre, dose=fr, margin=val)
            if b_i == 0:
                ax.add_patch(Rectangle((bx - 1.6, top + i * rh + 0.1), 1.1, rh - 0.3, facecolor=pal.CUE[cue], edgecolor="none"))
        hb = top + len(hist) * rh
        for j, (kind, fr) in enumerate(cols):
            xj = bx + j * cwid + (0.35 if j >= 4 else 0) + (0.35 if j >= 1 else 0) + cwid / 2
            ax.text(xj, hb + 0.4, "0" if fr == 0 else f"{fr:g}".lstrip("0"), ha="center", va="top", fontsize=fs.PT_SMALL, rotation=0)
        # median trend strip
        mx = cv.axes(x0 + bx, y0 + hb + 4.6, bw, h - hb - 13.0)
        doses = [0.0, 0.25, 0.5, 1.0]
        base = d[d["kind"] == "baseline"]["margin"]
        for kind, ls, col in (("cue", "-", fs.TEXT), ("scrambled", (0, (1.6, 1.0)), "#7A7A7A")):
            med = [float(base.median())] + [float(d[(d["kind"] == kind) & np.isclose(d["fraction"], a)]["margin"].median()) for a in doses[1:]]
            q1 = [float(base.quantile(0.25))] + [float(d[(d["kind"] == kind) & np.isclose(d["fraction"], a)]["margin"].quantile(0.25)) for a in doses[1:]]
            q3 = [float(base.quantile(0.75))] + [float(d[(d["kind"] == kind) & np.isclose(d["fraction"], a)]["margin"].quantile(0.75)) for a in doses[1:]]
            if kind == "cue":
                mx.fill_between(doses, q1, q3, color="#BDBDBD", alpha=0.45, lw=0)
            mx.plot(doses, med, color=col, lw=0.8, ls=ls)
        mx.axhline(0, color=fs.AXIS, lw=fs.LW_HAIR)
        mx.set_xlim(-0.05, 1.05)
        mx.set_ylim(-4, 4)
        mx.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
        mx.set_yticks([-4, 0, 4], [fs.MINUS + "4", "0", "4"] if b_i == 0 else ["", "", ""])
        fs.clean(mx)
        if not (u == "S2" and ctx == 0):
            p = float(tests.loc[f"fig4f.page.{u}", "p_adjusted"])
            mx.text(1.0, 1.0, fs.p_text(p), transform=mx.transAxes, ha="right", va="bottom", fontsize=fs.PT_SMALL)
            sd.add("c", "Page test", unit=u, p_holm=p, test_id=f"fig4f.page.{u}", note="S2 test spans both orientations")
        if b_i == 0:
            mx.set_ylabel("Margin", labelpad=1)
    ax.text(inset_w - 2.2, top + 6 * rh * 0.5, "cue 0", ha="right", va="center", fontsize=fs.PT_SMALL)
    ax.text(inset_w - 2.2, top + 6 * rh * 1.5, "cue 1", ha="right", va="center", fontsize=fs.PT_SMALL)
    yk = y0 + h - 1.2
    kx = cv.axes(x0 + inset_w, yk - 1.6, 14.0, 1.3)
    kx.imshow(cmap(MARGIN(np.linspace(-8, 8, 128)))[None, :, :3], aspect="auto", extent=(-8, 8, 0, 1))
    kx.set_yticks([])
    kx.set_xticks([-8, 0, 8], [fs.MINUS + "8", "0", "8"])
    kx.tick_params(axis="x", length=1.0, pad=0.4, labelsize=fs.PT_SMALL)
    for side in ("top", "right", "left"):
        kx.spines[side].set_visible(False)
    cv.text(x0 + inset_w + 15.2, yk - 1.0, "first-decision margin (orange: towards the other push).  Columns: baseline | towards the "
            "other cue, dose a = .25, .5, 1 | scrambled change of the same size.  ", size=fs.PT_SMALL, va="center")
    cv.text(x0 + inset_w + 15.2, yk + 1.4, r"$s_a = s + a\,(s' - s)$ for the whole carried state; lines: median, band: interquartile range, "
            "dashed: scrambled.", size=fs.PT_SMALL, va="center")


# ---------------------------------------------------------------- d: hidden vs visible, by component

def panel_d(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "d")
    v = cm.table("main/fig4/fig4d_visibility_contrast.csv")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    comps = [("all_state", "whole state"), ("voltage", "voltage"), ("current", "current"), ("adaptation", "adaptation"),
             ("delay_buffer", "delay buffer"), ("resources", "resources")]
    rows = [(f"{op}_{c}", f"{op} {lab}") for op in ("clear", "swap") for c, lab in comps]
    windows = ["hold", "read", "write"]
    cmap = colormaps["YlOrBr"]
    ax = bd.mm_axes(cv, x0, y0, w, h)
    lab_w = 21.0
    cw = (w - lab_w - 18.0) / 6
    top, rh = 7.6, (h - 13.0) / len(rows)
    for j, wdw in enumerate(windows):
        ax.text(lab_w + (2 * j + 1) * cw, top - 3.4, f"{wdw}", ha="center", va="bottom", fontsize=fs.PT_SMALL)
        ax.text(lab_w + (2 * j + 0.5) * cw, top - 0.6, "hid.", ha="center", va="bottom", fontsize=fs.PT_SMALL)
        ax.text(lab_w + (2 * j + 1.5) * cw, top - 0.6, "vis.", ha="center", va="bottom", fontsize=fs.PT_SMALL)
    tot_x = lab_w + 6 * cw + 2.0
    ax.text(tot_x, top - 0.6, "lost, all windows", ha="left", va="bottom", fontsize=fs.PT_SMALL)
    vv = v[v["source"] == "mechanisms_v1"]
    for i, (iv, lab) in enumerate(rows):
        yy = top + i * rh + (1.0 if i >= 6 else 0.0)
        ax.text(lab_w - 0.8, yy + rh / 2, lab, ha="right", va="center", fontsize=fs.PT_SMALL)
        sums = {"hidden": 0, "visible": 0}
        for j, wdw in enumerate(windows):
            g = vv[(vv["intervention"] == iv) & (vv["window"] == wdw)]
            for k, vis in enumerate(("hidden", "visible")):
                elig = int(g[f"{vis}_baseline_solved"].astype(str).str.lower().eq("true").sum())
                lost = int((g[f"effect_{vis}"] < 0).sum())
                frac = lost / elig if elig else 0.0
                ax.add_patch(Rectangle((lab_w + (2 * j + k) * cw + 0.1, yy + 0.1), cw - 0.2, rh - 0.2, facecolor=cmap(0.10 + 0.55 * frac) if lost else "white",
                                       edgecolor="#D0D0D0" if not lost else "none", linewidth=0.25))
                if lost:
                    ax.text(lab_w + (2 * j + k + 0.5) * cw, yy + rh / 2, str(lost), ha="center", va="center", fontsize=fs.PT_SMALL)
                sums[vis] += lost
                sd.add("d", "lost", intervention=iv, window=wdw, goal=vis, lost=lost, eligible=elig)
        for k, (vis, col) in enumerate((("hidden", "#5E5E5E"), ("visible", pal.LEARNER["visible goal"]))):
            ax.add_patch(Rectangle((tot_x, yy + 0.25 + k * (rh - 0.5) / 2), sums[vis] * 0.42, (rh - 0.5) / 2, facecolor=col, edgecolor="none"))
        ax.text(tot_x + max(sums.values()) * 0.42 + 0.6, yy + rh / 2, f"{sums['hidden']} | {sums['visible']}", ha="left", va="center",
                fontsize=fs.PT_SMALL)
    ps = [float(tests.loc[t, "p_adjusted"]) for t in tests.index if str(t).startswith("fig4d.sign.")]
    ax.text(lab_w, h - 0.3, f"hidden {fs.MINUS} visible: no difference survives Holm ({len(ps)} sign tests, smallest {fs.P_SYM} = {fs.fmt_p(min(ps))}); "
            "grey bar hidden, blue visible", ha="left", va="bottom", fontsize=fs.PT_SMALL)
    sd.add("d", "tests", tests=len(ps), smallest_p_holm=min(ps), family="F4d and F4d.supplement")


# ---------------------------------------------------------------- e: which cells carry the cue

def panel_e(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "e")
    cells = cm.table("figure_data/fd_cells.csv")
    st = cm.table("figure_data/fd_soma_state.csv")
    d = cells[["cell", "superclass"]].merge(st[["cell", "dv_m0_mv"]], on="cell")
    d["group"] = d["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other")
    rows = []
    for g, gg in d.groupby("group"):
        n = len(gg)
        neg = float((gg["dv_m0_mv"] < -1).mean())
        pos = float((gg["dv_m0_mv"] > 1).mean())
        rows.append((g, n, neg, pos))
        sd.add("e", "share over 1 mV", group=g, cells=n, after_cue0=neg, after_cue1=pos)
    rows = sorted([r for r in rows if r[0] not in ("other", "sensory")], key=lambda r: -(r[2] + r[3]))
    ax = cv.axes(x0 + 23.0, y0 + 6.6, w - 31.0, h - 13.0)
    for i, (g, n, neg, pos) in enumerate(rows):
        ax.barh(i, 100 * neg, color=pal.CUE[0], height=0.66, linewidth=0)
        ax.barh(i, 100 * pos, left=100 * neg, color=pal.CUE[1], height=0.66, linewidth=0)
        ax.text(100 * (neg + pos) + 2, i, f"{100 * (neg + pos):.0f}%", va="center", fontsize=fs.PT_SMALL)
    ax.set_yticks(range(len(rows)), [pal.SHORT_GROUP[r[0]] for r in rows])
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(fs.TEXT)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, 75)
    ax.set_xticks([0, 25, 50, 75])
    ax.set_xlabel("Cells over 1 mV (%)", labelpad=1)
    fs.clean(ax)
    ax.tick_params(axis="y", length=0, pad=7)
    for i, r in enumerate(rows):
        cv.mark(x0 + 21.9, y0 + 6.6 + (i + 0.5) * (h - 13.0) / len(rows) - 0.05, s=5.0, fc=pal.on_white(pal.ATLAS_ON_DARK[r[0]]),
                ec=pal.on_white(pal.ATLAS_ON_DARK[r[0]]), lw=0.3)
    cv.key(x0 + 23.0, y0 + 3.0, [([{"kind": "box", "fc": pal.CUE[0]}], "after cue 0"), ([{"kind": "box", "fc": pal.CUE[1]}], "after cue 1")],
           gap=2.0)


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("fig5")
    for rel in ("figure_data/fd_cells.csv", "figure_data/fd_soma_state.csv", "main/fig4/fig4e_directional_transfer.csv",
                "main/fig4/fig4f_dose_response.csv", "main/fig4/fig4d_visibility_contrast.csv", "stats/statistical_tests.csv"):
        sd.source(cm.RESULTS / rel)
    for f in ("cue0.png", "cue1.png", "cue0.json", "cue1.json"):
        sd.source(TILES / f)
    panel_a(cv, sd)
    bx = 2 * TW + fs.GUTTER + 2.0
    panel_b(cv, sd, bx, 0.0, fs.FULL - bx, TH)
    panel_c(cv, sd, 0.0, ROW2, fs.FULL, ROW3 - ROW2 - 3.0)
    panel_d(cv, sd, 0.0, ROW3, 118.0, H - ROW3)
    panel_e(cv, sd, 121.0, ROW3, fs.FULL - 121.0, H - ROW3)
    cm.finish(cv, "fig5", sd)


if __name__ == "__main__":
    main()
