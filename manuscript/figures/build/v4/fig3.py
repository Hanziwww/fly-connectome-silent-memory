"""v4 Fig. 3 (Cell) - The trace is distributed, and no circuit is special (174 mm wide).

A  dark band, "which one is the memory circuit?": the scene-1 cue ensemble (top 1% of cells by |cue difference|) with its
   internal connections beside three of its 20 matched random sets drawn by the same rule; the four most cue-tuned cell types
B  the ensemble is densely wired, in the form of MaleCNS Figs 7M and 7N: the scene-1 ensemble's adjacency matrix (sources
   down, targets across, cells by superclass group, each connection a dot in its source's group colour) with a red zoom on
   its densest block (group blocks outlined on the diagonal), and its internal connections by source group against the mean
   of its 20 matched sets, scenes 1 and 2, counts inside
C  yet silencing it at the decision changes the margin no more than silencing a matched set: the ensemble's effect relative
   to the mean of its matched sets, the quantity tested (X6; family-wise Holm)
D  and removing its internal synapses changes it no more than removing as many matched synapses (X7; family-wise Holm)
E  whole populations are dispensable: Kenyon cells or GABAergic antennal-lobe local neurons removed, half or all, leave the
   trained histories solved exactly as in the healthy network (TOST equivalence, family-wise Holm)
F  where the cue lives: dark maps of the cue difference at the decision (all cells, the readout cells, the six olfactory
   circuits in gray), and each circuit's share of the cue difference against its share of cells (three scenes)
G  robust to the connection rules: dark maps of where each rule acts (the share of each cell's input synapses it changes),
   and under the seven variants, each plotted under its rule, every run learns and the Kv4 lesion takes the memory

Every number in a label is read from manuscript/results (plan_v2 via P12) or from the render metadata. Reader-facing
names only (NARRATIVE_V4_OUTLINE.md section 12). Not shown (RESULTS_INVENTORY_V4.md, class c): the matched random-cell
removals and the descending-neuron control of the population test, and the strict timing lock under the variants (D9).

Run after P12, render/v4_fig3_tiles.py and render/v4_fig3_maps.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\v4\\fig3.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
from lib import circuit as ci, figstyle as fs, palette as pal, qc  # noqa: E402

cm.use_v4()
TILES = cm.RENDERS / "v4" / "fig3"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
MAPS_DIR = TILES / "maps"
MAPS = json.loads((MAPS_DIR / "maps.json").read_text(encoding="utf-8"))
BAND_H = META["band_mm"]
TW = META["tile_mm"][0]
NW, NH = META["neuron_mm"]
TITLE_MM = META["title_mm"]
MAP_W, MAP_H = MAPS["tile_mm"]
U = fs.unit(1)
XS = [fs.unit_x(i) for i in range(4)]
ROW_B = BAND_H + 4.2
B_H = 31.2
ROW2 = ROW_B + B_H + 3.4
ROW_H = 44.0
ROW3 = ROW2 + ROW_H + 3.4
W3 = (fs.FULL - 2 * fs.GUTTER) / 3                            # C, D, E: a third of the width each
X3 = [k * (W3 + fs.GUTTER) for k in range(3)]
MAP_Y = ROW3 + 4.6
PLOT_Y = MAP_Y + MAP_H + 1.6
PLOT_H = 46.0
H = PLOT_Y + PLOT_H
WHITE = fs.ON_DARK
NL_ = chr(10)
TARGET, MATCHED = fs.TEXT, "#B5B5B5"
SCENE_COL = {"scene 1": pal.STATE_SET["m0"], "scene 2": pal.STATE_SET["s1"], "scene 3": pal.STATE_SET["s2_o0"]}
COMBOS = [("scene 1", "scene 1"), ("scene 1", "scene 2"), ("scene 2", "scene 2"), ("scene 2", "scene 1")]
SCENE_CODE = {"scene 1": "m0", "scene 2": "s1"}                 # repository codes in the test names (lookup only)
OLFACTORY = ["Kenyon cells", "uniglomerular PNs (ACh ALPN)", "GABAergic ALLNs", "MBONs", "APL", "ORNs"]
FAMILIES = [("receptor rule", ["presynaptic signs"], "default: receptor signs"),
            ("same-type gain", ["same-type gain 0.05", "same-type gain 0.2", "same-type gain 1"], "default 0.1"),
            ("synapse cap", ["synapse cap 30", "synapse cap 120", "no synapse cap"], "default 60")]
MATRIX_GROUPS = ["optic lobe intrinsic", "visual projection & centrifugal", "central brain intrinsic", "descending & ascending",
                 "VNC intrinsic", "motor & efferent", "sensory", "other"]
ZOOM_K = 128                                                  # B: the zoomed block, cells per side
SHORT_VARIANT = {"presynaptic signs": f"presynaptic{NL_}signs", "same-type gain 0.05": "0.05", "same-type gain 0.2": "0.2", "same-type gain 1": "1",
                 "synapse cap 30": "30", "synapse cap 120": "120", "no synapse cap": "none"}


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


def scene_no(s):
    return s.split()[-1]


def gradient(cv, x, y, w, h, colours):
    """Vertical key bar, first colour at the top."""
    ax = cv.axes(x, y, w, h)
    ax.imshow(np.asarray(colours)[:, None, :3] * np.ones((1, 2, 1)), aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax.set_axis_off()


# ---------------------------------------------------------------- A: band

def panel_band(cv, sd):
    wiring = cm.table("plan_v2/fig3_set_wiring.csv")
    wiring = wiring[wiring["ensemble_scene"] == "scene 1"].set_index("set")
    xs = [k * (TW + fs.GUTTER) for k in range(4)]
    titles = {"ensemble": "Cue ensemble, top 1%", "matched 1": "Matched set 1", "matched 2": "Matched set 2", "matched 3": "Matched set 3"}
    for x, t in zip(xs, META["sets"]):
        img = np.asarray(Image.open(TILES / f"{t['name']}.png").convert("RGB"))
        cv.tile(x, 0, img, width=TW, height=BAND_H, group="band", label=t["name"])
        tile_text(cv, x + (4.2 if x == 0 else 1.2), 1.0, titles[t["set"]], va="top", size=fs.PT)
        n = int(wiring.loc[t["set"], "internal_connections"])
        tile_text(cv, x + 1.0, BAND_H - 8.6, f"{t['cells']:,} cells", va="center")
        tile_text(cv, x + 1.0, BAND_H - 6.2, f"{n:,}", va="center")
        tile_text(cv, x + 1.0, BAND_H - 3.8, "connections", va="center")
        sd.add("A", "set", set=t["set"], cells=int(t["cells"]), located=int(t["located"]), internal_connections=n,
               depolarized_more_by_cue0=int(t["cue0"]), by_cue1=int(t["cue1"]), no_difference=int(t["no_difference"]))
    white_letter(cv, 1.0, 0.8, "A")
    # key (right of the VNC in the first tile)
    kx, ky = xs[0] + TW - 11.6, BAND_H - 11.0
    for i, (c, lab) in enumerate(((pal.CUE_ON_DARK[0], "cue 0"), (pal.CUE_ON_DARK[1], "cue 1"))):
        cv.mark(kx + 0.7, ky + 2.4 * i, s=6.0, fc=c, ec=c, lw=0.3)
        tile_text(cv, kx + 1.6, ky + 2.4 * i, lab, va="center")
    tile_text(cv, kx, ky - 5.0, "depolarized", va="center")
    tile_text(cv, kx, ky - 2.6, "more by", va="center")
    # scale bar and orientation on the last set tile
    t = META["sets"][-1]
    px_per_mm = np.asarray(Image.open(TILES / f"{t['name']}.png")).shape[1] / TW
    length = 100 / t["scale_um_per_px"] / px_per_mm
    cv.line([xs[3] + TW - 1.4 - length, xs[3] + TW - 1.4], [BAND_H - 1.8, BAND_H - 1.8], WHITE, lw=1.0, solid_capstyle="butt")
    ox, oy = xs[3] + TW - 8.0, BAND_H - 9.4
    cv.line([ox, ox + 2.6], [oy, oy], WHITE, lw=0.5)
    tile_text(cv, ox + 3.0, oy, "L", va="center")
    cv.line([ox, ox], [oy, oy + 2.6], WHITE, lw=0.5)
    tile_text(cv, ox, oy + 3.0, "V", ha="center", va="top")
    # the four most cue-tuned cell types
    nx0 = 4 * (TW + fs.GUTTER)
    for i, n in enumerate(META["neurons"]):
        x, y = nx0 + (i % 2) * (NW + fs.GUTTER), (i // 2) * (NH + fs.GUTTER)
        img = np.asarray(Image.open(TILES / f"{n['name']}.png").convert("RGB"))
        cv.tile(x, y, img, width=NW, height=NH, group="band", label=n["name"])
        tile_text(cv, x + 1.0, y + 1.0, n["type"], va="top")
        ppm = img.shape[1] / NW
        ln = n["scale_bar_um"] / n["scale_um_per_px"] / ppm
        cv.line([x + NW - 1.2 - ln, x + NW - 1.2], [y + NH - 1.4, y + NH - 1.4], WHITE, lw=0.8, solid_capstyle="butt")
        sd.add("A", "cell type", rank=int(n["rank"]), type=n["type"], body_id=int(n["body_id"]), mean_abs_cue_difference_mv=float(n["type_mean_abs_dv_mv"]),
               scale_bar_um=int(n["scale_bar_um"]))
    sd.add("A", "view", scale_bar_um=100, um_per_px=t["scale_um_per_px"])


# ---------------------------------------------------------------- B: dense wiring

def group_colour(g):
    return ci.tint(pal.on_white(pal.ATLAS_ON_DARK[g]), 0.25) if g in pal.ATLAS_ON_DARK else "#9E9E9E"


def group_of(superclass):
    return pd.Series(superclass).map(pal.SUPERCLASS_GROUP).fillna("other").to_numpy()


def panel_matrix(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "B", "Densely wired")
    edges = cm.table("plan_v2/fig3_ensemble_edges.csv")
    bands = cm.table("plan_v2/fig3_band_sets.csv")
    by_group = cm.table("plan_v2/fig3_set_groups.csv")
    wiring = cm.table("plan_v2/fig3_set_wiring.csv").set_index(["ensemble_scene", "set"])["internal_connections"]
    sup = cm.table("figure_data/fd_cells.csv").set_index("cell")["superclass"]
    tot = by_group.groupby(["ensemble_scene", "set"])["connections"].sum()
    if not (tot.reindex(wiring.index) == wiring).all() or len(edges) != int(wiring.loc[("scene 1", "ensemble")]):   # gate
        raise RuntimeError("B: connections by source group or the ensemble's edges disagree with fig3_set_wiring")
    # M-like: the adjacency matrix, cells ordered by group
    cells = bands[(bands["ensemble_scene"] == "scene 1") & (bands["set"] == "ensemble")]["cell"].to_numpy()
    g = group_of(sup.reindex(cells).to_numpy())
    rank = np.array([MATRIX_GROUPS.index(x) if x in MATRIX_GROUPS else len(MATRIX_GROUPS) for x in g])
    order = np.lexsort((cells, rank))
    cells, g = cells[order], g[order]
    pos = pd.Series(np.arange(len(cells)), index=cells)
    src, tgt = pos.reindex(edges["pre"]).to_numpy(), pos.reindex(edges["post"]).to_numpy()
    n = len(cells)
    s, mx, my = 22.0, x0 + 6.4, y0 + 5.4
    ax = cv.axes(mx, my, s, s)
    cols = np.array([group_colour(g[i]) for i in src])
    ax.scatter(tgt + 0.5, src + 0.5, s=0.08, c=cols, marker="s", linewidths=0, rasterized=True)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.35)
        sp.set_edgecolor("#9E9E9E")
    ax0, ay0 = mx - 1.4, my + s + 1.2                       # MaleCNS axis arrows: sources down the left, targets along the bottom
    ci.arrow(cv, [(ax0, my + 2.0), (ax0, ay0)], 0.25, fs.TEXT, "arrow", gap=0.0)
    ci.arrow(cv, [(ax0, ay0), (mx + 7.0, ay0)], 0.25, fs.TEXT, "arrow", gap=0.0)
    cv.text(mx + 7.8, ay0, "target (postsynaptic)", size=fs.PT_SMALL, va="center")
    cv.text(ax0 - 0.7, my + s / 2, "source (presynaptic)", size=fs.PT_SMALL, va="center", ha="right", rotation=90)
    sd.add("B", "matrix", ensemble_scene="scene 1", cells=int(n), connections=int(len(edges)))
    # the densest block, enlarged, with the group blocks on its diagonal
    k = ZOOM_K
    dens = np.zeros((n // k + 1, n // k + 1))
    np.add.at(dens, (src // k, tgt // k), 1)
    r0, c0 = (np.array(np.unravel_index(np.argmax(dens), dens.shape)) * k).tolist()
    zx = mx + s + 5.0
    zax = cv.axes(zx, my, s, s)
    m = (src >= r0) & (src < r0 + k) & (tgt >= c0) & (tgt < c0 + k)
    zax.scatter(tgt[m] - c0 + 0.5, src[m] - r0 + 0.5, s=1.1, c=cols[m], marker="s", linewidths=0)
    zax.set_xlim(0, k)
    zax.set_ylim(k, 0)
    zax.set_xticks([])
    zax.set_yticks([])
    bounds = np.flatnonzero(np.r_[True, g[1:] != g[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        lo, hi = max(a, r0, c0), min(b, r0 + k, c0 + k)
        if hi > lo:
            zax.add_patch(Rectangle((lo - c0, lo - r0), hi - lo, hi - lo, facecolor="none", edgecolor=fs.TEXT, linewidth=0.6))
    ci.zoom(cv, ax, (c0, r0, c0 + k, r0 + k), zax)
    cv.text(zx + s / 2, my - 0.9, f"{k} × {k} cells: {int(m.sum()):,} connections", size=fs.PT_SMALL, va="bottom", ha="center")
    sd.add("B", "zoom", first_source=int(r0), first_target=int(c0), cells_per_side=k, connections=int(m.sum()))
    # N-like: internal connections by source group, the ensemble against the mean of its 20 matched sets, counts inside
    by_group = by_group.assign(group=group_of(by_group["source_superclass"].to_numpy()))
    piv = by_group.groupby(["ensemble_scene", "set", "group"])["connections"].sum().unstack("group").fillna(0)
    shown = [x for x in MATRIX_GROUPS if x in piv.columns]
    bx, bw = x0 + 92.0, 54.0
    bax = cv.axes(bx, y0 + 10.6, bw, 15.8)
    rows = []
    for k_, scene in enumerate(("scene 1", "scene 2")):
        q = piv.loc[scene]
        matched = q.loc[[i for i in q.index if i != "ensemble"]]
        totals = wiring.loc[scene].drop("ensemble")
        rows.append((scene, "ensemble", q.loc["ensemble", shown].to_numpy(float), 3.6 - 2.6 * k_,
                     f"{int(wiring.loc[(scene, 'ensemble')]):,} · ×{wiring.loc[(scene, 'ensemble')] / totals.mean():.1f}"))
        rows.append((scene, "matched", matched[shown].mean().to_numpy(float), 2.6 - 2.6 * k_,
                     f"{int(totals.min()):,}–{int(totals.max()):,}"))
        cv.text(x0 + 60.0, y0 + 10.6 + 15.8 * (4.2 - (3.1 - 2.6 * k_)) / 4.8, scene, size=fs.PT_SMALL, va="center", fontweight="bold")
        sd.add("B", "internal connections", ensemble_scene=scene, ensemble=int(wiring.loc[(scene, "ensemble")]),
               matched_min=int(totals.min()), matched_max=int(totals.max()), matched_mean=float(totals.mean()),
               ratio_to_matched_mean=float(wiring.loc[(scene, "ensemble")] / totals.mean()), matched_sets=int(len(totals)))
    span = 100000.0
    bax.set_xlim(0, span)
    bax.set_ylim(-0.6, 4.2)
    cv.fig.canvas.draw()
    for scene, which, vals, yy, total in rows:
        lefts = np.concatenate([[0.0], np.cumsum(vals)[:-1]])
        fills = [group_colour(x) for x in shown]
        for l0, v, f in zip(lefts, vals, fills):
            bax.barh(yy, v, left=l0, height=0.78, color=f, linewidth=0)
        ci.bar_counts(cv, bax, lefts, vals, yy, np.rint(vals), fills, min_frac=0.12)
        bax.text(vals.sum() + 0.012 * span, yy, total, fontsize=fs.PT_SMALL, va="center", ha="left",
                 fontweight="bold" if which == "ensemble" else "normal")
        for x_, v in zip(shown, vals):
            sd.add("B", "by source group", ensemble_scene=scene, set=which if which == "ensemble" else "matched sets (mean of 20)", group=x_,
                   connections=float(v))
    bax.set_yticks([r[3] for r in rows], ["ensemble" if r[1] == "ensemble" else "matched, mean of 20" for r in rows])
    bax.set_xticks([0, 40000, 80000], ["0", "40k", "80k"])
    fs.clean(bax)
    bax.tick_params(axis="y", length=0)
    bax.set_xlabel("Internal connections, by source group", labelpad=0.8)
    items = [([dict(kind="box", size=1.8, fc=group_colour(x))], pal.SHORT_GROUP.get(x, x)) for x in shown]
    half = (len(items) + 1) // 2
    cv.key(bx, y0 + 4.6, items[:half], size=fs.PT_SMALL, gap=1.4)                # the source groups, over the bars
    cv.key(bx, y0 + 7.4, items[half:], size=fs.PT_SMALL, gap=1.4)


# ---------------------------------------------------------------- C, D: X6, X7, relative to the matched sets

def panel_sets(cv, sd, x0, y0, w, h, letter, kind, heading, matched_label, family):
    title(cv, x0, y0, letter, heading)
    d = cm.table("plan_v2/fig3_ensemble_sets.csv")
    d = d[d["kind"] == kind]
    t = tests(family)
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 18.0)
    rng = np.random.default_rng(32 if kind == "x6" else 33)
    lim = 0.0
    for i, (es, ts) in enumerate(COMBOS):
        q = d[(d["ensemble_scene"] == es) & (d["test_scene"] == ts)]
        m = q[q["set"] != "ensemble"]["mean_margin_effect"].to_numpy(float)
        e = float(q.loc[q["set"] == "ensemble", "mean_margin_effect"].iloc[0])
        rel = e - m.mean()
        row = t[t["test"].astype(str).str.contains(f"ensemble from {SCENE_CODE[es]} tested in {SCENE_CODE[ts]}:", regex=False)]
        if len(row) != 1 or abs(rel - float(row["estimate"].iloc[0])) > 1e-9:
            raise RuntimeError(f"{letter}: the ensemble's effect relative to its matched sets differs from the tested estimate ({es} -> {ts})")
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(m)), m - m.mean(), s=4, facecolor=MATCHED, edgecolor="none", zorder=2)
        ax.scatter([i], [rel], s=14, facecolor=TARGET, edgecolor=TARGET, zorder=3)
        lim = max(lim, np.abs(m - m.mean()).max(), abs(rel))
        sd.add(letter, kind, ensemble_scene=es, test_scene=ts, ensemble_minus_matched_mean=rel, matched_minus_matched_mean=(m - m.mean()).tolist(),
               ensemble_effect=e, matched_mean_effect=float(m.mean()), holm_p=float(row["p_adjusted"].iloc[0]))
    ax.axhline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=0)
    top = np.ceil(lim * 1.25 * 2) / 2
    ax.set_ylim(-top, top)
    ax.set_yticks([-1, 0, 1] if top >= 1 else [-0.5, 0, 0.5])
    ax.set_xlim(-0.6, len(COMBOS) - 0.4)
    ax.set_xticks(range(len(COMBOS)), [f"{scene_no(es)}→{scene_no(ts)}" for es, ts in COMBOS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Ensemble's scene → tested in", labelpad=0.8)
    ax.set_ylabel(f"Margin change,{NL_}relative to matched sets", labelpad=0.8, linespacing=1.0)
    fs.clean(ax)
    p_lo, p_hi = float(t["p_adjusted"].min()), float(t["p_adjusted"].max())
    p_txt = f"{p_lo:.2g}" if p_hi - p_lo < 1e-9 else f"{p_lo:.2g}–{p_hi:.2g}"
    ax.text(len(COMBOS) - 0.5, top * 0.94, f"Holm {fs.P_SYM} = {p_txt}", fontsize=fs.PT_SMALL, ha="right", va="top", color=fs.TEXT2)
    cv.key(x0 + 9.0, y0 + 6.0, [([dict(kind="mark", s=10, fc=TARGET, ec=TARGET)], "ensemble"),
                                ([dict(kind="mark", s=6, fc=MATCHED, ec=MATCHED)], matched_label)], size=fs.PT_SMALL, gap=1.6)
    sd.add(letter, "test", family=family, holm_p_min=p_lo, holm_p_max=p_hi, tests=int(len(t)),
           quantity="ensemble effect minus the mean effect of its matched sets (positive: smaller effect than the matched sets)")


# ---------------------------------------------------------------- E: populations

def panel_populations(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "E", "Whole populations dispensable")
    r = cm.table("plan_v2/fig3_removals.csv")
    healthy = int(r.loc[r["kind"] == "healthy", "solved"].iloc[0])
    n_hist = int(r.loc[r["kind"] == "healthy", "histories"].iloc[0])
    groups = [("Kenyon cells", 0.5, "½"), ("Kenyon cells", 1.0, "all"), ("GABAergic antennal-lobe local neurons", 0.5, "½"),
              ("GABAergic antennal-lobe local neurons", 1.0, "all")]
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 18.0)
    changed = 0
    for i, (pop, frac, lab) in enumerate(groups):
        tg = r[(r["population"] == pop) & (r["fraction_removed"] == frac) & (r["kind"] == "whole population")].iloc[0]
        ax.scatter([i], [tg["solved"]], s=14, facecolor=TARGET, edgecolor=TARGET, zorder=3)
        ax.text(i, tg["solved"] - 10, f"{int(tg['cells']):,}", fontsize=fs.PT_SMALL, ha="center", va="top")
        changed += int(tg["solved_only_after_removal"]) + int(tg["solved_only_healthy"])
        sd.add("E", "removal", population=pop, fraction=frac, cells=int(tg["cells"]), solved=int(tg["solved"]), histories=int(tg["histories"]),
               solved_only_after_removal=int(tg["solved_only_after_removal"]), solved_only_healthy=int(tg["solved_only_healthy"]))
    ax.axhline(healthy, color=fs.TEXT2, lw=0.5, ls=(0, (2, 1.5)), zorder=1)
    ax.text(len(groups) - 0.5, healthy + 2, "healthy", fontsize=fs.PT_SMALL, ha="right", va="bottom", color=fs.TEXT2)
    ax.text((len(groups) - 1) / 2, healthy - 21, "cells removed", fontsize=fs.PT_SMALL, ha="center", va="top", color=fs.TEXT2)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, n_hist)
    ax.set_yticks([0, 60, 120])
    ax.set_xticks(range(len(groups)), [g[2] for g in groups])
    ax.tick_params(axis="x", length=0, labelsize=fs.PT_SMALL)
    ax.set_ylabel(f"Trained histories solved (of {n_hist})", labelpad=0.8)
    fs.clean(ax)
    below = ax.get_xaxis_transform()
    for xc, lab in ((0.5, "KC"), (2.5, "AL LN")):
        ax.text(xc, -0.115, lab, transform=below, fontsize=fs.PT_SMALL, ha="center", va="top")
    eq = tests("F12.dependency.equivalence")
    cv.text(x0 + 2.0, y0 + h - 1.2, f"TOST, Holm {fs.P_SYM} = {fs.fmt_p(float(eq['p_adjusted'].max()))}", size=fs.PT_SMALL, va="center")
    cv.key(x0 + 9.0, y0 + 6.0, [([dict(kind="mark", s=10, fc=TARGET, ec=TARGET)], "whole population removed")], size=fs.PT_SMALL)
    sd.add("E", "tests", equivalence_holm_p=float(eq["p_adjusted"].max()), healthy_solved=healthy)


# ---------------------------------------------------------------- F: where the cue lives

def map_tile(cv, sd, letter, x, name, lines):
    meta = json.loads((MAPS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    cv.tile(x, MAP_Y, np.asarray(Image.open(MAPS_DIR / f"{name}.png").convert("RGB")), width=MAP_W, height=MAP_H, group=f"maps {letter}", label=name)
    for k, (s, size) in enumerate(zip(lines, (fs.PT, fs.PT_SMALL, fs.PT_SMALL))):
        tile_text(cv, x + 1.2, MAP_Y + 1.1 + (0 if k == 0 else 1.0 + 2.35 * k), s, va="top", size=size)
    return meta


def map_key(cv, x, lo_lab, hi_lab, label):
    """Magma key in the black space right of the VNC of a map tile."""
    bx, by, bh = x + MAP_W - 2.2, MAP_Y + MAP_H - 13.4, 8.0
    gradient(cv, bx, by, 1.1, bh, pal.MAGNITUDE_ON_DARK(np.linspace(1.0, 0.25, 96)))
    tile_text(cv, bx - 0.4, by + 0.4, hi_lab, va="center", ha="right")
    tile_text(cv, bx - 0.4, by + bh - 0.4, lo_lab, va="center", ha="right")
    tile_text(cv, bx + 1.1, by - 0.8, label, va="bottom", ha="right", linespacing=1.0)


def panel_circuits(cv, sd, x0, w):
    title(cv, x0, ROW3, "F", "Cue in the readout cells, not in olfactory circuits")
    c = cm.table("plan_v2/fig3_cue_energy.csv")
    cells = cm.table("plan_v2/fig3_cue_cells.csv")
    s1 = c[c["scene"] == "scene 1"].set_index("circuit")
    ro, olf = s1.loc["readout cells"], s1.loc[OLFACTORY]
    olf_pct = 100 * float(olf["share_of_cue_energy"].sum())
    lo, hi = MAPS["dv_range_mv"]
    tiles = [("cue_all", ["All cells", f"{len(cells):,} cells", "the cue at the decision"]),
             ("cue_readout", ["Readout cells", f"{int(ro['cells']):,} cells", f"{100 * ro['share_of_cue_energy']:.0f}% of the cue"]),
             ("cue_olfactory", ["Olfactory circuits", f"{int(olf['cells'].sum()):,} cells, in gray", f"{olf_pct:.2f}% of the cue"])]
    for k, (name, lines) in enumerate(tiles):
        meta = map_tile(cv, sd, "F", x0 + k * (MAP_W + fs.GUTTER), name, lines)
        sd.add("F", "map", tile=name, lines=lines, cells_drawn=int(meta["cells_drawn"]), dv_range_mv=[lo, hi], learner="scene 1, readout trained")
    map_key(cv, x0, f"{lo:g}", f"{hi:g} mV", f"|cue{NL_}difference|")
    if int(olf["cells"].sum()) != int(cells["olfactory_cell"].sum()) or int(ro["cells"]) != int(cells["readout_cell"].sum()):
        raise RuntimeError("F: the map cells differ from the circuit table")
    # each circuit's share of the cue against its share of cells (three scenes)
    order = (c.groupby("circuit")["enrichment"].median().sort_values(ascending=False)).index.tolist()
    ax = cv.axes(x0 + 38.0, PLOT_Y + 4.0, w - 40.0, PLOT_H - 11.0)
    for j, circ in enumerate(order):
        for scene, colour in SCENE_COL.items():
            q = c[(c["circuit"] == circ) & (c["scene"] == scene)]
            if len(q):
                v = max(float(q["enrichment"].iloc[0]), 1e-9)
                ax.scatter([v], [j], s=9, facecolor=colour, edgecolor=colour, zorder=3)
                sd.add("F", "circuit", circuit=circ, scene=scene, enrichment=float(q["enrichment"].iloc[0]),
                       share_of_cue_energy=float(q["share_of_cue_energy"].iloc[0]), share_of_cells=float(q["share_of_cells"].iloc[0]))
    ax.axvline(1, color=fs.GRID, lw=fs.LW_THIN, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(1e-9, 30)
    ax.set_xticks([1e-8, 1e-4, 1], ["10$^{-8}$", "10$^{-4}$", "1"])
    fs.minor(ax, x=False, y=False)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_yticks(range(len(order)), [o.replace("uniglomerular PNs (ACh ALPN)", "uniglomerular PNs").replace("GABAergic ALLNs", "GABAergic AL LNs")
                                      for o in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Cue share / cell share (at the decision)", labelpad=0.8)
    fs.clean(ax)
    kc = c[c["circuit"] == "Kenyon cells"]["share_of_cue_energy"]
    rr = c[c["circuit"] == "readout cells"]["enrichment"]
    cv.key(x0 + 4.0, PLOT_Y + 1.6, [([dict(kind="mark", s=8, fc=col, ec=col)], s.capitalize()) for s, col in SCENE_COL.items()], size=fs.PT_SMALL, gap=1.6)


# ---------------------------------------------------------------- G: connection rules

def panel_structure(cv, sd, x0, w):
    title(cv, x0, ROW3, "G", "Robust to the connection rules")
    rules = cm.table("plan_v2/fig3_rules.csv").set_index("rule")
    lo, hi = MAPS["share_range"]
    tiles = [("rule_receptor", "receptor rule", ["Receptor rule", "signs from receptors"]),
             ("rule_same_type", "same-type gain", ["Same-type gain", "same-type synapses"]),
             ("rule_cap", "synapse cap", ["Synapse cap", "beyond 60 per connection"])]
    centres = {}
    for k, (name, rule, lines) in enumerate(tiles):
        x = x0 + k * (MAP_W + fs.GUTTER)
        r = rules.loc[rule]
        pct = 100 * float(r["share_of_synapses"])
        lines = lines + [f"{pct:.2f}% of synapses" if pct < 1 else f"{pct:.1f}% of synapses"]
        meta = map_tile(cv, sd, "G", x, name, lines)
        centres[rule] = x + MAP_W / 2
        sd.add("G", "map", tile=name, rule=rule, connections=int(r["entries"]), synapses=float(r["synapses"]), share_of_synapses=float(r["share_of_synapses"]),
               cells=int(r["cells"]), cells_drawn=int(meta["cells_drawn"]), share_range=[lo, hi])
    map_key(cv, x0, f"{100 * lo:g}%", f"{100 * hi:g}%", f"share of{NL_}inputs")
    # every run under its rule: learning and the Kv4-lesion memory effect
    s = cm.table("plan_v2/fig3_structure.csv")
    v = s[s["code"] != "default"]
    if sorted(sum((f[1] for f in FAMILIES), [])) != sorted(v["variant"].unique()):
        raise ValueError("connection-rule variants differ from the plotting families")
    left, right = x0 + 9.0, x0 + w - 1.0
    ax = cv.axes(left, PLOT_Y + 4.0, right - left, PLOT_H - 14.6)
    ticks, labels = [], []
    for rule, variants, default in FAMILIES:
        xs = [centres[rule]] if len(variants) == 1 else [centres[rule] + d for d in (-8.0, 0.0, 8.0)]
        for xc, var in zip(xs, variants):
            q = v[v["variant"] == var].sort_values("replicate")
            for k, (_, r) in enumerate(q.iterrows()):
                xx = xc + (k - 1) * 1.6
                ax.plot([xx, xx], [r["solved_healthy"], r["solved_lesion"]], color=MATCHED, lw=0.6, zorder=2)
                ax.scatter([xx], [r["solved_healthy"]], s=8, facecolor=TARGET, edgecolor=TARGET, zorder=3)
                ax.scatter([xx], [r["solved_lesion"]], s=8, facecolor="white", edgecolor=TARGET, linewidth=0.5, zorder=3)
                sd.add("G", "run", variant=var, replicate=int(r["replicate"]), learned=bool(r["learned"]), updates=int(r["updates"]),
                       trained_net_use=float(r["trained_net_use"]), solved_healthy=int(r["solved_healthy"]), solved_lesion=int(r["solved_lesion"]))
            ticks.append(xc)
            labels.append(SHORT_VARIANT[var])
        cv.text(centres[rule], PLOT_Y + PLOT_H - 4.6, f"{rule}, {default}" if len(variants) > 1 else default, size=fs.PT_SMALL, ha="center",
                va="center", color=fs.TEXT2)
    ax.set_xlim(left, right)
    ax.set_ylim(0, 20)
    ax.set_yticks([0, 9, 18])
    ax.set_xticks(ticks, labels, linespacing=1.0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Trained histories solved", labelpad=0.8)
    fs.clean(ax)
    n_runs, n_learn = len(v), int(v["learned"].sum())
    n_eff = int(v["lesion_effect"].astype(str).str.lower().eq("true").sum())
    use = float(v["trained_net_use"].min())
    cv.key(x0 + 9.0, PLOT_Y + PLOT_H - 1.2, [([dict(kind="mark", s=8, fc=TARGET, ec=TARGET)], "healthy"),
                                             ([dict(kind="mark", s=8, fc="white", ec=TARGET)], "pan-neuronal Kv4 lesion")], size=fs.PT_SMALL, gap=1.6)
    sd.add("G", "summary", runs=n_runs, learned=n_learn, lesion_effect=n_eff, trained_net_use_min=use)


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("fig3")
    for rel in ("plan_v2/fig3_set_wiring.csv", "plan_v2/fig3_set_groups.csv", "plan_v2/fig3_ensemble_edges.csv", "figure_data/fd_cells.csv",
                "plan_v2/fig3_band_sets.csv", "plan_v2/fig3_ensemble_sets.csv", "plan_v2/fig3_removals.csv",
                "plan_v2/fig3_cue_energy.csv", "plan_v2/fig3_cue_cells.csv", "plan_v2/fig3_rule_cells.csv", "plan_v2/fig3_rules.csv",
                "plan_v2/fig3_structure.csv", "plan_v2/tests.csv", "plan_v2/names.csv"):
        sd.source(cm.RESULTS / rel)
    for f in sorted(TILES.glob("*.png")) + sorted(TILES.glob("*.json")) + sorted(MAPS_DIR.glob("*.png")) + sorted(MAPS_DIR.glob("*.json")):
        sd.source(f)
    panel_band(cv, sd)
    panel_matrix(cv, sd, XS[0], ROW_B, fs.FULL, B_H)
    panel_sets(cv, sd, X3[0], ROW2, W3, ROW_H, "C", "x6", "Silencing it: no worse", "20 matched sets", "F13.X6.ensemble")
    panel_sets(cv, sd, X3[1], ROW2, W3, ROW_H, "D", "x7", "Cutting its synapses: no worse", "matched synapses", "F13.X7.edges")
    panel_populations(cv, sd, X3[2], ROW2, W3, ROW_H)
    panel_circuits(cv, sd, XS[0], fs.unit(2))
    panel_structure(cv, sd, XS[2], fs.unit(2))
    cm.finish(cv, "fig3", sd)


if __name__ == "__main__":
    main()
