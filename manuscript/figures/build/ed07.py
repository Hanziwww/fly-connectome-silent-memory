"""Extended Data Fig. 7 - Single candidate cell groups are not needed for the learned behaviour or the memory (180 x 166 mm).

Band  a  the six candidate groups (top; three selected on learning data, three on memory data, each one cell type) and their
         matched control sets (bottom; matched on log(1 + total degree) and spike probability, with the same number of
         readout cells), every cell's skeleton on the unfolded CNS at one shared view, coloured by superclass group
Row 1 b  the two matching covariates of each candidate group and its matched set
      c  suppressing each group's outgoing propagation (strength 0.25, 0.5, 1 x write, hold, read window): histories solved with the
         candidate minus the matched group suppressed; learning groups on the four T1 maps, memory groups in M0 and the shared
         readout (discovery), S1 (confirmation: 27 McNemar tests with Holm) and S2 (scope)
      d  the MNad18/27 group's route to the choice: change in |first-decision margin| when its readout route is replaced
         (strength 1, all decisions), candidate minus matched (Hodges-Lehmann, 95% CI; Wilcoxon signed-rank with Holm)
Row 2 e  structural paths from each group to the readout cells (M0 readout-only model): |direct| and |two-hop| summed weights,
         readout cells reached in one hop, and the net sign of the two-hop paths
      f  recorded direct flux into readout cells over 400 ms for the memory groups, after cue 0 and cue 1
      g  histories lost and gained over all suppression conditions, candidate vs matched

  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\ed07.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import NullFormatter
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import boards as bd, figstyle as fs, palette as pal, qc  # noqa: E402

U = (fs.FULL - 3 * fs.GAP) / 4
XS = [k * (U + fs.GAP) for k in range(4)]
TILES = cm.RENDERS / "ed07"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
TW, TH = META["tile_mm"]
BAND_Y = 6.6
KEY_Y = BAND_Y + 2 * TH + fs.GUTTER + fs.GUTTER
KEY_H = 3.4
ROW1 = KEY_Y + KEY_H + 2.8
ROW_H = 41.0
ROW2 = ROW1 + ROW_H + 2.0
H = ROW2 + ROW_H
AX_TOP, AX_H = 6.5, 27.0
WHITE = fs.ON_DARK
GROUPS = META["groups"]
LEARN, MEM = GROUPS[:3], GROUPS[3:]
CMAP = colormaps["BrBG"]
NORM = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
CAND, MATCH = fs.TEXT, "white"
STRENGTHS = [0.25, 0.5, 1.0]
WINDOWS = ["write", "hold", "read"]
C_UNITS = {"learning": [("learning_confirmation_T1", "difficulty-D1-T1", "D1"), ("learning_confirmation_T1", "difficulty-D3-T1", "D3"),
                        ("learning_confirmation_T1", "difficulty-D5-T1", "D5"), ("learning_confirmation_T1", "difficulty-D7-T1", "D7")],
           "memory": [("memory_discovery_M0", "M0", "M0"), ("memory_discovery_M0", "M0vis.hidden", "hidden"),
                      ("memory_discovery_M0", "M0vis.visible", "visible"), ("memory_confirmation_S1", "S1", "S1"), ("memory_scope_S2", "S2", "S2")]}
ROUTE_UNITS = [("M0", "M0"), ("M0joint", "joint"), ("M0vis.hidden", "hidden"), ("M0vis.visible", "visible"), ("S1", "S1"), ("S2", "S2")]


def tile_text(cv, x, y, s, size=fs.PT_SMALL, **kw):
    t = cv.text(x, y, s, size=size, color=WHITE, **kw)
    t.set_path_effects(fs.dark_halo(0.8))
    return t


def scale_bar(cv, right, y, length_um, um_per_px, px_per_mm):
    length = length_um / um_per_px / px_per_mm
    cv.line([right - length, right], [y, y], WHITE, lw=1.0, solid_capstyle="butt")
    tile_text(cv, right, y - 0.5, f"{length_um} µm", ha="right", va="bottom")


def group_labels():
    a = cm.table("extended/ed07/ed07a_group_selection.csv").set_index("group")
    assert list(a.index) == GROUPS or set(a.index) == set(GROUPS)
    lab = {}
    for g in GROUPS:
        types = str(a.loc[g, "type_labels"]).replace(";", ",").split(",")
        lab[g] = types[0] if len(types) == 1 else "/".join([types[0]] + [t[len(os.path.commonprefix(types)):] for t in types[1:]])
    return lab, a


GLAB, SELECTION = group_labels()


def mark_pair(ax, x_cand, x_match, y, ms=2.6, dy=0.13, colour=fs.TEXT, floor=None):
    for x, fc, off in ((x_cand, colour, -dy), (x_match, "white", dy)):
        if x is None:
            continue
        ax.plot([x if floor is None else max(x, floor)], [y + off], marker="o", ms=ms, mfc=fc, mec=colour, mew=0.5, ls="none", clip_on=False,
                zorder=3)


def side_key(cv, x, y, extra=()):
    return cv.key(x, y, [([dict(kind="mark", s=6, fc=CAND, ec=CAND)], "candidate"),
                         ([dict(kind="mark", s=6, fc=MATCH, ec=fs.TEXT, lw=0.5)], "matched"), *extra], gap=1.6)


def group_axis(ax, groups, labels=True):
    ax.set_yticks(range(len(groups)), [GLAB[g] for g in groups] if labels else [""] * len(groups))
    ax.set_ylim(len(groups) - 0.45, -0.55)
    ax.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL)
    if len(groups) == 6:
        ax.axhline(2.5, color=fs.GRID, lw=fs.LW_HAIR, zorder=0)


# ---------------------------------------------------------------- a: skeleton band

def panel_a(cv, sd):
    cv.letter(0, 0, "a")
    sel = cm.table("figure_data/fd_neuron_selection.csv")
    heads = (("learning groups: selected on T0 maps, tested on T1 maps", 0), ("memory groups: selected on M0 histories, tested in S1 and S2", 3))
    for lab, c0 in heads:
        x0, x1 = c0 * (TW + fs.GUTTER), (c0 + 3) * (TW + fs.GUTTER) - fs.GUTTER
        cv.text((x0 + x1) / 2, 1.3, lab, size=fs.PT_SMALL, ha="center", va="center")
        cv.line([x0 + (4.0 if c0 == 0 else 0.4), x1 - 0.4], [2.6, 2.6], fs.TEXT, lw=0.35)
    by_name = {t["name"]: t for t in META["tiles"]}
    for c, g in enumerate(GROUPS):
        x = c * (TW + fs.GUTTER)
        n = int(SELECTION.loc[g, "cells"])
        cv.text(x + TW / 2, 4.7, f"{GLAB[g]}, {n} cells", size=fs.PT_SMALL, ha="center", va="center")
        for r, side in enumerate(("candidate", "matched")):
            t = by_name[f"{g}_{side}"]
            assert t["cells"] == n == int((sel["label"] == g).sum() / 2)
            y = BAND_Y + r * (TH + fs.GUTTER)
            img = np.asarray(Image.open(TILES / f"{t['name']}.png").convert("RGB"))
            cv.tile(x, y, img, width=TW, height=TH, group="ed07band", label=t["name"])
            if c == 0:
                tile_text(cv, x + 1.0, y + 15.0, side, size=fs.PT, va="center")
            tile_text(cv, x + 1.0, y + TH - 1.0, "1 type" if t["n_types"] == 1 else f"{t['n_types']} types", va="bottom")
            sd.add("a", "tile", group=g, side=side, cells=t["cells"], somata=t["somata"], types=";".join(t["types"]),
                   superclass_groups=";".join(t["superclass_groups"]), body_ids=";".join(map(str, t["body_ids"])), um_per_px=t["scale_um_per_px"])
    # orientation and scale in the last tile (all twelve tiles share one view)
    t = META["tiles"][-1]
    x, y = 5 * (TW + fs.GUTTER), BAND_Y + TH + fs.GUTTER
    ox, oy = x + TW - 7.4, y + TH - 8.6
    cv.line([ox, ox + 2.4], [oy, oy], WHITE, lw=0.5)
    tile_text(cv, ox + 2.8, oy, "L", va="center")
    cv.line([ox, ox], [oy, oy + 2.4], WHITE, lw=0.5)
    tile_text(cv, ox, oy + 2.8, "V", ha="center", va="top")
    img_w = Image.open(TILES / f"{t['name']}.png").size[0]
    scale_bar(cv, x + TW - 1.2, y + TH - 1.4, 100, t["scale_um_per_px"], img_w / TW)
    # superclass key on a black strip under the band
    strip = np.zeros((int(round(KEY_H * fs.MM * fs.DPI)), int(round(fs.FULL * fs.MM * fs.DPI)), 3), np.uint8)
    cv.tile(0, KEY_Y, strip, width=fs.FULL, height=KEY_H, group="ed07band", label="key")
    present = []
    for tt in META["tiles"]:
        present += [s for s in tt["superclass_groups"] if s not in present]
    order = [k for k in pal.ATLAS_ON_DARK if k in present]
    tile_text(cv, 1.2, KEY_Y + KEY_H / 2, "skeleton colour:", va="center")
    cv.key(1.2 + fs.text_width("skeleton colour:", fs.PT_SMALL) + 1.6, KEY_Y + KEY_H / 2,
           [([dict(kind="mark", s=6, fc=pal.ATLAS_ON_DARK[k], ec=pal.ATLAS_ON_DARK[k])], k) for k in order], gap=2.4, colour=WHITE,
           effects=fs.dark_halo(0.8))
    sd.add("a", "view", um_per_px=t["scale_um_per_px"], scale_bar_um=100, superclass_groups=";".join(order),
           unfold="brain frontal; VNC rotated 90 degrees about the neck (msa/cnsview.py)",
           skeletons="MaleCNS v1.0 precomputed skeletons (CC BY 4.0)")


# ---------------------------------------------------------------- b: matching

def panel_b(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "b")
    m = cm.table("extended/ed07/ed07b_group_matching.csv")
    side_key(cv, x0 + 3.6, y0 + 1.5)
    lab_w, gap = 11.6, 2.6
    aw = (w - lab_w - 0.6 - gap) / 2
    specs = [("log1p_total_degree", "log(1 + degree)", 1.0, (3.5, 7.2), [4, 5, 6, 7]), ("spike_probability", "Spike prob. (%)", 100.0, (0, 1.0),
                                                                                       [0, 0.5, 1.0])]
    for k, (cov, lab, scale, lim, ticks) in enumerate(specs):
        ax = cv.axes(x0 + lab_w + k * (aw + gap), y0 + AX_TOP, aw, AX_H)
        for i, g in enumerate(GROUPS):
            r = m[(m["group"] == g) & (m["covariate"] == cov)].iloc[0]
            mark_pair(ax, scale * float(r["candidate_mean"]), scale * float(r["matched_mean"]), i)
            sd.add("b", "covariate", group=g, covariate=cov, candidate_mean=float(r["candidate_mean"]), matched_mean=float(r["matched_mean"]),
                   match_mean_distance=float(r["match_mean_distance"]), observed_membership_exact=bool(r["observed_membership_exact"]))
        group_axis(ax, GROUPS, labels=k == 0)
        ax.set_xlim(*lim)
        ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
        fs.clean(ax)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(lab, labelpad=1)


# ---------------------------------------------------------------- c: suppression, candidate vs matched

def panel_c(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    e = cm.table("main/fig6/fig6e_candidate_groups.csv")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    cv.text(x0 + 3.6, y0 + 1.5, "solved with the candidate minus with the matched group suppressed (share of histories)", size=fs.PT_SMALL,
            va="center")
    lab_w, mid = 9.6, 2.4
    n_blocks = len(C_UNITS["learning"]) + len(C_UNITS["memory"])
    B = (w - 2 * lab_w - mid) / n_blocks
    cell = (B - 0.7) / 3
    top = 8.4
    ax = bd.mm_axes(cv, x0, y0, w, h)
    n_tests = n_sig = 0
    starts = {"learning": lab_w, "memory": lab_w + len(C_UNITS["learning"]) * B + mid + lab_w}
    for line, groups in (("learning", LEARN), ("memory", MEM)):
        xs0 = starts[line]
        for i, g in enumerate(groups):
            ax.text(xs0 - 0.6, top + (i + 0.5) * B - 0.35, GLAB[g], ha="right", va="center", fontsize=fs.PT_SMALL)
        for j, (st, unit, ulab) in enumerate(C_UNITS[line]):
            bx = xs0 + j * B
            ax.text(bx + 1.5 * cell, top - 0.5, ulab, ha="center", va="bottom", fontsize=fs.PT_SMALL)
            for i, g in enumerate(groups):
                by = top + i * B
                d = e[(e["set"] == st) & (e["unit"] == unit) & (e["group"] == g)]
                assert len(d) == 18, (st, unit, g, len(d))
                for si, s in enumerate(STRENGTHS):
                    for wi, win in enumerate(WINDOWS):
                        q = d[np.isclose(d["strength"], s) & (d["window"] == win)].set_index("side")
                        hist = int(q.loc["candidate", "histories"])
                        diff = (int(q.loc["candidate", "solved"]) - int(q.loc["matched", "solved"])) / hist
                        ax.add_patch(bd.Rectangle((bx + wi * cell, by + si * cell), cell - 0.12, cell - 0.12, facecolor=CMAP(NORM(diff)),
                                                  edgecolor="none", linewidth=0.5))
                        tid, p = None, None
                        if st == "memory_confirmation_S1":
                            tid = f"fig6e.mcnemar.S1.{g}.{s}.{win}"
                            p = float(tests.loc[tid, "p_adjusted"])
                            n_tests += 1
                            if p < 0.05:
                                n_sig += 1
                                ax.add_patch(bd.Rectangle((bx + wi * cell - 0.05, by + si * cell - 0.05), cell - 0.02, cell - 0.02, facecolor="none",
                                                          edgecolor=fs.TEXT, linewidth=0.7))
                        sd.add("c", "condition", set=st, unit=unit, group=g, strength=s, window=win, histories=hist,
                               candidate_solved=int(q.loc["candidate", "solved"]), matched_solved=int(q.loc["matched", "solved"]),
                               baseline_solved=int(q.loc["candidate", "baseline_solved"]), test_id=tid, p_holm=p)
    # section headers
    lx0 = starts["learning"]
    ax.plot([lx0 + 0.3, lx0 + 4 * B - 1.0], [4.6, 4.6], color=fs.TEXT, lw=0.35)
    ax.text(lx0 + 2 * B - 0.35, 4.3, "confirmation, T1 maps", ha="center", va="bottom", fontsize=fs.PT_SMALL)
    mx0 = starts["memory"]
    for a_, b_, lab in ((0, 3, "discovery"), (3, 4, "confirm."), (4, 5, "scope")):
        ax.plot([mx0 + a_ * B + 0.3, mx0 + b_ * B - 1.0], [4.6, 4.6], color=fs.TEXT, lw=0.35)
        ax.text(mx0 + (a_ + b_) / 2 * B - 0.35, 4.3, lab, ha="center", va="bottom", fontsize=fs.PT_SMALL)
    assert n_tests == 27
    yb = top + 3 * B + 1.6
    cv.text(x0 + lab_w, y0 + yb + 0.4, "each block: strength 0.25, 0.5, 1 (rows) × write, hold, read window (columns)", size=fs.PT_SMALL,
            va="center")
    cv.text(x0 + lab_w, y0 + yb + 2.8, f"S1: {n_tests} McNemar tests, {n_sig or 'none'} with Holm {fs.P_SYM} < 0.05 (outlined)", size=fs.PT_SMALL,
            va="center")
    k_ax = cv.axes(x0 + w - 24.0, y0 + yb + 4.6, 23.4, 1.1)
    vals = np.linspace(-1, 1, 128)
    k_ax.imshow(CMAP(NORM(vals))[None, :, :3], aspect="auto", extent=(-1, 1, 0, 1), interpolation="bilinear")
    k_ax.set_yticks([])
    k_ax.set_xticks([-1, 0, 1], [fs.MINUS + "1", "0", "1"])
    k_ax.tick_params(axis="x", length=1.0, pad=0.4, labelsize=fs.PT_SMALL)
    for s_ in ("top", "right", "left"):
        k_ax.spines[s_].set_visible(False)


# ---------------------------------------------------------------- d: routes of the MNad18/27 group

def panel_d(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "d")
    e = cm.table("main/fig6/fig6e_candidate_groups.csv")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    r = e[e["set"].str.startswith("route")]
    assert set(r["group"]) == {"memory-g8561"}
    net = r[r["set"] == "route_network"]["margin_mean_change"].abs().max()
    ro = r[r["set"] == "route_readout"].set_index(["unit", "side", "strength", "window"])["margin_mean_change"]
    bo = r[r["set"] == "route_both"].set_index(["unit", "side", "strength", "window"])["margin_mean_change"]
    same = (ro - bo.reindex(ro.index)).abs().max()
    assert net < 1e-4 and same < 1e-5, (net, same)
    cv.text(x0 + 3.6, y0 + 1.5, f"{GLAB['memory-g8561']}: readout route replaced", size=fs.PT_SMALL, va="center")
    cv.text(x0 + 3.6, y0 + 3.9, "network route: |change| < 10$^{−4}$", size=fs.PT_SMALL, va="center")
    lab_w, p_w = 8.4, 7.0
    ax = cv.axes(x0 + lab_w, y0 + AX_TOP, w - lab_w - p_w, AX_H)
    for i, (u, ulab) in enumerate(ROUTE_UNITS):
        t = tests.loc[f"fig6e.wilcoxon_routes.{u}"]
        est, lo, hi, p = float(t["effect"]), float(t["effect_ci_low"]), float(t["effect_ci_high"]), float(t["p_adjusted"])
        ax.plot([lo, hi], [i, i], color=fs.TEXT, lw=0.6, solid_capstyle="butt")
        ax.plot([est], [i], marker="o", ms=2.6, mfc=fs.TEXT, mec=fs.TEXT, ls="none")
        cv.text(x0 + w - p_w + 1.2, y0 + AX_TOP + (i + 0.5) * AX_H / len(ROUTE_UNITS), fs.num(p, 2) if p < 1 else "1", size=fs.PT_SMALL,
                va="center")
        sd.add("d", "route test", unit=u, test_id=f"fig6e.wilcoxon_routes.{u}", n=int(t["n_used"]), hodges_lehmann=est, ci_low=lo, ci_high=hi,
               p=float(t["p_value"]), p_holm=p, network_route_max_abs_margin_change=float(net), readout_vs_both_max_difference=float(same))
    cv.text(x0 + w - p_w + 1.2, y0 + AX_TOP - 1.2, f"Holm {fs.P_SYM}", size=fs.PT_SMALL, va="center")
    ax.axvline(0, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=0)
    ax.set_yticks(range(len(ROUTE_UNITS)), [lab for _, lab in ROUTE_UNITS])
    ax.set_ylim(len(ROUTE_UNITS) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL)
    ax.set_xlim(-0.05, 0.2)
    ax.set_xticks([0, 0.1, 0.2], ["0", "0.1", "0.2"])
    fs.clean(ax)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Δ|margin|, cand. − matched", labelpad=1)


# ---------------------------------------------------------------- e: structural paths to the readout

def panel_e(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "e")
    c = cm.table("extended/ed07/ed07c_signed_pathways.csv")
    p = c[c["record"] == "signed_structural_paths"]
    assert set(p["model"]) == {"remote-dynamic-M0-head_only"}
    side_key(cv, x0 + 3.6, y0 + 1.5)
    cv.text(x0 + 3.6, y0 + 3.9, "paths into the 4,096 readout cells, M0 readout-only model", size=fs.PT_SMALL, va="center")
    lab_w, gap = 11.6, 3.0
    specs = [("direct_absolute", "|Direct| weight", "log", (0.1, 3e3), [0.1, 1, 100]),
             ("two_hop_absolute", "|Two-hop| weight", "log", (100, 1e6), [100, 1e3, 1e5]),
             ("observed_reached_one_hop", "Cells reached, 1 hop", "log", (1, 1e3), [1, 10, 100, 1000]),
             ("two_hop_share", "Two-hop net sign", "lin", (-1, 1), [-1, 0, 1])]
    aw = (w - lab_w - 0.6 - (len(specs) - 1) * gap) / len(specs)
    for k, (col, lab, scale, lim, ticks) in enumerate(specs):
        ax = cv.axes(x0 + lab_w + k * (aw + gap), y0 + AX_TOP, aw, AX_H)
        floor = lim[0] * 1.0 if scale == "log" else None
        for i, g in enumerate(GROUPS):
            vals = []
            for side in ("indices", "controls"):
                r = p[(p["group"] == g) & (p["side"] == side)].iloc[0]
                if col == "two_hop_share":
                    v = float(r["two_hop_signed"]) / float(r["two_hop_absolute"]) if r["two_hop_absolute"] > 0 else None
                else:
                    v = float(r[col])
                vals.append(v)
                if k == 0:
                    sd.add("e", "paths", group=g, side="candidate" if side == "indices" else "matched", direct_signed=float(r["direct_signed"]),
                           direct_absolute=float(r["direct_absolute"]), two_hop_signed=float(r["two_hop_signed"]),
                           two_hop_absolute=float(r["two_hop_absolute"]), readout_cells_one_hop=float(r["observed_reached_one_hop"]),
                           readout_cells_two_hop=float(r["observed_reached_two_hop"]))
            mark_pair(ax, vals[0], vals[1], i, floor=floor)
        group_axis(ax, GROUPS, labels=k == 0)
        if scale == "log":
            ax.set_xscale("log")
            ax.set_xlim(lim[0] / 1.6, lim[1])
            ax.set_xticks(ticks, ["0"] + [f"{d:g}" if d < 100 else f"10$^{{{int(round(np.log10(d)))}}}$" for d in ticks[1:]])
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_xlim(-1.05, 1.05)
            ax.set_xticks([-1, 0, 1], [fs.MINUS + "1", "0", "1"])
            ax.axvline(0, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=0)
        fs.clean(ax)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(lab, labelpad=1)


# ---------------------------------------------------------------- f: recorded direct flux

def panel_f(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "f")
    c = cm.table("extended/ed07/ed07c_signed_pathways.csv")
    f = c[c["record"] == "observed_direct_flux"].copy()
    ms = sorted(set(f["ms"].astype(int)))
    assert ms == [400]
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="mark", s=6, fc=pal.CUE[k], ec=pal.CUE[k])], f"cue {k}") for k in (0, 1)] +
           [([dict(kind="mark", s=6, fc="white", ec=fs.TEXT, lw=0.5)], "matched")], gap=1.6)
    cv.text(x0 + 3.6, y0 + 3.9, f"into readout cells, {ms[0]} ms after the cue", size=fs.PT_SMALL, va="center")
    lab_w = 11.6
    ax = cv.axes(x0 + lab_w, y0 + AX_TOP, w - lab_w - 0.6, AX_H)
    floor = 0.1
    for i, g in enumerate(MEM):
        for k in (0, 1):
            vals = []
            for side in ("indices", "controls"):
                r = f[(f["group"] == f"{g}-{side}") & (f["cue"].astype(int) == k)].iloc[0]
                vals.append(float(r["abs_aggregate_sum"]))
                sd.add("f", "flux", group=g, side="candidate" if side == "indices" else "matched", cue=k, ms=int(r["ms"]),
                       signed_sum=float(r["signed_sum"]), abs_aggregate_sum=float(r["abs_aggregate_sum"]), peak_positive=float(r["peak_positive"]),
                       peak_negative=float(r["peak_negative"]), nonzero_timepoints=int(r["nonzero_timepoints"]))
            yy = i + (k - 0.5) * 0.36
            for v, fc, dy in ((vals[0], pal.CUE[k], -0.08), (vals[1], "white", 0.08)):
                ax.plot([max(v, floor)], [yy + dy], marker="o", ms=2.6, mfc=fc, mec=pal.CUE[k], mew=0.5, ls="none", clip_on=False, zorder=3)
    ax.set_yticks(range(len(MEM)), [GLAB[g] for g in MEM])
    ax.set_ylim(len(MEM) - 0.45, -0.55)
    ax.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL)
    ax.set_xscale("log")
    ax.set_xlim(floor / 1.6, 1e3)
    ax.set_xticks([floor, 1, 10, 100, 1000], ["0", "1", "10", "10$^{2}$", "10$^{3}$"])
    fs.clean(ax)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("|Direct flux| (summed)", labelpad=1)


# ---------------------------------------------------------------- g: lost and gained, pooled

def panel_g(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "g")
    e = cm.table("main/fig6/fig6e_candidate_groups.csv")
    sets = [("learning_confirmation_T1", "T1 maps"), ("memory_discovery_M0", "M0 discovery"), ("memory_confirmation_S1", "S1"),
            ("memory_scope_S2", "S2")]
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="box", fc=fs.TEXT)], "candidate"), ([dict(kind="box", fc="#BDBDBD")], "matched")], gap=1.6)
    cv.text(x0 + 3.6, y0 + 3.9, "all suppression conditions pooled", size=fs.PT_SMALL, va="center")
    lab_w = 15.4
    ax = cv.axes(x0 + lab_w, y0 + AX_TOP, w - lab_w - 0.6, AX_H)
    bh = 0.32
    for i, (st, lab) in enumerate(sets):
        d = e[e["set"] == st]
        for k, (side, col) in enumerate((("candidate", fs.TEXT), ("matched", "#BDBDBD"))):
            q = d[d["side"] == side]
            lost, gained, n = int(q["lost"].sum()), int(q["gained"].sum()), int(q["histories"].sum())
            y = i + (k - 0.5) * (bh + 0.04)
            ax.barh([y], [-lost], height=bh, color=col, edgecolor=fs.TEXT if side == "matched" else "none", linewidth=0.3)
            ax.barh([y], [gained], height=bh, color=col, edgecolor=fs.TEXT if side == "matched" else "none", linewidth=0.3)
            sd.add("g", "pooled", set=st, side=side, conditions=int(len(q)), condition_histories=n, lost=lost, gained=gained)
    ax.set_yticks(range(len(sets)), [f"{lab} ({int(e[(e['set'] == st) & (e['side'] == 'candidate')]['histories'].sum()):,})" for st, lab in sets])
    ax.set_ylim(len(sets) - 0.45, -0.55)
    ax.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL)
    ax.axvline(0, color=fs.AXIS, lw=fs.LW_THIN, zorder=0)
    ax.set_xlim(-36, 16)
    ax.set_xticks([-30, -20, -10, 0, 10], ["30", "20", "10", "0", "10"])
    fs.clean(ax)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Histories lost | gained", labelpad=1)


def cm_u(n):
    return n * U + (n - 1) * fs.GAP


def main():
    cv = fs.Canvas(fs.FULL, H, ppi_min=300)
    sd = qc.SourceData("ed07")
    for rel in ("extended/ed07/ed07a_group_selection.csv", "extended/ed07/ed07b_group_matching.csv", "extended/ed07/ed07c_signed_pathways.csv",
                "main/fig6/fig6e_candidate_groups.csv", "figure_data/fd_neuron_selection.csv", "figure_data/fd_skeletons.csv",
                "stats/statistical_tests.csv"):
        sd.source(cm.RESULTS / rel)
    panel_a(cv, sd)
    panel_b(cv, sd, XS[0], ROW1, cm_u(1), ROW_H)
    panel_c(cv, sd, XS[1], ROW1, cm_u(2), ROW_H)
    panel_d(cv, sd, XS[3], ROW1, cm_u(1), ROW_H)
    panel_e(cv, sd, XS[0], ROW2, cm_u(2), ROW_H)
    panel_f(cv, sd, XS[2], ROW2, cm_u(1), ROW_H)
    panel_g(cv, sd, XS[3], ROW2, cm_u(1), ROW_H)
    cm.finish(cv, "ed07", sd, kind="extended")


if __name__ == "__main__":
    main()
