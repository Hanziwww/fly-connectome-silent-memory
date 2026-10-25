"""Extended Data Fig. 4 - Where and how consistently the connectome parameters change (180 x 145 mm).

a  whole-CNS maps of the mean |change| of each cell's parameter group (mean over 24 maps, fraction of the parameter range) for
   output gain, input gain and threshold (shared log scale; grey: unchanged)
b  mean |change| by superclass (groups per superclass in brackets)
c  size of the change on each map vs the certified optimal moves (L2 norm; Spearman, Holm)
d  consistency through the curriculum: mean pairwise Spearman between maps and within-map agreement with the final change
e  mean |change| by membership, group size and transmitter
f  enrichment of the 1% most consistently changed groups by superclass and transmitter (observed vs expected; none significant
   after BH, family F3e.enrichment)
g  direction of change of those groups: maps (of 24) in which each parameter increased

  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\ed04.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import boards as bd, figstyle as fs, palette as pal, qc  # noqa: E402

U = (fs.FULL - 3 * fs.GAP) / 4
XS = [k * (U + fs.GAP) for k in range(4)]
TILES = cm.RENDERS / "ed04"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
BAND_H = META["band_mm"]
ROW2, ROW3, ROW_H = BAND_H + 4.0, BAND_H + 48.0, 41.0
H = ROW3 + ROW_H
AX_TOP, AX_H = 6.5, 27.0
PARAMS = [("post_log_gain", "output gain"), ("pre_log_gain", "input gain"), ("threshold_mv", "threshold")]
PCOL = {"post_log_gain": "#1A1A1A", "pre_log_gain": "#9A9A9A", "threshold_mv": pal.ACCENT}
WHITE = fs.ON_DARK
EDGE = "#ABABAB"


def tile_text(cv, x, y, s, size=fs.PT_SMALL, **kw):
    t = cv.text(x, y, s, size=size, color=WHITE, **kw)
    t.set_path_effects(fs.dark_halo(0.8))
    return t


def pkey(cv, x, y, **kw):
    return cv.key(x, y, [([dict(kind="mark", s=6, fc=PCOL[p], ec=PCOL[p])], lab) for p, lab in PARAMS], gap=1.6, **kw)


# ---------------------------------------------------------------- a: maps

def panel_a(cv, sd):
    mw, bh = META["map_mm"]
    for k, t in enumerate(META["tiles"]):
        x = k * (mw + fs.GUTTER)
        img = np.asarray(Image.open(TILES / f"{t['name']}.png").convert("RGB"))
        cv.tile(x, 0, img, width=mw, height=bh, group="ed04band", label=t["name"])
        if k == 0:
            cv.letter(x + 1.2, 0.9, "a")
            cv.fig.texts[-1].set_color(WHITE)
        tile_text(cv, x + mw - 1.2, 30.0, t["label"], size=fs.PT, ha="right", va="top")
        tile_text(cv, x + 1.4, bh - 8.0, f"{100 * t['zero_cells'] / t['cells']:.0f}% of cells" + "\n" + "unchanged", va="top", linespacing=1.05)
        sd.add("a", "map", parameter=t["parameter"], cells=t["cells"], unchanged_cells=t["zero_cells"], median_cell_value=t["median_cell_value"],
               lo=t["lo"], hi=t["hi"], um_per_px=t["scale_um_per_px"])
    # colour key in the black space of the first tile (left of the VNC)
    t0 = META["tiles"][0]
    kx, ky, kh = 1.6, 27.0, 14.0
    gax = cv.axes(kx, ky, 1.4, kh)
    lo, hi, cl = t0["lo"], t0["hi"], t0["cmap_lo"]
    vals = np.logspace(np.log10(hi), np.log10(lo), 256)
    tt = (np.log10(vals) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    gax.imshow(pal.MAGNITUDE_ON_DARK(cl + (1 - cl) * tt)[:, None, :3] * np.ones((1, 2, 1)), aspect="auto", extent=(0, 1, 0, 1),
               interpolation="bilinear")
    gax.set_axis_off()
    for v, lab in ((1e-2, "1%"), (1e-3, "0.1%"), (1e-4, "0.01%")):
        yv = ky + (np.log10(hi) - np.log10(v)) / (np.log10(hi) - np.log10(lo)) * kh
        cv.line([kx + 1.4, kx + 2.0], [yv, yv], WHITE, lw=0.4)
        tile_text(cv, kx + 2.4, yv, lab, va="center")
    tile_text(cv, kx, ky - 1.0, "mean |change|" + "\n" + "(of range)", va="bottom", linespacing=1.05)
    # orientation and scale in the last map (the three maps share one view)
    t = META["tiles"][-1]
    assert len({round(tt["scale_um_per_px"], 9) for tt in META["tiles"]}) == 1
    x = (len(META["tiles"]) - 1) * (mw + fs.GUTTER)
    px_per_mm = Image.open(TILES / f"{t['name']}.png").size[0] / mw
    ox, oy = x + mw - 12.6, bh - 6.4
    cv.line([ox, ox + 2.8], [oy, oy], WHITE, lw=0.5)
    tile_text(cv, ox + 3.2, oy, "L", va="center")
    cv.line([ox, ox], [oy, oy + 2.8], WHITE, lw=0.5)
    tile_text(cv, ox, oy + 3.2, "V", ha="center", va="top")
    length = 100 / t["scale_um_per_px"] / px_per_mm
    cv.line([x + mw - 1.4 - length, x + mw - 1.4], [bh - 1.6, bh - 1.6], WHITE, lw=1.0, solid_capstyle="butt")
    tile_text(cv, x + mw - 1.4, bh - 2.1, "100 µm", ha="right", va="bottom")
    sd.add("a", "view", um_per_px=t["scale_um_per_px"], scale_bar_um=100,
           unfold="brain frontal; VNC rotated 90 degrees about the neck (msa/cnsview.py)")


# ---------------------------------------------------------------- b: superclass strata

def panel_b(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "b")
    b = cm.table("extended/ed04/ed04b_parameter_change_anatomy.csv")
    s = b[b["stratum_type"] == "superclass"]
    n = s[s["parameter"] == "post_log_gain"].set_index("stratum")["n_groups"]
    keep = [k for k in n.index if n[k] >= 20]
    order = sorted(keep, key=lambda k: -float(s[(s["stratum"] == k) & (s["parameter"] == "post_log_gain")]["mean_normalized_abs_change"].iloc[0]))
    pkey(cv, x0 + 3.6, y0 + 1.5)
    lab_w = 23.0
    ax = cv.axes(x0 + lab_w, y0 + 5.2, w - lab_w - 0.6, h - 12.4)
    floor = 6e-5
    names = {"cb_intrinsic": "central brain intrinsic", "vnc_intrinsic": "VNC intrinsic", "ascending_neuron": "ascending",
             "descending_neuron": "descending", "visual_projection": "visual projection", "ol_intrinsic": "optic lobe intrinsic",
             "vnc_sensory": "VNC sensory", "cb_sensory": "central brain sensory", "vnc_motor": "VNC motor", "visual_centrifugal": "visual centrifugal",
             "cb_motor": "central brain motor", "vnc_efferent": "VNC efferent", "sensory_ascending": "sensory ascending"}
    for i, k in enumerate(order):
        for j, (p, _) in enumerate(PARAMS):
            v = float(s[(s["stratum"] == k) & (s["parameter"] == p)]["mean_normalized_abs_change"].iloc[0])
            ax.plot([max(v, floor)], [i + (j - 1) * 0.18], marker="o", ms=2.4, mfc=PCOL[p] if v > 0 else "white", mec=PCOL[p], mew=0.5, ls="none",
                    clip_on=False)
            sd.add("b", "superclass", superclass=k, parameter=p, mean_abs_change=v, groups=int(n[k]))
    ax.set_yticks(range(len(order)), [f"{names.get(k, k)} ({int(n[k]):,})" for k in order])
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xscale("log")
    ax.set_xlim(floor / 1.3, 2e-2)
    ax.set_xticks([floor, 1e-3, 1e-2], ["0", "0.1%", "1%"])
    fs.minor(ax, y=False)
    ax.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL)
    fs.clean(ax)
    ax.set_xlabel("Mean |change| (of range)", labelpad=1)


# ---------------------------------------------------------------- c: change vs optimum

def panel_c(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    a = cm.table("extended/ed04/ed04a_parameter_change_consistency.csv")
    opt = cm.table("main/fig2/fig2c_cost_vs_difficulty.csv").set_index("run_id")["optimal_moves"]
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    pkey(cv, x0 + 3.6, y0 + 1.5)
    ax = cv.axes(x0 + 8.4, y0 + AX_TOP, w - 8.8, AX_H)
    lines = []
    for p, lab in PARAMS:
        d = a[(a["parameter"] == p) & (a["checkpoint"] == "final") & a["metric"].str.startswith("l2_change:")]
        runs = d["metric"].str.split(":").str[1]
        x = opt.reindex(runs).to_numpy(float)
        y = d["observed"].to_numpy(float)
        assert not np.isnan(x).any()
        ax.scatter(x, y, s=5, color=PCOL[p], linewidth=0, zorder=3)
        t = tests.loc[f"ed04a.spearman.l2_{p}~optimum"]
        lines.append(f"{lab} {fs.RHO} = {float(t['effect']):.2f}")
        for rr, xx, yy in zip(runs, x, y):
            sd.add("c", p, run_id=rr, optimal_moves=float(xx), l2_change=float(yy))
        sd.add("c", "test", parameter=p, spearman=float(t["effect"]), ci_low=float(t["effect_ci_low"]), ci_high=float(t["effect_ci_high"]),
               p_holm=float(t["p_adjusted"]), test_id=t.name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(7, 80)
    ax.set_xticks([10, 20, 40, 80], ["10", "20", "40", "80"])
    ax.set_yticks([0.1, 0.2, 0.5], ["0.1", "0.2", "0.5"])
    fs.minor(ax)
    fs.clean(ax)
    ax.set_xlabel("Certified optimal moves", labelpad=1)
    ax.set_ylabel("Size of change (L2)", labelpad=1)
    pmax = max(float(tests.loc[f"ed04a.spearman.l2_{p}~optimum", "p_adjusted"]) for p, _ in PARAMS)
    cv.text(x0 + 3.6, y0 + 3.9, f"{fs.RHO} = " + ", ".join(t.split(" ")[-1] for t in lines) + f"; each {fs.P_SYM} = {fs.fmt_p(pmax)} (Holm)", size=fs.PT_SMALL,
            va="center")


# ---------------------------------------------------------------- d: consistency through training

def panel_d(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "d")
    a = cm.table("extended/ed04/ed04a_parameter_change_consistency.csv")
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="hline", w=2.6, lw=0.9)], "between maps"), ([dict(kind="hline", w=2.6, lw=0.9, ls=(0, (2, 1.2)))],
                                                                                         "within map, vs final")], gap=1.6)
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="mark", s=6, marker="_", fc=fs.TEXT, lw=0.8)], "null mean (final)")])
    ax = cv.axes(x0 + 8.4, y0 + AX_TOP, w - 8.8, AX_H)
    for p, lab in PARAMS:
        d = a[(a["parameter"] == p) & (a["metric"] == "mean_pairwise_spearman")].sort_values("depth_fraction")
        ax.plot(d["depth_fraction"], d["observed"], color=PCOL[p], lw=0.9, marker="o", ms=2.0, mew=0, zorder=3)
        w_ = a[(a["parameter"] == p) & (a["metric"] == "mean_within_map_spearman_with_final")].sort_values("depth_fraction")
        ax.plot(list(w_["depth_fraction"]) + [1.0], list(w_["observed"]) + [1.0], color=PCOL[p], lw=0.7, ls=(0, (2, 1.2)), zorder=2)
        null = float(d[d["checkpoint"] == "final"]["null_mean"].iloc[0])
        ax.plot([1.04], [null], marker="_", ms=5, mew=0.9, color=PCOL[p], clip_on=False, zorder=4)
        for r in d.itertuples():
            sd.add("d", "between maps", parameter=p, depth_fraction=float(r.depth_fraction), spearman=float(r.observed))
        for r in w_.itertuples():
            sd.add("d", "within map", parameter=p, depth_fraction=float(r.depth_fraction), spearman_with_final=float(r.observed))
        sd.add("d", "null", parameter=p, null_mean=null)
    ax.set_xlim(0.2, 1.06)
    ax.set_xticks([0.25, 0.5, 0.75, 1.0], ["25", "50", "75", "100"])
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 1])
    fs.clean(ax)
    ax.set_xlabel("Curriculum depth reached (%)", labelpad=1)
    ax.set_ylabel(f"Spearman {fs.RHO}", labelpad=1)


# ---------------------------------------------------------------- e: other strata

E_GROUPS = [("membership", [("input", "input"), ("input_and_observed", "input + readout"), ("observed", "readout"), ("neither", "neither")],
             "membership"),
            ("size_class", [("1", "1"), ("2-3", "2–3"), ("4-10", "4–10"), ("11-100", "11–100"), (">100", ">100")], "cells per group"),
            ("transmitter", [("acetylcholine", "ACh"), ("gaba", "GABA"), ("glutamate", "Glu"), ("histamine", "HA"), ("octopamine", "OA"),
                             ("serotonin", "5-HT"), ("dopamine", "DA")], "transmitter")]


def panel_e(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "e")
    b = cm.table("extended/ed04/ed04b_parameter_change_anatomy.csv")
    pkey(cv, x0 + 3.6, y0 + 1.5)
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="mark", s=6, fc="white", ec=fs.TEXT, lw=0.5)], "no change (plotted at 0)")])
    ax_l, ax_w, ax_h = x0 + 8.4, w - 8.8, AX_H - 5.0
    ax = cv.axes(ax_l, y0 + AX_TOP, ax_w, ax_h)
    floor = 6e-5
    xpos, labels, groups = [], [], []
    x = 0.0
    for stype, cats, glab in E_GROUPS:
        start = x
        for cat, lab in cats:
            for j, (p, _) in enumerate(PARAMS):
                r = b[(b["stratum_type"] == stype) & (b["stratum"] == cat) & (b["parameter"] == p)]
                v = float(r["mean_normalized_abs_change"].iloc[0])
                ax.plot([x + (j - 1) * 0.2], [max(v, floor)], marker="o", ms=2.4, mfc=PCOL[p] if v > 0 else "white", mec=PCOL[p], mew=0.5,
                        ls="none", clip_on=False)
                sd.add("e", stype, stratum=cat, parameter=p, mean_abs_change=v, groups=int(r["n_groups"].iloc[0]),
                       fraction_groups_changed=float(r["fraction_groups_changed"].iloc[0]))
            xpos.append(x)
            labels.append(lab)
            x += 1.0
        groups.append((start, x - 1.0, glab))
        x += 0.6
    ax.set_yscale("log")
    ax.set_ylim(floor / 1.3, 2e-2)
    ax.set_yticks([floor, 1e-3, 1e-2], ["0", "0.1%", "1%"])
    fs.minor(ax, x=False)
    lim = (-0.6, x - 1.0)
    ax.set_xlim(*lim)
    ax.set_xticks(xpos, [""] * len(xpos))
    ax.tick_params(axis="x", length=0)
    fs.clean(ax)
    ax.set_ylabel("Mean |change|", labelpad=1)
    xm = lambda v: ax_l + (v - lim[0]) / (lim[1] - lim[0]) * ax_w  # noqa: E731
    for xx, lab in zip(xpos, labels):
        cv.text(xm(xx) + 0.6, y0 + AX_TOP + ax_h + 1.0, lab, size=fs.PT_SMALL, ha="right", va="top", rotation=40, rotation_mode="anchor")
    for a_, b_, glab in groups:
        cv.line([xm(a_) - 1.0, xm(b_) + 1.0], [y0 + AX_TOP + ax_h + 8.9] * 2, fs.TEXT, lw=0.35)
        cv.text(xm((a_ + b_) / 2), y0 + AX_TOP + ax_h + 9.4, glab, size=fs.PT_SMALL, ha="center", va="top")


# ---------------------------------------------------------------- f: enrichment of the top 1%

def panel_f(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "f")
    c = cm.table("extended/ed04/ed04c_parameter_change_enrichment.csv")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    c = c[c["expected_in_top"] >= 1.0].copy()
    c["p_adjusted"] = c["test_id"].map(tests["p_adjusted"])
    c = c.sort_values(["variable", "expected_in_top"], ascending=[False, False])
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="box", fc="#4D4D4D", size=1.3)], "observed"), ([dict(kind="box", fc="white", ec=fs.TEXT, lw=0.4, size=1.3)],
                                                                                        "expected"), ], gap=1.6)
    cv.text(x0 + 3.6, y0 + 3.9, f"{int(c['top_k'].iloc[0])} most consistently changed groups; every {cm.adjusted_label(c['test_id'])} ≥ "
                                f"{np.floor(c['p_adjusted'].min() * 100) / 100:.2f}", size=fs.PT_SMALL, va="center")
    names = {"cb_intrinsic": "brain intrinsic", "vnc_intrinsic": "VNC intrinsic", "descending_neuron": "descending",
             "vnc_motor": "VNC motor", "visual_projection": "visual projection", "ascending_neuron": "ascending",
             "visual_centrifugal": "visual centrifugal", "vnc_efferent": "VNC efferent", "acetylcholine": "ACh", "gaba": "GABA", "glutamate": "Glu",
             "unknown": "unknown", "octopamine": "OA"}
    ax_l, ax_w, ax_h = x0 + 8.4, w - 8.8, AX_H - 5.0
    ax = cv.axes(ax_l, y0 + AX_TOP, ax_w, ax_h)
    for i, r in enumerate(c.itertuples()):
        ax.bar(i - 0.17, r.observed_in_top, width=0.32, color="#4D4D4D", linewidth=0)
        ax.bar(i + 0.17, r.expected_in_top, width=0.32, facecolor="white", edgecolor=fs.TEXT, linewidth=0.4)
        sd.add("f", r.variable, category=r.category, groups_in_category=int(r.groups_in_category), observed=int(r.observed_in_top),
               expected=float(r.expected_in_top), p_adjusted=float(r.p_adjusted), correction=cm.correction([r.test_id]), test_id=r.test_id)
    ax.set_xlim(-0.6, len(c) - 0.4)
    ax.set_xticks(range(len(c)), [""] * len(c))
    ax.tick_params(axis="x", length=0)
    fs.minor(ax, x=False)
    fs.clean(ax)
    ax.set_ylabel("Groups in the top 1%", labelpad=1)
    xm = lambda v: ax_l + (v + 0.6) / (len(c)) * ax_w  # noqa: E731
    for i, r in enumerate(c.itertuples()):
        cv.text(xm(i) + 0.6, y0 + AX_TOP + ax_h + 1.0, names.get(r.category, r.category), size=fs.PT_SMALL, ha="right", va="top", rotation=40,
                rotation_mode="anchor")


# ---------------------------------------------------------------- g: direction of change

def panel_g(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "g")
    g = cm.table("figure_data/fd_group_change.csv")
    top = g[g["top_1pct"].astype(str).str.lower() == "true"]
    pkey(cv, x0 + 3.6, y0 + 1.5)
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="hline", w=2.6, lw=0.6, ls=(0, (2, 1.2)))], "binomial, p = 0.5 (24 maps)")])
    ax = cv.axes(x0 + 9.4, y0 + AX_TOP, w - 11.4, AX_H)
    ks = np.arange(0, 25)
    for j, (p, lab) in enumerate(PARAMS):
        inc = top[f"maps_increased_{p}"].to_numpy(int)
        dec = top[f"maps_decreased_{p}"].to_numpy(int)
        moved = (inc + dec) == 24
        h_ = np.bincount(inc[moved], minlength=25)
        ax.step(ks, h_ / max(moved.sum(), 1), where="mid", color=PCOL[p], lw=0.8)
        sd.add("g", p, groups=int(len(inc)), groups_changed_on_every_map=int(moved.sum()), median_maps_increased=float(np.median(inc[moved])),
               counts=";".join(map(str, h_)))
    ax.plot(ks, binom.pmf(ks, 24, 0.5), color=fs.TEXT, lw=0.6, ls=(0, (2, 1.2)))
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_ylim(0, None)
    fs.clean(ax)
    ax.set_xlabel("Maps (of 24) in which the parameter increased", labelpad=1)
    ax.set_ylabel("Share of top-1% groups", labelpad=1)


def main():
    cv = fs.Canvas(fs.FULL, H, ppi_min=300)
    sd = qc.SourceData("ed04")
    for rel in ("figure_data/fd_group_change.csv", "figure_data/fd_cells.csv", "extended/ed04/ed04a_parameter_change_consistency.csv",
                "extended/ed04/ed04b_parameter_change_anatomy.csv", "extended/ed04/ed04c_parameter_change_enrichment.csv",
                "main/fig2/fig2c_cost_vs_difficulty.csv", "stats/statistical_tests.csv"):
        sd.source(cm.RESULTS / rel)
    panel_a(cv, sd)
    mw = META["map_mm"][0]
    bx = 3 * (mw + fs.GUTTER) + 1.4
    panel_b(cv, sd, bx, 0.0, fs.FULL - bx, BAND_H)
    panel_c(cv, sd, XS[0], ROW2, U, ROW_H)
    panel_d(cv, sd, XS[1], ROW2, U, ROW_H)
    panel_e(cv, sd, XS[2], ROW2, 2 * U + fs.GAP, ROW_H)
    panel_f(cv, sd, XS[0], ROW3, 2 * U + fs.GAP, ROW_H)
    panel_g(cv, sd, XS[2], ROW3, 2 * U + fs.GAP, ROW_H)
    cm.finish(cv, "ed04", sd, kind="extended")


if __name__ == "__main__":
    main()
