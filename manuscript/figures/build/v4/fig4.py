"""v4 Fig. 4 (Cell, full page) - Trace-carrying cells have a molecular signature: SK (174 mm wide).

A  dark band: the transcriptome painted on the connectome, an in silico triple stain (SK cyan, Shal magenta, slo yellow)
B  dark band: the scene-1 trace carriers glowing by SK; badges for the hemilineages richest in carriers
C  dark band: the "Rosetta", the adult VNC atlas in UMAP space joined through an SK-ordered spine of its 21 hemilineages to
   their cells in the MaleCNS VNC
D  carriers' mean SK z and adaptation index against matched sets and the mapping null, four learners
E  the VNC alone, on data that never shaped the SK hypothesis
F  across the 21 hemilineages, carrier share rises with SK
G  the channel panel across the 21 hemilineages, ordered by carrier share (dot size: detection; colour: z)
H  the validated bridge and the coverage it buys
I  carrier cell types with genetic access, in the grammar of MaleCNS Figs 4C and 5C: each type on the filled light-grey brain
   (one exemplar carrier's skeleton, the type's trace carriers as gold somata, its other cells pale), its split-GAL4 line, its
   cells (carriers of all), the share of its cue drive that goes into readout types, and its channel expression (z)
J  adaptation heterogeneity as the transcriptome sets it keeps memory, lock and mechanism (10 runs). The cue's routes to the
   motor neurons by hemilineage (the single-cell network of revision 3) are Figure S4G.

Every number in a label is read from manuscript/results (plan_v2 via P12) or from the band's render metadata. Reader-facing
names only. Not shown (class c, RESULTS_INVENTORY_V4.md): the adaptation index against the mapping null (M12), the scene-2
heterogeneity runs and the type-permuted null brains (M13), the earlier bridge runs and the H1 v1 null (M10, M11).

Run after P12, render/v4_fig4_tiles.py and render/v4_fig4_access_light.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\v4\\fig4.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
from lib import circuit as ci, figstyle as fs, palette as pal, qc  # noqa: E402

cm.use_v4()
TILES = cm.RENDERS / "v4" / "fig4"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
ROS = json.loads((TILES / "rosetta.json").read_text(encoding="utf-8"))
BAND_H = META["band_mm"]
CNS_W = META["cns_mm"]
ROS_W = META["rosetta_mm"]
TITLE_MM = META["title_mm"]
NW, NH = META["neuron_mm"]
U = fs.unit(1)
XS = [fs.unit_x(i) for i in range(4)]
ROW2 = BAND_H + 4.2
R2_H = 46.0
ROW3 = ROW2 + R2_H + 3.4
R3_H = 46.0
ROW4 = ROW3 + R3_H + 3.4
R4_H = 44.0
H = ROW4 + R4_H
WHITE = fs.ON_DARK
NL_ = chr(10)
LEARNER_COL = {("scene 1", "readout"): pal.STATE_SET["m0"], ("scene 1", "brain and readout"): pal.STATE_SET["m0_joint"],
               ("scene 2", "readout"): pal.STATE_SET["s1"], ("scene 3", "readout"): pal.STATE_SET["s2_o0"]}
NULL_MATCHED, NULL_MAPPING = "#9E9E9E", "#DCDCDC"
Z_CMAP = colormaps["PuOr_r"]
X_A, X_B, X_C = 0.0, CNS_W + fs.GUTTER, 2 * (CNS_W + fs.GUTTER)


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


def learner_label(scene, trained):
    return scene.capitalize() + (", brain + readout" if trained == "brain and readout" else "")


def tile_img(name):
    return np.asarray(Image.open(TILES / name).convert("RGB"))


# ---------------------------------------------------------------- A: in silico triple stain

def panel_stain(cv, sd):
    img = tile_img("stain.png")
    cv.tile(X_A, 0, img, width=CNS_W, height=BAND_H, group="band", label="stain")
    white_letter(cv, 1.0, 0.8, "A")
    tile_text(cv, 4.2, 1.0, "In silico triple stain", va="top", size=fs.PT)
    tile_text(cv, 4.2, 3.6, "atlas expression of each cell's type", va="top")
    e = cm.table("plan_v2/fig4_cell_expression.csv")
    mapped = int(e["tier"].ne("unmapped").sum())
    # colour-mixing triangle (black space right of the VNC)
    tri = tile_img("triangle.png")
    ts = META["triangle"]["size_mm"]
    tx, ty = X_A + CNS_W - ts - 1.4, BAND_H - ts - 12.0
    cv.tile(tx, ty, tri, width=ts, height=ts, group="key", label="triangle")
    v = META["triangle"]["vertices_mm"]
    tile_text(cv, tx + v[0][0], ty + v[0][1] - 0.8, "SK", ha="center", va="bottom")
    tile_text(cv, tx + v[1][0], ty + v[1][1] + 0.8, "Shal", ha="center", va="top")
    tile_text(cv, tx + v[2][0], ty + v[2][1] + 0.8, "slo", ha="center", va="top")
    for i, line in enumerate(["bright: cell type", "dim: hemilineage", "dimmer: class", "gray: unmapped"]):
        tile_text(cv, X_A + 1.0, BAND_H - 17.4 + 2.3 * i, line, va="center")
    tile_text(cv, X_A + 1.0, BAND_H - 5.2, f"{mapped:,} of {len(e):,}", va="center")
    tile_text(cv, X_A + 1.0, BAND_H - 2.9, "cells mapped", va="center")
    sd.add("A", "stain", mapped_cells=mapped, cells=int(len(e)), channels="SK cyan, Shal magenta, slo yellow",
           z_range=META["z_range"], gamma=META["gamma"], tier_gain=json.dumps(META["tier_gain"]))


# ---------------------------------------------------------------- B: carriers by SK

def panel_carriers(cv, sd):
    img = tile_img("carriers.png")
    meta = json.loads((TILES / "carriers.json").read_text(encoding="utf-8"))
    cv.tile(X_B, 0, img, width=CNS_W, height=BAND_H, group="band", label="carriers")
    white_letter(cv, X_B + 1.0, 0.8, "B")
    tile_text(cv, X_B + 4.2, 1.0, "Where the memory sits, by SK", va="top", size=fs.PT)
    e = cm.table("plan_v2/fig4_cell_expression.csv")
    car = e["carrier_scene1"].astype(str).str.lower().eq("true")
    n_car, n_map = int(car.sum()), int((car & e["tier"].ne("unmapped")).sum())
    tile_text(cv, X_B + 4.2, 3.6, f"{n_car:,} trace carriers, scene 1", va="top")
    hem = cm.table("plan_v2/fig4_hemilineages.csv").set_index("unit")
    ppm = img.shape[1] / CNS_W
    for i, u in enumerate(meta["badge_units"]):
        cx, cy = meta["badge_px"][u]
        x1, y1 = X_B + cx / ppm, cy / ppm
        lx, ly = X_B + 1.2, BAND_H - 21.0 + 3.0 * i
        cv.line([lx + 9.0, x1], [ly, y1], WHITE, lw=0.3)
        tile_text(cv, lx, ly, f"{u}: {100 * hem.loc[u, 'carrier_share']:.0f}%", va="center")
        sd.add("B", "badge", hemilineage=u, carrier_share=float(hem.loc[u, "carrier_share"]))
    tile_text(cv, X_B + 1.2, BAND_H - 24.4, "carrier share", va="center")
    # SK key: dark-centred SK map, as the rendered carriers
    kx, ky = X_B + CNS_W - 12.6, BAND_H - 11.0
    for i, (z, lab) in enumerate(((2.0, "SK high"), (-2.0, "SK low"))):
        uu = min(abs(z) / 2.0, 1.0)
        c = pal.SIGNED_ON_DARK(0.5 + np.sign(z) * (0.20 + 0.30 * uu))
        cv.mark(kx + 0.7, ky + 2.4 * i, s=6.0, fc=c, ec=c, lw=0.3)
        tile_text(cv, kx + 1.6, ky + 2.4 * i, lab, va="center")
    cv.mark(kx + 0.7, ky + 4.8, s=6.0, fc="#CFCFCF", ec="#CFCFCF", lw=0.3)
    tile_text(cv, kx + 1.6, ky + 4.8, "unmapped", va="center")
    sd.add("B", "carriers", carriers=n_car, mapped=n_map, located=int(meta["carriers_located"]))


# ---------------------------------------------------------------- C: Rosetta

def panel_rosetta(cv, sd):
    img = tile_img("rosetta.png")
    cv.tile(X_C, 0, img, width=ROS_W, height=BAND_H, group="band", label="rosetta")
    white_letter(cv, X_C + 1.0, 0.8, "C")
    tile_text(cv, X_C + 4.2, 1.0, "From atlas to connectome", va="top", size=fs.PT)
    hem = cm.table("plan_v2/fig4_hemilineages.csv").set_index("unit")
    um = cm.table("plan_v2/fig4_vnc_atlas_umap.csv")
    for u, (x, y) in ROS["spine_mm_xy"].items():
        strong = u in ROS["highlight_units"]
        t = tile_text(cv, X_C + x, y, u, va="center", fontweight="bold" if strong else "normal")
        sd.add("C", "hemilineage", unit=u, colour=ROS["unit_colours"][u], z_SK=float(hem.loc[u, "z_SK"]), spine_rank=ROS["spine_order"].index(u) + 1)
    sw0 = ROS["swatch_mm"][ROS["spine_order"][0]]
    tile_text(cv, X_C + sw0[0] + 1.1, sw0[1] - 1.9, "SK", ha="center", va="bottom")
    tile_text(cv, X_C + ROS["galaxy_mm"] + ROS["spine_mm"] + 1.0, TITLE_MM + 0.6, "MaleCNS VNC", va="top")
    gy = TITLE_MM + ROS["galaxy_mm"] + 4.0
    lines = ["Adult VNC atlas,", f"{len(um):,} cells; {int(um['unit'].notna().sum()):,} in", f"{len(hem)} hemilineage units", "",
             "fibers: atlas cluster →", "SK rank → cells in", "the connectome"]
    for i, s in enumerate(lines):
        if s:
            tile_text(cv, X_C + 1.0, gy + 2.3 * i, s, va="center")
    sd.add("C", "atlas", cells=int(len(um)), in_units=int(um["unit"].notna().sum()), units=int(len(hem)),
           vnc_cells_located=int(ROS["vnc_located"]), vnc_cells_in_units=int(ROS["in_units"]))




# ---------------------------------------------------------------- D: enrichment forest

FACETS = [("SK", "matched-sets", "SK, matched sets"), ("SK", "mapping-null", "SK, mapping null"), ("A", "matched-sets", "Index, matched sets")]
ROWS = [("all", "scene 1", "readout", "Scene 1"), ("all", "scene 1", "brain and readout", "Scene 1," + chr(10) + "brain + readout"),
        ("all", "scene 2", "readout", "Scene 2"), ("all", "scene 3", "readout", "Scene 3"), ("vnc", "scene 1", "readout", "VNC only, scene 1")]


def panel_forest(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "D", "Trace carriers are enriched for SK")
    t = cm.table("plan_v2/fig4_trace_tests.csv")
    lab_w = 24.0
    fw = (w - lab_w - 3.0 * 2) / 3
    ys = [0, 1, 2, 3, 4.6]                                          # the VNC-only row set apart: new data
    for k, (stat, null, head) in enumerate(FACETS):
        ax = cv.axes(x0 + lab_w + k * (fw + 3.0), y0 + 12.0, fw, h - 18.6)
        vals = []
        for yy, (comp, scene, trained, _) in zip(ys, ROWS):
            r = t[(t["compartment"] == comp) & (t["scene"] == scene) & (t["trained"] == trained) & (t["statistic_name"] == stat)
                  & (t["null"] == null)].iloc[0]
            eff = (float(r["statistic"]) - float(r["mean_reference"])) / float(r["null_sd"])
            vals.append(eff)
            c = LEARNER_COL[(scene, trained)]
            ax.plot([0, eff], [yy, yy], color=c, lw=0.6, zorder=2)
            ax.scatter([eff], [yy], s=12, facecolor=c if comp == "all" else "white", edgecolor=c, linewidth=0.7, zorder=3)
            sd.add("D", head, compartment=comp, learner=learner_label(scene, trained), carriers_mean=float(r["statistic"]), null_mean=float(r["mean_reference"]),
                   null_sd=float(r["null_sd"]), effect_sd=eff, p_adjusted=float(r["p_adjusted"]), family=r["family"], mapped_carriers=int(r["n"]))
        q = t[(t["statistic_name"] == stat) & (t["null"] == null)]["p_adjusted"]
        p_txt = f"{fs.P_SYM} ≤ {fs.fmt_p(q.max())}" if q.max() < 0.001 else f"{fs.P_SYM} = {fs.fmt_p(q.min())}–{fs.fmt_p(q.max())}"
        ax.axvline(0, color=fs.TEXT2, lw=fs.LW_THIN, zorder=1)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_ylim(max(ys) + 0.6, -0.6)
        ax.set_yticks([])
        ax.set_title(f"{head}{NL_}{p_txt}", fontsize=fs.PT_SMALL, pad=1.5, linespacing=1.0)
        fs.clean(ax, left=False)
        if k == 1:
            ax.set_xlabel("Carriers above the null mean (null SD)", labelpad=0.8)
    for yy, (_, scene, trained, name) in zip(ys, ROWS):
        frac = (yy + 0.6) / (max(ys) + 1.2)
        cv.text(x0 + lab_w - 1.0, y0 + 12.0 + frac * (h - 18.6), name, size=fs.PT_SMALL, ha="right", va="center", linespacing=1.0)


# ---------------------------------------------------------------- E: carrier share vs SK

def panel_share(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "E", "More SK, more carriers")
    hem = cm.table("plan_v2/fig4_hemilineages.csv")
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 15.6)
    colours = META["unit_colours"]
    show = set(META["highlight_units"]) | set(hem.nlargest(2, "carrier_share")["unit"])
    for _, r in hem.iterrows():
        c = pal.on_white(colours[r["unit"]])
        ax.scatter([r["z_SK"]], [100 * r["carrier_share"]], s=12, facecolor=c, edgecolor="white", linewidth=0.3, zorder=3)
        if r["unit"] in show:
            ax.text(r["z_SK"] + 0.14, 100 * r["carrier_share"], r["unit"], fontsize=fs.PT_SMALL, ha="left", va="center")
        sd.add("E", "hemilineage", unit=r["unit"], z_SK=float(r["z_SK"]), carrier_share=float(r["carrier_share"]),
               carriers=int(r["carriers_scene1"]), malecns_cells=int(r["malecns_cells"]))
    rho = spearmanr(hem["z_SK"], hem["carrier_share"]).statistic
    ax.text(0.97, 0.03, f"{fs.RHO} = {rho:.2f}{NL_}{len(hem)} hemilineages", transform=ax.transAxes, fontsize=fs.PT_SMALL, ha="right", va="bottom",
            linespacing=1.0)
    ax.set_xlim(-2.6, 2.9)
    ax.set_xlabel("SK z", labelpad=0.8)
    ax.set_ylabel("Carriers (% of cells)", labelpad=0.8)
    fs.clean(ax)
    sd.add("E", "summary", spearman_rho=float(rho), units=int(len(hem)))


# ---------------------------------------------------------------- F: coverage

def panel_coverage(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "F", "Carriers mapped")
    cov = cm.table("plan_v2/fig4_coverage.csv").set_index("cell_set")
    bars = [("cell type", "trace carriers (m0)", "type_or_fuzzy", "#BDBDBD"), ("+ hemilineage", "trace carriers (m0)", "type_lineage", "#6E6E6E"),
            ("VNC carriers", "trace carriers (m0), VNC compartment", "type_lineage", "#1A1A1A")]
    ax = cv.axes(x0 + 9.0, y0 + 7.4, w - 10.6, h - 15.6)
    for i, (lab, row, col, c) in enumerate(bars):
        v = 100 * float(cov.loc[row, col])
        ax.bar(i, v, width=0.62, color=c, linewidth=0)
        ax.text(i, v + 1.5, f"{v:.0f}%", fontsize=fs.PT_SMALL, ha="center", va="bottom")
        sd.add("F", "coverage", level=lab, cells=int(cov.loc[row, "cells"]), share=v / 100)
    ax.set_xticks(range(len(bars)), [f"cell{NL_}type", f"+ hemi-{NL_}lineage", f"VNC{NL_}carriers"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 65)
    ax.set_yticks([0, 30, 60])
    ax.set_ylabel("Scene-1 carriers mapped (%)", labelpad=0.8)
    fs.clean(ax)


# ---------------------------------------------------------------- G: channel dot plot

def panel_dots(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "G", "Channel expression by hemilineage")
    hem = cm.table("plan_v2/fig4_hemilineages.csv").sort_values("carrier_share", ascending=False)
    genes = ["SK", "slo", "KCNQ", "Shab", "Shal"]
    units = hem["unit"].tolist()
    left, aw = x0 + 12.0, w - 13.4
    ax_b = cv.axes(left, y0 + 7.0, aw, 7.4)
    ax_b.bar(range(len(units)), 100 * hem["carrier_share"], width=0.7, color="#7A7A7A", linewidth=0)
    ax_b.set_xlim(-0.6, len(units) - 0.4)
    ax_b.set_xticks([])
    ax_b.set_yticks([0, 40], ["0", "40"])
    ax_b.set_ylabel("Carriers (%)", labelpad=0.6)
    fs.clean(ax_b)
    ax = cv.axes(left, y0 + 15.4, aw, h - 27.4)
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    for j, g in enumerate(genes + ["index"]):
        for i, (_, r) in enumerate(hem.iterrows()):
            z = float(r["adaptation_index"] if g == "index" else r[f"z_{g}"])
            frac = None if g == "index" else float(r[f"detected_{g}"])
            colour = Z_CMAP(norm(np.clip(z, -2, 2)))
            if frac is None:
                ax.scatter([i], [j], s=10, marker="s", facecolor=colour, edgecolor="none", zorder=3)
            else:
                ax.scatter([i], [j], s=2 + 22 * frac ** 1.5, facecolor=colour, edgecolor="none", zorder=3)
            sd.add("G", g, unit=r["unit"], z=z, detected=frac)
    ax.set_xlim(-0.6, len(units) - 0.4)
    ax.set_ylim(len(genes) + 0.5, -0.5)
    ax.set_yticks(range(len(genes) + 1), genes + ["index"])
    ax.set_xticks(range(len(units)), units, rotation=90)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # keys: z colour bar and detection sizes, one row under the plot
    ky = y0 + h - 1.6
    bar = cv.axes(left, ky - 0.7, 14.0, 1.4)
    bar.imshow(np.linspace(-2, 2, 128)[None], cmap=Z_CMAP, norm=norm, aspect="auto", extent=(-2, 2, 0, 1))
    bar.set_axis_off()
    cv.text(left - 0.6, ky, f"z {fs.MINUS}2", size=fs.PT_SMALL, ha="right", va="center")
    cv.text(left + 14.6, ky, "+2", size=fs.PT_SMALL, ha="left", va="center")
    items = [([dict(kind="mark", s=2 + 22 * f ** 1.5, fc="#7A7A7A", ec="#7A7A7A")], f"{int(100 * f)}%") for f in (0.5, 0.7, 0.9)]
    end = cv.key(left + 22.0, ky, items, size=fs.PT_SMALL, gap=1.4)
    cv.text(end + 1.2, ky, "detected", size=fs.PT_SMALL, va="center", color=fs.TEXT2)


# ---------------------------------------------------------------- H: bridge validation

def panel_bridge(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "H", "A validated bridge")
    cl = cm.table("plan_v2/fig4_bridge_clusters.csv")
    hem = cm.table("plan_v2/fig4_hemilineages.csv")
    from matplotlib.ticker import NullFormatter
    aw = 25.0
    panels = [(cl["published_cells"], cl["recovered_cells"], "Published cluster", "Recovered", f"{len(cl)} clusters", [30, 100, 300]),
              (hem["atlas_cells"], hem["malecns_cells"], "Atlas cells", "MaleCNS cells", f"{len(hem)} hemilineages", [200, 500, 1000])]
    for k, (x, y, xl, yl, n_txt, ticks) in enumerate(panels):
        ax = cv.axes(x0 + 9.0 + k * (aw + 10.0), y0 + 7.4, aw, h - 15.6)
        ax.scatter(x, y, s=4, facecolor="#5E5E5E", edgecolor="none", zorder=3)
        lo, hi = min(x.min(), y.min()) * 0.7, max(x.max(), y.max()) * 1.4
        ax.plot((lo, hi), (lo, hi), color=fs.GRID, lw=fs.LW_THIN, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        for a in (ax.xaxis, ax.yaxis):
            a.set_minor_formatter(NullFormatter())
        ticks = [v for v in ticks if lo <= v <= hi]
        ax.set_xticks(ticks, [f"{v:,}" for v in ticks])
        ax.set_yticks(ticks, [f"{v:,}" for v in ticks])
        ax.set_xlabel(xl, labelpad=0.8)
        ax.set_ylabel(yl, labelpad=0.8)
        fs.clean(ax)
        rho = spearmanr(x, y).statistic
        ax.text(0.05, 0.97, f"{fs.RHO} = {rho:.2f}{NL_}{n_txt}", transform=ax.transAxes, fontsize=fs.PT_SMALL, ha="left", va="top", linespacing=1.0)
        sd.add("H", yl, spearman_rho=float(rho), n=int(len(x)))
    same_nt = int((hem["em_transmitter"].str.lower() == hem["transcriptome_transmitter"].str.lower()).sum())
    marked = int((hem["markers"] > 0).sum())
    tx = x0 + 9.0 + 2 * aw + 14.0
    for i, (num, lab) in enumerate(((f"{same_nt}/{len(hem)}", f"transmitter{NL_}= EM"), (f"{marked}/{len(hem)}", f"markers{NL_}enriched"))):
        cv.text(tx, y0 + 10.4 + 11.0 * i, num, size=fs.PT, va="center", fontweight="bold")
        cv.text(tx, y0 + 12.6 + 11.0 * i, lab, size=fs.PT_SMALL, va="top", color=fs.TEXT2, linespacing=1.0)
    for _, r in cl.iterrows():
        sd.add("H", "cluster", cluster=int(r["cluster"]), published=int(r["published_cells"]), recovered=int(r["recovered_cells"]))
    for _, r in hem.iterrows():
        sd.add("H", "hemilineage", unit=r["unit"], atlas_cells=int(r["atlas_cells"]), malecns_cells=int(r["malecns_cells"]),
               em_transmitter=r["em_transmitter"], atlas_transmitter=r["transcriptome_transmitter"], markers=int(r["markers"]))


# ---------------------------------------------------------------- I: carrier types with genetic access

def panel_access(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "I", "Carrier types with driver lines")
    types = cm.table("plan_v2/fig4_access_types.csv").set_index("type")
    tiles = cm.table("plan_v2/fig4_access_tiles.csv").set_index("type")
    light = json.loads((TILES / "light.json").read_text(encoding="utf-8"))
    genes = ["SK", "slo", "KCNQ", "Shab", "Shal"]
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    order = [n["type"] for n in META["neurons"]]
    tw = (w - 3 * fs.GUTTER) / 4
    drive_fill, rest_fill, other_fill = "#5E5E5E", "#E3E3E3", "#D6D6D6"
    for k, tname in enumerate(order):
        r, tl, lm = types.loc[tname], tiles.loc[tname], light["types"][tname]
        if not (int(tl["carriers"]) == int(r["carriers_scene1"]) == int(lm["carriers"]) and int(tl["cells"]) == int(lm["cells"])):   # gate
            raise RuntimeError(f"I: {tname}: carriers or cells disagree between fig4_access_types, fig4_access_tiles and the render")
        x = x0 + k * (tw + fs.GUTTER)
        img = tile_img(f"light_{tname}.png")
        th = img.shape[0] / fs.DPI / fs.MM
        cv.tile(x, y0 + 5.0, img, dark=False, label=f"light_{tname}")             # native size: 600 ppi
        cv.text(x + 0.4, y0 + 5.4, tname, size=fs.PT, va="top", fontweight="bold")
        yb = y0 + 5.0 + th + 1.0
        ci.label_chip(cv, x, yb + 1.25, str(r["driver_lines"]), ci.GREY_CHIP, text_colour=fs.TEXT, size=fs.PT_SMALL, h=2.5)
        cells, car = int(tl["cells"]), int(tl["carriers"])
        yc = yb + 3.4                                                          # cells: carriers (gold) and the rest of the type
        cv.box(x, yc, tw * car / cells, 2.2, light["carrier"])
        cv.box(x + tw * car / cells, yc, tw * (1 - car / cells), 2.2, other_fill)
        cv.text(x + tw * car / cells + tw * (1 - car / cells) / 2, yc + 1.1, f"{car} of {cells} cells", size=fs.PT_SMALL, ha="center", va="center")
        share = float(tl["drive_to_readout_share"])                            # cue drive: into readout types, and elsewhere
        yd = yc + 2.8
        cv.box(x, yd, tw * share, 2.2, drive_fill)
        cv.box(x + tw * share, yd, tw * (1 - share), 2.2, rest_fill)
        cv.fills.append(dict(rect=(x, yd, tw * share, 2.2), fill=drive_fill))
        cv.text(x + tw * share / 2, yd + 1.1, f"{100 * share:.0f}% to readout", size=fs.PT_SMALL, ha="center", va="center", color=WHITE)
        sw = tw / len(genes)
        ys = yd + 2.8
        for j, g in enumerate(genes):
            cv.box(x + j * sw, ys, sw - 0.3, 1.6, fc=Z_CMAP(norm(np.clip(float(r[f"z_{g}"]), -2, 2))))
        sd.add("I", "type", type=tname, body_id=int(lm["body_id"]), driver_lines=r["driver_lines"], cells=cells, carriers=car,
               cue_drive_to_readout_share=share, readout_cells=int(tl["readout_cells"]), input_cells=int(tl["input_cells"]),
               **{f"z_{g}": float(r[f"z_{g}"]) for g in genes})
    ky = ys + 4.4
    cv.key(x0 + 0.4, ky, [([dict(kind="box", size=1.8, fc=light["carrier"])], "trace carriers"),
                          ([dict(kind="box", size=1.8, fc=other_fill)], "other cells of the type"),
                          ([dict(kind="box", size=1.8, fc=drive_fill)], "cue drive into readout types")], size=fs.PT_SMALL, gap=1.4)
    cv.text(x0 + w - 0.8, y0 + 0.6, "strip: " + ", ".join(genes) + " (z)", size=fs.PT_SMALL, ha="right", va="top", color=fs.TEXT2)


# ---------------------------------------------------------------- J: heterogeneity

def panel_hetero(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "J", "Heterogeneity keeps the memory")
    het = cm.table("plan_v2/fig4_heterogeneity.csv")
    hom = cm.table("plan_v2/x1_runs.csv")
    hom = hom[(hom["scene"] == "scene 1") & (hom["trained"] == "readout")]
    t = cm.table("plan_v2/tests.csv")
    main_t = t[t["family"] == "F14.SC.variant"].set_index("test")
    sec = t[(t["family"] == "F14.SC.variant.secondary") & t["test"].astype(str).str.contains("vs X1, M0: lock")]
    # (i) memory and lock: trained vs untrained use, homogeneous vs heterogeneous brains
    ax = cv.axes(x0 + 9.0, y0 + 9.0, 36.0, h - 17.0)
    rng = np.random.default_rng(41)
    for i, (d, lab) in enumerate(((hom, "homogeneous"), (het, "heterogeneous"))):
        tr, un = d["trained_net_use"].to_numpy(float), d["untrained_net_use"].to_numpy(float)
        jit = rng.uniform(-0.05, 0.05, len(d))
        for xx, yt, yu in zip(i + jit, tr, un):
            ax.plot([xx - 0.18, xx + 0.18], [yt, yu], color="#CFCFCF", lw=0.4, zorder=2)
        ax.scatter(i - 0.18 + jit, tr, s=7, facecolor=pal.ACCENT, edgecolor=pal.ACCENT, zorder=3)
        ax.scatter(i + 0.18 + jit, un, s=7, facecolor="white", edgecolor=fs.TEXT2, linewidth=0.5, zorder=3)
        for _, r in d.iterrows():
            sd.add("J", lab, replicate=int(r["replicate"]), trained_net_use=float(r["trained_net_use"]), untrained_net_use=float(r["untrained_net_use"]))
    ax.axhline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.4, 1.3)
    ax.set_xticks([0, 1], ["homogeneous", "heterogeneous"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Net memory use", labelpad=0.8)
    fs.clean(ax)
    p_lock = float(sec["p_adjusted"].iloc[0])
    ax.text(1.0, 1.24, f"lock: BH {fs.P_SYM} = {fs.fmt_p(p_lock)}", fontsize=fs.PT_SMALL, ha="center", va="top")
    cv.key(x0 + 9.0, y0 + 6.2, [([dict(kind="mark", s=7, fc=pal.ACCENT, ec=pal.ACCENT)], "trained timings"),
                                ([dict(kind="mark", s=7, fc="white", ec=fs.TEXT2)], "untrained")], size=fs.PT_SMALL, gap=1.6)
    # (ii) the mechanism in the heterogeneous brains: memory lost without the adaptation difference; flipped by a state swap
    ax2 = cv.axes(x0 + 58.0, y0 + 9.0, w - 59.4, h - 17.0)
    for i, (col, lab) in enumerate((("adaptation_drop", f"adaptation{NL_}difference{NL_}removed"), ("swap_drop", f"state{NL_}swapped"))):
        v = het[col].to_numpy(float)
        ax2.scatter(i + rng.uniform(-0.12, 0.12, len(v)), -v, s=7, facecolor=fs.TEXT, edgecolor="none", zorder=3)
        ax2.plot([i - 0.2, i + 0.2], [-np.median(v)] * 2, color=fs.TEXT, lw=0.9, zorder=4)
        for r in v:
            sd.add("J", col, change_in_net_use=-float(r))
    ax2.axhline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=0)
    ax2.set_xlim(-0.6, 1.6)
    ax2.set_ylim(-2.3, 0.4)
    ax2.set_yticks([-2, -1, 0])
    ax2.set_xticks([0, 1], [f"adaptation{NL_}removed", f"state{NL_}swapped"])
    ax2.tick_params(axis="x", length=0)
    ax2.set_ylabel("Change in net use", labelpad=0.8)
    fs.clean(ax2)
    pm = float(main_t["p_adjusted"].max())
    cv.text(x0 + 58.0, y0 + 6.2, f"Holm {fs.P_SYM} = {fs.fmt_p(pm)}", size=fs.PT_SMALL, va="center", color=fs.TEXT2)
    sd.add("J", "tests", main_holm_p_max=pm, lock_vs_homogeneous_bh_p=p_lock, runs=int(len(het)), homogeneous_runs=int(len(hom)))


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("fig4")
    for rel in ("plan_v2/fig4_cell_expression.csv", "plan_v2/fig4_hemilineages.csv", "plan_v2/fig4_vnc_atlas_umap.csv", "plan_v2/fig4_bridge_clusters.csv",
                "plan_v2/fig4_coverage.csv", "plan_v2/fig4_trace_tests.csv", "plan_v2/fig4_access_types.csv", "plan_v2/fig4_heterogeneity.csv",
                "plan_v2/x1_runs.csv", "plan_v2/tests.csv", "plan_v2/fig4_access_tiles.csv"):
        sd.source(cm.RESULTS / rel)
    for f in sorted(TILES.glob("*.png")) + sorted(TILES.glob("*.json")):
        sd.source(f)
    panel_stain(cv, sd)
    panel_carriers(cv, sd)
    panel_rosetta(cv, sd)
    panel_forest(cv, sd, XS[0], ROW2, fs.unit(2), R2_H)
    panel_share(cv, sd, XS[2], ROW2, U, R2_H)
    panel_coverage(cv, sd, XS[3], ROW2, U, R2_H)
    panel_dots(cv, sd, XS[0], ROW3, fs.unit(2), R3_H)
    panel_bridge(cv, sd, XS[2], ROW3, fs.unit(2), R3_H)
    panel_access(cv, sd, XS[0], ROW4, fs.unit(2), R4_H)
    panel_hetero(cv, sd, XS[2], ROW4, fs.unit(2), R4_H)
    cm.finish(cv, "fig4", sd)


if __name__ == "__main__":
    main()
