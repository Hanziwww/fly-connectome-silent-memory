"""Extended Data Fig. 8 - The cue state sits in the same cell types in every model: type and transmitter atlas (180 x 129 mm).

Band  a  signed cue difference of the membrane potential (cue 1 - cue 0, cells over 1 mV) in each of the six distinct state sets
         (last window at 160-ms delay and 4 prefix windows) on the unfolded CNS; one shared view
Row 1 b  agreement of the type-level mean |dv| between the state sets (Spearman rho over all cell types)
      c  the 30 cell types with the largest mean |dv| over the six state sets: |dv| per state set, superclass group, transmitter
         and number of cells
      d  enrichment of transmitters and superclasses among the 1% of cells with the largest |dv|, for every category and state set
         (odds ratio; dot: BH q < 0.05, family F8.enrichment; grey: none observed and fewer than one expected; ED8b)
Row 2 e  type-level activity in the 24-map static task before vs after training: change against the value before training, for
         spike probability, membrane potential and adaptation (every cell type)

  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\ed08.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import boards as bd, figstyle as fs, palette as pal, qc  # noqa: E402

U = (fs.FULL - 3 * fs.GAP) / 4
XS = [k * (U + fs.GAP) for k in range(4)]
TILES = cm.RENDERS / "ed08"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
TW, TH = META["tile_mm"]
HEAD = 4.0
KEY_Y = HEAD + TH + fs.GUTTER
KEY_H = 3.4
ROW1 = KEY_Y + KEY_H + 2.8
ROW_H = 41.0
ROW2 = ROW1 + ROW_H + 2.0
H = ROW2 + ROW_H
AX_TOP, AX_H = 6.5, 27.0
WHITE = fs.ON_DARK
SETS = [("m0", "M0 readout-only (hidden)", "M0 readout-only", "M0"), ("m0_joint", "M0 joint (hidden)", "M0 joint", "joint"),
        ("m0vis_visible", "M0 shared readout (visible)", "M0 shared, goal visible", "vis."), ("s1", "S1 readout-only (hidden)", "S1 readout-only", "S1"),
        ("s2_o0", "S2 readout-only, orientation 0 (hidden)", "S2, orientation 0°", "S2\n0°"),
        ("s2_o180", "S2 readout-only, orientation 180 (hidden)", "S2, orientation 180°", "S2\n180°")]
NT = {"acetylcholine": "ACh", "gaba": "GABA", "glutamate": "Glu", "histamine": "HA", "dopamine": "DA", "serotonin": "5-HT", "octopamine": "OA",
      "unknown": "unknown"}
SC_NAME = {"ENS": "enteric", "ascending_neuron": "ascending", "cb_efferent": "CB efferent", "cb_endocrine": "CB endocrine", "cb_intrinsic": "CB intrinsic",
           "cb_motor": "CB motor", "cb_sensory": "CB sensory", "cb_sensory_tbc": "CB sensory (tbc)", "descending_neuron": "descending",
           "efferent_ascending": "efferent ascending", "efferent_descending": "efferent descending", "ol_intrinsic": "OL intrinsic",
           "ol_sensory": "OL sensory", "sensory_ascending": "sensory ascending", "sensory_ascending_tbc": "sensory ascending (tbc)",
           "sensory_descending": "sensory descending", "unannotated": "unannotated", "visual_centrifugal": "visual centrifugal",
           "visual_projection": "visual projection", "visual_projection_tbc": "visual projection (tbc)", "vnc_efferent": "VNC efferent",
           "vnc_endocrine": "VNC endocrine", "vnc_intrinsic": "VNC intrinsic", "vnc_motor": "VNC motor", "vnc_sensory": "VNC sensory",
           "vnc_sensory_tbc": "VNC sensory (tbc)", "vnc_tbc": "VNC (tbc)"}
ENRICH = colormaps["PRGn_r"]
NA = "#E6E6E6"
RHO_CMAP = LinearSegmentedColormap.from_list("rho", [colormaps["Blues"](0.08), colormaps["Blues"](0.55)])
MAG_LO, MAG_HI = 0.1, 50.0


def qc_luminance(colour) -> float:
    from matplotlib.colors import to_rgb
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def tile_text(cv, x, y, s, size=fs.PT_SMALL, **kw):
    t = cv.text(x, y, s, size=size, color=WHITE, **kw)
    t.set_path_effects(fs.dark_halo(0.8))
    return t


def gradient_h(cv, x, y, w, h, colours):
    ax = cv.axes(x, y, w, h)
    ax.imshow(np.asarray(colours)[None, :, :3], aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax.set_axis_off()
    return ax


def mag_colour(v):
    t = np.clip((np.log10(np.maximum(np.asarray(v, float), 1e-9)) - np.log10(MAG_LO)) / (np.log10(MAG_HI) - np.log10(MAG_LO)), 0, 1)
    return pal.MAGNITUDE_ON_DARK(t)


def atlas():
    a = cm.table("extended/ed08/ed08a_type_atlas.csv")
    c = a[a["record"] == "cue_difference_by_type"]
    piv = c.pivot_table(index="type", columns="phase_or_state_set", values="mean_abs_voltage_cue_difference")[[s[1] for s in SETS]]
    assert not piv.isna().any().any()
    return piv, c.groupby("type")["cells"].first()


def type_info():
    c = cm.table("figure_data/fd_cells.csv")
    g = c.dropna(subset=["type"]).groupby("type")
    return g["superclass"].agg(lambda x: x.mode().iloc[0] if x.notna().any() else "other"), \
        g["transmitter"].agg(lambda x: x.mode().iloc[0] if x.notna().any() else "unknown")


# ---------------------------------------------------------------- a: six state sets

def panel_a(cv, sd):
    cv.letter(0, 0, "a")
    xs = [k * (TW + fs.GUTTER) for k in range(len(SETS))]
    by = {t["name"]: t for t in META["tiles"]}
    for x, (key, _, long, _) in zip(xs, SETS):
        t = by[key]
        cv.text(x + TW / 2, 1.9, long, size=fs.PT_SMALL, ha="center", va="center")
        img = np.asarray(Image.open(TILES / f"{key}.png").convert("RGB"))
        cv.tile(x, HEAD, img, width=TW, height=TH, group="ed08band", label=key)
        for i, (k, n) in enumerate(((1, t["positive"]), (0, t["negative"]))):
            yy = HEAD + TH - 4.0 + i * 2.3
            cv.mark(x + 1.6, yy, s=6.0, fc=pal.CUE_ON_DARK[k], ec="#000000", lw=0.3)
            tile_text(cv, x + 2.6, yy, f"{n:,}", va="center")
        sd.add("a", "state set", state_set=key, cells_over_1mv=t["cells_above"], cue1_higher=t["positive"], cue0_higher=t["negative"],
               located=t["located"], cells_all=t["cells_all"], over_1mv_all_cells=t["above_all"], um_per_px=t["scale_um_per_px"])
    t = META["tiles"][-1]
    x = xs[-1]
    px_per_mm = Image.open(TILES / f"{t['name']}.png").size[0] / TW
    length = 100 / t["scale_um_per_px"] / px_per_mm
    cv.line([x + TW - 1.2 - length, x + TW - 1.2], [HEAD + TH - 1.4] * 2, WHITE, lw=1.0, solid_capstyle="butt")
    tile_text(cv, x + TW - 1.2, HEAD + TH - 1.9, "100 µm", ha="right", va="bottom")
    ox, oy = x + TW - 7.4, HEAD + TH - 8.6
    cv.line([ox, ox + 2.4], [oy, oy], WHITE, lw=0.5)
    tile_text(cv, ox + 2.8, oy, "L", va="center")
    cv.line([ox, ox], [oy, oy + 2.4], WHITE, lw=0.5)
    tile_text(cv, ox, oy + 2.8, "V", ha="center", va="top")
    # signed key on a black strip
    strip = np.zeros((int(round(KEY_H * fs.MM * fs.DPI)), int(round(fs.FULL * fs.MM * fs.DPI)), 3), np.uint8)
    cv.tile(0, KEY_Y, strip, width=fs.FULL, height=KEY_H, group="ed08band", label="key")
    yc = KEY_Y + KEY_H / 2
    lab = "membrane potential, cue 1 minus cue 0 (cells over 1 mV):"
    tile_text(cv, 1.2, yc, lab, va="center")
    x = 1.2 + fs.text_width(lab, fs.PT_SMALL) + 1.6
    u = np.linspace(0, 1, 128)
    tile_text(cv, x, yc, fs.MINUS + "20", va="center")
    x += fs.text_width(fs.MINUS + "20", fs.PT_SMALL) + 0.6
    gradient_h(cv, x, yc - 0.6, 14.0, 1.2, pal.SIGNED_ON_DARK(0.5 - 0.26 - 0.24 * u[::-1]))
    x += 14.6
    tile_text(cv, x, yc, fs.MINUS + "1", va="center")
    x += fs.text_width(fs.MINUS + "1", fs.PT_SMALL) + 1.6
    tile_text(cv, x, yc, "+1", va="center")
    x += fs.text_width("+1", fs.PT_SMALL) + 0.6
    gradient_h(cv, x, yc - 0.6, 14.0, 1.2, pal.SIGNED_ON_DARK(0.5 + 0.26 + 0.24 * u))
    x += 14.6
    tile_text(cv, x, yc, "+20 mV", va="center")
    x += fs.text_width("+20 mV", fs.PT_SMALL) + 4.0
    cv.key(x, yc, [([dict(kind="mark", s=6, fc=pal.CUE_ON_DARK[1], ec="#000000", lw=0.3)], "cells with cue 1 higher"),
                   ([dict(kind="mark", s=6, fc=pal.CUE_ON_DARK[0], ec="#000000", lw=0.3)], "cue 0 higher")], gap=2.4, colour=WHITE,
           effects=fs.dark_halo(0.8))
    sd.add("a", "view", um_per_px=t["scale_um_per_px"], scale_bar_um=100, colour_range_mv="1 to 20 (log), dark-centred cue map",
           unfold="brain frontal; VNC rotated 90 degrees about the neck (msa/cnsview.py)")


# ---------------------------------------------------------------- b: agreement between state sets

def panel_b(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "b")
    piv, cells = atlas()
    rho = piv.corr(method="spearman")
    big = cells[cells >= 4].index
    rho4 = piv.loc[big].corr(method="spearman")
    cv.text(x0 + 3.6, y0 + 1.5, f"type mean |Δv|, Spearman {fs.RHO}, {len(piv):,} types", size=fs.PT_SMALL, va="center")
    n = len(SETS)
    lab_w = 8.4
    cs = min((w - lab_w - 1.0) / (n - 1), (h - 7.0 - 6.4) / (n - 1))
    ax = bd.mm_axes(cv, x0, y0, w, h)
    top = 6.0
    off4 = np.array([rho4.iloc[i, j] for i in range(n) for j in range(i)])
    lo, hi = 0.70, 0.95
    for i in range(1, n):
        ax.text(lab_w - 0.6, top + (i - 0.5) * cs, SETS[i][3].replace("\n", " "), ha="right", va="center", fontsize=fs.PT_SMALL)
        for j in range(i):
            v = float(rho.iloc[i, j])
            ax.add_patch(bd.Rectangle((lab_w + j * cs, top + (i - 1) * cs), cs - 0.15, cs - 0.15, facecolor=RHO_CMAP((v - lo) / (hi - lo)),
                                      edgecolor="none", linewidth=0.5))
            ax.text(lab_w + (j + 0.5) * cs - 0.075, top + (i - 0.5) * cs - 0.075, f"{v:.2f}", ha="center", va="center", fontsize=fs.PT_SMALL)
            sd.add("b", "spearman", state_set_a=SETS[i][1], state_set_b=SETS[j][1], rho_all_types=v, types=int(len(piv)),
                   rho_types_4_or_more_cells=float(rho4.iloc[i, j]), types_4_or_more_cells=int(len(big)))
    for j in range(n - 1):
        ax.text(lab_w + (j + 0.5) * cs, top + (n - 1) * cs + 0.3, SETS[j][3], ha="center", va="top", fontsize=fs.PT_SMALL, linespacing=1.0)
    cv.text(x0 + lab_w, y0 + h - 1.6, f"types with ≥ 4 cells ({len(big):,}): {fs.RHO} {off4.min():.2f}–{off4.max():.2f}", size=fs.PT_SMALL,
            va="center")


# ---------------------------------------------------------------- c: the most cue-responsive types

def panel_c(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    piv, cells = atlas()
    sc_of, nt_of = type_info()
    top = piv.mean(axis=1).sort_values(ascending=False, kind="mergesort")
    names = list(top.index[:30])
    cv.text(x0 + 3.6, y0 + 1.5, "mean |Δv| (mV) of the 30 most cue-responsive cell types", size=fs.PT_SMALL, va="center")
    half = len(names) // 2
    bw = (w - fs.GAP) / 2
    name_w = max(fs.text_width(n, fs.PT_SMALL) for n in names) + 0.6
    cs = (bw - name_w - 1.8 - 6.4 - 5.0) / len(SETS)
    hdr = 9.9
    rh = (h - hdr - 0.2) / half
    ax = bd.mm_axes(cv, x0, y0, w, h)
    for b in range(2):
        bx = b * (bw + fs.GAP)
        cx = bx + name_w + 1.8 + 6.4 + 5.0
        for j, s in enumerate(SETS):
            ax.text(cx + (j + 0.5) * cs, hdr - 0.4, s[3], ha="center", va="bottom", fontsize=fs.PT_SMALL, linespacing=1.0)
        ax.text(bx + name_w + 1.8 + 6.4 + 4.4, hdr - 0.4, "cells", ha="right", va="bottom", fontsize=fs.PT_SMALL)
        for i, t in enumerate(names[b * half:(b + 1) * half]):
            y = hdr + i * rh
            ax.text(bx + name_w - 0.4, y + rh / 2, t, ha="right", va="center", fontsize=fs.PT_SMALL)
            grp = pal.SUPERCLASS_GROUP.get(sc_of.get(t, "other"), "other")
            col = pal.on_white(pal.ATLAS_ON_DARK[grp])
            ax.add_patch(bd.Rectangle((bx + name_w + 0.2, y + rh / 2 - 0.55), 1.1, 1.1, facecolor=col, edgecolor="none", linewidth=0.5))
            ax.text(bx + name_w + 1.8, y + rh / 2, NT.get(nt_of.get(t, "unknown"), "unknown") if nt_of.get(t, "unknown") != "unknown" else "unk.",
                    ha="left", va="center", fontsize=fs.PT_SMALL)
            ax.text(bx + name_w + 1.8 + 6.4 + 4.4, y + rh / 2, f"{int(cells.loc[t])}", ha="right", va="center", fontsize=fs.PT_SMALL)
            for j, s in enumerate(SETS):
                v = float(piv.loc[t, s[1]])
                ax.add_patch(bd.Rectangle((cx + j * cs, y + 0.06), cs - 0.15, rh - 0.12, facecolor=mag_colour(v), edgecolor="none", linewidth=0.5))
                sd.add("c", "type", type=t, rank=b * half + i + 1, state_set=s[1], mean_abs_dv_mv=v, cells=int(cells.loc[t]),
                       superclass=sc_of.get(t, ""), transmitter=nt_of.get(t, ""))
    # colour key in the header line, right
    kx = x0 + w - 30.0
    gradient_h(cv, kx, y0 + 0.9, 18.0, 1.2, mag_colour(np.logspace(np.log10(MAG_LO), np.log10(MAG_HI), 128)))
    cv.text(kx - 0.6, y0 + 1.5, f"{MAG_LO:g}", size=fs.PT_SMALL, ha="right", va="center")
    cv.text(kx + 18.6, y0 + 1.5, f"{MAG_HI:g} mV", size=fs.PT_SMALL, ha="left", va="center")
    cv.key(x0 + 3.6, y0 + 3.9, [([dict(kind="box", fc=pal.on_white(pal.ATLAS_ON_DARK[g]))], pal.SHORT_GROUP[g])
                                for g in pal.ATLAS_ON_DARK if g != "other" and g != "visual projection & centrifugal"] +
           [([dict(kind="box", fc=pal.on_white(pal.ATLAS_ON_DARK["visual projection & centrifugal"]))], "visual proj.")], gap=1.4)


# ---------------------------------------------------------------- d: enrichment of every category

def panel_d(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "d")
    b = cm.table("extended/ed08/ed08b_atlas_enrichment.csv")
    tests = cm.table("stats/statistical_tests.csv").set_index("test_id")
    m0 = b[b["state_set"] == SETS[0][1]]
    nts = [c for c in NT if c in set(m0[m0["variable"] == "nt"]["category"])]
    scs = list(m0[m0["variable"] == "superclass"].sort_values(["cells_in_category", "category"], ascending=[False, True])["category"])
    rows = [("nt", c, NT[c]) for c in nts] + [("superclass", c, SC_NAME.get(c, c)) for c in scs]
    assert len(rows) * len(SETS) == len(b)
    cv.text(x0 + 3.6, y0 + 1.5, "top 1% by |Δv| vs all cells", size=fs.PT_SMALL, va="center")
    lab_w = max(fs.text_width(r[2], fs.PT_SMALL) for r in rows) + 1.2
    cs = (w - lab_w - 0.6) / len(SETS)
    hdr, gap, foot = 7.6, 1.2, 9.6
    rh = (h - hdr - gap - foot) / len(rows)
    ax = bd.mm_axes(cv, x0, y0, w, h)
    for j, s in enumerate(SETS):
        ax.text(lab_w + (j + 0.5) * cs, hdr - 0.4, s[3], ha="center", va="bottom", fontsize=fs.PT_SMALL, linespacing=1.0)
    n_sig = n_few = 0
    for i, (var, cat, lab) in enumerate(rows):
        y = hdr + i * rh + (gap if var == "superclass" else 0.0)
        ax.text(lab_w - 0.5, y + rh / 2, lab, ha="right", va="center", fontsize=fs.PT_SMALL)
        for j, s in enumerate(SETS):
            r = b[(b["state_set"] == s[1]) & (b["variable"] == var) & (b["category"] == cat)].iloc[0]
            orr = float(r["odds_ratio"])
            l2 = np.clip(np.log2(orr), -3, 3) if orr > 0 else -3.0
            p = float(tests.loc[r["test_id"], "p_adjusted"])
            few = float(r["expected_in_top"]) < 1.0 and int(r["top_in_category"]) == 0
            n_few += few
            col = NA if few else ENRICH((l2 + 3) / 6)
            ax.add_patch(bd.Rectangle((lab_w + j * cs, y + 0.05), cs - 0.15, rh - 0.1, facecolor=col, edgecolor="none", linewidth=0.5))
            if p < 0.05:
                n_sig += 1
                dark = qc_luminance(col) < 0.35
                ax.plot([lab_w + (j + 0.5) * cs - 0.075], [y + rh / 2], marker="o", ms=1.5, mfc="white" if dark else fs.TEXT, mec="none", ls="none")
            sd.add("d", "enrichment", state_set=s[1], variable=var, category=cat, cells_in_category=int(r["cells_in_category"]),
                   top_cells=int(r["top_cells"]), top_in_category=int(r["top_in_category"]), odds_ratio=orr, ci_low=float(r["ci_low"]),
                   ci_high=float(r["ci_high"]), p=float(r["p_value"]), p_adjusted=p, correction=cm.correction([r["test_id"]]),
                   test_id=r["test_id"])
    yb = h - foot + 1.4
    gradient_h(cv, x0 + lab_w, y0 + yb, len(SETS) * cs - 0.15, 1.1, ENRICH(np.linspace(0, 1, 128)))
    for v, lab, ha in ((-3, "≤1/8", "left"), (0, "1", "center"), (3, "≥8", "right")):
        cv.text(x0 + lab_w + (v + 3) / 6 * (len(SETS) * cs - 0.15), y0 + yb + 2.3, lab, size=fs.PT_SMALL, ha=ha, va="center")
    cv.text(x0 + lab_w - 0.6, y0 + yb + 0.55, "odds ratio", size=fs.PT_SMALL, ha="right", va="center")
    cv.text(x0 + 3.6, y0 + yb + 4.4, f"dot: {cm.adjusted_label(b['test_id'])} < 0.05 ({n_sig} of {len(b)} tests)", size=fs.PT_SMALL, ha="left",
            va="center")
    cv.text(x0 + 3.6, y0 + yb + 6.6, "grey: none observed, fewer than 1 expected", size=fs.PT_SMALL, ha="left", va="center")
    sd.add("d", "summary", tests=int(len(b)), significant_adjusted=int(n_sig), correction=cm.correction(b["test_id"]),
           fewer_than_one_expected=int(n_few))


# ---------------------------------------------------------------- e: activity before vs after training

def panel_e(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "e")
    a = cm.table("extended/ed08/ed08a_type_atlas.csv")
    s = a[a["record"] == "static_activity_24_maps"]
    maps = sorted(set(s["maps"].dropna().astype(int)))
    assert maps == [24]
    sc_of, _ = type_info()
    groups = [g for g in pal.ATLAS_ON_DARK if g != "other"] + ["other"]
    cv.key(x0 + 3.6, y0 + 1.5, [([dict(kind="mark", s=6, fc=pal.on_white(pal.ATLAS_ON_DARK[g]), ec=pal.on_white(pal.ATLAS_ON_DARK[g]))],
                                 pal.SHORT_GROUP[g]) for g in groups], gap=1.6)
    cv.text(x0 + 3.6, y0 + 3.9, f"every cell type, mean over the {maps[0]} maps; change = after minus before training", size=fs.PT_SMALL, va="center")
    specs = [("spike_probability", "Spike probability", 1e3, "Δ spike prob. (10$^{−3}$)"), ("voltage_mv", "Membrane potential (mV)", 1.0,
                                                                                                 "Δ potential (mV)"),
             ("adaptation", "Adaptation", 1e2, "Δ adaptation (10$^{−2}$)")]
    gap, lab_w = 12.0, 10.0
    aw = (w - lab_w - (len(specs) - 1) * gap - 0.6) / len(specs)
    for k, (col, xlab, scale, ylab) in enumerate(specs):
        p = s.pivot_table(index="type", columns="phase_or_state_set", values=col)
        before, after = p["before"].to_numpy(float), p["after"].to_numpy(float)
        d = after - before
        ds = d * scale
        grp = np.array([pal.SUPERCLASS_GROUP.get(sc_of.get(t, "other"), "other") for t in p.index])
        ax = cv.axes(x0 + lab_w + k * (aw + gap), y0 + AX_TOP, aw, AX_H)
        for g in ["other"] + [g for g in groups if g != "other"]:
            m = grp == g
            ax.scatter(before[m], ds[m], s=1.2, color=pal.on_white(pal.ATLAS_ON_DARK[g]), linewidth=0, alpha=0.7, rasterized=True, zorder=2)
        ax.axhline(0, color=fs.AXIS, lw=fs.LW_HAIR, ls=(0, (2, 1.5)), zorder=1)
        ratio = float(np.std(d) / np.std(before))
        ax.text(0.97, 0.97, f"SD of change = {100 * ratio:.1f}%" + "\n" + "of SD across types", transform=ax.transAxes, ha="right", va="top",
                fontsize=fs.PT_SMALL, linespacing=1.1)
        lim = float(np.max(np.abs(ds))) * 1.08
        ax.set_ylim(-lim, lim)
        fs.clean(ax)
        ax.set_xlabel(xlab + ", before", labelpad=1)
        ax.set_ylabel(ylab, labelpad=1)
        sd.add("e", "activity", variable=col, types=int(len(p)), maps=maps[0], sd_before=float(np.std(before)), sd_change=float(np.std(d)),
               sd_ratio=ratio, max_abs_change=float(np.max(np.abs(d))), median_change=float(np.median(d)),
               spearman_before_after=float(p[["before", "after"]].corr(method="spearman").iloc[0, 1]) if np.std(before) > 0 else None)


def verified_type_names():
    """Displayed cell-type names that contain an underscore, each checked against the cell table (published nomenclature)."""
    piv, _ = atlas()
    shown = list(piv.mean(axis=1).sort_values(ascending=False, kind="mergesort").index[:30])
    types = set(cm.table("figure_data/fd_cells.csv")["type"].dropna())
    names = tuple(t for t in shown if "_" in t)
    assert all(t in types for t in names)
    return names


def cm_u(n):
    return n * U + (n - 1) * fs.GAP


def main():
    cv = fs.Canvas(fs.FULL, H, ppi_min=300)
    sd = qc.SourceData("ed08")
    for rel in ("extended/ed08/ed08a_type_atlas.csv", "extended/ed08/ed08b_atlas_enrichment.csv", "figure_data/fd_soma_state.csv",
                "figure_data/fd_cells.csv", "stats/statistical_tests.csv"):
        sd.source(cm.RESULTS / rel)
    panel_a(cv, sd)
    panel_b(cv, sd, XS[0], ROW1, cm_u(1), ROW_H)
    panel_c(cv, sd, XS[1], ROW1, cm_u(2), ROW_H)
    panel_d(cv, sd, XS[3], ROW1, cm_u(1), H - ROW1)
    panel_e(cv, sd, XS[0], ROW2, cm_u(3), ROW_H)
    cm.finish(cv, "ed08", sd, kind="extended", allow_names=verified_type_names())


if __name__ == "__main__":
    main()
