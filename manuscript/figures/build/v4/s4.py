"""v4 Figure S4 (Cell), related to Figure 4 - The molecular atlas behind the SK signature (174 mm wide).

A  dark gallery: single-gene in silico stains on the unfolded CNS (SK, slo, KCNQ, Shab, Shal and the adaptation index), each
   mapped cell glowing by its atlas unit's z, mapping tier as brightness
B  where the cells map: share of each cell set mapped by cell type, by hemilineage and by hemilineage pair
C  the atlases agree with the connectome: transmitter of the mapped types against the EM prediction, and the map rows present,
   per atlas; the central-brain expression reproduced from the published data
D  the atlases agree with each other: expression correlation of unit pairs mapped to the same cell type against different types
E  marker genes light up in the mapped units (Kenyon cells, projection neurons, GABAergic ring neurons)
F  SK enrichment of the scene-1 carriers with and without the hemilineage-pair tier, against matched sets and the mapping null
G  the cue's routes to the motor neurons by hemilineage, the level at which the atlas maps the VNC, in the grammar of MaleCNS
   Fig. 5F: hemilineage modules tinted by their atlas SK (z), types gold when they held the trace, motor neurons outlined,
   edges the cue drive at the first decision (the learner's effective connectome); beside it, each hemilineage's SK and
   carrier share

Every number in a label is read from manuscript/results (plan_v2 via P12) or from the render metadata. Reader-facing names
only. Not shown: the per-atlas UMAPs of the outline (only the VNC atlas has routed cell coordinates; it is Fig. 4C), the
adaptation index against the mapping null (class c, M12), H1 v1 and the failed bridge runs (class c, M10-M11).

Run after P12 and render/v4_s4_tiles.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\v4\\s4.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm, to_hex
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
from lib import circuit as ci, figstyle as fs, palette as pal, qc  # noqa: E402

cm.use_v4()
TILES = cm.RENDERS / "v4" / "s4"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
GW, GH = META["tile_mm"]
U = fs.unit(1)
XS = [fs.unit_x(i) for i in range(4)]
ROW2, R_H = GH + 9.8, 46.0
ROW3 = ROW2 + R_H + 3.4
ROW4, R4_H = ROW3 + R_H + 3.4, 48.0
Z_CMAP = colormaps["PuOr_r"]                            # G: atlas z, the Fig. 4G scale
H = ROW4 + R4_H
WHITE = fs.ON_DARK
NL_ = chr(10)
FAMILY = {"SK": "Ca$^{2+}$-activated K$^+$", "slo": "BK channel", "KCNQ": "M current", "Shab": "Kv2", "Shal": "Kv4"}
TIERS = [("type_or_fuzzy", "cell type", "#1A1A1A"), ("lineage", "hemilineage", "#7A7A7A"), ("lineage_pair", "hemilineage pair", "#BDBDBD")]


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


def sci(x):
    e = int(np.floor(np.log10(abs(x))))
    return f"{x / 10 ** e:.0f} × 10$^{{{e}}}$"


def gradient(cv, x, y, w, h, colours):
    ax = cv.axes(x, y, w, h)
    ax.imshow(np.asarray(colours)[:, None, :3] * np.ones((1, 2, 1)), aspect="auto", extent=(0, 1, 0, 1), interpolation="bilinear")
    ax.set_axis_off()


# ---------------------------------------------------------------- A: single-gene stains

def panel_gallery(cv, sd):
    for k, name in enumerate(META["stains"]):
        x = k * (GW + fs.GUTTER)
        t = META["tiles"][name]
        cv.tile(x, 0, np.asarray(Image.open(TILES / f"{name}.png").convert("RGB")), width=GW, height=GH, group="gallery", label=name)
        tx = x + (4.4 if k == 0 else 1.2)
        head = META["labels"][name] + (f" · {FAMILY[name]}" if name in FAMILY else "")
        tile_text(cv, tx, 1.1, head, va="top", size=fs.PT_SMALL if name == "SK" else fs.PT)
        sd.add("A", "stain", stain=META["labels"][name], mapped_located=int(t["mapped_located"]), above_mean_located=int(t["above_mean_located"]),
               z_range=t["z_range"], tier_gain=t["tier_gain"])
    white_letter(cv, 1.2, 0.9, "A")
    bx, by, bh = GW - 2.2, GH - 21.0, 9.0
    gradient(cv, bx, by, 1.1, bh, pal.MAGNITUDE_ON_DARK(np.linspace(1.0, 0.15, 96)))
    z0, z1 = META["tiles"]["SK"]["z_range"]
    tile_text(cv, bx - 0.4, by + 0.3, f"z {z1:g}", va="center", ha="right")
    tile_text(cv, bx - 0.4, by + bh - 0.3, f"{z0:g}", va="center", ha="right")
    gains = META["tiles"]["SK"]["tier_gain"]
    base = np.asarray(pal.MAGNITUDE_ON_DARK(0.75)[:3])                # brightness by mapping tier, as swatches
    cv.key(0.0, GH + 2.4, [([dict(kind="box", size=1.8, fc=tuple(base * gains[k]))], lab) for k, lab in
                           (("exact", "cell type"), ("lineage", "hemilineage"), ("lineage_pair", "pair"), ("class", "class"))]
           + [([dict(kind="box", size=1.8, fc="#7F7F7F")], "unmapped")], size=fs.PT_SMALL, gap=1.6)


# ---------------------------------------------------------------- B: where the cells map

def panel_coverage(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "B", "Where the cells map")
    c = cm.table("plan_v2/fig4_coverage.csv")
    lab = {"all": "all cells", "VNC compartment (vnc_*, ascending)": "VNC cells", "readout cells": "readout cells",
           "trace carriers (m0)": "carriers, scene 1", "trace carriers (m0_joint)": "carriers, scene 1, brain + readout",
           "trace carriers (s1)": "carriers, scene 2", "trace carriers (s2)": "carriers, scene 3", "trace carriers (m0), VNC compartment": "carriers, scene 1, VNC"}
    c = c.set_index("cell_set").loc[[k for k in ["all", "VNC compartment (vnc_*, ascending)", "readout cells", "trace carriers (m0)",
                                                 "trace carriers (m0), VNC compartment", "trace carriers (m0_joint)", "trace carriers (s1)",
                                                 "trace carriers (s2)"]]].reset_index()
    lab_w = 46.0
    ax = cv.axes(x0 + lab_w, y0 + 9.4, w - lab_w - 7.0, h - 17.4)
    for i, (_, r) in enumerate(c.iterrows()):
        left = 0.0
        for col, _, colour in TIERS:
            v = float(r[col])
            ax.add_patch(Rectangle((left, i - 0.32), v, 0.64, facecolor=colour, edgecolor="none"))
            left += v
        ax.text(left + 0.01, i, f"{100 * left:.0f}%", fontsize=fs.PT_SMALL, ha="left", va="center")
        sd.add("B", "coverage", cell_set=lab[r["cell_set"]], cells=int(r["cells"]), **{t: float(r[col]) for col, t, _ in TIERS})
    ax.set_yticks(range(len(c)), [f"{lab[s]} ({int(n):,})" for s, n in zip(c["cell_set"], c["cells"])])
    ax.set_ylim(len(c) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "50", "100%"])
    ax.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL)
    fs.clean(ax)
    ax.spines["left"].set_visible(False)
    cv.key(x0 + 4.0, y0 + 6.0, [([dict(kind="box", fc=colour)], t) for _, t, colour in TIERS], size=fs.PT_SMALL, gap=1.4)


# ---------------------------------------------------------------- C: the atlases agree with the connectome

def panel_concordance(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "C", "Atlases agree with the connectome")
    q = cm.table("plan_v2/s4_mapping_qc.csv")
    tc = q[q["kind"] == "transmitter concordance"].set_index("source")
    mr = q[q["kind"] == "map rows present"].set_index("source")
    order = ["Davis 2020", "Özel 2021", "Davie 2018", "AFCA 2023", "AD-FCA 2025"]
    lab_w, txt = 19.0, 12.0
    hw = (w - lab_w - 2 * txt - 1.0) / 2
    for j, (tab, head) in enumerate(((tc, "transmitter = EM"), (mr, "map rows present"))):
        ax = cv.axes(x0 + lab_w + j * (hw + txt), y0 + 10.4, hw, h - 19.4)
        for i, s in enumerate(order):
            a, t = float(tab.loc[s, "agree"]), float(tab.loc[s, "total"])
            ax.add_patch(Rectangle((0, i - 0.3), a / t, 0.6, facecolor=fs.TEXT, edgecolor="none"))
            ax.add_patch(Rectangle((a / t, i - 0.3), 1 - a / t, 0.6, facecolor="#E0E0E0", edgecolor="none"))
            ax.text(1.03, i, f"{int(a)}/{int(t)}", fontsize=fs.PT_SMALL, ha="left", va="center")
            sd.add("C", head, atlas=s, agree=int(a), total=int(t))
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 1], ["0", "100%"])
        ax.set_yticks(range(len(order)), order if j == 0 else [""] * len(order))
        ax.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL)
        fs.clean(ax)
        ax.spines["left"].set_visible(False)
        cv.text(x0 + lab_w + j * (hw + txt) + hw / 2, y0 + 9.6, head, size=fs.PT_SMALL, ha="center", va="bottom")
    agree, total = int(tc["agree"].sum()), int(tc["total"].sum())
    rep = q[q["kind"] == "reproduction"].iloc[0]
    sd.add("C", "summary", transmitter_agree=agree, transmitter_total=total, reproduction_units=int(rep["units"]), reproduction_genes=int(rep["genes"]),
           reproduction_max_abs_diff=float(rep["max_abs_diff"]))


# ---------------------------------------------------------------- D: the atlases agree with each other

def panel_cross(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "D", "Atlases agree with each other")
    q = cm.table("plan_v2/s4_mapping_qc.csv")
    q = q[q["kind"] == "cross-source agreement"].reset_index(drop=True)
    lab_w = 22.0
    ax = cv.axes(x0 + lab_w, y0 + 9.4, w - lab_w - 1.6, h - 17.4)
    for i, r in q.iterrows():
        ax.plot([r["mismatched_rho"], r["matched_rho"]], [i, i], color=fs.GRID, lw=0.8, zorder=1)
        ax.scatter([r["mismatched_rho"]], [i], s=10, facecolor="white", edgecolor=fs.TEXT, linewidth=0.6, zorder=3)
        ax.scatter([r["matched_rho"]], [i], s=10, facecolor=fs.TEXT, edgecolor=fs.TEXT, zorder=3)
        sd.add("D", "pair", atlas=r["source"], compared_with=r["compared_with"], matched_pairs=int(r["matched_pairs"]), matched_rho=float(r["matched_rho"]),
               mismatched_pairs=int(r["mismatched_pairs"]), mismatched_rho=float(r["mismatched_rho"]))
    ax.set_yticks(range(len(q)), [f"{a} vs{NL_}{b}" for a, b in zip(q["source"], q["compared_with"])], linespacing=1.0)
    ax.set_ylim(len(q) - 0.5, -0.5)
    ax.set_xlim(0.4, 1.0)
    ax.set_xticks([0.4, 0.7, 1.0])
    ax.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL)
    ax.set_xlabel(f"Median expression {fs.RHO}", labelpad=0.8)
    fs.clean(ax)
    ax.spines["left"].set_visible(False)
    cv.key(x0 + 4.0, y0 + 6.0, [([dict(kind="mark", s=8, fc=fs.TEXT, ec=fs.TEXT)], "same type"),
                                ([dict(kind="mark", s=8, fc="white", ec=fs.TEXT)], "different")], size=fs.PT_SMALL, gap=1.2)


# ---------------------------------------------------------------- E: markers in the mapped units

def panel_markers(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "E", "Markers light up where expected")
    q = cm.table("plan_v2/s4_mapping_qc.csv")
    q = q[q["kind"] == "marker genes"]
    pops = [("Kenyon cells", "Kenyon cells"), ("projection neurons", "projection neurons"), ("ring neurons (GABAergic)", "ring neurons")]
    order = ["Davis 2020", "Özel 2021", "Davie 2018", "AFCA 2023", "AD-FCA 2025"]
    ax = cv.axes(x0 + 9.0, y0 + 9.4, w - 10.6, h - 18.4)
    for i, (pop, lab) in enumerate(pops):
        r = q[q["population"] == pop]
        for j, (_, rr) in enumerate(r.iterrows()):
            ax.scatter([i + (j - (len(r) - 1) / 2) * 0.16], [rr["mean_z"]], s=12, facecolor=fs.TEXT, edgecolor="none", zorder=3)
            sd.add("E", "marker", population=lab, atlas=rr["source"], genes=rr["genes"], units=int(rr["units"]), mean_z=float(rr["mean_z"]))
    ax.axhline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
    ax.set_xlim(-0.6, len(pops) - 0.4)
    ax.set_ylim(0, 5.2)
    ax.set_xticks(range(len(pops)), [f"{lab.split()[0]}{NL_}{' '.join(lab.split()[1:])}" for _, lab in pops], linespacing=1.0)
    ax.tick_params(axis="x", length=0, labelsize=fs.PT_SMALL)
    ax.set_ylabel("Marker genes, mean z", labelpad=0.8)
    fs.clean(ax)
    present = [s for s in order if (q["source"] == s).any()]
    cv.text(x0 + 4.0, y0 + 6.0, "one dot per atlas; ey, Pka-C1 · acj6 · Gad1, VGAT", size=fs.PT_SMALL, va="center", color=fs.TEXT2)
    sd.add("E", "atlases", atlases=present)


# ---------------------------------------------------------------- F: SK with and without the hemilineage pairs

def panel_strict(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "F", "SK enrichment does not hinge on the hemilineage pairs")
    s = cm.table("plan_v2/s4_sk_mapping.csv")
    rows = ["all cells, any tier", "all cells, without hemilineage pairs", "VNC, type and hemilineage", "VNC, without hemilineage pairs"]
    lab_w = 40.0
    hw = (w - lab_w - 4.4 - 1.0) / 2
    for j, null in enumerate(("matched sets", "mapping null")):
        ax = cv.axes(x0 + lab_w + j * (hw + 4.4), y0 + 10.4, hw, h - 18.4)
        for i, m in enumerate(rows):
            r = s[(s["mapping"] == m) & (s["null"] == null)].iloc[0]
            ax.plot([0, r["z"]], [i, i], color=pal.STATE_SET["m0"], lw=0.8, zorder=2)
            ax.scatter([r["z"]], [i], s=12, facecolor=pal.STATE_SET["m0"] if "without" not in m else "white", edgecolor=pal.STATE_SET["m0"], linewidth=0.6, zorder=3)
            sd.add("F", null, mapping=m, carriers=int(r["carriers"]), coverage=float(r["coverage"]), z=float(r["z"]))
        ax.axvline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=1)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_yticks(range(len(rows)), [f"{m} ({int(s[(s['mapping'] == m) & (s['null'] == null)]['carriers'].iloc[0]):,})" for m in rows] if j == 0 else [""] * len(rows))
        ax.set_xlim(0, 20 if null == "matched sets" else 3)
        ax.tick_params(axis="y", length=0, labelsize=fs.PT_SMALL)
        fs.clean(ax)
        ax.spines["left"].set_visible(False)
        cv.text(x0 + lab_w + j * (hw + 4.4) + hw / 2, y0 + 9.6, f"SK, {null}", size=fs.PT_SMALL, ha="center", va="bottom")
    cv.text(x0 + lab_w + (2 * hw + 4.4) / 2, y0 + h - 2.4, "Above the null mean (null SDs), scene 1", size=fs.PT_SMALL + 0.5,
            ha="center", va="center")


# ---------------------------------------------------------------- G: the cue's routes in the VNC, by hemilineage (revision 3)

HELD = "#EBC05A"                                         # as Fig. 2K: the type held the trace through the silent delay
J_THRESH, J_MAX_EDGES = 50.0, 22                         # cue drive (mV) of a candidate edge; edges drawn
J_CHIP_H, J_PITCH, J_COL_GAP, J_PAD_TOP, J_PAD, J_CHANNEL = 2.8, 3.9, 2.4, 3.4, 0.8, 1.8
J_NET_W = 52.0
J_WRAP = 16.0                                            # motor-neuron names wider than this (mm) take two lines
MN = "motor neurons"


def j_units(fd, x):
    """VNC types mapped through their hemilineage (more than half their cells), and the hemilineage of each."""
    x = x.merge(fd[["cell", "type", "superclass"]], on="cell")
    x = x[x["superclass"].astype(str).str.startswith("vnc") & x["tier"].isin(["lineage", "lineage_pair"])]
    share = x.groupby("type").size() / fd[fd["type"].isin(x["type"].unique())].groupby("type").size()
    unit = x.groupby("type")["unit"].agg(lambda s: s.value_counts().index[0])
    return unit[share.reindex(unit.index) > 0.5]


def j_select(ty, e, unit, ro):
    sc_ = ty["superclass"].astype(str)
    mn = set(sc_[sc_.isin(["vnc_motor", "vnc_efferent"])].index) & ro
    mapped = set(unit.index) - mn
    e = e[(e["abs_drive"] >= J_THRESH) & e["pre"].isin(mapped) & (e["post"].isin(mapped) | e["post"].isin(mn))]
    targets = list(e[e["post"].isin(mn)].groupby("post")["abs_drive"].sum().sort_values(ascending=False).index[:4])
    senders = list(e[e["post"].isin(targets)].groupby("pre")["abs_drive"].sum().sort_values(ascending=False).index[:6])
    upstream = list(e[e["post"].isin(senders) & ~e["pre"].isin(senders)].groupby("pre")["abs_drive"].sum().sort_values(ascending=False).index[:3])
    nodes = list(dict.fromkeys(targets + senders + upstream))
    shown = e[e["pre"].isin(nodes) & e["post"].isin(nodes)].sort_values("abs_drive", ascending=False).head(J_MAX_EDGES)
    used = set(shown["pre"]) | set(shown["post"])
    return [n for n in nodes if n in used], shown, mn


def j_wrap(name):
    """A motor-neuron name wider than J_WRAP on two lines, broken at the space nearest its middle."""
    if fs.text_width(name, fs.PT_SMALL) <= J_WRAP or " " not in name:
        return name
    cuts = [i for i, ch in enumerate(name) if ch == " "]
    i = min(cuts, key=lambda c: abs(c - len(name) / 2))
    return name[:i] + NL_ + name[i + 1:]


def panel_hemilineages(cv, sd, x0, y0, w, h):
    title(cv, x0, y0, "G", "Through SK-rich hemilineages to the motor neurons")
    ty = cm.table("plan_v2/fig2_cue_flow_types.csv").set_index("type")
    e = cm.table("plan_v2/fig2_cue_flow_edges.csv")
    e = e[(e["pre"] != "untyped") & (e["post"] != "untyped") & (e["pre"] != e["post"])]
    ro = set(ty[ty["role"].astype(str).str.contains("readout")].index)
    hem = cm.table("plan_v2/fig4_hemilineages.csv").set_index("unit")
    unit = j_units(cm.table("figure_data/fd_cells.csv"), cm.table("plan_v2/fig4_cell_expression.csv")[["cell", "tier", "unit"]])
    nodes, shown, mn = j_select(ty, e, unit, ro)
    module = {n: (MN if n in mn else unit[n]) for n in nodes}
    members = {}
    for n in nodes:
        members.setdefault(module[n], []).append(n)
    has_in, has_out = set(shown["post"]), set(shown["pre"])
    depth = {m: np.mean([2 if n in mn else (1 if n in has_in else 0) for n in ns]) for m, ns in members.items()}
    hl = sorted([m for m in members if m != MN], key=lambda m: (depth[m], -len(members[m])))
    label = {n: j_wrap(n) if n in mn else n for n in nodes}
    widths = {n: max(fs.text_width(s, fs.PT_SMALL) for s in label[n].split(NL_)) + 2 * ci.CHIP_PAD for n in nodes}
    size = {}
    for m, ns in members.items():
        cols = 1 if len(ns) <= 4 else 2
        rows = int(np.ceil(len(ns) / cols))
        colw = max(widths[n] for n in ns) + J_COL_GAP
        size[m] = (cols, rows, colw, max(cols * colw - J_COL_GAP + 2 * J_PAD, fs.text_width(m, fs.PT_SMALL) + 2.6),
                   J_PAD_TOP + rows * J_PITCH + J_PAD - 0.6)
    def stack(ms):
        return sum(size[m][4] for m in ms) + J_CHANNEL * max(len(ms) - 1, 0)
    from itertools import product                          # two stacked columns: the assignment with the lowest taller column,
    best = None                                            # upstream hemilineages preferred in the left one
    for bits in product((0, 1), repeat=len(hl)):
        cols_ = [[m for m, c in zip(hl, bits) if c == k_] for k_ in (0, 1)]
        if not all(cols_):
            continue
        key = (round(max(stack(cols_[0]), stack(cols_[1])), 3), sum(depth[m] for m in cols_[0]) - sum(depth[m] for m in cols_[1]))
        if best is None or key < best[0]:
            best = (key, bits)
    col_of = dict(zip(hl, best[1])) if best else {m: 0 for m in hl}
    top = y0 + 5.6
    col_w = [max([size[m][3] for m in hl if col_of[m] == c] or [0]) for c in (0, 1)]
    mn_w = max(widths[n] for n in members[MN]) + 2 * J_PAD
    chan = (J_NET_W - 0.6 - sum(col_w) - mn_w) / 2
    if chan < 2.0:
        raise RuntimeError(f"S4G: the network does not fit in {J_NET_W} mm (channel {chan:.1f} mm)")
    xs = [x0 + 0.3, x0 + 0.3 + col_w[0] + chan, x0 + J_NET_W - 0.3 - mn_w]
    boxes, heights = {}, [top, top]
    for m in hl:
        c = col_of[m]
        boxes[m] = (xs[c], heights[c], xs[c] + col_w[c], heights[c] + size[m][4])
        heights[c] += size[m][4] + J_CHANNEL
    bottom = max(heights) - J_CHANNEL
    boxes[MN] = (xs[2], top, x0 + J_NET_W - 0.3, bottom)
    ky = y0 + h - 4.2
    if bottom > ky - 4.0:
        raise RuntimeError(f"S4G: the network ({bottom - y0:.1f} mm) runs into its keys")
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    pos = {}
    for m, (bx0, by0, bx1, by1) in boxes.items():
        if m == MN:
            ci.module(cv, bx0, by0, bx1 - bx0, by1 - by0, MN, size=fs.PT_SMALL)
            continue
        z = float(hem.loc[m, "z_SK"])
        ci.module(cv, bx0, by0, bx1 - bx0, by1 - by0, None, fill=ci.tint(to_hex(Z_CMAP(norm(np.clip(z, -2, 2)))), 0.62))
        cv.text(bx0 + 1.0, by0 + 0.9, m, size=fs.PT_SMALL, va="top", style="italic")
        cols, rows, colw, _, _ = size[m]
        ns = sorted(members[m], key=lambda n: (n in has_in and n not in has_out, n in has_in,
                                               -shown[(shown["pre"] == n) | (shown["post"] == n)]["abs_drive"].sum()))
        for k_, n in enumerate(ns):
            c, rr = divmod(k_, rows)
            gx = (bx1 - bx0 - cols * colw + J_COL_GAP) / 2
            pos[n] = (bx0 + gx + colw * c + (colw - J_COL_GAP) / 2, by0 + J_PAD_TOP + J_PITCH * (rr + 0.5))
    bx0, by0, bx1, by1 = boxes[MN]                          # motor neurons: down their module, level with their senders
    n_slots = int((by1 - by0 - J_PAD_TOP - J_PAD) // 5.6)                 # room for two-line chips
    if n_slots < len(members[MN]):
        raise RuntimeError("S4G: the motor-neuron module is too short for its chips")
    slots_y = list(by0 + J_PAD_TOP + (by1 - by0 - J_PAD_TOP - J_PAD) * (np.arange(n_slots) + 0.5) / n_slots)

    def want(n):
        x = shown[shown["post"] == n]
        return float(np.average([pos[p][1] for p in x["pre"]], weights=x["abs_drive"])) if len(x) else by0
    taken = set()
    for n in sorted(members[MN], key=want):
        s_ = min((s for s in range(n_slots) if s not in taken), key=lambda s: abs(slots_y[s] - want(n)))
        taken.add(s_)
        pos[n] = ((bx0 + bx1) / 2, slots_y[s_])
    labels = [(a0 + 0.8, b0 + 0.6, a1 - 0.8, b0 + 3.2) for m, (a0, b0, a1, b1) in boxes.items()]
    net = ci.Board(cv, (x0, y0 + 5.0, x0 + J_NET_W, bottom + 1.2), cell=0.45, margin=0.35, margin_in=1.0, obstacles=labels)
    for n in nodes:
        two = NL_ in label[n]
        net.add(n, label[n], *pos[n], w=widths[n], h=5.0 if two else J_CHIP_H, size=fs.PT_SMALL,
                fill=HELD if ty.loc[n, "hold"] > 0 else ("#FFFFFF" if n in mn else ci.GREY_CHIP), edge=fs.TEXT if n in mn else "none")
        sd.add("G", "type", type=n, module=module[n], held_trace=bool(ty.loc[n, "hold"] > 0), motor_neuron=n in mn)
    bins = list(np.linspace(J_THRESH, float(shown["abs_drive"].max()), 5))
    bins[-1] += 1e-6
    widths_e = tuple(v * 0.85 for v in ci.WIDTHS)
    for r in shown.itertuples():
        b = ci.bin_of(r.abs_drive, bins)
        tr = str(ty.loc[r.pre, "transmitter"])
        net.link(r.pre, r.post, widths_e[b], ci.colour_for(tr, b), ci.head_for(tr))
        sd.add("G", "edge", pre=r.pre, post=r.post, cue_drive=float(r.abs_drive), width_bin=b + 1, transmitter=tr)
    if net.draw():
        raise RuntimeError("S4G: edges could not be routed")
    # keys
    cv.text(x0 + 0.4, ky - 2.8, "cue drive (mV)", size=fs.PT_SMALL, va="center", color=fs.TEXT2)
    xe = ci.width_key(cv, x0 + 0.4, ky, [f"{bins[i]:.0f}–{bins[i + 1]:.0f}" for i in range(4)], widths=widths_e, length=3.0)
    ci.transmitter_key(cv, xe + 1.6, ky - 2.8)
    xk = cv.key(xe + 1.6, ky, [([dict(kind="box", size=1.8, fc=HELD)], "held the trace"),
                               ([dict(kind="box", size=1.8, fc="#FFFFFF", ec=fs.TEXT, lw=0.6)], "motor neuron")], size=fs.PT_SMALL, gap=1.4)
    cv.text(xe + 1.6, ky + 3.0, "tint: SK (z)", size=fs.PT_SMALL, va="center")
    tx = xe + 1.6 + fs.text_width("tint: SK (z)", fs.PT_SMALL) + 4.6
    for j, z in enumerate(np.linspace(-2, 2, 9)):
        cv.box(tx + j * 1.1, ky + 2.45, 1.1, 1.1, ci.tint(to_hex(Z_CMAP(norm(z))), 0.62))
    cv.text(tx - 0.4, ky + 3.0, fs.MINUS + "2", size=fs.PT_SMALL, va="center", ha="right")
    cv.text(tx + 9 * 1.1 + 0.4, ky + 3.0, "+2", size=fs.PT_SMALL, va="center")
    # subplot: each hemilineage's SK (z) and its share of trace carriers, rows by SK
    units = sorted([m for m in boxes if m != MN], key=lambda u: -float(hem.loc[u, "z_SK"]))
    n_u = len(units)
    sx = x0 + J_NET_W + 11.0
    yy = np.arange(n_u)[::-1]
    ah = min(bottom - top - 6.0, 3.6 * n_u)
    a1 = cv.axes(sx, top + 2.6, 9.0, ah)
    for yv, u in zip(yy, units):
        z = float(hem.loc[u, "z_SK"])
        a1.barh(yv, z, height=0.64, color=to_hex(Z_CMAP(norm(np.clip(z, -2, 2)))), linewidth=0)
        sd.add("G", "hemilineage", unit=u, z_SK=z, carrier_share=float(hem.loc[u, "carrier_share"]))
    a1.axvline(0, color=fs.AXIS, linewidth=0.5)
    a1.set_xlim(-1.5, 1.5)
    a1.set_xticks([-1, 1], [fs.MINUS + "1", "1"])
    a1.set_yticks(yy, units)
    a1.set_ylim(-0.6, n_u - 0.4)
    a1.set_xlabel("SK (z)", labelpad=0.8)
    fs.clean(a1)
    a1.tick_params(axis="y", length=0)
    a2 = cv.axes(sx + 10.4, top + 2.6, w - (sx + 10.4 - x0) - 1.6, ah)
    for yv, u in zip(yy, units):
        a2.barh(yv, 100 * float(hem.loc[u, "carrier_share"]), height=0.64, color=HELD, linewidth=0)
    a2.set_xlim(0, 40)
    a2.set_xticks([0, 40])
    a2.set_yticks([])
    a2.set_ylim(-0.6, n_u - 0.4)
    a2.set_xlabel("Carriers (%)", labelpad=0.8)
    fs.clean(a2, left=False)
    cv.text(sx - 9.0, top + 0.6, "Hemilineages", size=fs.PT_SMALL, va="center", color=fs.TEXT2)


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("s4")
    for rel in ("plan_v2/s4_cell_genes.csv", "plan_v2/fig4_coverage.csv", "plan_v2/s4_mapping_qc.csv", "plan_v2/s4_sk_mapping.csv",
                "plan_v2/fig2_cue_flow_types.csv", "plan_v2/fig2_cue_flow_edges.csv", "plan_v2/fig4_hemilineages.csv",
                "plan_v2/fig4_cell_expression.csv", "figure_data/fd_cells.csv"):
        sd.source(cm.RESULTS / rel)
    for f in sorted(TILES.glob("*.png")) + sorted(TILES.glob("*.json")):
        sd.source(f)
    panel_gallery(cv, sd)
    panel_coverage(cv, sd, XS[0], ROW2, fs.unit(2), R_H)
    panel_concordance(cv, sd, XS[2], ROW2, fs.unit(2), R_H)
    panel_cross(cv, sd, XS[0], ROW3, U, R_H)
    panel_markers(cv, sd, XS[1], ROW3, U, R_H)
    panel_strict(cv, sd, XS[2], ROW3, fs.unit(2), R_H)
    panel_hemilineages(cv, sd, XS[0], ROW4, fs.unit(2), R4_H)
    cm.finish(cv, "s4", sd, kind="supplement")


if __name__ == "__main__":
    main()
