"""Fig. 1 - A trainable spiking model of the entire male Drosophila CNS (180 x 168 mm).

a  unfolded CNS atlas: 140,024 cell bodies by superclass group (dark tile)
b  dorsal oblique view: 4,096 readout arbors, input cell bodies, DNp01 highlighted, LC12 and i2 MN insets (dark tile)
c  model schematic (trained terms in the accent colour)
d  parameter budget (log scale)
e  task library: 34 boards by tier and family, with certified optimal moves
f  D3-T0 curriculum: depth staircase, entry margin and loss
g  memory task: cue - delay - prefix - decision timeline and the 7 x 5 timing grid

Run after render/fig1_tiles.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\fig1.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from lib import boards as bd, figstyle as fs, palette as pal, qc  # noqa: E402

TILES = cm.RENDERS / "fig1"
BAND = json.loads((TILES / "band.json").read_text(encoding="utf-8"))
BAND_H = BAND["band_height_mm"]
ATLAS_W = BAND["atlas_mm"][0]
IFACE_X = ATLAS_W + fs.GUTTER
IFACE_W = BAND["interface_mm"][0]
ROW2 = BAND_H + 3.4
WHITE = fs.ON_DARK
NL_ = chr(10)
GROUP_ORDER = ["optic lobe intrinsic", "central brain intrinsic", "VNC intrinsic", "visual projection & centrifugal",
               "descending & ascending", "motor & efferent", "sensory", "other"]


def tile_text(cv, x, y, s, size=fs.PT_SMALL, **kw):
    t = cv.text(x, y, s, size=size, color=WHITE, **kw)
    t.set_path_effects(fs.dark_halo(0.8))
    return t


def scale_bar(cv, x_right, y, um, um_per_px, px_per_mm, label=None):
    """White 1-pt scale bar ending at x_right mm (bar length in the legend unless label is given)."""
    length = um / um_per_px / px_per_mm
    cv.line([x_right - length, x_right], [y, y], WHITE, lw=1.0, solid_capstyle="butt")
    if label:
        tile_text(cv, x_right - length / 2, y - 0.6, label, ha="center", va="bottom")
    return length


def arrow_mm(ax, x0, y0, x1, y1, colour=fs.TEXT, lw=0.5, head=2.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=f"-|>,head_length={head * 0.4},head_width={head * 0.25}",
                                 mutation_scale=4, color=colour, linewidth=lw, shrinkA=0, shrinkB=0, zorder=5))


# ---------------------------------------------------------------- a, b: hero band

def panel_atlas(cv, sd):
    img = np.asarray(Image.open(TILES / "atlas.png").convert("RGB"))
    meta = json.loads((TILES / "atlas.json").read_text(encoding="utf-8"))
    cv.tile(0, 0, img, width=ATLAS_W, height=BAND_H, group="band", label="atlas")
    px_per_mm = img.shape[1] / ATLAS_W
    cv.letter(1.2, 1.0, "a")
    cv.fig.texts[-1].set_color(WHITE)
    counts = meta["group_counts"]
    y = 50.0
    tile_text(cv, 1.6, y - 3.4, "Cell bodies by" + NL_ + "superclass group", va="center", linespacing=1.0)
    for g in GROUP_ORDER:
        n = counts.get(g, 0)
        cv.mark(2.4, y, s=7.0, fc=pal.ATLAS_ON_DARK[g], ec=pal.ATLAS_ON_DARK[g], lw=0.3)
        tile_text(cv, 3.6, y, pal.SHORT_GROUP[g], va="center")
        tile_text(cv, 19.6, y, f"{n:,}", va="center", ha="right")
        sd.add("a", "cell bodies by group", group=g, cells=n)
        y += 2.55
    tile_text(cv, ATLAS_W - 1.4, BAND_H - 12.2, f"{meta['somata']:,} cell bodies", ha="right", va="center")
    tile_text(cv, ATLAS_W - 1.4, BAND_H - 9.9, f"of {meta['cells']:,} neurons", ha="right", va="center")
    scale_bar(cv, ATLAS_W - 1.6, BAND_H - 2.0, 100, meta["scale_um_per_px"], px_per_mm)
    # orientation: brain frontal (fly left on the right), VNC ventral (posterior down)
    ox, oy = ATLAS_W - 13.0, BAND_H - 6.2
    cv.line([ox, ox + 3.0], [oy, oy], WHITE, lw=0.5)
    tile_text(cv, ox + 3.4, oy, "L", va="center")
    cv.line([ox, ox], [oy, oy + 3.0], WHITE, lw=0.5)
    tile_text(cv, ox, oy + 3.4, "V", ha="center", va="top")
    sd.add("a", "view", unfold="brain frontal; VNC rotated 90 degrees about the neck (msa/cnsview.py)",
           um_per_px=meta["scale_um_per_px"], scale_bar_um=100)


def panel_interface(cv, sd):
    img = np.asarray(Image.open(TILES / "interface.png").convert("RGB"))
    meta = json.loads((TILES / "interface.json").read_text(encoding="utf-8"))
    cv.tile(IFACE_X, 0, img, width=IFACE_W, height=BAND_H, group="band", label="interface")
    px_per_mm = img.shape[1] / IFACE_W
    to_mm = lambda px_, py_: (IFACE_X + px_ / px_per_mm, py_ / px_per_mm)  # noqa: E731
    cv.letter(IFACE_X + 1.2, 1.0, "b")
    cv.fig.texts[-1].set_color(WHITE)
    for ins in meta["insets"]:
        x0, y0, x1, y1 = ins["box_px"]
        (ax0, ay0), (ax1, ay1) = to_mm(x0, y0), to_mm(x1, y1)
        role = "input" if ins["role"] == "input" else "readout, motor neuron"
        tile_text(cv, ax0, ay0 - 0.7, f"{ins['label']} ({role})", va="bottom")
        um = 20 if ins["scale_um_per_px"] < 0.2 else 50
        scale_bar(cv, ax1 - 1.2, ay1 - 1.4, um, ins["scale_um_per_px"], px_per_mm)
        sd.add("b", "inset", body_id=ins["body_id"], label=ins["label"], type=ins["type"], superclass=ins["superclass"],
               role=ins["role"], scale_bar_um=um, skeleton=ins["skeleton"])
    # key strip below the VNC
    ky = BAND_H - 9.4
    kx = IFACE_X + 0.40 * IFACE_W
    cv.mark(kx + 0.8, ky, s=6.0, fc=WHITE, ec=WHITE, lw=0.3)
    tile_text(cv, kx + 2.0, ky, f"input cell bodies ({meta['input_cells']:,} of 3,200 located)", va="center")
    ky2 = ky + 3.0
    seg = 0.55
    for k, g in enumerate(["central brain intrinsic", "optic lobe intrinsic", "visual projection & centrifugal",
                           "descending & ascending", "motor & efferent"]):
        cv.line([kx + k * seg, kx + (k + 1) * seg], [ky2, ky2], pal.ATLAS_ON_DARK[g], lw=1.0)
    tile_text(cv, kx + 3.4, ky2, "4,096 readout arbors, coloured as in a", va="center")
    cv.line([kx, kx + 2.75], [ky2 + 3.0, ky2 + 3.0], WHITE, lw=1.0)
    tile_text(cv, kx + 3.4, ky2 + 3.0, "DNp01 (giant fibre), a readout cell", va="center")
    scale_bar(cv, IFACE_X + IFACE_W - 1.6, BAND_H - 2.0, 100, meta["scale_um_per_px"], px_per_mm)
    # orientation glyph from the view: anterior and fly-left screen directions
    o = meta["orientation"]
    ox, oy = IFACE_X + 0.40 * IFACE_W - 8.0, BAND_H - 7.0
    for key, d in (("A", o["A"]), ("L", o["L"])):
        dx, dy = 3.0 * d[0], 3.0 * d[1]
        cv.line([ox, ox + dx], [oy, oy + dy], WHITE, lw=0.5)
        tile_text(cv, ox + dx * 1.35, oy + dy * 1.35, key, ha="center", va="center")
    for g, n in sorted(meta["readout_group_counts"].items()):
        sd.add("b", "readout arbors by group", group=g, cells=n)
    sd.add("b", "highlight", body_id=meta["highlight"]["body_id"], label="DNp01", skeleton=meta["highlight"]["skeleton"])
    sd.add("b", "view", um_per_px=meta["scale_um_per_px"], scale_bar_um=100, orientation_A=o["A"], orientation_L=o["L"])


# ---------------------------------------------------------------- c: model schematic

def action_cross(ax, cx, cy, size=1.25, gap=0.55, colour=fs.TEXT):
    """Four triangles pointing up, down, left and right: the four actions."""
    from matplotlib.patches import Polygon
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        tip = (cx + dx * (gap + size), cy + dy * (gap + size))
        base = (cx + dx * gap, cy + dy * gap)
        px_, py_ = -dy * size * 0.55, dx * size * 0.55
        ax.add_patch(Polygon([tip, (base[0] + px_, base[1] + py_), (base[0] - px_, base[1] - py_)], closed=True,
                             facecolor=colour, edgecolor="none"))


def badge(ax, x, y, text, pad=1.0, h=None):
    """Accent-outlined box sized to its text; returns (width, height)."""
    lines = text.split(NL_)
    w = max(fs.text_width(t, fs.PT_SMALL) for t in lines) + 2 * pad
    h = h or len(lines) * 2.1 + 1.2
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7", facecolor="white",
                                edgecolor=pal.ACCENT, linewidth=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs.PT_SMALL, linespacing=1.05)
    return w, h


def panel_schematic(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "c")
    ax = bd.mm_axes(cv, x0, y0, w, h)
    spec = cm.table("main/fig1/fig1a_model_specification.csv").set_index(["section", "quantity"])["value"]
    budget = cm.table("main/fig1/fig1b_parameter_budget.csv")
    cells = cm.table("figure_data/fd_cells.csv")
    boards_t = cm.table("figure_data/fd_boards.csv")
    n_cells = int(float(spec[("network", "neurons")]))
    n_entries = int(float(spec[("network", "stored_connectome_entries")]))
    n_in = int(float(spec[("network", "input_cells")]))
    n_obs = int(float(spec[("network", "observed_cells")]))
    core_rows = budget[(budget.model_kind == "connectome_linear") & (budget.component == "brain")]
    core = int(core_rows["n_elements"].sum())
    groups = int(core_rows["n_elements"].iloc[0])
    head = int(budget[(budget.model_kind == "connectome_linear") & (budget.component == "readout") &
                      cm.truthy(budget.trainable_in_joint)]["n_elements"].sum())
    row_y = 3.6
    # board
    b = bd.Board.from_table(boards_t, "D3-T0")
    cell = 0.9
    bw, bh = b.size_mm(cell)
    bd.draw(ax, b, 0.6, row_y, cell)
    mid = row_y + bh / 2
    ax.text(0.6 + bw / 2, row_y + bh + 1.0, "Board", ha="center", va="top", fontsize=fs.PT_SMALL)
    # input cells
    ix = 11.4
    vis = pal.on_white(pal.ATLAS_ON_DARK["visual projection & centrifugal"])
    for i in range(7):
        for j in range(4):
            ax.add_patch(plt_circle((ix + j * 1.0, row_y + 0.45 + i * 0.98), 0.34, vis))
    ax.text(ix + 1.5, row_y + bh + 1.0, f"{n_in:,}\ninputs", ha="center", va="top", fontsize=fs.PT_SMALL, linespacing=1.0)
    arrow_mm(ax, 0.6 + bw + 0.5, mid, ix - 0.7, mid)
    # the connectome (mini atlas tile)
    atlas = np.asarray(Image.open(TILES / "atlas.png").convert("RGB"))
    tw = 12.6
    th = tw * atlas.shape[0] / atlas.shape[1]
    tx, ty = 18.6, 0.8
    cv.tile(x0 + tx, y0 + ty, atlas, width=tw, height=th, group=None, label="schematic CNS")
    arrow_mm(ax, ix + 3.9, mid, tx - 0.5, mid)
    ax.text(tx + tw / 2, ty + th + 0.8, f"{n_cells:,}\nspiking neurons\n{n_entries / 1e6:.1f} M synaptic\nweights (fixed)",
            ha="center", va="top", fontsize=fs.PT_SMALL, linespacing=1.05)
    # readout pools (256 x 16 cells), coloured by their majority group
    ro = cells[cells["role"] == "readout"].copy()
    ro["group"] = ro["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other")
    major = ro.groupby("readout_population")["group"].agg(lambda v: v.value_counts().index[0])
    q = 0.56
    px0, py0 = 36.4, row_y - 0.4
    for k, (pid, g) in enumerate(major.items()):
        r, c = divmod(k, 16)
        ax.add_patch(Rectangle((px0 + c * q, py0 + r * q), q * 0.84, q * 0.84, facecolor=pal.on_white(pal.ATLAS_ON_DARK[g]),
                               edgecolor="none"))
    arrow_mm(ax, tx + tw + 0.5, mid, px0 - 0.6, mid)
    ax.text(px0 + 8 * q, py0 + 16 * q + 0.8, f"{n_obs:,} readout cells\n(256 pools of 16)", ha="center", va="top",
            fontsize=fs.PT_SMALL, linespacing=1.0)
    # 512 features -> trained readout -> actions
    cxm = px0 + 8 * q
    fy = py0 + 16 * q + 5.0
    arrow_mm(ax, cxm, fy, cxm, fy + 2.6)
    ax.text(cxm + 0.9, fy + 1.3, "512 features", ha="left", va="center", fontsize=fs.PT_SMALL)
    label = "readout (trained)\n2,052 parameters".replace("2,052", f"{head:,}")
    bw2 = max(fs.text_width(t, fs.PT_SMALL) for t in label.split(NL_)) + 2.0
    bx2 = cxm - bw2 / 2
    _, bh2 = badge(ax, bx2, fy + 3.0, label)
    arrow_mm(ax, bx2 + bw2 + 0.4, fy + 3.0 + bh2 / 2, bx2 + bw2 + 2.6, fy + 3.0 + bh2 / 2)
    action_cross(ax, bx2 + bw2 + 5.0, fy + 3.0 + bh2 / 2, size=1.0, gap=0.45)
    ax.text(bx2 + bw2 + 5.0, fy + 3.0 + bh2 / 2 + 3.0, "action", ha="center", va="top", fontsize=fs.PT_SMALL)
    # trained core
    cw, ch = badge(ax, 0.6, h - 6.6, f"trained: input gain, output gain and\nthreshold of {groups:,} cell groups ({core:,})")
    arrow_mm(ax, tx + tw / 2, h - 6.9, tx + tw / 2, ty + th + 9.6, colour=pal.ACCENT)
    sd.add("c", "model", neurons=n_cells, synaptic_entries=n_entries, input_cells=n_in, readout_cells=n_obs,
           trained_core=core, cell_groups=groups, trained_readout=head)


def plt_circle(xy, r, colour):
    from matplotlib.patches import Circle
    return Circle(xy, r, facecolor=colour, edgecolor="none")


# ---------------------------------------------------------------- d: parameter budget

def panel_budget(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "d")
    spec = cm.table("main/fig1/fig1a_model_specification.csv").set_index(["section", "quantity"])["value"]
    b = cm.table("main/fig1/fig1b_parameter_budget.csv")
    tot = b[b.component == "total"].set_index(["model_kind", "parameter"])["n_elements"]
    core = int(b[(b.model_kind == "connectome_linear") & (b.component == "brain")]["n_elements"].sum())
    head = int(b[(b.model_kind == "connectome_linear") & (b.component == "readout") & cm.truthy(b.trainable_in_joint)]["n_elements"].sum())
    rows = [("Synaptic weights (fixed)", float(spec[("network", "stored_connectome_entries")]), pal.FIXED),
            ("Gains and thresholds", core, pal.ACCENT),
            ("Readout", head, pal.ACCENT),
            ("Board MLP (comparison)", int(tot[("board_mlp", "trainable")]), "#5E5E5E"),
            ("GRU (comparison)", int(tot[("gru", "trainable")]), "#5E5E5E")]
    ax = cv.axes(x0 + 23.5, y0 + 4.0, w - 27.5, h - 10.5)
    ys = np.arange(len(rows))[::-1]
    for yv, (lab, n, c) in zip(ys, rows):
        ax.barh(yv, n, height=0.62, color=c, left=1, linewidth=0)
        ax.text(n * 1.25, yv, f"{n:,.0f}", va="center", ha="left", fontsize=fs.PT_SMALL)
        sd.add("d", "parameters", item=lab, n=n)
    ax.set_xscale("log")
    ax.set_xlim(1e3, 4e8)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([1e3, 1e5, 1e7])
    fs.minor(ax, y=False)
    ax.set_xlabel("Parameters")
    fs.clean(ax)
    ax.tick_params(axis="y", length=0)


# ---------------------------------------------------------------- e: task library

def panel_boards(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "e")
    lib = cm.table("main/fig1/fig1d_task_library.csv")
    boards_t = cm.table("figure_data/fd_boards.csv")
    cols = ["T0", "T1", "T2", "H", "C"]
    col_names = ["T0", "T1", "T2", "held-out", "calib."]
    top = y0 + 8.6
    pitch = (y0 + h - top) / 8
    cell = (pitch - 1.6) / 10
    colw = 10 * cell + 0.9
    left = x0 + 4.2
    ax = bd.mm_axes(cv, x0, y0, w, h)
    for j, name in enumerate(col_names):
        ax.text(left - x0 + j * colw + 5 * cell, top - y0 - 1.0, name, ha="center", va="bottom", fontsize=fs.PT_SMALL)
    ax.text(0.2, top - y0 - 1.0, "Tier", ha="left", va="bottom", fontsize=fs.PT_SMALL)
    for _, r in lib.iterrows():
        tier = int(r["tier"])
        fam = r["map_id"].split("-")[1] if "-" in r["map_id"] else "C"
        j = cols.index(fam)
        b = bd.Board.from_table(boards_t, r["map_id"])
        bw, bh = b.size_mm(cell)
        bx = left - x0 + j * colw + (10 * cell - bw) / 2
        by = top - y0 + tier * pitch + (10 * cell - bh) / 2
        bd.draw(ax, b, bx, by, cell)
        sd.add("e", "board", map_id=r["map_id"], tier=tier, family=fam, optimal_moves=int(r["optimal_moves"]),
               boxes=int(r["boxes"]), board_size=int(r["board_size"]))
    for t in range(8):
        ax.text(1.4, top - y0 + t * pitch + 5 * cell, str(t), ha="center", va="center", fontsize=fs.PT_SMALL)
    # certified optimum per board, aligned with the tier rows
    sx = left + 5 * colw + 1.6
    sax = cv.axes(sx, top, x0 + w - sx - 0.6, 8 * pitch - 1.6)
    for _, r in lib.iterrows():
        tier = int(r["tier"])
        fam = r["map_id"].split("-")[1] if "-" in r["map_id"] else "C"
        style = dict(T0=("o", fs.TEXT), T1=("o", fs.TEXT), T2=("o", fs.TEXT), H=("o", "white"), C=("D", "#9E9E9E"))[fam]
        sax.scatter([r["optimal_moves"]], [tier + 0.42], s=7, marker=style[0], facecolor=style[1], edgecolor=fs.TEXT if fam != "C" else "#9E9E9E",
                    linewidth=0.45, zorder=3)
    sax.set_ylim(7.95, -0.15)
    sax.set_xlim(0, 64)
    sax.set_yticks([])
    sax.xaxis.tick_top()
    sax.xaxis.set_label_position("top")
    sax.set_xticks([0, 30, 60])
    fs.minor(sax, y=False)
    for side in ("bottom", "left", "right"):
        sax.spines[side].set_visible(False)
    sax.spines["top"].set_visible(True)
    sax.set_xlabel("Optimal moves", labelpad=1.0, fontsize=fs.PT_SMALL)
    for t in range(8):
        sax.axhline(t + 0.42, color=fs.GRID, lw=fs.LW_HAIR, zorder=0)


# ---------------------------------------------------------------- f: curriculum

def panel_curriculum(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "f")
    c = cm.table("main/fig1/fig1c_curriculum_example.csv")
    c = c[cm.truthy(c["featured"])].sort_values("assessment")
    lib = cm.table("main/fig1/fig1d_task_library.csv").set_index("map_id")
    L = int(lib.loc["D3-T0", "optimal_moves"])
    lx, aw = x0 + 10.5, w - 12.0
    a1 = cv.axes(lx, y0 + 4.4, aw, h - 21.1)
    a2 = cv.axes(lx, y0 + h - 15.0, aw, 4.6)
    a3 = cv.axes(lx, y0 + h - 9.6, aw, 4.6)
    u = c["updates"].to_numpy()
    d = c["depth"].to_numpy()
    a1.fill_between(u, 0, d, step="post", color=pal.ACCENT, alpha=0.12, lw=0)
    a1.step(u, d, where="post", color=pal.ACCENT, lw=0.9)
    passed = c[cm.truthy(c["stage_passed"])]
    a1.scatter(passed["updates"], passed["depth"], s=5, facecolor="white", edgecolor=pal.ACCENT, linewidth=0.5, zorder=4)
    last = c.iloc[-1]
    a1.scatter([last["updates"]], [last["depth"]], s=16, marker="*", facecolor=fs.TEXT, edgecolor=fs.TEXT, linewidth=0.3, zorder=5)
    a1.text(last["updates"] - 4, last["depth"] - 5.2, f"solved after\n{int(last['updates'])} updates", ha="right", va="top",
            fontsize=fs.PT_SMALL, linespacing=1.0)
    a1.set_ylim(0, L + 1.5)
    a1.set_yticks([0, 10, 20, L])
    a1.set_ylabel("Depth\n(moves)", linespacing=1.0)
    m = c["entry_margin"].to_numpy()
    a2.fill_between(u, 0, np.maximum(m, 0), step="post", color="#5E5E5E", lw=0)
    a2.fill_between(u, 0, np.minimum(m, 0), step="post", color="#C8C8C8", lw=0)
    a2.axhline(0, color=fs.AXIS, lw=fs.LW_HAIR)
    a2.set_ylabel("Margin", rotation=0, ha="right", va="center", labelpad=2)
    a2.set_ylim(-2.2, 2.2)
    a2.set_yticks([-2, 0, 2])
    a3.semilogy(u, np.maximum(c["loss"].to_numpy(), 1e-4), color="#5E5E5E", lw=0.5, drawstyle="steps-post")
    a3.set_ylabel("Loss", rotation=0, ha="right", va="center", labelpad=2)
    a3.set_ylim(1e-4, 10)
    a3.set_yticks([1e-4, 1e-2, 1])
    a3.set_yticklabels(["10$^{-4}$", "10$^{-2}$", "1"])
    for a in (a1, a2, a3):
        a.set_xlim(0, u.max() * 1.01)
        fs.clean(a)
    for a in (a1, a2):
        a.tick_params(axis="x", labelbottom=False)
    a3.set_xlabel("Parameter updates", labelpad=1.0)
    cv.text(x0 + 3.2, y0 + 0.35, "Curriculum on D3-T0 (stages grow back from the goal)", size=fs.PT_SMALL, va="top")
    for _, r in c.iterrows():
        sd.add("f", "assessment", assessment=int(r["assessment"]), updates=int(r["updates"]), depth=int(r["depth"]),
               entry_margin=float(r["entry_margin"]), loss=float(r["loss"]), stage_passed=str(r["stage_passed"]).lower() == "true")


# ---------------------------------------------------------------- g: memory task

def panel_memory(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "g")
    boards_t = cm.table("figure_data/fd_boards.csv")
    design = cm.table("main/fig1/fig1e_memory_task_design.csv")
    ax = bd.mm_axes(cv, x0, y0, w, h)
    cell = 0.78
    m0 = {cue: bd.Board.from_table(boards_t, f"M0:c0-cue{cue}") for cue in (0, 1)}
    bw, bh = m0[0].size_mm(cell)
    segs = [("cue", "cue\n40 ms"), ("delay", "delay\n0-640 ms"), ("prefix", "prefix\n0-8 windows"), ("decision", "decision\nfirst push")]
    seg_w = 12.4
    gap = (w - 3.2 - 4 * seg_w) / 3
    xs = [3.2 + k * (seg_w + gap) for k in range(4)]
    ty = 3.0
    for k, cue in enumerate((0, 1)):                      # both cue frames, goal in the cue colour
        bd.draw(ax, m0[cue], xs[0] + k * (bw + 0.8) + (seg_w - 2 * bw - 0.8) / 2, ty, cell, goal_colour=pal.CUE[cue])
    ax.add_patch(Rectangle((xs[1] + (seg_w - bw) / 2, ty), bw, bh, facecolor="#2B2B2B", edgecolor="none"))
    b = m0[0]
    bx = xs[2] + (seg_w - bw) / 2
    bd.draw(ax, b, bx, ty, cell, show_goals=False)
    bd.path(ax, b, [44, 45, 44, 43, 44], bx, ty, cell, colour=pal.ACCENT, lw=0.5, z=6)
    dx = xs[3] + (seg_w - bw) / 2
    bd.draw(ax, b, dx, ty, cell, show_goals=False)
    bd.arrow(ax, b, 34, 2, dx, ty, cell, colour=pal.CUE[0], lw=0.8, length=2.6, head=1.0)
    bd.arrow(ax, b, 34, 3, dx, ty, cell, colour=pal.CUE[1], lw=0.8, length=2.6, head=1.0)
    yb = ty + bh + 1.4
    shades = ["#E4E4E4", "#3A3A3A", "#BDBDBD", "#E4E4E4"]
    for x, c, (_, lab) in zip(xs, shades, segs):
        ax.add_patch(Rectangle((x, yb), seg_w, 1.1, facecolor=c, edgecolor=fs.AXIS, linewidth=0.3))
        ax.text(x + seg_w / 2, yb + 1.7, lab, ha="center", va="top", fontsize=fs.PT_SMALL, linespacing=1.0)
    t = design[design["section"] == "timing"].copy()
    delays = sorted(t["delay_ms"].astype(int).unique())
    prefixes = sorted(t["prefix_windows"].astype(int).unique())
    gx0, gy0 = 11.0, yb + 9.6
    gw = (w - gx0 - 0.6) / len(delays)
    gh = (h - gy0 - 5.4) / len(prefixes)
    for _, r in t.iterrows():
        i = delays.index(int(r["delay_ms"]))
        j = prefixes.index(int(r["prefix_windows"]))
        trained = r["split"] == "train"
        ax.add_patch(Rectangle((gx0 + i * gw + 0.2, gy0 + j * gh + 0.2), gw - 0.4, gh - 0.4, facecolor="#4D4D4D" if trained else "white",
                               edgecolor="#4D4D4D" if trained else "#A8A8A8", linewidth=0.35))
        sd.add("g", "timing", delay_ms=int(r["delay_ms"]), prefix_windows=int(r["prefix_windows"]), split=r["split"])
    for i, dly in enumerate(delays):
        ax.text(gx0 + (i + 0.5) * gw, gy0 + len(prefixes) * gh + 0.5, str(dly), ha="center", va="top", fontsize=fs.PT_SMALL)
    for j, p in enumerate(prefixes):
        ax.text(gx0 - 0.7, gy0 + (j + 0.5) * gh, str(p), ha="right", va="center", fontsize=fs.PT_SMALL)
    ax.text(gx0 + len(delays) * gw / 2, h, "Delay (ms)", ha="center", va="bottom", fontsize=fs.PT)
    ax.text(0.2, gy0 + len(prefixes) * gh / 2, "Prefix\nwindows", ha="left", va="center", fontsize=fs.PT, linespacing=1.0)
    n_tr = int((t["split"] == "train").sum())
    ax.add_patch(Rectangle((gx0, gy0 - 2.6), 1.6, 1.0, facecolor="#4D4D4D", edgecolor="none"))
    ax.text(gx0 + 2.2, gy0 - 2.1, f"trained timing ({n_tr} of {len(t)}); the rest are untrained", ha="left", va="center", fontsize=fs.PT_SMALL)
    for cue in (0, 1):
        r = design[(design["section"] == "case") & (design["family"] == "M0") & (design["cue"] == cue)].iloc[0]
        sd.add("g", "M0 case", cue=cue, goal=r["goals"], first_push_action=int(r["first_push_action"]))


def main():
    cv = fs.Canvas(fs.FULL, 168.0)
    sd = qc.SourceData("fig1")
    for rel in ("figure_data/fd_cells.csv", "figure_data/fd_boards.csv", "main/fig1/fig1a_model_specification.csv",
                "main/fig1/fig1b_parameter_budget.csv", "main/fig1/fig1c_curriculum_example.csv", "main/fig1/fig1d_task_library.csv",
                "main/fig1/fig1e_memory_task_design.csv"):
        sd.source(cm.RESULTS / rel)
    for f in ("atlas.png", "interface.png", "atlas.json", "interface.json"):
        sd.source(TILES / f)
    panel_atlas(cv, sd)
    panel_interface(cv, sd)
    panel_schematic(cv, sd, 0.0, ROW2, fs.span(2), 39.5)
    panel_budget(cv, sd, 0.0, ROW2 + 42.0, fs.span(2), 168.0 - ROW2 - 42.0)
    panel_boards(cv, sd, fs.col_x(2), ROW2, fs.span(2), 168.0 - ROW2)
    panel_curriculum(cv, sd, fs.col_x(4), ROW2, fs.span(2), 38.0)
    panel_memory(cv, sd, fs.col_x(4), ROW2 + 40.0, fs.span(2), 168.0 - ROW2 - 40.0)
    cm.finish(cv, "fig1", sd)


if __name__ == "__main__":
    main()
