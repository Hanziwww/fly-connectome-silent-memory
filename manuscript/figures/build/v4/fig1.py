"""v4 Fig. 1 (Cell) - A whole-CNS connectome learns to plan and to remember (174 mm wide).

A  hero band, left: the unfolded CNS, every located soma glowing by superclass group (dark tile)
B  hero band, right: dorsal-oblique interface, readout arbors, input somata, DNp01, LC12 and i2 MN insets (dark tile)
C  dark filmstrip, one memory trial in the whole CNS: the cells that spike in the cue, the delay and the input (scene 1,
   readout trained), in their group colours; the brain falls almost silent in the delay
D  the same trial ms by ms: cells spiking per ms, by superclass group
E  the model at a glance, in the form of MaleCNS Fig. 6A: the brain, half anatomy (an input cell, LC12; a readout cell, the
   giant fiber DNp01) and half schematic (the visual path: input -> visual projection neurons -> descending neurons), the
   fly, and count bars (MaleCNS Fig. 4B): parameters fixed vs trained, readout cells by their shortest route from the input
F  task suite: three boards across the ladder and the delayed-cue protocol (trained and untrained timings)
G  learning cost scales with certified difficulty in every replicate set (X10)
H  only the final brain with the final readout solves the maps (X9 pairings)
I  delayed-cue memory locked to the trained timings in every replicate of four learners (X1), paired by replicate

X8 (readout-only vs joint training from a neutral brain) is not shown: class (c) since the Fig. 1 review of 2026-10-22,
reported in STAR Methods and Table S2 only (RESULTS_INVENTORY_V4.md, L4). Every number in a label is read from
manuscript/results or from the render metadata of the band.

Run after render/v4_fig1_tiles.py, render/v4_fig1_glance.py, render/v4_trial_tiles.py and the analysis phase P12:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\v4\\fig1.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
from lib import boards as bd, circuit as ci, figstyle as fs, fly, palette as pal, qc  # noqa: E402

cm.use_v4()
TILES = cm.RENDERS / "v4" / "fig1"
BAND = json.loads((TILES / "band.json").read_text(encoding="utf-8"))
BAND_H = BAND["band_height_mm"]
ATLAS_W = BAND["atlas_mm"][0]
IFACE_X = ATLAS_W + fs.GUTTER
IFACE_W = BAND["interface_mm"][0]
TRIAL = json.loads((cm.RENDERS / "v4" / "trial" / "tiles.json").read_text(encoding="utf-8"))
U = fs.unit(1)
XS = [fs.unit_x(i) for i in range(4)]
TRIAL_PHASES = ("cue", "delay", "input")
TRIAL_W = (fs.unit(3) - 2 * fs.GUTTER) / 3
TRIAL_H = TRIAL_W * TRIAL["activity_mm"][1] / TRIAL["activity_mm"][0]
ROW_TRIAL = BAND_H + 4.2
ROW2 = ROW_TRIAL + TRIAL_H + 4.2
ROW_H = 47.0
ROW3 = ROW2 + ROW_H + 3.4
H = ROW3 + ROW_H
WHITE = fs.ON_DARK
NL_ = chr(10)
GROUP_ORDER = ["optic lobe intrinsic", "central brain intrinsic", "VNC intrinsic", "visual projection & centrifugal",
               "descending & ascending", "motor & efferent", "sensory", "other"]
SETS = list(range(6))
SET_COLOUR = {0: fs.TEXT, 1: "#2E6E9E", 2: "#4C9BC8", 3: "#7DB9DE", 4: "#3A8A6E", 5: "#79B89F"}
LEARNERS = [("scene 1", "readout"), ("scene 1", "brain and readout"), ("scene 2", "readout"), ("scene 3", "readout")]
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
PAIR_LINE = "#CFCFCF"
GLANCE = json.loads((TILES / "glance.json").read_text(encoding="utf-8"))
FLOW = {"input": ("#F0A030", "#B85C00"), "vpn": ("#3BAA7E", "#2E7D5B"), "dn": ("#5B8FD0", "#3A6EA5")}   # E: arrow, chip (white text)
FIXED_BAR = "#BDBDBD"


def spec_value(quantity, section="network"):
    spec = cm.table("main/fig1/fig1a_model_specification.csv").set_index(["section", "quantity"])["value"]
    return int(float(spec[(section, quantity)]))


def tile_text(cv, x, y, s, size=None, **kw):
    t = cv.text(x, y, s, size=size or fs.PT_SMALL, color=WHITE, **kw)
    t.set_path_effects(fs.dark_halo(0.9))
    return t


def scale_bar(cv, x_right, y, um, um_per_px, px_per_mm):
    length = um / um_per_px / px_per_mm
    cv.line([x_right - length, x_right], [y, y], WHITE, lw=1.0, solid_capstyle="butt")
    return length


def white_letter(cv, x, y, s):
    cv.letter(x, y, s)
    cv.fig.texts[-1].set_color(WHITE)


def test_row(family, test=None):
    t = cm.table("plan_v2/tests.csv")
    t = t[t["family"] == family]
    if test is not None:
        t = t[t["test"].astype(str).str.contains(test, regex=False)]
    return t


# ---------------------------------------------------------------- A, B: hero band

def panel_atlas(cv, sd):
    img = np.asarray(Image.open(TILES / "atlas.png").convert("RGB"))
    meta = json.loads((TILES / "atlas.json").read_text(encoding="utf-8"))
    cv.tile(0, 0, img, width=ATLAS_W, height=BAND_H, group="band", label="atlas")
    px_per_mm = img.shape[1] / ATLAS_W
    white_letter(cv, 1.2, 1.0, "A")
    counts = meta["group_counts"]
    # the VNC fills x 20-42 mm: group names to its left, their counts to its right, row by row
    y, kx, cx = 47.4, 1.6, ATLAS_W - 1.4
    tile_text(cv, 0.9, y - 4.6, "Cell bodies by" + NL_ + "superclass group", va="center", linespacing=1.0)
    tile_text(cv, cx, y - 3.6, "cells", va="center", ha="right")
    for g in GROUP_ORDER:
        n = counts.get(g, 0)
        cv.mark(kx, y, s=7.0, fc=pal.ATLAS_ON_DARK[g], ec=pal.ATLAS_ON_DARK[g], lw=0.3)
        tile_text(cv, kx + 1.2, y, pal.SHORT_GROUP[g], va="center")
        tile_text(cv, cx, y, f"{n:,}", va="center", ha="right")
        sd.add("A", "cell bodies by group", group=g, cells=n)
        y += 2.9
    tile_text(cv, ATLAS_W - 1.4, BAND_H - 12.6, f"{meta['somata']:,} cell bodies", ha="right", va="center")
    tile_text(cv, ATLAS_W - 1.4, BAND_H - 10.0, f"of {meta['cells']:,} neurons", ha="right", va="center")
    scale_bar(cv, ATLAS_W - 1.6, BAND_H - 2.0, 100, meta["scale_um_per_px"], px_per_mm)
    ox, oy = ATLAS_W - 14.0, BAND_H - 6.4
    cv.line([ox, ox + 3.0], [oy, oy], WHITE, lw=0.5)
    tile_text(cv, ox + 3.5, oy, "L", va="center")
    cv.line([ox, ox], [oy, oy + 3.0], WHITE, lw=0.5)
    tile_text(cv, ox, oy + 3.4, "V", ha="center", va="top")
    sd.add("A", "view", unfold="brain frontal; VNC rotated 90 degrees about the neck (msa/cnsview.py)",
           um_per_px=meta["scale_um_per_px"], scale_bar_um=100)


def panel_interface(cv, sd):
    img = np.asarray(Image.open(TILES / "interface.png").convert("RGB"))
    meta = json.loads((TILES / "interface.json").read_text(encoding="utf-8"))
    cv.tile(IFACE_X, 0, img, width=IFACE_W, height=BAND_H, group="band", label="interface")
    px_per_mm = img.shape[1] / IFACE_W
    to_mm = lambda px_, py_: (IFACE_X + px_ / px_per_mm, py_ / px_per_mm)  # noqa: E731
    white_letter(cv, IFACE_X + 1.2, 1.0, "B")
    for ins in meta["insets"]:
        x0, y0, x1, y1 = ins["box_px"]
        (ax0, ay0), (ax1, ay1) = to_mm(x0, y0), to_mm(x1, y1)
        role = "input" if ins["role"] == "input" else "readout"
        tile_text(cv, ax0, ay0 - 0.7, f"{ins['label']} ({role})", va="bottom")
        um = 20 if ins["scale_um_per_px"] < 0.2 else 50
        scale_bar(cv, ax1 - 1.2, ay1 - 1.4, um, ins["scale_um_per_px"], px_per_mm)
        sd.add("B", "inset", body_id=ins["body_id"], label=ins["label"], type=ins["type"], superclass=ins["superclass"],
               role=ins["role"], scale_bar_um=um, skeleton=ins["skeleton"])
    n_in, n_out = spec_value("input_cells"), spec_value("observed_cells")
    if sum(meta["readout_group_counts"].values()) != n_out:
        raise ValueError("the rendered readout arbors do not match the model's readout cells")
    ky = BAND_H - 10.4
    kx = IFACE_X + 0.40 * IFACE_W
    cv.mark(kx + 0.8, ky, s=6.0, fc=WHITE, ec=WHITE, lw=0.3)
    tile_text(cv, kx + 2.2, ky, f"input cell bodies ({meta['input_cells']:,} of {n_in:,} located)", va="center")
    ky2 = ky + 3.2
    seg = 0.55
    for k, g in enumerate(["central brain intrinsic", "optic lobe intrinsic", "visual projection & centrifugal",
                           "descending & ascending", "motor & efferent"]):
        cv.line([kx + k * seg, kx + (k + 1) * seg], [ky2, ky2], pal.ATLAS_ON_DARK[g], lw=1.0)
    tile_text(cv, kx + 3.4, ky2, f"{n_out:,} readout arbors, colored as in A", va="center")
    cv.line([kx, kx + 2.75], [ky2 + 3.2, ky2 + 3.2], WHITE, lw=1.0)
    tile_text(cv, kx + 3.4, ky2 + 3.2, "DNp01 (giant fiber), a readout cell", va="center")
    scale_bar(cv, IFACE_X + IFACE_W - 1.6, BAND_H - 2.0, 100, meta["scale_um_per_px"], px_per_mm)
    o = meta["orientation"]
    ox, oy = IFACE_X + 0.40 * IFACE_W - 8.4, BAND_H - 7.4
    for key, d in (("A", o["A"]), ("L", o["L"])):
        dx, dy = 3.0 * d[0], 3.0 * d[1]
        cv.line([ox, ox + dx], [oy, oy + dy], WHITE, lw=0.5)
        tile_text(cv, ox + dx * 1.4, oy + dy * 1.4, key, ha="center", va="center")
    for g, n in sorted(meta["readout_group_counts"].items()):
        sd.add("B", "readout arbors by group", group=g, cells=n)
    sd.add("B", "highlight", body_id=meta["highlight"]["body_id"], label="DNp01", skeleton=meta["highlight"]["skeleton"])


# ---------------------------------------------------------------- C, D: one memory trial in the whole CNS

PHASE_LABEL = {"cue": ("Cue", "0–40 ms"), "delay": ("Delay", "160 ms"), "input": ("Prefix", "160 ms"), "decision": ("Decision", "40 ms")}


def panel_trial(cv, sd):
    cells = cm.table("plan_v2/trial_cells.csv")
    tdir = cm.RENDERS / "v4" / "trial"
    for k, ph in enumerate(TRIAL_PHASES):
        x = k * (TRIAL_W + fs.GUTTER)
        img = np.asarray(Image.open(tdir / f"activity_{ph}.png").convert("RGB"))
        cv.tile(x, ROW_TRIAL, img, width=TRIAL_W, height=TRIAL_H, group="trial", label=f"activity_{ph}")
        n = int((cells[f"spikes_{ph}_cue0"] > 0).sum())
        spikes = int(cells[f"spikes_{ph}_cue0"].sum())
        name, dur = PHASE_LABEL[ph]
        tile_text(cv, x + TRIAL_W - 1.2, ROW_TRIAL + 1.2, f"{name} · {dur}", ha="right", va="top", size=fs.PT)
        tile_text(cv, x + TRIAL_W - 1.2, ROW_TRIAL + 4.3, f"{n:,} cells spike", ha="right", va="top")
        sd.add("C", "phase", phase=ph, cells_spiking=n, spikes=spikes, cue=0)
    white_letter(cv, 1.2, ROW_TRIAL + 0.9, "C")
    tile_text(cv, TRIAL_W - 1.2, ROW_TRIAL + TRIAL_H - 2.0, "scene 1, readout trained, cue 0", ha="right", va="center")
    sd.add("C", "view", rotation="unfolded CNS turned 90 degrees, brain left", colours="superclass group, as A")


def panel_trial_classes(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "D")
    cv.text(x0 + 4.0, y0 + 0.5, "The trial by class", size=fs.PT, va="top", fontweight="bold")
    tc = cm.table("plan_v2/trial_timecourse.csv")
    tc = tc.assign(group=tc["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other"))
    g = tc.groupby(["ms", "group"])["spiking_cue0"].sum().unstack(fill_value=0)
    ax = cv.axes(x0 + 9.0, y0 + 6.0, w - 12.6, h - 12.4)
    ms = g.index.to_numpy()
    k = 5                                                          # 5-ms running mean, for legibility
    bottom = np.zeros(len(ms))
    for grp in GROUP_ORDER:
        if grp not in g:
            continue
        v = np.convolve(g[grp].to_numpy(float), np.ones(k) / k, mode="same")
        ax.fill_between(ms, bottom, bottom + v, color=pal.on_white(pal.ATLAS_ON_DARK[grp]), lw=0, zorder=2)
        bottom += v
        sd.add("D", "cells spiking per ms", group=grp, ms=ms.tolist(), cells=g[grp].astype(int).tolist())
    ax.axvspan(40, 200, color="#F2F2F2", lw=0, zorder=0)
    ax.text(120, bottom.max() * 0.95, "delay", fontsize=fs.PT_SMALL, ha="center", va="top")
    ax.set_xlim(0, 400)
    ax.set_ylim(0, bottom.max() * 1.05)
    ax.set_xticks([0, 200, 400])
    ax.set_xlabel("ms from cue onset", labelpad=0.8)
    ax.set_ylabel(f"Cells spiking{NL_}per ms", labelpad=0.8, linespacing=1.0)
    fs.clean(ax)


# ---------------------------------------------------------------- E: model at a glance

def panel_model(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "E")
    cv.text(x0 + 4.0, y0 + 0.5, "The model at a glance", size=fs.PT, va="top", fontweight="bold")
    spec = cm.table("main/fig1/fig1a_model_specification.csv").set_index(["section", "quantity"])["value"]
    b = cm.table("main/fig1/fig1b_parameter_budget.csv")
    paths_t = cm.table("plan_v2/graph_paths.csv")
    routes = cm.table("plan_v2/fig6_routes.csv")
    # the brain: half anatomy (left), half schematic (right), from render/v4_fig1_glance.py
    img = np.asarray(Image.open(TILES / "glance.png").convert("RGB"))
    tw, th = img.shape[1] / fs.DPI / fs.MM, img.shape[0] / fs.DPI / fs.MM
    tx, ty = x0 + (w - tw) / 2, y0 + 4.4
    cv.tile(tx, ty, img, dark=False, label="glance")
    if not GLANCE["anatomy_on_left"]:
        raise RuntimeError("glance tile: the build expects the anatomical half on the left")
    mid = tx + tw / 2
    cv.line([mid, mid], [ty + 0.3, ty + th - 0.3], "#B0B0B0", lw=0.5, linestyle=(0, (2.0, 1.6)))

    def flow(p0, p1, key, bend):
        a = np.array([mid + p0[0], ty + p0[1]])
        e = np.array([mid + p1[0], ty + p1[1]])
        ci.ribbon_curve(cv, a, e, 0.8, FLOW[key][0], bend=bend)
        cv.mark(a[0], a[1], s=8.0, fc=FLOW[key][0], ec=FLOW[key][0], lw=0.0)

    flow((19.6, 2.6), (15.4, 8.4), "input", 0.2)                      # from the eye into the optic lobe
    flow((12.6, 13.2), (6.2, 10.6), "vpn", 0.25)                      # optic lobe -> central brain
    flow((4.0, 12.2), (1.8, th - 0.6), "dn", -0.1)                    # central brain -> the neck, to the VNC
    ci.label_chip(cv, mid + 13.6, ty + 1.0, "input", FLOW["input"][1], size=fs.PT_SMALL, h=2.5)
    ci.label_chip(cv, mid + 11.4, ty + 15.8, "VPNs", FLOW["vpn"][1], size=fs.PT_SMALL, h=2.5)
    ci.label_chip(cv, mid + 3.6, ty + th - 2.0, "DNs", FLOW["dn"][1], size=fs.PT_SMALL, h=2.5)
    for name, (nx, ny) in GLANCE["neuropils"].items():               # the path's neuropils, named on the schematic half
        dx, dy = {"ME": (1.6, 0.0), "LO": (-0.8, 3.2), "PVLP": (-1.4, -1.4)}[name]
        cv.text(tx + nx + dx, ty + ny + dy, name, size=fs.PT_SMALL, ha="center", va="center")
    for name, (sx, sy) in GLANCE["neurons"].items():                 # the two cells, named on the anatomical half
        ny = ty + (1.6 if name == "LC12" else th - 2.0)
        end = ci.label_chip(cv, tx + 0.3, ny, name, (FLOW["input"] if name == "LC12" else FLOW["dn"])[1], size=fs.PT_SMALL, h=2.5)
        cv.line([end + 0.2, tx + sx], [ny, ty + sy], "#8C8C8C", lw=0.3)
    sd.add("E", "glance tile", neurons=list(GLANCE["neurons"]), neuropils=list(GLANCE["neuropils"]), tile_mm=[tw, th])
    # the fly (the cue is visual: red eyes)
    fs_y = ty + th + 1.8
    fly.inset(cv, x0 + 0.3, fs_y + 0.4, 9.6)
    bx0 = x0 + 11.4
    bw = x0 + w - 0.3 - bx0
    # count bars (MaleCNS Fig. 4B): parameters fixed vs trained, on a log scale
    core = int(b[(b.model_kind == "connectome_linear") & (b.component == "brain")]["n_elements"].sum())
    head = int(b[(b.model_kind == "connectome_linear") & (b.component == "readout") & cm.truthy(b.trainable_in_joint)]["n_elements"].sum())
    entries = float(spec[("network", "stored_connectome_entries")])
    cv.text(bx0, fs_y + 0.2, "Fixed vs trained parameters", size=fs.PT_SMALL, va="top")
    ax = cv.axes(bx0, fs_y + 2.8, bw, 7.8)
    lo, hi = 1e2, 3e8
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(2.5, -0.5)
    rows = [("connections", entries, FIXED_BAR, f"{entries / 1e6:.1f} M connections", None),
            ("gains, thresholds", core, pal.ACCENT, f"{core:,}", "gains, thresholds"),
            ("readout weights", head, pal.ACCENT, f"{head:,}", "readout weights")]
    for i, (lab, n, fill, inside, right) in enumerate(rows):
        ax.barh(i, n - lo, left=lo, height=0.8, color=fill, linewidth=0)
        xin = lo * 1.35 if right is None else float(np.sqrt(lo * n))
        _bar_text(cv, ax, xin, i, inside, fill, ha="left" if right is None else "center")
        if right:
            ax.text(n * 1.3, i, right, fontsize=fs.PT_SMALL, va="center", ha="left", color=fs.TEXT)
        sd.add("E", "parameters", item=lab, n=n, trained=right is not None)
    _bare(ax)
    ax.axvline(lo, color=fs.AXIS, lw=fs.LW_THIN)
    # count bars: the readout cells by their shortest route from the input cells
    p = paths_t[paths_t["pathway"] == "visual inputs -> readout"].sort_values("k")
    n_out = int(float(spec[("network", "observed_cells")]))
    within = np.rint(p["targets_reached_within_k"].to_numpy()[:3] * n_out).astype(int)
    by_k = np.diff(np.concatenate([[0], within]))
    ro = routes[routes["role"] == "readout"]
    if not np.array_equal(by_k, [int((ro["hops_0"] == k).sum()) for k in (1, 2, 3)]):     # gate: graph analysis = route table
        raise RuntimeError("Fig. 1E: readout cells by shortest route disagree between graph_paths and fig6_routes")
    yb = fs_y + 11.6
    cv.text(bx0, yb, "Readout cells by shortest route", size=fs.PT_SMALL, va="top")
    lab_w = fs.text_width("1 synapse", fs.PT_SMALL) + 1.0
    ax = cv.axes(bx0 + lab_w, yb + 2.6, bw - lab_w - 4.2, 7.8)
    ax.set_xlim(0, n_out)
    ax.set_ylim(2.5, -0.5)
    shades = ("#5E5E5E", "#9E9E9E", "#CFCFCF")
    for i, (n, fill) in enumerate(zip(by_k, shades)):
        ax.barh(i, n, height=0.8, color=fill, linewidth=0)
        if n / n_out > 0.25:
            _bar_text(cv, ax, n / 2, i, f"{n:,}", fill, ha="center")
        else:
            ax.text(n + 0.02 * n_out, i, f"{n:,}", fontsize=fs.PT_SMALL, va="center", ha="left", color=fs.TEXT)
        cv.text(bx0 + lab_w - 1.0, yb + 2.6 + 7.8 * (i + 0.5) / 3, "1 synapse" if i == 0 else f"{i + 1}", size=fs.PT_SMALL,
                ha="right", va="center")
        sd.add("E", "readout cells by shortest route", synapses=i + 1, cells=int(n), of=n_out)
    _bare(ax)
    ax.axvline(0, color=fs.AXIS, lw=fs.LW_THIN)


def _bare(ax):
    """No ticks, labels or spines (count bars carry their values)."""
    from matplotlib.ticker import NullLocator
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(NullLocator())
        a.set_minor_locator(NullLocator())
    for s in ax.spines.values():
        s.set_visible(False)
    ax.patch.set_visible(False)


def _bar_text(cv, ax, x, y, text, fill, ha="center"):
    """A count inside a bar: white on dark fills (registered for the QC contrast check), black on pale ones."""
    from matplotlib.colors import to_rgb
    lum = sum(k * (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for k, c in zip((0.2126, 0.7152, 0.0722), to_rgb(fill)))
    white = 1.05 / (lum + 0.05) >= 4.5
    tt = ax.text(x, y, text, fontsize=fs.PT_SMALL, va="center", ha=ha, color="#FFFFFF" if white else fs.TEXT)
    if white:
        cv.fig.canvas.draw()
        bb = tt.get_window_extent(cv.fig.canvas.get_renderer())
        inv = cv.fig.transFigure.inverted()
        (fx0, fy0), (fx1, fy1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        cv.fills.append(dict(rect=(fx0 * cv.w - 0.2, (1 - fy1) * cv.h - 0.2, (fx1 - fx0) * cv.w + 0.4, (fy1 - fy0) * cv.h + 0.4), fill=fill))


# ---------------------------------------------------------------- F: task suite

def panel_tasks(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "F")
    cv.text(x0 + 4.0, y0 + 0.5, "Two task families", size=fs.PT, va="top", fontweight="bold")
    boards_t = cm.table("figure_data/fd_boards.csv")
    lib = cm.table("main/fig1/fig1d_task_library.csv").set_index("map_id")
    design = cm.table("main/fig1/fig1e_memory_task_design.csv")
    ax = bd.mm_axes(cv, x0, y0, w, h)
    picks = ["D1-T0", "D4-T0", "D7-T0"]
    cell = 1.0
    n_maps = cm.table("plan_v2/x10_maps.csv")["map"].nunique()
    ax.text(1.0, 4.6, f"Planning: {n_maps} Sokoban maps", ha="left", va="top", fontsize=fs.PT_SMALL)
    # boards and their labels justified across the panel, each item as wide as the wider of board and label
    items = []
    for m in picks:
        b = bd.Board.from_table(boards_t, m)
        lab = f"{int(lib.loc[m, 'optimal_moves'])} moves"
        items.append((m, b, b.size_mm(cell), lab, max(b.size_mm(cell)[0], fs.text_width(lab, fs.PT_SMALL))))
    gap = (w - 1.8 - sum(it[4] for it in items)) / (len(items) - 1)
    x = 1.0
    for m, b, (bw, bh), lab, iw in items:
        cx = x + iw / 2
        bd.draw(ax, b, cx - bw / 2, 8.0 + (10 * cell - bh) / 2, cell, lw=0.25)
        ax.text(cx, 8.0 + 10 * cell + 0.6, lab, ha="center", va="top", fontsize=fs.PT_SMALL)
        sd.add("F", "board", map_id=m, optimal_moves=int(lib.loc[m, "optimal_moves"]))
        x += iw + gap
    # memory protocol: timeline and the 7 x 5 timing grid
    ty = 22.6
    ax.text(1.0, ty, "Memory: a cue held across a delay", ha="left", va="top", fontsize=fs.PT_SMALL)
    segs = [("cue", "#E4E4E4"), ("delay", "#3A3A3A"), ("prefix", "#BDBDBD"), ("decision", "#E4E4E4")]
    sw = (w - 2.0) / 4
    for k, (lab, c) in enumerate(segs):
        ax.add_patch(Rectangle((1.0 + k * sw, ty + 2.6), sw - 0.4, 1.0, facecolor=c, edgecolor=fs.AXIS, linewidth=0.3))
        ax.text(1.0 + k * sw + (sw - 0.4) / 2, ty + 3.9, lab, ha="center", va="top", fontsize=fs.PT_SMALL)
    t = design[design["section"] == "timing"].copy()
    delays = sorted(t["delay_ms"].astype(int).unique())
    prefixes = sorted(t["prefix_windows"].astype(int).unique())
    gx0, gy0 = 1.0 + fs.text_width("windows", fs.PT_SMALL) + 0.6 + fs.text_width("8", fs.PT_SMALL) + 0.5, ty + 9.4
    gw = (w - gx0 - 0.8) / len(delays)
    gh = (h - gy0 - 5.6) / len(prefixes)
    for _, r in t.iterrows():
        i = delays.index(int(r["delay_ms"]))
        j = prefixes.index(int(r["prefix_windows"]))
        trained = r["split"] == "train"
        ax.add_patch(Rectangle((gx0 + i * gw + 0.15, gy0 + j * gh + 0.15), gw - 0.3, gh - 0.3,
                               facecolor="#4D4D4D" if trained else "white", edgecolor="#4D4D4D" if trained else "#A8A8A8",
                               linewidth=0.3))
        sd.add("F", "timing", delay_ms=int(r["delay_ms"]), prefix_windows=int(r["prefix_windows"]), split=r["split"])
    for i in (0, 3, 6):
        ax.text(gx0 + (i + 0.5) * gw, gy0 + len(prefixes) * gh + 0.4, str(delays[i]), ha="center", va="top", fontsize=fs.PT_SMALL)
    for j in (0, 2, 4):
        ax.text(gx0 - 0.5, gy0 + (j + 0.5) * gh, str(prefixes[j]), ha="right", va="center", fontsize=fs.PT_SMALL)
    ax.text(gx0 + len(delays) * gw / 2, h, "Delay (ms)", ha="center", va="bottom", fontsize=fs.PT_SMALL)
    ax.text(1.0, gy0 + len(prefixes) * gh / 2, "Prefix" + NL_ + "windows", ha="left", va="center", fontsize=fs.PT_SMALL,
            linespacing=1.0)
    n_tr = int((t["split"] == "train").sum())
    lab = f"trained timing ({n_tr} of {len(t)})"
    lx = w - 0.8 - fs.text_width(lab, fs.PT_SMALL)
    ax.add_patch(Rectangle((lx - 2.0, gy0 - 2.2), 1.3, 1.1, facecolor="#4D4D4D", edgecolor="none"))
    ax.text(w - 0.8, gy0 - 1.65, lab, ha="right", va="center", fontsize=fs.PT_SMALL)


# ---------------------------------------------------------------- G: learning cost vs difficulty

def panel_cost(cv, sd, x0, y0, w, h):
    d = cm.table("plan_v2/x10_maps.csv")
    if sorted(d["replicate"].unique()) != SETS:
        raise ValueError("X10 replicate sets differ from SETS")
    cv.letter(x0, y0, "G")
    cv.text(x0 + 4.0, y0 + 0.5, f"Learning cost tracks certified difficulty, in {WORDS[len(SETS)]} replicate sets", size=fs.PT,
            va="top", fontweight="bold")
    tests = test_row("F13.X10.fig2")
    ax = cv.axes(x0 + 9.0, y0 + 9.0, w - 11.0, h - 16.0)
    for s in SETS:
        r = d[d["replicate"] == s].sort_values("optimal_moves")
        xx, yy = r["optimal_moves"].to_numpy(float), r["updates"].to_numpy(float)
        ax.scatter(xx, yy, s=6 if s else 9, facecolor=SET_COLOUR[s] if s == 0 else "white", edgecolor=SET_COLOUR[s],
                   linewidth=0.5, zorder=4 if s == 0 else 3)
        k, c = np.polyfit(np.log10(xx), np.log10(yy), 1)
        gx = np.array([xx.min(), xx.max()])
        ax.plot(gx, 10 ** (c + k * np.log10(gx)), color=SET_COLOUR[s], lw=0.6 if s == 0 else 0.45, zorder=2)
        for _, rr in r.iterrows():
            sd.add("G", "map", replicate_set=int(s), map_id=rr["map"], optimal_moves=int(rr["optimal_moves"]),
                   updates=int(rr["updates"]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(7.5, 70)
    ax.set_xticks([10, 20, 50], ["10", "20", "50"])
    fs.minor(ax)
    ax.set_xlabel("Certified optimal moves")
    ax.set_ylabel("Updates to solve")
    fs.clean(ax)
    rho = tests[tests["test"].astype(str).str.contains("replicate set")]
    pmax = float(pd.to_numeric(rho["p_value"]).max())
    r_lo, r_hi = pd.to_numeric(rho["estimate"]).min(), pd.to_numeric(rho["estimate"]).max()
    items = [([dict(kind="mark", s=9, fc=SET_COLOUR[0], ec=SET_COLOUR[0])], "original set")] + \
            [([dict(kind="mark", s=6, fc="white", ec=SET_COLOUR[s], lw=0.5, space=0.2) for s in SETS[1:]],
              f"replicate sets {SETS[1]}–{SETS[-1]}")]
    cv.key(x0 + 9.0, y0 + 6.3, items, size=fs.PT_SMALL)
    per = {int(str(t).split("replicate set ")[1].split(":")[0]): float(e) for t, e in zip(rho["test"], pd.to_numeric(rho["estimate"]))
           if "replicate set " in str(t)}
    ix = cv.axes(x0 + w - 19.0, y0 + h - 15.4, 15.6, 2.2)        # Spearman rho of each set, on the empty lower right
    for s_, v in sorted(per.items()):
        ix.scatter([v], [0], s=7 if s_ else 9, facecolor=SET_COLOUR[s_] if s_ == 0 else "white", edgecolor=SET_COLOUR[s_], linewidth=0.5, zorder=3)
    ix.set_xlim(0.9, 1.0)
    ix.set_xticks([0.9, 0.95, 1.0], ["0.9", "0.95", "1"])
    ix.set_ylim(-1, 1)
    ix.set_yticks([])
    ix.tick_params(axis="x", length=1.5, pad=0.8)
    fs.clean(ix, left=False)
    ix.set_title(f"Spearman {fs.RHO} per set", fontsize=fs.PT_SMALL, pad=1.2)
    ix.set_xlabel(f"each {fs.P_SYM} ≤ {fs.fmt_p(pmax)}", fontsize=fs.PT_SMALL, labelpad=0.6)
    sd.add("G", "test", family="F13.X10.fig2", rho_min=float(r_lo), rho_max=float(r_hi), p_max=pmax, n_sets=len(rho),
           n_maps=int(d["map"].nunique()))


# ---------------------------------------------------------------- H: X9

def panel_x9(cv, sd, x0, y0, w, h):
    cv.letter(x0, y0, "H")
    cv.text(x0 + 4.0, y0 + 0.5, "Brain and readout co-adapt", size=fs.PT, va="top", fontweight="bold")
    d = cm.table("plan_v2/x9_pairings.csv")
    t = test_row("F13.X9.pairing")
    order = ["before", "mid", "after"]
    n_maps = d["map"].nunique()
    m = np.zeros((3, 3), int)
    for (bk, rk), g in d.groupby(["brain", "readout"]):
        m[order.index(bk), order.index(rk)] = int(g["solved"].astype(str).str.lower().eq("true").sum())
    ax = cv.axes(x0 + 9.4, y0 + 8.6, w - 11.4, h - 18.0)
    ax.imshow(m, cmap=colormaps["Greys"], vmin=0, vmax=n_maps * 2.2, aspect="auto")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i, j]}", ha="center", va="center", fontsize=fs.PT_SMALL, color=fs.TEXT)
            sd.add("H", "maps solved", brain=order[i], readout=order[j], solved=int(m[i, j]), maps=int(n_maps))
    ax.add_patch(Rectangle((1.5, 1.5), 1.0, 1.0, facecolor="none", edgecolor=pal.ACCENT, linewidth=0.9))
    ax.set_xticks(range(3), order)
    ax.set_yticks(range(3), order)
    ax.set_xlabel("Readout checkpoint")
    ax.set_ylabel("Brain checkpoint")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    q = float(pd.to_numeric(t["p_adjusted"]).max())
    cv.text(x0 + w - 1.0, y0 + h - 0.2, f"maps solved of {n_maps}; BH {fs.Q_SYM} ≤ {fs.fmt_p(q)}", size=fs.PT_SMALL, ha="right",
            va="bottom")
    sd.add("H", "test", family="F13.X9.pairing", q_max=q)


# ---------------------------------------------------------------- I: X1

def panel_x1(cv, sd, x0, y0, w, h):
    d = cm.table("plan_v2/x1_runs.csv")
    tests = cm.table("plan_v2/tests.csv")
    design = cm.table("main/fig1/fig1e_memory_task_design.csv")
    timing = design[design["section"] == "timing"]
    n_tr = int((timing["split"] == "train").sum())
    if sorted(set(zip(d["scene"], d["trained"]))) != sorted(LEARNERS):
        raise ValueError("X1 learners differ from LEARNERS")
    n_pos = int((d["lock"] > 0).sum())
    lock_all = f"in all {len(d)}" if n_pos == len(d) else f"in {n_pos} of {len(d)}"
    cv.letter(x0, y0, "I")
    cv.text(x0 + 4.0, y0 + 0.5, f"Memory is locked to the trained timings {lock_all} replicates of {WORDS[len(LEARNERS)]} learners",
            size=fs.PT, va="top", fontweight="bold")
    ax = cv.axes(x0 + 9.0, y0 + 9.0, w - 11.0, h - 18.0)
    rng = np.random.default_rng(20261022)
    half = 0.2
    for i, (scene, trained) in enumerate(LEARNERS):
        r = d[(d["scene"] == scene) & (d["trained"] == trained)]
        lrn = f"{scene}, {trained} trained"
        tr, un = r["trained_net_use"].to_numpy(float), r["untrained_net_use"].to_numpy(float)
        jit = rng.uniform(-0.05, 0.05, len(r))
        xt, xu = i - half + jit, i + half + jit
        for a, b, ya, yb in zip(xt, xu, tr, un):
            ax.plot([a, b], [ya, yb], color=PAIR_LINE, lw=0.4, zorder=2)
        ax.scatter(xt, tr, s=7, facecolor=pal.ACCENT, edgecolor=pal.ACCENT, linewidth=0.5, zorder=3)
        ax.scatter(xu, un, s=7, facecolor="white", edgecolor=fs.TEXT2, linewidth=0.5, zorder=3)
        for xc, v, colour in ((i - half, tr, pal.ACCENT), (i + half, un, fs.TEXT2)):
            ax.plot([xc - 0.11, xc + 0.11], [np.median(v)] * 2, color=colour, lw=0.9, zorder=4)
        lock = tests[(tests["family"] == "F13.X1.lock") & (tests["test"] == r["code"].iloc[0])]
        p = float(lock["p_adjusted"].iloc[0])
        ax.text(i, 1.13, f"n = {len(r)}{NL_}lock {fs.p_text(p)}", ha="center", va="bottom", fontsize=fs.PT_SMALL, linespacing=1.0)
        for _, rr in r.iterrows():
            sd.add("I", "replicate", learner=lrn, replicate=int(rr["replicate"]), trained_net_use=float(rr["trained_net_use"]),
                   untrained_net_use=float(rr["untrained_net_use"]), lock=float(rr["lock"]))
        sd.add("I", "test", learner=lrn, family="F13.X1.lock", p_adjusted=p)
    ax.axhline(0, color=fs.GRID, lw=fs.LW_THIN, zorder=0)
    ax.set_xticks(range(len(LEARNERS)), [f"{s.capitalize()}{NL_}{'readout' if t == 'readout' else 'brain + readout'} trained"
                                          for s, t in LEARNERS])
    ax.set_xlim(-0.6, len(LEARNERS) - 0.4)
    ax.set_ylim(-0.4, 1.55)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Net memory use")
    fs.clean(ax)
    cv.key(x0 + 9.0, y0 + 6.3, [([dict(kind="mark", s=7, fc=pal.ACCENT, ec=pal.ACCENT)], f"trained timings ({n_tr})"),
                                ([dict(kind="mark", s=7, fc="white", ec=fs.TEXT2)], f"untrained timings ({len(timing) - n_tr})"),
                                ([dict(kind="hline", colour=PAIR_LINE, lw=0.6, w=2.4)], "one replicate")], size=fs.PT_SMALL)
    cv.text(x0 + w - 1.0, y0 + 6.3, "lock = trained − untrained; Holm", size=fs.PT_SMALL, ha="right", va="center")
    sd.add("I", "lock positive", replicates=len(d), positive=n_pos)


def main():
    cv = fs.Canvas(fs.FULL, H)
    sd = qc.SourceData("fig1")
    for rel in ("figure_data/fd_cells.csv", "figure_data/fd_boards.csv", "main/fig1/fig1a_model_specification.csv",
                "main/fig1/fig1b_parameter_budget.csv", "main/fig1/fig1d_task_library.csv",
                "main/fig1/fig1e_memory_task_design.csv", "plan_v2/x10_maps.csv", "plan_v2/x9_pairings.csv",
                "plan_v2/x1_runs.csv", "plan_v2/graph_paths.csv", "plan_v2/graph_flow.csv", "plan_v2/tests.csv",
                "plan_v2/trial_cells.csv", "plan_v2/trial_timecourse.csv", "plan_v2/fig6_routes.csv"):
        sd.source(cm.RESULTS / rel)
    for f in ("atlas.png", "interface.png", "atlas.json", "interface.json", "band.json", "glance.png", "glance.json"):
        sd.source(TILES / f)
    for ph in TRIAL_PHASES:
        sd.source(cm.RENDERS / "v4" / "trial" / f"activity_{ph}.png")
        sd.source(cm.RENDERS / "v4" / "trial" / f"activity_{ph}.json")
    panel_atlas(cv, sd)
    panel_interface(cv, sd)
    panel_trial(cv, sd)
    panel_trial_classes(cv, sd, XS[3], ROW_TRIAL, U, TRIAL_H)
    panel_model(cv, sd, XS[0], ROW2, U, ROW_H)
    panel_tasks(cv, sd, XS[1], ROW2, U, ROW_H)
    panel_cost(cv, sd, XS[2], ROW2, fs.unit(2), ROW_H)
    panel_x9(cv, sd, XS[0], ROW3, U, ROW_H)
    panel_x1(cv, sd, XS[1], ROW3, fs.unit(3), ROW_H)
    cm.finish(cv, "fig1", sd)


if __name__ == "__main__":
    main()
