"""v4 graphical abstract (Cell: one square panel of 1,200 x 1,200 px at 300 dpi; Arial 12-16 pt; reads left to right and top
to bottom; no data items, no title, panel letters or white border; no caricatures; drawn by code from the model's records,
never by a generative image tool, as the author guide requires).

The story in three bands, read left to right and top to bottom:
  1  the model and the mechanism: the fly (the stylised top view of lib/fly.py, the MaleCNS inset form, in a dark palette)
     with its nervous system seen through head and thorax (render: fly_cns), unlabelled; beside it one
     trace cell through a trial: the cue, spikes that write the cue into adaptation (gold), a silent hold in which no spike
     carries it, the next input that reads it, and the choice of the cued target
  2  the trace in the whole CNS, in the frontal view of connectome papers: the trained connectome (render: learned, silver)
     and, after a disease-like shift (the Kv4-loss lesion at dose 1), the trace written into other and many more cells
     (render: shifted, purple); an arrow from silver to purple between them
  3  what the learned readout can do, under each state: a key cut in a state opens only that state's lock. The same old key
     opens the learned state's lock (recall) and fails the shifted state's (recall lost: the trained histories solved fall to
     the level of a cue-blind policy); a key cut after the shift opens it (new learning: spared under the lesion)

Design rules: margins of 4 mm; colour only in the graphics (gold the cue, the trace and adaptation; silver the state of
learning; purple the shifted state), every word in one neutral colour, Arial 12 pt (the guide's minimum); one stroke weight
for every glyph; each symbol drawn by one function and reused unchanged.

Each element states a finding of the paper: training of the whole CNS (Figure 1), the silent adaptation trace written by
spikes and read by the next input (Figure 2), recall locked to the state of learning (Figures 5-7), the lesion rewriting the
trace (Figure 7J) and learning spared under the lesion (Figure 7C). Nothing is plotted: the strip is a schematic of one cell,
with no axis, number or scale bar.

Run after P12 and render/v4_ga_tiles.py:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\build\\v4\\graphical_abstract.py
Outputs in manuscript/figures/v4/ga: graphical_abstract.tif (the submission file), .pdf, .jpg and .png, colour-vision
previews, and graphical_abstract_qc.json (checks, texts, input hashes).
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as cm  # noqa: E402
from lib import fly as fl, figstyle as fs, qc  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (after figstyle has chosen the Agg backend)

TILES = cm.RENDERS / "v4" / "ga"
META = json.loads((TILES / "tiles.json").read_text(encoding="utf-8"))
OUT = cm.FIG / "v4" / "ga"
NAME = "graphical_abstract"
S, PPI = META["ga_mm"], META["ppi"]                 # 101.6 mm = 1,200 px at 300 ppi
PT_MIN, PT_MAX = 12.0, 16.0                         # the guide's Arial sizes
LABEL = 12.0
TEXT = "#E2E2E2"                                    # every word: one neutral colour
GREY, LINE = "#5E6770", "#8A939C"                   # the cue's other target; spikes' baseline and quiet strokes
SILVER, PURPLE, GOLD = "#CDD6DE", "#B98BE6", "#FFC46B"
BODY = {SILVER: "#14181C", PURPLE: "#1B1226"}       # lock bodies: the state colour, nearly black
LW = 1.3                                            # every glyph's stroke (pt)
M = 4.0                                             # outer margin (mm)
LUMA = np.array([0.2126, 0.7152, 0.0722])

# band 1: the fly and one trace cell's trial
FLY_BOX = (3.0, 2.6, 19.0, 20.6)                    # left, top, width, height (mm): the drawing's 12 x 13 units
STRIP_X0, BASE_Y, TICK_TOP = 26.5, 16.5, 12.0       # the strip's left edge; the spike baseline and tick tops (mm)
CUE_Y, PHASE_Y, UNDER_Y = 14.4, 6.0, 22.5           # the cue and choice icons; phase names above; names below
WRITE_X = [37.0 + 1.25 * k for k in range(6)]       # spikes that write the cue
READ_X = [74.5 + 1.25 * k for k in range(5)]        # the next input, read
TRACE_PEAK, TRACE_TAU = 6.4, 46.0                   # the adaptation trace's height and decay (mm): it outlasts the hold
CHOICE_X = 84.5
# band 2: the two states
HEAD_Y, RENDER_TOP = 29.8, 32.6
COL_X = {"learned": 26.4, "shifted": 75.2}          # column centres (mm)
# band 3: what the learned readout can do
LOCK_W, LOCK_H, KEY_L = 6.0, 4.8, 7.4               # mm
LINE_Y = (84.6, 95.4)                               # the outcome lines' centres (mm)
LOCK_X = {"learned": 22.6, "shifted": 67.3}         # lock centres; each label starts 5 mm right of its lock


def mm(px):
    return px / PPI * 25.4


def snap(v):
    return round(v / 25.4 * PPI) * 25.4 / PPI


def canvas():
    fs.setup()
    fig = plt.figure(figsize=(S * fs.MM, S * fs.MM), dpi=PPI)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor("black")
    ax.set_axis_off()
    return fig, ax


def lock_view(ax):
    ax.set_xlim(0, S)
    ax.set_ylim(S, 0)
    ax.set_aspect("auto")


def text(ax, x, y, s, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    return ax.text(x, y, s, fontsize=LABEL, color=TEXT, fontfamily=fs.FONT, zorder=20, **kw)


def stroke(ax, xs, ys, colour, lw=LW, z=8, halo=True):
    """A line with a faint halo, the soft light of the renders."""
    if halo:
        ax.plot(xs, ys, color=colour, lw=lw * 3.2, alpha=0.10, solid_capstyle="round", solid_joinstyle="round", zorder=z)
    ax.plot(xs, ys, color=colour, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=z + 0.1)


def image(ax, path, left, top):
    """An RGB image at its pixel size (300 ppi), its top-left corner on the pixel grid; returns its extent (mm)."""
    img = np.asarray(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    left, top = snap(left), snap(top)
    ax.imshow(img, extent=(left, left + mm(w), top + mm(h), top), interpolation="none", zorder=1)
    return left, top, mm(w), mm(h)


def faded(ax, xs, y, colour, a0, a1, lw=0.9):
    """A horizontal thread of light whose opacity runs from a0 to a1."""
    segs = [[(xs[i], y), (xs[i + 1], y)] for i in range(len(xs) - 1)]
    a = np.linspace(a0, a1, len(segs))
    rgb = to_rgb(colour)
    ax.add_collection(LineCollection(segs, colors=[(*rgb, v * 0.12) for v in a], linewidths=lw * 3.3, capstyle="round", zorder=6))
    ax.add_collection(LineCollection(segs, colors=[(*rgb, v) for v in a], linewidths=lw, capstyle="round", zorder=7))


# ---- band 1 -----------------------------------------------------------------------------------------------------------------

def fly(fig):
    """The fly of lib/fly.py in a dark palette, its nervous system (the learned state, small) seen through head and thorax.
    Unlabelled: the left column's header names the model."""
    l, t, w, h = FLY_BOX
    fa = fig.add_axes((l / S, 1 - (t + h) / S, w / S, h / S))
    fa.set_facecolor("none")
    body, edge, legc, dark = "#2A3037", "#76808A", "#6E7781", "#20252B"
    for s in (-1, 1):                                          # legs: the drawing's own joints
        fl._leg(fa, [(0.7 * s, 2.3), (1.9 * s, 3.0), (2.3 * s, 4.1), (2.6 * s, 5.2)], legc)
        fl._leg(fa, [(0.9 * s, 1.6), (2.4 * s, 1.9), (3.5 * s, 1.2), (4.4 * s, 0.6)], legc)
        fl._leg(fa, [(0.8 * s, 0.9), (2.0 * s, -0.1), (2.7 * s, -1.9), (3.1 * s, -3.2)], legc)
    abd = Ellipse((0, -1.95), 2.4, 4.6, facecolor=body, edgecolor=edge, linewidth=0.5, zorder=2)
    fa.add_patch(abd)
    for y in (-0.95, -1.8, -2.65, -3.45):                      # tergite bands
        band = Rectangle((-1.4, y - 0.2), 2.8, 0.36, facecolor=dark, edgecolor="none", zorder=2.1)
        fa.add_patch(band)
        band.set_clip_path(abd)
    for s in (-1, 1):                                          # wings: faint glass over the abdomen
        path, *_ = fl._wing_path(s)
        fa.add_patch(PathPatch(path, facecolor="#FFFFFF", edgecolor=edge, linewidth=0.4, alpha=0.10, zorder=3))
        fa.add_patch(PathPatch(path, facecolor="none", edgecolor=edge, linewidth=0.4, alpha=0.8, zorder=3))
    fa.add_patch(Ellipse((0, 1.55), 2.6, 2.8, facecolor=body, edgecolor=edge, linewidth=0.5, zorder=4))     # thorax
    fa.add_patch(Ellipse((0, 3.5), 2.4, 1.5, facecolor=body, edgecolor=edge, linewidth=0.5, zorder=4))      # head
    for s in (-1, 1):                                          # red compound eyes: the fruit fly's mark; antennae
        fa.add_patch(Ellipse((1.0 * s, 3.6), 0.95, 1.4, angle=-12 * s, facecolor="#8B4A44", edgecolor="none", zorder=4.3))
        fl._leg(fa, [(0.25 * s, 4.15), (0.42 * s, 4.7)], legc, lw=0.6)
    cns = np.asarray(Image.open(TILES / "fly_cns.png").convert("RGB")).astype(float) / 255
    alpha = np.clip((cns @ LUMA) / 0.16, 0, 1)                 # black is see-through
    cw = 3.0                                                   # the CNS as wide as head and eyes (fly units)
    ch = cw * cns.shape[0] / cns.shape[1]
    fa.imshow(np.dstack([cns, alpha]), extent=(-cw / 2, cw / 2, 4.35 - ch, 4.35), interpolation="bilinear", zorder=5)
    fa.set_xlim(-6, 6)
    fa.set_ylim(-6.5, 6.5)
    fa.set_aspect("equal")
    fa.set_axis_off()


def cue(ax, x, cy, chosen=False):
    """The task's two targets as rings, the cued one lit gold (the cue; with chosen, the choice of it). x is the icon's left
    edge; returns its right edge."""
    r, step = 1.05, 3.5
    other, lit = (x + r, cy), (x + r + step, cy)
    ax.add_patch(Circle(other, r, facecolor="none", edgecolor=GREY, linewidth=1.0, zorder=8))
    for rr, a in ((2.0, 0.06), (1.5, 0.12), (1.2, 0.20)):
        ax.add_patch(Circle(lit, rr, facecolor=GOLD, edgecolor="none", alpha=a, zorder=7))
    ax.add_patch(Circle(lit, r, facecolor="none", edgecolor=GOLD, linewidth=1.0, zorder=8))
    ax.add_patch(Circle(lit, 0.62 if chosen else 0.32, facecolor=GOLD, edgecolor="none", zorder=8))
    return x + 2 * r + step


def strip(ax):
    """One trace cell through a trial: the cue, spikes that write it into adaptation (gold), a silent hold, the next input
    that reads it, and the choice. A schematic: no axis, number or scale bar."""
    right = cue(ax, STRIP_X0, CUE_Y)
    faded(ax, np.linspace(right + 0.4, WRITE_X[0] - 0.6, 16), CUE_Y, GOLD, 0.95, 0.4)
    # the trace: adaptation rises with the spikes and outlasts the silent hold
    x = np.linspace(WRITE_X[0], READ_X[-1] + 1.0, 400)
    rise = np.clip((x - WRITE_X[0]) / (WRITE_X[-1] - WRITE_X[0]), 0, 1)
    g = np.where(x <= WRITE_X[-1], 1 - (1 - rise) ** 2, np.exp(-(x - WRITE_X[-1]) / TRACE_TAU))
    y = BASE_Y - TRACE_PEAK * g
    ax.add_patch(Polygon(np.vstack([np.column_stack([x, y]), [[x[-1], BASE_Y], [x[0], BASE_Y]]]), closed=True, facecolor=GOLD,
                         edgecolor="none", alpha=0.22, zorder=5))
    stroke(ax, x, y, GOLD, lw=1.2, z=6)
    # spikes on the cell's baseline: written at the cue, none in the hold, the next input read
    stroke(ax, [WRITE_X[0] - 1.5, CHOICE_X - 1.6], [BASE_Y, BASE_Y], LINE, lw=0.6, z=7, halo=False)
    for xs in (WRITE_X, READ_X):
        for xx in xs:
            stroke(ax, [xx, xx], [TICK_TOP, BASE_Y], TEXT, lw=1.0, z=9, halo=False)
    faded(ax, np.linspace(READ_X[-1] + 1.0, CHOICE_X - 0.5, 16), CUE_Y, GOLD, 0.4, 0.95)
    cue(ax, CHOICE_X, CUE_Y, chosen=True)
    # the three phases above, the names of the icons and of the gold below
    hold = (WRITE_X[-1] + READ_X[0]) / 2
    text(ax, np.mean(WRITE_X), PHASE_Y, "write")
    text(ax, hold, PHASE_Y, "silent hold")
    text(ax, np.mean(READ_X), PHASE_Y, "read")
    text(ax, STRIP_X0 + 3.3, UNDER_Y, "cue")
    text(ax, hold, UNDER_Y, "adaptation")
    text(ax, CHOICE_X + 3.3, UNDER_Y, "choice")


# ---- band 2 -----------------------------------------------------------------------------------------------------------------

def header(ax, cx, colour, label):
    """A column's name after a short line in its state colour, centred on the column."""
    w = 3.6 + 1.6 + fs.text_width(label, LABEL)
    x0 = cx - w / 2
    stroke(ax, [x0, x0 + 3.6], [HEAD_Y, HEAD_Y], colour, lw=2.4, z=8, halo=False)
    text(ax, x0 + 5.2, HEAD_Y, label, ha="left")


def states(ax):
    """The trace in the whole CNS: the learned state and, after the shift, the rewritten trace; an arrow between them."""
    edges = {}
    for name, colour, label in (("learned", SILVER, "trained connectome"), ("shifted", PURPLE, "disease-like shift")):
        tw, th = META["tile_mm"]
        left, top, w, h = image(ax, TILES / f"{name}.png", COL_X[name] - tw / 2, RENDER_TOP)
        edges[name] = (left, left + w, top, top + h)
        header(ax, COL_X[name], colour, label)
    y = (edges["learned"][2] + edges["learned"][3]) / 2
    x0, x1 = edges["learned"][1] + 1.5, edges["shifted"][0] - 1.5
    xs = np.linspace(x0, x1 - 1.6, 40)
    a, b = np.array(to_rgb(SILVER)), np.array(to_rgb(PURPLE))
    cols = [(1 - f) * a + f * b for f in np.linspace(0, 1, len(xs) - 1)]
    segs = [[(xs[i], y), (xs[i + 1], y)] for i in range(len(xs) - 1)]
    ax.add_collection(LineCollection(segs, colors=[(*c, 0.12) for c in cols], linewidths=4.2, capstyle="round", zorder=6))
    ax.add_collection(LineCollection(segs, colors=[(*c, 0.95) for c in cols], linewidths=1.3, capstyle="round", zorder=7))
    ax.add_patch(Polygon([(x1 - 2.2, y - 1.3), (x1, y), (x1 - 2.2, y + 1.3)], closed=True, facecolor=PURPLE, edgecolor="none", zorder=8))


# ---- band 3 -----------------------------------------------------------------------------------------------------------------

def lock(ax, cx, body_top, colour, opened):
    """A padlock centred on cx with its body's top edge at body_top (mm); opened lifts the shackle's left leg free."""
    w, h = LOCK_W, LOCK_H
    r, leg, lift = 0.29 * w, 0.24 * w, (0.30 * w if opened else 0.0)
    ac = body_top - leg - lift                       # centre of the shackle's arc
    t = np.linspace(np.pi, 0, 48)
    xs = np.concatenate([[cx - r], cx + r * np.cos(t), [cx + r]])
    ys = np.concatenate([[body_top - lift + (0.0 if opened else 0.4)], ac - r * np.sin(t), [body_top + 0.4]])
    stroke(ax, xs, ys, colour, z=7)
    ax.add_patch(FancyBboxPatch((cx - w / 2, body_top), w, h, boxstyle=f"round,pad=0,rounding_size={0.2 * w}", facecolor="none",
                                edgecolor=colour, linewidth=LW * 3.2, alpha=0.10, zorder=8))
    ax.add_patch(FancyBboxPatch((cx - w / 2, body_top), w, h, boxstyle=f"round,pad=0,rounding_size={0.2 * w}", facecolor=BODY[colour],
                                edgecolor=colour, linewidth=LW, zorder=9))
    kr = 0.10 * w
    ax.add_patch(Circle((cx, body_top + 0.40 * h), kr, facecolor=colour, edgecolor="none", zorder=10))
    ax.add_patch(Polygon([(cx - 0.32 * kr, body_top + 0.44 * h), (cx + 0.32 * kr, body_top + 0.44 * h),
                          (cx + 0.55 * kr, body_top + 0.72 * h), (cx - 0.55 * kr, body_top + 0.72 * h)], closed=True,
                         facecolor=colour, edgecolor="none", zorder=10))
    return body_top + 0.40 * h                       # the keyhole's height


def key(ax, tip, cy, colour, angle=0.0, z=11):
    """A key pointing right with its tip at (tip, cy): a ring bow, a shaft and a two-step bit; angle (degrees, positive
    clockwise on the page) turns it about the tip. Below the lock's body (z < 9) the body hides its bit and tip."""
    L, R = KEY_L, 1.15
    t = np.linspace(0, 2 * np.pi, 72)
    bow = np.column_stack([R + R * np.cos(t), R * np.sin(t)])          # local coordinates: x from the bow's left edge
    shaft = np.array([[2 * R, 0.0], [L, 0.0]])
    bit = np.array([[L - 2.5, 0.0], [L - 2.5, 1.05], [L - 1.55, 1.05], [L - 1.55, 0.6], [L - 0.65, 0.6], [L - 0.65, 0.0]])
    a = np.radians(angle)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    for part in (bow, shaft, bit):
        p = (part - [L, 0.0]) @ rot.T + [tip, cy]
        stroke(ax, p[:, 0], p[:, 1], colour, z=z)


def outcome(ax, lock_x, cy, key_colour, lock_colour, label):
    """One test of the learned readout, on one line: a key cut in one state at the lock of the current state, then the
    outcome's name. The key opens only the lock of the state it was cut in."""
    fits = key_colour == lock_colour
    hole = lock(ax, lock_x, cy - LOCK_H / 2, lock_colour, opened=fits)
    if fits:                                         # the key slides in behind the body: only its bow and shaft show
        key(ax, lock_x, hole, key_colour, z=7.5)
    else:                                            # the key stops short of the lock, turned away
        key(ax, lock_x - LOCK_W / 2 - 1.2, hole + 0.4, key_colour, angle=-16.0)
    text(ax, lock_x + LOCK_W / 2 + 2.0, cy, label, ha="left")


def outcomes(ax):
    outcome(ax, LOCK_X["learned"], LINE_Y[0], SILVER, SILVER, "recall")
    outcome(ax, LOCK_X["shifted"], LINE_Y[0], SILVER, PURPLE, "recall lost")
    outcome(ax, LOCK_X["shifted"], LINE_Y[1], PURPLE, PURPLE, "new learning")


def build():
    fig, ax = canvas()
    fly(fig)
    strip(ax)
    states(ax)
    outcomes(ax)
    lock_view(ax)
    return fig, ax


def check(fig, files):
    """The guide's rules on the finished abstract; returns (issues, texts)."""
    issues, found = [], []
    w, h = fig.get_size_inches()
    if abs(w * 25.4 - S) > 1e-6 or abs(h * 25.4 - S) > 1e-6:
        issues.append(f"canvas {w * 25.4:.2f} x {h * 25.4:.2f} mm (expected {S} square)")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for t in qc._texts(fig):
        s, size = t.get_text(), t.get_fontsize()
        found.append(dict(text=s, size_pt=size, weight=str(t.get_fontweight()), colour=t.get_color()))
        if not PT_MIN - 1e-6 <= size <= PT_MAX + 1e-6:
            issues.append(f"font size {size} pt for {s!r} (the guide: {PT_MIN:g}-{PT_MAX:g} pt)")
        if t.get_fontname() != fs.FONT:
            issues.append(f"font {t.get_fontname()} for {s!r}")
        if t.get_color() != TEXT:
            issues.append(f"text {s!r} is not in the one text colour")
        if qc.INTERNAL.search(s) or qc.DEV_LABELS.search(s):
            issues.append(f"internal label in {s!r}")
        for word, us in qc.british(s):
            issues.append(f"British spelling {word!r} in {s!r} (American: {us})")
        bb = t.get_window_extent(r)
        if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fig.bbox.width or bb.y1 > fig.bbox.height:
            issues.append(f"text {s!r} leaves the canvas")
        boxes.append((s, bb))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                issues.append(f"texts overlap: {boxes[i][0]!r} and {boxes[j][0]!r}")
    img = Image.open(files["tif"])
    if img.size != (round(S / 25.4 * PPI),) * 2:
        issues.append(f"TIFF is {img.size} px (expected 1,200 square)")
    if tuple(round(v) for v in img.info.get("dpi", (0, 0))) != (PPI, PPI):
        issues.append(f"TIFF resolution {img.info.get('dpi')} (expected {PPI} dpi)")
    a = np.asarray(img.convert("RGB"), float)
    ring = np.concatenate([a[:4].reshape(-1, 3), a[-4:].reshape(-1, 3), a[:, :4].reshape(-1, 3), a[:, -4:].reshape(-1, 3)])
    if ring.max() > 60:
        issues.append(f"the outer 4 px are not dark (max {ring.max():.0f} of 255): no white border allowed")
    return issues, found


def main():
    fig, ax = build()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{NAME}.png"
    fig.savefig(png, dpi=PPI, facecolor="black", metadata={"Software": None})
    rgb = Image.open(png).convert("RGB")
    files = {"png": png, "tif": OUT / f"{NAME}.tif", "jpg": OUT / f"{NAME}.jpg", "pdf": OUT / f"{NAME}.pdf"}
    rgb.save(files["tif"], compression="tiff_lzw", dpi=(PPI, PPI))
    rgb.save(files["jpg"], quality=95, dpi=(PPI, PPI))
    rgb.save(png, dpi=(PPI, PPI))
    fig.savefig(files["pdf"], format="pdf", facecolor="black", metadata={"CreationDate": None, "ModDate": None,
                                                                       "Creator": "manuscript/figures", "Title": NAME})
    issues, found = check(fig, files)
    qc.simulate_png(png, OUT / "vision")
    sd = qc.SourceData(NAME)
    sd.source(cm.RESULTS / "plan_v2" / "link_cells.csv", *sorted(TILES.glob("*.png")), *sorted(TILES.glob("*.json")))
    report = dict(figure=NAME, issues=issues, texts=found, words=sum(len(t["text"].split()) for t in found),
                  size_mm=[S, S], ppi=PPI, files={k: v.relative_to(cm.FIG).as_posix() for k, v in files.items()}, inputs=sd.inputs)
    (OUT / f"{NAME}_qc.json").write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    plt.close(fig)
    print(f"[{NAME}] {len(issues)} QC issue(s); {len(found)} labels, {report['words']} words")
    for i in issues:
        print("   -", i)


if __name__ == "__main__":
    main()
