"""A stylised top view of a fly for organism insets (the MaleCNS Figs 5A, 6A and 6I insets: a grey fly in a light-grey square,
the relevant organ in red). Drawn as vector patches in an axes whose data units are arbitrary (head up, about 12 units tall).

Wings follow the fly's own shape: narrow at the hinge on the side of the thorax, broad and rounded at the tip, folded back
over the abdomen in a narrow V; translucent, outlined, with four longitudinal veins and two cross-veins clipped to the blade.
"""
import numpy as np
from matplotlib.patches import Ellipse, PathPatch, Rectangle
from matplotlib.path import Path as MPath

BODY, DARK, LEG, WING_EDGE, VEIN = "#A3A3A3", "#6F6F6F", "#7A7A7A", "#9A9A9A", "#B9B9B9"
WING_L, WING_W, WING_ANGLE = 5.7, 2.5, 13.0                  # length, breadth (units), angle from the midline (deg)


def _leg(ax, pts, colour, lw=0.6):
    pts = np.asarray(pts, float)
    ax.plot(pts[:, 0], pts[:, 1], color=colour, linewidth=lw, solid_capstyle="round", solid_joinstyle="round", zorder=1)


def _frame(side):
    """Hinge, axis and outer-side unit vectors of one wing (side -1 left, +1 right)."""
    th = np.radians(WING_ANGLE)
    hinge = np.array([0.55 * side, 0.95])
    d = np.array([side * np.sin(th), -np.cos(th)])            # toward the tip
    n = np.array([side * np.cos(th), np.sin(th)])              # toward the leading (outer) edge
    return hinge, d, n


def _wing_path(side):
    hinge, d, n = _frame(side)
    L, W = WING_L, WING_W
    uv = [(0.00, 0.06), (0.25, 0.30), (0.70, 0.50), (0.93, 0.38),      # leading edge
          (1.03, 0.26), (1.03, -0.24), (0.92, -0.38),                    # rounded tip
          (0.70, -0.52), (0.32, -0.40), (0.12, -0.17),                   # trailing edge
          (0.05, -0.10), (0.00, -0.04), (0.00, 0.06)]                    # back to the hinge
    pts = [hinge + u * L * d + v * W * n for u, v in uv]
    codes = [MPath.MOVETO] + [MPath.CURVE4] * 12
    return MPath(np.array(pts), codes), hinge, d, n


def _veins(ax, side, clip):
    """Three longitudinal veins arching gently toward the tip, and one cross-vein; thin and pale, clipped to the blade."""
    hinge, d, n = _frame(side)
    L, W = WING_L, WING_W
    for v0, v1 in ((0.10, 0.30), (0.00, 0.03), (-0.09, -0.27)):
        u = np.linspace(0.08, 1.02, 40)
        t = (u - 0.08) / 0.94
        v = v0 + (v1 - v0) * t + 0.05 * np.sin(np.pi * t)
        pts = np.array([hinge + a * L * d + b * W * n for a, b in zip(u, v)])
        ln, = ax.plot(pts[:, 0], pts[:, 1], color=VEIN, linewidth=0.3, zorder=3.1)
        ln.set_clip_path(clip)
    a = hinge + 0.56 * L * d + 0.06 * W * n
    b = hinge + 0.60 * L * d + -0.13 * W * n
    ln, = ax.plot([a[0], b[0]], [a[1], b[1]], color=VEIN, linewidth=0.3, zorder=3.1)
    ln.set_clip_path(clip)


def draw(ax, highlight=None, eyes=None, antennae=None, legs=None):
    """Draw the fly into ax (limits -6..6 by -6.5..6.5, equal aspect). highlight colours: eyes, antennae, legs (or None)."""
    eyes = eyes or highlight or "#8C8C8C"
    lc = legs or LEG
    for s in (-1, 1):                                          # legs: coxa at the thorax, femur, tibia, tarsus
        _leg(ax, [(0.7 * s, 2.3), (1.9 * s, 3.0), (2.3 * s, 4.1), (2.6 * s, 5.2)], lc)
        _leg(ax, [(0.9 * s, 1.6), (2.4 * s, 1.9), (3.5 * s, 1.2), (4.4 * s, 0.6)], lc)
        _leg(ax, [(0.8 * s, 0.9), (2.0 * s, -0.1), (2.7 * s, -1.9), (3.1 * s, -3.2)], lc)
    abd = Ellipse((0, -1.95), 2.4, 4.6, facecolor="#B8B8B8", edgecolor="none", zorder=2)
    ax.add_patch(abd)
    for y in (-0.95, -1.8, -2.65, -3.45):                      # tergite bands, clipped to the abdomen
        band = Rectangle((-1.4, y - 0.2), 2.8, 0.36, facecolor=DARK, edgecolor="none", zorder=2.1)
        ax.add_patch(band)
        band.set_clip_path(abd)
    for s in (-1, 1):                                          # wings: translucent blades over the abdomen
        path, *_ = _wing_path(s)
        blade = PathPatch(path, facecolor="#FFFFFF", edgecolor=WING_EDGE, linewidth=0.4, alpha=0.72, zorder=3)
        ax.add_patch(blade)
        _veins(ax, s, blade)
    ax.add_patch(Ellipse((0, 1.55), 2.6, 2.8, facecolor=BODY, edgecolor="none", zorder=4))          # thorax
    ax.add_patch(Ellipse((0, 0.3), 1.2, 0.7, facecolor="#999999", edgecolor="none", zorder=4.1))    # scutellum
    for x in (-0.42, 0.42):                                    # thoracic stripes
        ax.plot([x, x * 0.8], [2.6, 0.95], color="#8A8A8A", linewidth=0.3, zorder=4.2)
    ax.add_patch(Ellipse((0, 3.5), 2.4, 1.5, facecolor="#9C9C9C", edgecolor="none", zorder=4))      # head
    for s in (-1, 1):
        ax.add_patch(Ellipse((1.0 * s, 3.6), 0.95, 1.4, angle=-12 * s, facecolor=eyes, edgecolor="none", zorder=4.3))
        _leg(ax, [(0.25 * s, 4.15), (0.42 * s, 4.7)], antennae or "#7A7A7A", lw=0.6)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6.5, 6.5)
    ax.set_aspect("equal")
    ax.set_axis_off()


def inset(cv, x, y, size, highlight="#D7191C", frame="#D0D0D0", **kw):
    """The fly in a light-grey framed square at (x, y) mm (top-left), size mm wide."""
    cv.box(x, y, size, size, "none", ec=frame, lw=1.0)
    ax = cv.axes(x + 0.6, y + 0.6, size - 1.2, size - 1.2)
    ax.set_zorder(5)
    draw(ax, highlight=highlight, **kw)
    return ax
