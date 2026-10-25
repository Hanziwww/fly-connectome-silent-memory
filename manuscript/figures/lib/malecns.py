"""The visual language of the MaleCNS Cell article (Berg et al. 2026), measured from its figures, and the drawing tools that
reproduce its forms with our data.

STYLE_2B holds colours measured from Fig. 2B rendered at 1200 dpi (2026-10-23): flat pastel strands per transmitter, black
hairline node bars, light grey sensory/motor strips and neck band, a grey axis arrow. FlowMap2B draws a flow map in that form:
nodes placed freely (input strip left, readout strip right, relays by anatomy between them), forward ribbons that leave and
enter horizontally with one long straight diagonal between rounded bends, loops for flows that run back or stay within a
class (orthogonal, rounded, with white chevrons), strands side by side by presynaptic transmitter, thin flows as single
curves, and - our addition, in the grammar of MaleCNS Fig. 2D (max flow inside capacity) - a darker core in each strand for
the flow that remains after a perturbation.
"""
import numpy as np
from matplotlib.lines import Line2D

from . import figstyle as fs
from .circuit import _poly, _rect, offset_path, rounded_path

STYLE_2B = {
    "acetylcholine": "#F9C387", "gaba": "#87CFB7", "glutamate": "#D5B1C9", "histamine": "#B8E076", "other": "#999999",
    "core": {"acetylcholine": "#E99A3F", "gaba": "#3FA584", "glutamate": "#A87A9E", "histamine": "#7FB53A", "other": "#6B6B6B"},
    "node": "#030303", "strip": "#EDEDED", "strip_edge": "#C8C8C8", "neck": "#E7E7E7", "axis": "#B4B2B2", "chevron": "#FFFFFF",
}
ORDER = ("acetylcholine", "gaba", "glutamate", "histamine", "other")


class FlowMap2B:
    """Flow map in the form of MaleCNS Fig. 2B (see the module docstring). Coordinates in mm; scale in mm per unit of flow."""

    def __init__(self, cv, scale, bar_w=0.35, thin=0.3, clearance=1.4):
        self.cv, self.scale, self.bar_w, self.thin, self.clear = cv, scale, bar_w, thin, clearance
        self.nodes, self.flows = {}, []

    # ---- content
    def node(self, key, x, y, label=None, label_at=None, label_dx=0.0, label_dy=0.0):
        """x: bar position; y: bar top (mm). label_at: 'above', 'below', 'left' or 'right' (multi-line labels allowed)."""
        self.nodes[key] = dict(x=x, y0=y, label=label, label_at=label_at, ldx=label_dx, ldy=label_dy)

    def flow(self, a, b, strands, lane=None):
        """strands: {transmitter: (intact, remaining)}; lane: x of the diagonal's middle (forward flows)."""
        st = [(t, float(v[0]), float(v[1])) for t, v in sorted(strands.items(), key=lambda q: ORDER.index(q[0]) if q[0] in ORDER else 9)
              if v[0] > 0]
        if st:
            self.flows.append(dict(a=a, b=b, strands=st, lane=lane))

    # ---- layout
    def _w(self, f):
        return sum(s[1] for s in f["strands"]) * self.scale

    def _back(self, f):
        return self.nodes[f["b"]]["x"] <= self.nodes[f["a"]]["x"]

    def layout(self):
        for k, n in self.nodes.items():
            vin = sum(self._w(f) for f in self.flows if f["b"] == k)
            vout = sum(self._w(f) for f in self.flows if f["a"] == k)
            n["h"] = max(vin, vout, 0.4)
            n["y1"] = n["y0"] + n["h"]
        for k, n in self.nodes.items():
            outs = [f for f in self.flows if f["a"] == k]
            outs.sort(key=lambda f: (self._back(f), self.nodes[f["b"]]["y0"]))
            y = n["y0"] + (n["h"] - sum(self._w(f) for f in outs)) / 2
            for f in outs:
                f["ya"] = y
                y += self._w(f)
            ins = [f for f in self.flows if f["b"] == k]
            ins.sort(key=lambda f: (self._back(f), self.nodes[f["a"]]["y0"]))
            y = n["y0"] + (n["h"] - sum(self._w(f) for f in ins)) / 2
            for f in ins:
                f["yb"] = y
                y += self._w(f)
        loops = sorted([f for f in self.flows if self._back(f)], key=lambda f: self._span(f)[1] - self._span(f)[0])
        placed = []
        for f in loops:                                         # each loop hugs its own two nodes, nested by span, and drops
            lo, hi = self._span(f)                              # lower only past a node or loop its run would cut
            w = self._w(f)
            floor = max(self.nodes[f["a"]]["y1"], self.nodes[f["b"]]["y1"]) + self.clear
            moved = True
            while moved:
                moved = False
                for n in self.nodes.values():
                    if lo - 0.5 < n["x"] < hi + 0.5 and n["y0"] - 0.5 < floor + w and n["y1"] + self.clear > floor:
                        floor, moved = n["y1"] + self.clear, True
                for g in placed:
                    gy0, gy1 = g["loop_y"] - 0.5, g["loop_y"] + self._w(g) + 0.5
                    if g["span"][0] < hi and lo < g["span"][1] and gy0 < floor + w and gy1 > floor:
                        floor, moved = gy1, True
            f["span"], f["loop_y"] = (lo, hi), floor
            placed.append(f)

    def _span(self, f):
        a, b = self.nodes[f["a"]], self.nodes[f["b"]]
        e = self._e(f)
        return b["x"] - e - self._w(f) / 2, a["x"] + self.bar_w + e + self._w(f) / 2

    def _e(self, f):
        return max(self._w(f) / 2 + 0.8, 1.6) + 0.9

    # ---- geometry
    def path(self, f):
        a, b = self.nodes[f["a"]], self.nodes[f["b"]]
        w = self._w(f)
        ya, yb = f["ya"] + w / 2, f["yb"] + w / 2
        xa, xb = a["x"] + self.bar_w, b["x"]
        if not self._back(f):
            dy = abs(yb - ya)
            if dy < 0.05:
                return rounded_path([(xa, ya), (xb, yb)], 1.0)
            span = xb - xa
            tw = float(np.clip(0.9 * dy + 2.5, 2.0, 0.85 * span))
            xm = f["lane"] if f["lane"] is not None else xa + 0.5 * span
            x1 = float(np.clip(xm - tw / 2, xa + 0.3, xb - tw - 0.3))
            x2 = x1 + tw
            r = min(3.2, 0.45 * tw, 0.5 * dy + 0.8)
            return rounded_path([(xa, ya), (x1, ya), (x2, yb), (xb, yb)], r)
        e = self._e(f)
        r = max(w / 2 + 0.6, 1.2)
        yl = f["loop_y"] + w / 2
        return rounded_path([(xa, ya), (xa + e, ya), (xa + e, yl), (xb - e, yl), (xb - e, yb), (xb, yb)], r)

    # ---- drawing
    def draw(self, chevrons=True, core=True):
        self.layout()
        for f in sorted(self.flows, key=lambda f: -self._w(f)):
            c = self.path(f)
            w = self._w(f)
            if w < self.thin:                                   # thin flows: one smooth curve in the dominant transmitter's colour
                t = max(f["strands"], key=lambda s: s[1])[0]
                if not self._back(f):
                    a_, b_ = self.nodes[f["a"]], self.nodes[f["b"]]
                    p0 = np.array([a_["x"] + self.bar_w, f["ya"] + w / 2])
                    p3 = np.array([b_["x"], f["yb"] + w / 2])
                    k = 0.5 * (p3[0] - p0[0])
                    tt = np.linspace(0, 1, 80)[:, None]
                    c = ((1 - tt) ** 3 * p0 + 3 * (1 - tt) ** 2 * tt * (p0 + [k, 0]) + 3 * (1 - tt) * tt ** 2 * (p3 - [k, 0])
                         + tt ** 3 * p3)
                self.cv.fig.add_artist(Line2D(c[:, 0] / self.cv.w, 1 - c[:, 1] / self.cv.h, transform=self.cv.fig.transFigure,
                                              color=STYLE_2B.get(t, STYLE_2B["other"]), linewidth=0.45, zorder=1, solid_capstyle="butt"))
                continue
            off = -w / 2
            for t, v0, v1 in f["strands"]:
                sw = v0 * self.scale
                mid = off + sw / 2
                _poly(self.cv, np.vstack([offset_path(c, mid + sw / 2), offset_path(c, mid - sw / 2)[::-1]]),
                      STYLE_2B.get(t, STYLE_2B["other"]), 1.0)
                if core and v1 > 0 and sw >= 0.25:
                    cw = sw * v1 / v0
                    _poly(self.cv, np.vstack([offset_path(c, mid + cw / 2), offset_path(c, mid - cw / 2)[::-1]]),
                          STYLE_2B["core"].get(t, STYLE_2B["core"]["other"]), 1.01)
                off += sw
            if chevrons and self._back(f) and w >= 0.6:
                self._chevrons(c, w)
        for n in self.nodes.values():
            _rect(self.cv, n["x"], n["y0"], n["x"] + self.bar_w, n["y1"], STYLE_2B["node"], zorder=4)
        return self

    def _chevrons(self, c, w):
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
        size = min(0.5, 0.3 * w)
        for s_ in np.arange(1.0, L[-1] - 1.0, 1.1):
            i = int(np.searchsorted(L, s_))
            d = c[min(i + 2, len(c) - 1)] - c[max(i - 2, 0)]
            d = d / (np.linalg.norm(d) + 1e-12)
            nrm = np.array([-d[1], d[0]])
            tip = c[i] + d * size / 2
            pts = [tip - d * size + nrm * size * 0.8, tip, tip - d * size - nrm * size * 0.8]
            self.cv.fig.add_artist(Line2D([q[0] / self.cv.w for q in pts], [1 - q[1] / self.cv.h for q in pts], transform=self.cv.fig.transFigure,
                                          color=STYLE_2B["chevron"], linewidth=0.35, zorder=2))

    def labels(self, size=None):
        size = size or fs.PT
        for n in self.nodes.values():
            if not n["label"]:
                continue
            at, x = n["label_at"], n["x"] + n["ldx"]
            kw = dict(size=size, linespacing=0.95)
            if at == "above":
                t = self.cv.text(x - 0.3, n["y0"] - 0.5 + n["ldy"], n["label"], va="bottom", ha="left", **kw)
            elif at == "below":
                t = self.cv.text(x - 0.3, n["y1"] + 0.5 + n["ldy"], n["label"], va="top", ha="left", **kw)
            elif at == "left":
                t = self.cv.text(x - 0.8, (n["y0"] + n["y1"]) / 2 + n["ldy"], n["label"], va="center", ha="right", **kw)
            else:
                t = self.cv.text(x + self.bar_w + 0.8, (n["y0"] + n["y1"]) / 2 + n["ldy"], n["label"], va="center", ha="left", **kw)
            t.set_path_effects(fs.halo(1.2))
            t.set_zorder(6)


def strip(cv, x, y, w, h):
    """A light grey column (the sensory and motor strips of Fig. 2B)."""
    cv.box(x, y, w, h, STYLE_2B["strip"], ec=STYLE_2B["strip_edge"], lw=0.3)


def axis_arrow(cv, x0, x1, y, left, right, size=None):
    """'left ------> right' in the grey of Fig. 2B's sensory-to-motor axis."""
    from .circuit import arrow
    size = size or fs.PT
    cv.text(x0, y, left, size=size, va="center")
    cv.text(x1, y, right, size=size, va="center", ha="right")
    a = x0 + fs.text_width(left, size) + 2.4
    b = x1 - fs.text_width(right, size) - 2.4
    arrow(cv, [(a, y), (b, y)], 0.4, STYLE_2B["axis"], "arrow", gap=0.0)


def zone_labels(cv, x, zones, size=None):
    """Rotated 'brain', 'neck', 'VNC' along the left margin, joined by grey double arrows (Fig. 2B)."""
    from .circuit import arrow
    size = size or fs.PT_SMALL
    items = list(zones.items())
    for name, (y0, y1) in items:
        cv.text(x, (y0 + y1) / 2, name, size=size, rotation=90, ha="center", va="center")
    for (n0, (a0, a1)), (n1, (b0, b1)) in zip(items[:-1], items[1:]):
        ya = (a0 + a1) / 2 + fs.text_width(n0, size) / 2 + 1.0
        yb = (b0 + b1) / 2 - fs.text_width(n1, size) / 2 - 1.0
        if yb - ya > 2.0:
            arrow(cv, [(x - 0.35, ya), (x - 0.35, yb)], 0.22, STYLE_2B["axis"], "arrow", gap=0.0)
            arrow(cv, [(x + 0.35, yb), (x + 0.35, ya)], 0.22, STYLE_2B["axis"], "arrow", gap=0.0)
