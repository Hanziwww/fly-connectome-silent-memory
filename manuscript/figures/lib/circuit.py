"""Circuit-diagram elements in the visual grammar of the MaleCNS Cell article (Berg et al. 2026, Figs 2D, 4H-I, 5F, 7M-N; the
user's request of 2026-10-23, specification in design/MALECNS_STYLE_SPEC.md), carrying our own results.

- Network diagrams: cell-type chips in columns under black-ruled headers; orthogonal edges in parallel lanes (ordered to
  minimise crossings, ports spread along the chip side); edge colour by the sender's transmitter, paler for thinner bins;
  widths in four bins with a 2 x 2 key of grey arrows; triangle heads (acetylcholine, glutamate) or flat bars (GABA).
- Grey module grounds with italic labels, a thick dashed Brain/VNC rule, pathway chips, a red zoom box with dotted connectors,
  counts inside bars.

Placement is in mm from the top-left corner of a figs.Canvas. Edges, heads and dashes are filled polygons, never stroked
lines, so the QC rule on line widths (0.25-1 pt) holds at any edge width. Light text sits only on fills registered in
cv.fills (QC checks its contrast).
"""
from dataclasses import dataclass, field

import numpy as np
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from . import figstyle as fs

CHIP_H, CHIP_PAD, CHIP_R = 3.4, 1.1, 0.55
GREY_CHIP = "#E4E4E4"
MODULE_FILL = "#E6E6E6"
RULE = "#1A1A1A"
DASH = "#D2D2D2"
ZOOM = "#D7191C"
TRANSMITTER = {"acetylcholine": "#F29B30", "gaba": "#3BAE85", "glutamate": "#B87AAE"}
TRANSMITTER_LABEL = {"acetylcholine": "acetylcholine", "gaba": "GABA", "glutamate": "glutamate"}   # as Fig. 6G and MaleCNS Fig. 2B
WIDTHS = (0.35, 0.6, 0.9, 1.25)
PALE = (0.55, 0.35, 0.15, 0.0)                 # mixed with white, thinnest bin first
KEY_GREY = "#5A5A5A"


def tint(colour, k):
    """colour mixed with white by k (0: unchanged, 1: white)."""
    c = np.asarray(to_rgb(colour))
    return to_hex(c + (1 - c) * k)


def head_for(transmitter):
    return "bar" if str(transmitter).lower() == "gaba" else "arrow"


def colour_for(transmitter, b=3):
    return tint(TRANSMITTER.get(str(transmitter).lower(), "#9E9E9E"), PALE[b])


def bin_of(value, edges_):
    """Index of value's bin; edges_ are the n+1 bin edges."""
    return int(np.clip(np.searchsorted(edges_, value, side="right") - 1, 0, len(edges_) - 2))


# ---------------------------------------------------------------- primitives

def _poly(cv, pts, colour, zorder=2, alpha=1.0):
    pts = np.asarray(pts, float)
    cv.fig.patches.append(Polygon(np.column_stack([pts[:, 0] / cv.w, 1 - pts[:, 1] / cv.h]), closed=True, transform=cv.fig.transFigure,
                                  facecolor=colour, edgecolor="none", linewidth=0.0, zorder=zorder, alpha=alpha))


def _rect(cv, x0, y0, x1, y1, colour, zorder=2):
    _poly(cv, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], colour, zorder)


def thick_path(cv, pts, width, colour, zorder=2):
    """An axis-aligned polyline `width` mm wide as rectangles with square joins (each inner joint extended by half a width)."""
    pts = np.asarray(pts, float)
    h = width / 2
    n = len(pts) - 1
    for k in range(n):
        (x0, y0), (x1, y1) = pts[k], pts[k + 1]
        e0 = h if k > 0 else 0.0
        e1 = h if k < n - 1 else 0.0
        if abs(y1 - y0) < 1e-9:                  # horizontal
            lo, hi = (x0 - e0, x1 + e1) if x1 >= x0 else (x1 - e1, x0 + e0)
            _rect(cv, lo, y0 - h, hi, y0 + h, colour, zorder)
        else:                                    # vertical
            lo, hi = (y0 - e0, y1 + e1) if y1 >= y0 else (y1 - e1, y0 + e0)
            _rect(cv, x0 - h, lo, x0 + h, hi, colour, zorder)


def head_size(width, head):
    if head == "arrow":
        return max(1.0, 1.8 * width), max(0.9, 1.2 * width)
    if head == "bar":
        return max(1.2, 2.2 * width), 0.35
    return 0.0, 0.0


def head(cv, tip, direction, width, colour, kind="arrow", zorder=3):
    """A triangle (tip at `tip`) or a flat bar (outer face at `tip`) pointing along direction (dx, dy)."""
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    nrm = np.array([-d[1], d[0]])
    hw, hl = head_size(width, kind)
    tip = np.asarray(tip, float)
    base = tip - d * hl
    if kind == "arrow":
        _poly(cv, [base + nrm * hw / 2, tip, base - nrm * hw / 2], colour, zorder)
    elif kind == "bar":
        _poly(cv, [base + nrm * hw / 2, tip + nrm * hw / 2, tip - nrm * hw / 2, base - nrm * hw / 2], colour, zorder)


def arrow(cv, pts, width, colour, kind="arrow", gap=0.08, zorder=2):
    """An orthogonal polyline whose last segment ends in a head; pts end at the target's border, gap mm short of it."""
    pts = np.asarray(pts, float).copy()
    d = pts[-1] - pts[-2]
    d = d / (np.linalg.norm(d) + 1e-12)
    tip = pts[-1] - d * gap
    _, hl = head_size(width, kind)
    pts[-1] = tip - d * (hl - (0.02 if kind == "arrow" else 0.0))
    thick_path(cv, pts, width, colour, zorder)
    if kind:
        head(cv, tip, d, width, colour, kind, zorder + 0.1)


def ribbon_curve(cv, p0, p1, width, colour, bend=0.0, head_kind="arrow", n=48, zorder=3):
    """A thick curved flow arrow (quadratic Bezier, control point offset by bend x the chord length), tapered head at p1: the
    flow arrows of the MaleCNS overview schematics."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    ctrl = (p0 + p1) / 2 + bend * np.array([-d[1], d[0]])
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1
    hw, hl = head_size(width, head_kind)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    body = pts[cum <= cum[-1] - hl] if head_kind else pts
    tan = np.gradient(body, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
    _poly(cv, np.vstack([body + nrm * width / 2, (body - nrm * width / 2)[::-1]]), colour, zorder)
    if head_kind:
        tip = pts[-1]
        dirn = tip - body[-1]
        head(cv, tip, dirn, width, colour, head_kind, zorder + 0.1)


# ---------------------------------------------------------------- chips

@dataclass
class Node:
    key: str
    label: str
    x: float                                     # centre (mm)
    y: float
    w: float = 0.0                               # 0: fitted to the label
    h: float = CHIP_H
    fill: str = GREY_CHIP
    edge: str = "none"
    size: float = 0.0                            # font size (0: PT)
    text: str = "#1A1A1A"
    col: int = 0
    out_edges: list = field(default_factory=list)
    in_edges: list = field(default_factory=list)

    def __post_init__(self):
        self.size = self.size or fs.PT
        if not self.w:
            self.w = fs.text_width(self.label, self.size) + 2 * CHIP_PAD

    @property
    def left(self):
        return self.x - self.w / 2

    @property
    def right(self):
        return self.x + self.w / 2

    @property
    def top(self):
        return self.y - self.h / 2

    @property
    def bottom(self):
        return self.y + self.h / 2


def chip(cv, n: Node, lw: float = 0.6, radius: float = CHIP_R, zorder: float = 4) -> None:
    """A rounded label box centred on (n.x, n.y); a light label is registered with its fill for the QC contrast check."""
    cv.fig.patches.append(FancyBboxPatch((n.left / cv.w, 1 - n.bottom / cv.h), n.w / cv.w, n.h / cv.h,
                                         boxstyle=f"round,pad=0,rounding_size={radius / cv.w}", transform=cv.fig.transFigure,
                                         facecolor=n.fill, edgecolor=n.edge, linewidth=lw if n.edge != "none" else 0.0,
                                         mutation_aspect=cv.w / cv.h, zorder=zorder))
    if n.label:
        t = cv.text(n.x, n.y + 0.05, n.label, size=n.size, ha="center", va="center", color=n.text)
        t.set_zorder(zorder + 0.1)
        if n.text.upper() in ("#FFFFFF", "WHITE"):
            cv.fills.append(dict(rect=(n.left, n.top, n.w, n.h), fill=n.fill))


def label_chip(cv, x, y, text, fill, text_colour="#FFFFFF", size=None, ha="left", pad=0.7, h=2.8):
    """A small coloured label box (MaleCNS names flows and neurons this way); x is its left edge (ha='left') or centre."""
    size = size or fs.PT_SMALL
    w = fs.text_width(text, size) + 2 * pad
    cx = x + w / 2 if ha == "left" else x
    chip(cv, Node(text, text, cx, y, w=w, h=h, fill=fill, size=size, text=text_colour), radius=0.4)
    return cx + w / 2


# ---------------------------------------------------------------- network with orthogonal routing

@dataclass
class Edge:
    a: Node
    b: Node
    width: float
    colour: str
    kind: str = "arrow"
    ys: float = 0.0
    yd: float = 0.0
    segs: list = field(default_factory=list)      # vertical runs: [gap (c, c+1), enter y, exit y, lane x]
    side: bool = False                            # both ends in the last column: routed down its right side


class Network:
    """Chips in columns and orthogonal edges between them: out of the sender's right side, along vertical lanes in the gaps
    between columns, into the receiver's left side. An edge that skips columns runs straight through a free row, or doglegs
    through a free corridor between the chips it passes. Edges between two chips of the last column run down a lane to its
    right and enter the receiver from the right (MaleCNS Fig. 4H routes such links beside the column)."""

    def __init__(self, cv):
        self.cv, self.nodes, self.edges = cv, {}, []

    def add(self, key, label, x, y, col, **style) -> Node:
        n = Node(key, label, x, y, col=col, **style)
        self.nodes[key] = n
        return n

    def link(self, a, b, width, colour, kind="arrow") -> Edge:
        e = Edge(self.nodes[a], self.nodes[b], width, colour, kind)
        last = max(n.col for n in self.nodes.values())
        e.side = e.a.col == e.b.col == last
        if e.b.col <= e.a.col and not e.side:
            raise ValueError(f"edges run left to right, or within the last column: {a} -> {b}")
        self.edges.append(e)
        e.a.out_edges.append(e)
        e.b.in_edges.append(e)
        return e

    @staticmethod
    def _ports(node, edges, attr, other, gap):
        """Spread the edges' ports along one chip side, ordered by the other end's row; spacing by edge widths."""
        es = sorted(edges, key=lambda e: (getattr(e, other).y, getattr(e, other).x))
        if not es:
            return
        widths = sum(e.width for e in es)
        room = node.h - 0.4
        g = gap if widths + gap * (len(es) - 1) <= room or len(es) == 1 else max(0.04, (room - widths) / (len(es) - 1))
        need = widths + g * (len(es) - 1)
        y = node.y - need / 2
        for e in es:
            setattr(e, attr, y + e.width / 2)
            y += e.width + g

    @staticmethod
    def _crossings(p, q):
        """Crossings when vertical run p is left of run q in one gap (each entered from the left at its first y and left to
        the right at its second)."""
        (_, p0, p1, _), (_, q0, q1, _) = p, q
        c = 0
        if min(p0, p1) < q0 < max(p0, p1):
            c += 1
        if min(q0, q1) < p1 < max(q0, q1):
            c += 1
        return c

    def _order(self, runs):
        order = sorted(runs, key=lambda r: (-(r[1] if r[2] >= r[1] else -r[1]), r[2]))
        improved = True
        while improved:
            improved = False
            for i in range(len(order) - 1):
                p, q = order[i], order[i + 1]
                if self._crossings(q, p) < self._crossings(p, q):
                    order[i], order[i + 1] = q, p
                    improved = True
        return order

    def _free(self, y, width, cols):
        return all(abs(y - n.y) >= n.h / 2 + width / 2 + 0.25 for n in self.nodes.values() if n.col in cols)

    def _corridor(self, y_mid, width, cols):
        ys = sorted({n.y for n in self.nodes.values() if n.col in cols})
        hs = {n.y: n.h for n in self.nodes.values() if n.col in cols}
        cand = [ys[0] - hs[ys[0]] / 2 - width / 2 - 0.6, ys[-1] + hs[ys[-1]] / 2 + width / 2 + 0.6] + [(a + b) / 2 for a, b in zip(ys[:-1], ys[1:])]
        cand = [c for c in cand if self._free(c, width, cols)]
        return min(cand, key=lambda c: abs(c - y_mid))

    def route(self, port_gap=0.3, lane_gap=0.4, margin=1.0):
        for n in self.nodes.values():
            self._ports(n, [e for e in n.out_edges if not e.side], "ys", "b", port_gap)
            self._ports(n, [e for e in n.in_edges if not e.side], "yd", "a", port_gap)
        side = [e for e in self.edges if e.side]
        self._side_ports(side, port_gap)
        if side:
            x0 = max(n.right for n in self.nodes.values() if n.col == side[0].a.col) + margin
            for k, e in enumerate(sorted(side, key=lambda e: abs(e.ys - e.yd))):
                e.segs = [[None, e.ys, e.yd, x0 + k * (max(x.width for x in side) + lane_gap) + e.width / 2]]
        runs = {}
        for e in self.edges:
            if e.side:
                continue
            ca, cb = e.a.col, e.b.col
            mid = list(range(ca + 1, cb))
            if abs(e.ys - e.yd) <= 1e-6 and (not mid or self._free(e.ys, e.width, mid)):
                e.segs = []
            elif not mid:
                e.segs = [[(ca, cb), e.ys, e.yd, 0.0]]
            elif self._free(e.ys, e.width, mid):
                e.segs = [[(cb - 1, cb), e.ys, e.yd, 0.0]]
            elif self._free(e.yd, e.width, mid):
                e.segs = [[(ca, ca + 1), e.ys, e.yd, 0.0]]
            else:
                yc = self._corridor((e.ys + e.yd) / 2, e.width, mid)
                e.segs = [[(ca, ca + 1), e.ys, yc, 0.0], [(cb - 1, cb), yc, e.yd, 0.0]]
            for sgm in e.segs:
                runs.setdefault(sgm[0], []).append((e, sgm))
        for (ca, cb), items in runs.items():
            x0 = max(n.right for n in self.nodes.values() if n.col == ca) + margin
            x1 = min(n.left for n in self.nodes.values() if n.col == cb) - margin
            keyed = {id(sgm): (e, sgm) for e, sgm in items}
            order = self._order([(id(sgm), sgm[1], sgm[2], e.width) for e, sgm in items])
            widths = sum(r[3] for r in order)
            g = lane_gap if widths + lane_gap * max(len(order) - 1, 0) <= x1 - x0 or len(order) < 2 else max(0.08, (x1 - x0 - widths) / (len(order) - 1))
            total = widths + g * max(len(order) - 1, 0)
            x = (x0 + x1) / 2 - total / 2
            for r in order:
                e, sgm = keyed[r[0]]
                sgm[3] = x + e.width / 2
                x += e.width + g

    def _side_ports(self, side, gap):
        """Right-side ports of every chip in the last column: its outgoing side links and its incoming side links share the
        side, ordered by the other end's row."""
        nodes = {id(e.a): e.a for e in side} | {id(e.b): e.b for e in side}
        for n in nodes.values():
            items = [(e, "ys", e.b) for e in side if e.a is n] + [(e, "yd", e.a) for e in side if e.b is n]
            items.sort(key=lambda it: it[2].y)
            widths = sum(it[0].width for it in items)
            room = n.h - 0.4
            g = gap if widths + gap * (len(items) - 1) <= room or len(items) == 1 else max(0.04, (room - widths) / (len(items) - 1))
            y = n.y - (widths + g * (len(items) - 1)) / 2
            for e, attr, _ in items:
                setattr(e, attr, y + e.width / 2)
                y += e.width + g

    def path(self, e):
        pts = [(e.a.right, e.ys)]
        for _, y_in, y_out, lane in e.segs:
            pts += [(lane, y_in), (lane, y_out)]
        pts.append((e.b.right if e.side else e.b.left, e.yd))
        clean = [pts[0]]                                     # drop zero-length steps
        for p in pts[1:]:
            if abs(p[0] - clean[-1][0]) > 1e-9 or abs(p[1] - clean[-1][1]) > 1e-9:
                clean.append(p)
        return clean

    def draw(self, chip_lw=0.6):
        self.route()
        for e in sorted(self.edges, key=lambda e: e.width):
            arrow(self.cv, self.path(e), e.width, e.colour, e.kind)
        for n in self.nodes.values():
            chip(self.cv, n, lw=chip_lw)


class Board:
    """A network with chips placed anywhere (modules in two dimensions, as MaleCNS Fig. 5F), its edges routed orthogonally on
    a grid by lib/router.py, strongest first: they leave and enter chips from any side (right out and left in preferred), never
    run along each other and cross at right angles only."""

    def __init__(self, cv, box, cell=0.6, margin=0.5, margin_in=1.1, obstacles=()):
        self.cv, self.box, self.cell, self.margin, self.margin_in = cv, box, cell, margin, margin_in
        self.nodes, self.edges, self.failed, self.obstacles = {}, [], [], list(obstacles)

    def add(self, key, label, x, y, **style) -> Node:
        n = Node(key, label, x, y, **style)
        self.nodes[key] = n
        return n

    def link(self, a, b, width, colour, kind="arrow"):
        self.edges.append(Edge(self.nodes[a], self.nodes[b], width, colour, kind))

    def draw(self, chip_lw=0.6):
        from . import router as rt
        g = rt.Grid(*self.box, self.cell)
        for n in self.nodes.values():
            g.block_box(n.left, n.top, n.right, n.bottom, self.margin)
        for l, t, r, b in self.obstacles:                  # e.g. module labels
            g.block_box(l, t, r, b, 0.2)
        for e in sorted(self.edges, key=lambda e: -e.width):
            src = (e.a.left, e.a.top, e.a.right, e.a.bottom)
            dst = (e.b.left, e.b.top, e.b.right, e.b.bottom)
            wide = max(1, int(np.ceil(e.width / self.cell - 0.15)))
            grow = self.margin_in - self.margin
            path = rt.route(g, (src[0] - grow, src[1] - grow, src[2] + grow, src[3] + grow),
                            (dst[0] - grow, dst[1] - grow, dst[2] + grow, dst[3] + grow), self.margin, wide)
            if path is None:
                self.failed.append(e)
                continue
            rt.claim(g, path, wide)
            hw, _ = head_size(e.width, e.kind)
            i, j, d = path[-1]
            half = int(np.ceil(hw / 2 / self.cell))
            for k in range(-half, half + 1):          # keep the head's breadth free of other arrivals
                ii, jj = (i, j + k) if d in (0, 2) else (i + k, j)
                if 0 <= ii < g.ni and 0 <= jj < g.nj:
                    g.occ[ii, jj] |= rt.H_BIT if d in (0, 2) else rt.V_BIT
            pts = rt.polyline(g, path, src, dst)
            arrow(self.cv, pts, e.width, e.colour, e.kind)
        for n in self.nodes.values():
            chip(self.cv, n, lw=chip_lw)
        return self.failed


# ---------------------------------------------------------------- headers, modules, keys

def column_header(cv, x0, x1, y, text, size=None):
    """Header text centred over a black rule from x0 to x1 (rule at y)."""
    cv.text((x0 + x1) / 2, y - 0.5, text, size=size or fs.PT, ha="center", va="bottom")
    cv.line([x0, x1], [y, y], RULE, lw=0.8, solid_capstyle="butt")


def module(cv, x, y, w, h, label=None, fill=MODULE_FILL, radius=1.4, size=None):
    """A light-grey rounded ground grouping chips, with an italic label at its top-left."""
    cv.fig.patches.append(FancyBboxPatch((x / cv.w, 1 - (y + h) / cv.h), w / cv.w, h / cv.h, boxstyle=f"round,pad=0,rounding_size={radius / cv.w}",
                                         transform=cv.fig.transFigure, facecolor=fill, edgecolor="none", linewidth=0.0, zorder=0,
                                         mutation_aspect=cv.w / cv.h))
    if label:
        cv.text(x + 1.4, y + 1.2, label, size=size or fs.PT, va="top", style="italic", color=fs.TEXT)


def dashed_rule(cv, x0, x1, y, above="Brain", below="VNC", dash=1.6, gap=1.0, thick=0.5, size=None):
    """The MaleCNS Brain/VNC separator: thick light-grey dashes with stacked labels at the right end."""
    x = x0
    while x < x1 - 0.2:
        _rect(cv, x, y - thick / 2, min(x + dash, x1), y + thick / 2, DASH, zorder=0.5)
        x += dash + gap
    if above:
        cv.text(x1, y - 0.9, above, size=size or fs.PT, ha="right", va="bottom")
    if below:
        cv.text(x1, y + 0.9, below, size=size or fs.PT, ha="right", va="top")


def width_key(cv, x, y, labels, widths=WIDTHS, colour=KEY_GREY, length=4.6, col_w=None, row_h=3.0):
    """The 2 x 2 key of grey arrows (thin and pale to thick and dark) with bin labels; returns its right end."""
    col_w = col_w or max(fs.text_width(s, fs.PT_SMALL) for s in labels) + length + 2.4
    for k, (lab, w) in enumerate(zip(labels, widths)):
        cx, cy = x + (k // 2) * col_w, y + (k % 2) * row_h
        arrow(cv, [(cx, cy), (cx + length, cy)], w, tint(colour, PALE[k]), "arrow", gap=0.0)
        cv.text(cx + length + 0.6, cy, lab, size=fs.PT_SMALL, va="center")
    return x + 2 * col_w


def transmitter_key(cv, x, y, which=("acetylcholine", "gaba", "glutamate"), gap=1.8):
    """Head glyphs in the transmitter colours with their names; returns the right end."""
    for t in which:
        kind = head_for(t)
        if kind == "arrow":
            head(cv, (x + 1.4, y), (1, 0), 0.6, TRANSMITTER[t], "arrow")
        else:
            _rect(cv, x, y - 0.3, x + 1.0, y + 0.3, TRANSMITTER[t])
            head(cv, (x + 1.4, y), (1, 0), 0.6, TRANSMITTER[t], "bar")
        cv.text(x + 1.9, y, TRANSMITTER_LABEL[t], size=fs.PT_SMALL, va="center")
        x += 1.9 + fs.text_width(TRANSMITTER_LABEL[t], fs.PT_SMALL) + gap
    return x - gap


def pathway(cv, x, y, labels, fills, links, size=None, h=3.2):
    """Chips joined by black arrows ('arrow') or bars ('bar'), left to right from x, centred on y; returns the right end."""
    size = size or fs.PT
    for i, (lab, fill) in enumerate(zip(labels, fills)):
        n = Node(lab, lab, 0, y, h=h, fill=fill, size=size)
        n.x = x + n.w / 2
        chip(cv, n, radius=0.45)
        x += n.w
        if i < len(labels) - 1:
            arrow(cv, [(x + 0.15, y), (x + 2.3, y)], 0.3, fs.TEXT, links[i], gap=0.12)
            x += 2.45
    return x


# ---------------------------------------------------------------- matrices and bars

def zoom(cv, src_ax, rect, dst_ax, colour=ZOOM):
    """Mark rect = (x0, y0, x1, y1) in src_ax data coordinates (red box) and join two of its corners to the matching corners of
    dst_ax with red dotted lines; dst_ax gets a red frame."""
    x0, y0, x1, y1 = rect
    src_ax.add_patch(Rectangle((min(x0, x1), min(y0, y1)), abs(x1 - x0), abs(y1 - y0), facecolor="none", edgecolor=colour, linewidth=0.8, zorder=5))
    inv = cv.fig.transFigure.inverted()
    bb = dst_ax.get_position()
    a = [inv.transform(src_ax.transData.transform(p)) for p in ((max(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)))]
    top_first = a[0][1] >= a[1][1]
    targets = [(bb.x0, bb.y1), (bb.x0, bb.y0)] if top_first else [(bb.x0, bb.y0), (bb.x0, bb.y1)]
    for (fx, fy), (tx, ty) in zip(a, targets):
        cv.fig.add_artist(Line2D([fx, tx], [fy, ty], transform=cv.fig.transFigure, color=colour, linewidth=0.7, linestyle=(0, (1.2, 1.4))))
    for sp in dst_ax.spines.values():
        sp.set_edgecolor(colour)
        sp.set_linewidth(0.8)
        sp.set_visible(True)


def l_axes(cv, x, y, length, xlabel, ylabel):
    """MaleCNS axis arrows at a matrix's bottom-left corner: down for sources, right for targets (labels beside them)."""
    arrow(cv, [(x, y - length), (x, y)], 0.25, fs.TEXT, "arrow", gap=0.0)
    arrow(cv, [(x, y), (x + length, y)], 0.25, fs.TEXT, "arrow", gap=0.0)
    cv.text(x + length + 0.8, y, xlabel, size=fs.PT_SMALL, va="center")
    cv.text(x - 0.8, y - length, ylabel, size=fs.PT_SMALL, va="top", ha="right", rotation=90)


def bar_counts(cv, ax, lefts, widths, y, counts, fills, min_frac=0.05, fmt="{:,}"):
    """Counts printed inside horizontal bar segments: white on dark fills (registered for the QC contrast check), black on pale."""
    from matplotlib.colors import to_rgba
    span = np.diff(ax.get_xlim())[0]
    for l0, w, c, f in zip(lefts, widths, counts, fills):
        if w / span < min_frac or not c:
            continue
        lum = 0.2126 * _lin(to_rgba(f)[0]) + 0.7152 * _lin(to_rgba(f)[1]) + 0.0722 * _lin(to_rgba(f)[2])
        white = (1.05) / (lum + 0.05) >= 4.5
        label = fmt.format(int(round(c))) if fmt == "{:,}" else fmt.format(c)
        t = ax.text(l0 + w / 2, y, label, ha="center", va="center", fontsize=fs.PT_SMALL, color="#FFFFFF" if white else fs.TEXT)
        if white:
            bb = t.get_window_extent(cv.fig.canvas.get_renderer())
            inv = cv.fig.transFigure.inverted()
            (fx0, fy0), (fx1, fy1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
            cv.fills.append(dict(rect=(fx0 * cv.w - 0.2, (1 - fy1) * cv.h - 0.2, (fx1 - fx0) * cv.w + 0.4, (fy1 - fy0) * cv.h + 0.4), fill=f))


def stacked_hbar(cv, ax, y, values, fills, height=0.66, left=0.0, min_frac=0.06, counts=True, fmt="{:,}", labels=None):
    """One horizontal stacked bar (MaleCNS Figs 4B, 7N): segments in fills, each value (or label) printed inside its segment
    when it fits (white on dark fills, registered for the QC contrast check; dark on pale). Returns the right end."""
    lefts, widths = [], []
    x = left
    for v, f in zip(values, fills):
        ax.barh(y, v, left=x, height=height, color=f, linewidth=0)
        lefts.append(x)
        widths.append(v)
        x += v
    if counts:
        cv.fig.canvas.draw()
        bar_counts(cv, ax, lefts, widths, y, labels if labels is not None else values, fills, min_frac=min_frac, fmt=fmt)
    return x


def rounded_path(points, radius, step=0.15):
    """Dense points along an orthogonal polyline whose corners are rounded with the given radius (clipped to half of the
    shorter adjacent segment)."""
    pts = [np.asarray(p, float) for p in points]
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        u, v = b - a, c - b
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        if lu < 1e-9 or lv < 1e-9:
            continue
        u, v = u / lu, v / lv
        r = min(radius, lu / 2, lv / 2)
        p0, p1 = b - u * r, b + v * r
        seg = np.linspace(out[-1], p0, max(2, int(np.linalg.norm(p0 - out[-1]) / step)))
        out.extend(seg[1:])
        t = np.linspace(0, 1, max(6, int(r * np.pi / 2 / step)))[:, None]
        arc = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * b + t ** 2 * p1          # quadratic corner (close to a circular arc)
        out.extend(arc[1:])
    seg = np.linspace(out[-1], pts[-1], max(2, int(np.linalg.norm(pts[-1] - out[-1]) / step)))
    out.extend(seg[1:])
    return np.array(out)


def offset_path(c, d):
    tan = np.gradient(c, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    return c + np.column_stack([-tan[:, 1], tan[:, 0]]) * d


def _lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
