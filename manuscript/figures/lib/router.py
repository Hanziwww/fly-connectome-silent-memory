"""Orthogonal edge routing on a grid for network diagrams with chips placed anywhere (the MaleCNS Fig. 5F layout: modules in
two dimensions, edges entering chips from any side).

Each edge is routed by A* over (cell, heading) states: straight steps cost 1, a bend costs BEND, crossing an earlier edge at
right angles costs CROSS, running along an earlier edge (or turning on it) is forbidden, and chips (with a margin) are
obstacles. Edges are routed in the given order (strongest first), each claiming its cells, so later edges find parallel tracks
beside earlier ones; running along an earlier edge or turning on it is not forbidden but costs OVERLAP or BEND_ON, so a
route always exists in a crowded layout and such overlaps stay rare. A path leaves its sender from a side cell and arrives
perpendicular to a side of its receiver; the right side out and the left side in are preferred (SIDE costs), so the reading
direction stays left to right.
"""
import heapq

import numpy as np

H_BIT, V_BIT = 1, 2
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))          # right, down, left, up (grid i to the right, j downward)
BEND, CROSS = 3.0, 1.5
OVERLAP, BEND_ON = 60.0, 25.0                   # soft limits: running along or turning on another edge
SIDE_OUT = {0: 0.0, 1: 4.0, 3: 4.0, 2: 12.0}       # leaving from the right (heading right) is free; top/bottom cost; left more
SIDE_IN = {0: 0.0, 1: 4.0, 3: 4.0, 2: 12.0}        # arriving heading right (into the left side) is free


class Grid:
    def __init__(self, x0, y0, x1, y1, cell):
        self.x0, self.y0, self.cell = x0, y0, cell
        self.ni, self.nj = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
        self.blocked = np.zeros((self.ni, self.nj), bool)
        self.occ = np.zeros((self.ni, self.nj), np.uint8)

    def ij(self, x, y):
        return int((x - self.x0) / self.cell), int((y - self.y0) / self.cell)

    def xy(self, i, j):
        return self.x0 + (i + 0.5) * self.cell, self.y0 + (j + 0.5) * self.cell

    def block_box(self, l, t, r, b, margin):
        i0, j0 = self.ij(l - margin, t - margin)
        i1, j1 = self.ij(r + margin, b + margin)
        self.blocked[max(i0, 0):min(i1 + 1, self.ni), max(j0, 0):min(j1 + 1, self.nj)] = True

    def box_cells(self, l, t, r, b, margin):
        i0, j0 = self.ij(l - margin, t - margin)
        i1, j1 = self.ij(r + margin, b + margin)
        return max(i0, 0), max(j0, 0), min(i1, self.ni - 1), min(j1, self.nj - 1)


def _ports(grid, box, margin, table, outward):
    """Side cells just outside a chip box: (i, j, heading, cost). outward: heading away from the chip (sender) or into it."""
    i0, j0, i1, j1 = grid.box_cells(*box, margin)
    out = []
    for j in range(j0 + 1, j1):                    # right side (heading right when leaving, left when arriving)
        if i1 + 1 < grid.ni:
            out.append((i1 + 1, j, 0 if outward else 2, table[0 if outward else 2]))
        if i0 - 1 >= 0:
            out.append((i0 - 1, j, 2 if outward else 0, table[2 if outward else 0]))
    for i in range(i0 + 1, i1):
        if j1 + 1 < grid.nj:
            out.append((i, j1 + 1, 1 if outward else 3, table[1 if outward else 3]))
        if j0 - 1 >= 0:
            out.append((i, j0 - 1, 3 if outward else 1, table[3 if outward else 1]))
    return out


def _bit(d):
    return H_BIT if d in (0, 2) else V_BIT


def route(grid, src_box, dst_box, margin, cells_wide=1):
    """A* from the sender's side cells to a side cell of the receiver, arriving perpendicular to that side. Returns the cell
    path [(i, j, heading)] or None."""
    starts = [(i, j, d, c + (OVERLAP if grid.occ[i, j] else 0.0)) for i, j, d, c in _ports(grid, src_box, margin, SIDE_OUT, True)
              if not grid.blocked[i, j]]
    goals = {}
    for i, j, d, c in _ports(grid, dst_box, margin, SIDE_IN, False):
        if not grid.blocked[i, j]:
            goals[(i, j, d)] = c + (OVERLAP if grid.occ[i, j] else 0.0)
    if not starts or not goals:
        return None
    gi = np.array([k[0] for k in goals])
    gj = np.array([k[1] for k in goals])
    tgt = (int(np.median(gi)), int(np.median(gj)))

    def h(i, j):
        return abs(i - tgt[0]) + abs(j - tgt[1])

    best, prev, heap = {}, {}, []
    for i, j, d, c in starts:
        s = (i, j, d)
        best[s] = c
        heapq.heappush(heap, (c + h(i, j), c, s))
    while heap:
        f, g, s = heapq.heappop(heap)
        if g > best.get(s, np.inf) + 1e-9:
            continue
        i, j, d = s
        if s in goals:
            path = [s]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            return path[::-1]
        for nd in (d, (d + 1) % 4, (d + 3) % 4):
            di, dj = DIRS[nd]
            ni, nj = i + di, j + dj
            if not (0 <= ni < grid.ni and 0 <= nj < grid.nj) or grid.blocked[ni, nj]:
                continue
            o = grid.occ[ni, nj]
            cost = 1.0 + (BEND if nd != d else 0.0)
            if nd != d and grid.occ[i, j]:
                cost += BEND_ON                         # turning on another edge
            if o & _bit(nd):
                cost += OVERLAP                         # running along another edge
            elif o:
                cost += CROSS
            ns = (ni, nj, nd)
            ng = g + cost
            if ns in goals:
                ng += goals[ns]
            if ng < best.get(ns, np.inf) - 1e-9:
                best[ns] = ng
                prev[ns] = s
                heapq.heappush(heap, (ng + h(ni, nj), ng, ns))
    return None


def claim(grid, path, cells_wide=1):
    """Mark a routed path's cells (and, for wide edges, their neighbours across the heading) as used in its orientation."""
    for i, j, d in path:
        bit = _bit(d)
        grid.occ[i, j] |= bit
        for k in range(1, cells_wide):
            for s in (-1, 1):
                ii, jj = (i, j + s * k) if d in (0, 2) else (i + s * k, j)
                if 0 <= ii < grid.ni and 0 <= jj < grid.nj:
                    grid.occ[ii, jj] |= bit
    for (i, j, d), (i2, j2, d2) in zip(path[:-1], path[1:]):      # a corner cell is used both ways
        if d != d2:
            grid.occ[i, j] |= H_BIT | V_BIT


def polyline(grid, path, src_box, dst_box):
    """Corner points in mm, from the sender's border to the receiver's border."""
    pts = []
    for k, (i, j, d) in enumerate(path):
        if k == 0 or k == len(path) - 1 or d != path[k + 1][2]:
            pts.append(grid.xy(i, j))
    i, j, d = path[0]
    x, y = grid.xy(i, j)
    l, t, r, b = src_box
    first = {0: (r, y), 2: (l, y), 1: (x, b), 3: (x, t)}[d]
    i, j, d = path[-1]
    x, y = grid.xy(i, j)
    l, t, r, b = dst_box
    last = {0: (l, y), 2: (r, y), 1: (x, t), 3: (x, b)}[d]
    pts = [first] + pts + [last]
    clean = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - clean[-1][0]) > 1e-9 or abs(p[1] - clean[-1][1]) > 1e-9:
            clean.append(p)
    out = [clean[0]]                                # merge collinear points
    for k in range(1, len(clean) - 1):
        a, b_, c = out[-1], clean[k], clean[k + 1]
        if (abs(a[0] - b_[0]) < 1e-9 and abs(b_[0] - c[0]) < 1e-9) or (abs(a[1] - b_[1]) < 1e-9 and abs(b_[1] - c[1]) < 1e-9):
            continue
        out.append(b_)
    out.append(clean[-1])
    return out
