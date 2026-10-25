"""Orthographic anatomy renderer for the dark image tiles (numba, CPU; no GPU, no network).

Scene units are micrometres in MaleCNS coordinates (+x fly left, +y ventral, +z posterior). A View maps them to
pixels: either the unfolded CNS (msa/cnsview.py: brain frontal, VNC ventral below the neck) or a rotated 3-D view.
Layers are accumulated in linear light at a supersampled resolution and resolved to sRGB at the end:

* shell: order-independent rim glow of the neuropil surfaces (front and back faces), like a glass brain;
* points: somata as discs, either alpha-composited far-to-near (depth order) or by value (largest on top), or
  max-projected (a maximum-intensity projection, as for a confocal stack);
* lines: skeleton segments, alpha-composited far-to-near.

Every tile writes a JSON sidecar with its view, scale and layer statistics.
"""
from __future__ import annotations

import json
import os
import math
import sys
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "manuscript" / "analysis"))
from msa import cnsview  # noqa: E402

RESULTS = Path(os.environ.get("FLYBRAIN_RESULTS", ROOT / "manuscript" / "results"))   # preview renders may read another results root
ANATOMY = ROOT / "sokoban" / "artifacts" / "anatomy"
MORPHOLOGY = ROOT / "sokoban" / "artifacts" / "morphology"


# ---------------------------------------------------------------- colour helpers

def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def hex_linear(h: str) -> np.ndarray:
    from matplotlib.colors import to_rgb
    return srgb_to_linear(np.array(to_rgb(h)))


# ---------------------------------------------------------------- views

def rotation(azimuth: float = 0.0, elevation: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Camera rotation (rows = screen right, screen down, depth) for a view that starts frontal (looking along +z,
    fly left on the right, ventral down), then turns about the vertical axis by azimuth degrees (positive: camera
    moves towards the fly's right side; -90 = lateral view from the left, anterior on the left) and tilts by
    elevation degrees (positive: camera moves ventral)."""
    a, e, r = (math.radians(v) for v in (azimuth, elevation, roll))
    # world -> frontal camera: identity (u = x, v = y, w = z)
    ry = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(e), math.sin(e)], [0, -math.sin(e), math.cos(e)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


def look(towards_camera, down) -> np.ndarray:
    """Camera rotation from the direction pointing from the scene to the camera and a screen-down hint (world
    coordinates; +x fly left, +y ventral, +z posterior). Rows: screen right, screen down, depth (right-handed)."""
    w = -np.asarray(towards_camera, np.float64)
    w /= np.linalg.norm(w)
    v = np.asarray(down, np.float64)
    v = v - w * (v @ w)
    v /= np.linalg.norm(v)
    u = np.cross(v, w)
    return np.vstack([u, v, w])


class View:
    """Orthographic camera. unfold=True applies the unfolded-CNS transform first (then R is usually identity)."""

    def __init__(self, width_px: int, height_px: int, um_per_px: float, centre, R=None, unfold: bool = False, ss: int = 2):
        self.width, self.height, self.ss = int(width_px), int(height_px), int(ss)
        self.um_per_px = float(um_per_px)
        self.centre = np.asarray(centre, np.float64)    # screen-space centre (camera coordinates, um)
        self.R = np.eye(3) if R is None else np.asarray(R, np.float64)
        self.unfold = bool(unfold)

    def camera(self, xyz, normals=None, side=None):
        p = np.asarray(xyz, np.float64).reshape(-1, 3)
        n = None if normals is None else np.asarray(normals, np.float64).reshape(-1, 3)
        if self.unfold:
            p, side = cnsview.unfold(p, side)
            if n is not None:
                n = cnsview.unfold_vectors(n, side)
        c = p @ self.R.T
        if n is not None:
            n = n @ self.R.T
        return c, n

    def pixels(self, xyz, normals=None, side=None):
        """Supersampled pixel coordinates (x right, y down), depth (um, larger = farther) and rotated normals."""
        c, n = self.camera(xyz, normals, side)
        s = self.ss / self.um_per_px
        px = (c[:, 0] - self.centre[0]) * s + self.width * self.ss / 2
        py = (c[:, 1] - self.centre[1]) * s + self.height * self.ss / 2
        return px, py, c[:, 2], n

    @property
    def shape(self):
        return self.height * self.ss, self.width * self.ss

    def describe(self) -> dict:
        return dict(width_px=self.width, height_px=self.height, supersample=self.ss, um_per_px=self.um_per_px,
                    centre_um=self.centre.tolist(), rotation=self.R.round(9).tolist(), unfold=self.unfold,
                    neck_z_um=cnsview.NECK_Z if self.unfold else None, neck_y_um=cnsview.NECK_Y if self.unfold else None)


def fit_view(xyz, width_px: int, height_px: int, R=None, unfold: bool = False, margin: float = 0.04, ss: int = 2,
             anchor: str = "centre") -> View:
    """A view that fits the camera-space bounding box of xyz into the tile with a relative margin."""
    tmp = View(width_px, height_px, 1.0, (0, 0, 0), R, unfold, ss)
    c, _ = tmp.camera(xyz)
    lo, hi = c.min(0), c.max(0)
    span = hi - lo
    um = max(span[0] / (width_px * (1 - 2 * margin)), span[1] / (height_px * (1 - 2 * margin)))
    centre = (lo + hi) / 2
    if anchor == "top":
        centre[1] = lo[1] - margin * height_px * um + height_px * um / 2
    return View(width_px, height_px, um, centre, R, unfold, ss)


# ---------------------------------------------------------------- assets

def load_regions():
    """List of (metadata dict, vertices um (n x 3), faces (m x 3)) for the 114 neuropil display meshes."""
    meta = json.loads((ANATOMY / "metadata.json").read_text(encoding="utf-8"))
    raw = (ANATOMY / "regions.bin").read_bytes()
    out = []
    for r in meta["regions"]:
        o = r["byte_offset"]
        nv, nf = (int(v) for v in np.frombuffer(raw, "<u4", 2, o))
        v = np.frombuffer(raw, "<f4", nv * 3, o + 8).reshape(-1, 3).astype(np.float64)
        f = np.frombuffer(raw, "<u4", nf * 3, o + 8 + nv * 12).reshape(-1, 3).astype(np.int64)
        out.append((r, v, f))
    return out


def vertex_normals(v, f):
    n = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(norm > 0, norm, 1)


def load_arbors():
    """(segment endpoints um (2m x 3) float32, per-arbor offsets, counts, body ids) of the cached arbor sample."""
    meta = json.loads((MORPHOLOGY / "metadata.json").read_text(encoding="utf-8"))
    pts = np.fromfile(MORPHOLOGY / "arbors.bin", "<f4").reshape(-1, 3)
    offs = np.array([a["offset"] if a else -1 for a in meta["arbors"]], np.int64)
    cnts = np.array([a["count"] if a else 0 for a in meta["arbors"]], np.int64)
    return pts, offs, cnts, np.asarray(meta["body_ids"], np.int64)


def load_skeleton(path: Path):
    """A neuroglancer precomputed skeleton (no vertex attributes): vertices in um, edges."""
    data = Path(path).read_bytes()
    nv, ne = (int(v) for v in np.frombuffer(data, "<u4", 2))
    if len(data) != 8 + 12 * nv + 8 * ne:
        raise ValueError(f"{path}: unexpected skeleton layout")
    v = np.frombuffer(data, "<f4", nv * 3, 8).reshape(-1, 3).astype(np.float64) / 1000.0
    e = np.frombuffer(data, "<u4", ne * 2, 8 + 12 * nv).reshape(-1, 2).astype(np.int64)
    return v, e


# ---------------------------------------------------------------- kernels

@njit(cache=True)
def _rim_accumulate(px, py, nz, faces, acc, gamma, fill):
    """Add rim glow (1 - |n_depth|)^gamma + fill for every covered pixel of every triangle (front and back)."""
    H, W = acc.shape
    for t in range(faces.shape[0]):
        i0, i1, i2 = faces[t, 0], faces[t, 1], faces[t, 2]
        x0, y0, x1, y1, x2, y2 = px[i0], py[i0], px[i1], py[i1], px[i2], py[i2]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        xmin = max(int(math.floor(min(x0, x1, x2))), 0)
        xmax = min(int(math.ceil(max(x0, x1, x2))), W - 1)
        ymin = max(int(math.floor(min(y0, y1, y2))), 0)
        ymax = min(int(math.ceil(max(y0, y1, y2))), H - 1)
        for y in range(ymin, ymax + 1):
            sy = y + 0.5
            for x in range(xmin, xmax + 1):
                sx = x + 0.5
                w0 = ((x1 - sx) * (y2 - sy) - (x2 - sx) * (y1 - sy)) / area
                w1 = ((x2 - sx) * (y0 - sy) - (x0 - sx) * (y2 - sy)) / area
                w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                n = abs(w0 * nz[i0] + w1 * nz[i1] + w2 * nz[i2])
                if n > 1.0:
                    n = 1.0
                acc[y, x] += (1.0 - n) ** gamma + fill


@njit(cache=True)
def _zbuffer(px, py, pz, faces, region, depth, rid):
    """Front-most surface per pixel: depth and region id (for crisp outlines)."""
    H, W = depth.shape
    for t in range(faces.shape[0]):
        i0, i1, i2 = faces[t, 0], faces[t, 1], faces[t, 2]
        x0, y0, x1, y1, x2, y2 = px[i0], py[i0], px[i1], py[i1], px[i2], py[i2]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        xmin = max(int(math.floor(min(x0, x1, x2))), 0)
        xmax = min(int(math.ceil(max(x0, x1, x2))), W - 1)
        ymin = max(int(math.floor(min(y0, y1, y2))), 0)
        ymax = min(int(math.ceil(max(y0, y1, y2))), H - 1)
        for y in range(ymin, ymax + 1):
            sy = y + 0.5
            for x in range(xmin, xmax + 1):
                sx = x + 0.5
                w0 = ((x1 - sx) * (y2 - sy) - (x2 - sx) * (y1 - sy)) / area
                w1 = ((x2 - sx) * (y0 - sy) - (x0 - sx) * (y2 - sy)) / area
                w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                z = w0 * pz[i0] + w1 * pz[i1] + w2 * pz[i2]
                if z < depth[y, x]:
                    depth[y, x] = z
                    rid[y, x] = region


@njit(cache=True)
def _discs_over(px, py, radius, rgb, alpha, order, img, cov):
    """Alpha-composite anti-aliased discs in the given order (later on top). img is linear RGB, cov the coverage."""
    H, W = cov.shape
    for k in range(order.shape[0]):
        i = order[k]
        r = radius[i]
        cx, cy = px[i], py[i]
        x0 = max(int(math.floor(cx - r - 1)), 0)
        x1 = min(int(math.ceil(cx + r + 1)), W - 1)
        y0 = max(int(math.floor(cy - r - 1)), 0)
        y1 = min(int(math.ceil(cy + r + 1)), H - 1)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                d = math.sqrt((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2)
                a = r + 0.5 - d
                if a <= 0:
                    continue
                if a > 1:
                    a = 1.0
                a *= alpha[i]
                for c in range(3):
                    img[y, x, c] = img[y, x, c] * (1 - a) + rgb[i, c] * a
                cov[y, x] = cov[y, x] * (1 - a) + a


@njit(cache=True)
def _discs_add(px, py, radius, rgb, alpha, img):
    """Add anti-aliased discs in linear light (order-independent; dark colours add nothing)."""
    H, W = img.shape[0], img.shape[1]
    for i in range(px.shape[0]):
        r = radius[i]
        cx, cy = px[i], py[i]
        x0 = max(int(math.floor(cx - r - 1)), 0)
        x1 = min(int(math.ceil(cx + r + 1)), W - 1)
        y0 = max(int(math.floor(cy - r - 1)), 0)
        y1 = min(int(math.ceil(cy + r + 1)), H - 1)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                d = math.sqrt((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2)
                a = r + 0.5 - d
                if a <= 0:
                    continue
                if a > 1:
                    a = 1.0
                a *= alpha[i]
                for c in range(3):
                    img[y, x, c] += rgb[i, c] * a


@njit(cache=True)
def _gauss_add(px, py, sigma, rgb, weight, img):
    """Add Gaussian splats (peak weight, standard deviation sigma pixels) in linear light."""
    H, W = img.shape[0], img.shape[1]
    for i in range(px.shape[0]):
        s = sigma[i]
        r = 3.0 * s
        cx, cy = px[i], py[i]
        x0 = max(int(math.floor(cx - r)), 0)
        x1 = min(int(math.ceil(cx + r)), W - 1)
        y0 = max(int(math.floor(cy - r)), 0)
        y1 = min(int(math.ceil(cy + r)), H - 1)
        inv = 1.0 / (2.0 * s * s)
        for y in range(y0, y1 + 1):
            dy = y + 0.5 - cy
            for x in range(x0, x1 + 1):
                dx = x + 0.5 - cx
                g = math.exp(-(dx * dx + dy * dy) * inv) * weight[i]
                if g < 1e-4:
                    continue
                for c in range(3):
                    img[y, x, c] += rgb[i, c] * g


@njit(cache=True)
def _discs_max(px, py, radius, value, out):
    """Maximum projection of disc-shaped footprints carrying value (soft edge)."""
    H, W = out.shape
    for i in range(px.shape[0]):
        r = radius[i]
        cx, cy = px[i], py[i]
        x0 = max(int(math.floor(cx - r - 1)), 0)
        x1 = min(int(math.ceil(cx + r + 1)), W - 1)
        y0 = max(int(math.floor(cy - r - 1)), 0)
        y1 = min(int(math.ceil(cy + r + 1)), H - 1)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                d = math.sqrt((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2)
                a = r + 0.5 - d
                if a <= 0:
                    continue
                v = value[i] * (a if a < 1 else 1.0)
                if v > out[y, x]:
                    out[y, x] = v


@njit(cache=True)
def _segments_over(x0s, y0s, x1s, y1s, width, rgb, alpha, order, img, cov):
    """Alpha-composite anti-aliased line segments (capsules) in the given order."""
    H, W = cov.shape
    for k in range(order.shape[0]):
        i = order[k]
        ax, ay, bx, by = x0s[i], y0s[i], x1s[i], y1s[i]
        hw = width[i] / 2
        xa = max(int(math.floor(min(ax, bx) - hw - 1)), 0)
        xb = min(int(math.ceil(max(ax, bx) + hw + 1)), W - 1)
        ya = max(int(math.floor(min(ay, by) - hw - 1)), 0)
        yb = min(int(math.ceil(max(ay, by) + hw + 1)), H - 1)
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        for y in range(ya, yb + 1):
            sy = y + 0.5
            for x in range(xa, xb + 1):
                sx = x + 0.5
                if L2 > 0:
                    t = ((sx - ax) * dx + (sy - ay) * dy) / L2
                    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
                else:
                    t = 0.0
                ex, ey = ax + t * dx - sx, ay + t * dy - sy
                d = math.sqrt(ex * ex + ey * ey)
                a = hw + 0.5 - d
                if a <= 0:
                    continue
                if a > 1:
                    a = 1.0
                a *= alpha[i]
                for c in range(3):
                    img[y, x, c] = img[y, x, c] * (1 - a) + rgb[i, c] * a
                cov[y, x] = cov[y, x] * (1 - a) + a


# ---------------------------------------------------------------- canvas

class Tile:
    """Linear-light layer stack for one image tile."""

    def __init__(self, view: View, background: str = "#000000"):
        self.view = view
        H, W = view.shape
        self.img = np.zeros((H, W, 3), np.float64)
        self.img[:] = hex_linear(background)
        self.log = dict(view=view.describe(), layers=[])

    # ---- neuropil shell
    def shell(self, regions, colour: str = "#9AA4AE", strength: float = 0.06, gamma: float = 3.0, fill: float = 0.02,
              select=None, outline: float = 0.0, outline_colour: str | None = None):
        """Rim glow of the neuropil surfaces, added in linear light; optional crisp outlines from a z-buffer."""
        H, W = self.view.shape
        acc = np.zeros((H, W), np.float64)
        depth = np.full((H, W), np.inf)
        rid = np.full((H, W), -1, np.int64)
        n_used = 0
        for k, (meta, v, f) in enumerate(regions):
            if select is not None and not select(meta):
                continue
            n = vertex_normals(v, f)
            px, py, pz, nn = self.view.pixels(v, n, side=meta["dataset"] == "vnc")
            _rim_accumulate(px, py, nn[:, 2].copy(), f, acc, gamma, fill)
            if outline > 0:
                _zbuffer(px, py, pz, f, k, depth, rid)
            n_used += 1
        glow = 1.0 - np.exp(-strength * acc)
        self.img += glow[..., None] * hex_linear(colour)[None, None, :]
        if outline > 0:
            edge = np.zeros((H, W), bool)
            edge[:, 1:] |= rid[:, 1:] != rid[:, :-1]
            edge[1:, :] |= rid[1:, :] != rid[:-1, :]
            self.img += (outline * edge)[..., None] * hex_linear(outline_colour or colour)[None, None, :]
        self.log["layers"].append(dict(kind="shell", regions=n_used, strength=strength, gamma=gamma, fill=fill, outline=outline,
                                       colour=colour, max_glow=float(glow.max())))

    def silhouette(self, regions, fill: str = "#EDEDED", line: str = "#A6A6A6", fill_alpha: float = 1.0, line_alpha: float = 1.0,
                   line_px: int = 2, select=None, highlight=None, highlight_fill: str = "#F3C7D3"):
        """Flat neuropil silhouette for light tiles (the MaleCNS figure style): the front-most region per pixel (z-buffer),
        filled and outlined by alpha-over compositing rather than added light, so it reads on a white ground. highlight(meta)
        selects regions filled in highlight_fill (e.g. a lesion site)."""
        from scipy.ndimage import binary_dilation

        H, W = self.view.shape
        depth = np.full((H, W), np.inf)
        rid = np.full((H, W), -1, np.int64)
        lit = []
        for k, (meta, v, f) in enumerate(regions):
            if select is not None and not select(meta):
                continue
            px, py, pz, _ = self.view.pixels(v, side=meta["dataset"] == "vnc")
            _zbuffer(px, py, pz, f, k, depth, rid)
            if highlight is not None and highlight(meta):
                lit.append(k)
        inside = rid >= 0
        col = np.empty((H, W, 3))
        col[:] = hex_linear(fill)
        if lit:
            col[np.isin(rid, lit)] = hex_linear(highlight_fill)
        a = (inside * fill_alpha)[..., None]
        self.img = self.img * (1 - a) + col * a
        edge = np.zeros((H, W), bool)
        edge[:, 1:] |= rid[:, 1:] != rid[:, :-1]
        edge[1:, :] |= rid[1:, :] != rid[:-1, :]
        if line_px > 1:
            edge = binary_dilation(edge, iterations=line_px - 1)
        b = (edge * line_alpha)[..., None]
        self.img = self.img * (1 - b) + hex_linear(line)[None, None, :] * b
        self.log["layers"].append(dict(kind="silhouette", regions=int(len(np.unique(rid[inside]))), highlighted=len(lit), fill=fill, line=line))

    def glass(self, regions, colour: str = "#8A9096", strength: float = 0.05, gamma: float = 2.2, fill: float = 0.015,
              opacity: float = 0.55, select=None):
        """Translucent light neuropils for a white ground (the left, anatomical half of the MaleCNS overview panels): rim
        shading accumulated through all surfaces, blended toward a grey by 1 - exp(-strength * rim), at most `opacity`."""
        H, W = self.view.shape
        acc = np.zeros((H, W), np.float64)
        for meta, v, f in regions:
            if select is not None and not select(meta):
                continue
            n = vertex_normals(v, f)
            px, py, _, nn = self.view.pixels(v, n, side=meta["dataset"] == "vnc")
            _rim_accumulate(px, py, nn[:, 2].copy(), f, acc, gamma, fill)
        a = (opacity * (1.0 - np.exp(-strength * acc)))[..., None]
        self.img = self.img * (1 - a) + hex_linear(colour)[None, None, :] * a
        self.log["layers"].append(dict(kind="glass", colour=colour, strength=strength, opacity=opacity))

    def flat(self, regions, base: str = "#E6E3E0", select=None, highlight=None, hi_fill: str = "#FFFFFF", hi_line: str = "#6E6E6E",
             line_px: int = 3):
        """Flat schematic silhouette (the right half of the MaleCNS overview panels): every region in one warm grey without
        inner lines; highlighted regions white with a dark outline."""
        from scipy.ndimage import binary_dilation

        H, W = self.view.shape
        depth = np.full((H, W), np.inf)
        rid = np.full((H, W), -1, np.int64)
        hi = set()
        for k, (meta, v, f) in enumerate(regions):
            if select is not None and not select(meta):
                continue
            px, py, pz, _ = self.view.pixels(v, side=meta["dataset"] == "vnc")
            _zbuffer(px, py, pz, f, k, depth, rid)
            if highlight is not None and highlight(meta):
                hi.add(k)
        inside = rid >= 0
        is_hi = np.isin(rid, list(hi)) if hi else np.zeros_like(inside)
        col = np.empty((H, W, 3))
        col[:] = hex_linear(base)
        col[is_hi] = hex_linear(hi_fill)
        self.img = np.where(inside[..., None], col, self.img)
        edge = np.zeros((H, W), bool)
        lab = np.where(is_hi, rid, -2)
        edge[:, 1:] |= (lab[:, 1:] != lab[:, :-1]) & (is_hi[:, 1:] | is_hi[:, :-1])
        edge[1:, :] |= (lab[1:, :] != lab[:-1, :]) & (is_hi[1:, :] | is_hi[:-1, :])
        if line_px > 1:
            edge = binary_dilation(edge, iterations=line_px - 1)
        self.img[edge] = hex_linear(hi_line)
        self.log["layers"].append(dict(kind="flat", base=base, highlighted=len(hi)))

    # ---- points
    def discs(self, xyz, radius_um, colours, alpha, order: str = "depth", values=None, label: str = ""):
        """Discs composited over the image; order 'depth' (far first), 'value' (ascending values on top), 'given', or
        'add' (added in linear light, order-independent)."""
        px, py, pz, _ = self.view.pixels(xyz)
        n = len(px)
        r = np.broadcast_to(np.asarray(radius_um, np.float64) * self.view.ss / self.view.um_per_px, (n,)).copy()
        rgb = _colour_array(colours, n)
        a = np.broadcast_to(np.asarray(alpha, np.float64), (n,)).copy()
        if order == "depth":
            idx = np.argsort(-pz, kind="stable")
        elif order == "value":
            idx = np.argsort(np.asarray(values, np.float64), kind="stable")
        else:
            idx = np.arange(n)
        if order == "add":
            _discs_add(px, py, r, rgb, a, self.img)
        else:
            cov = np.zeros(self.view.shape, np.float64)
            _discs_over(px, py, r, rgb, a, idx.astype(np.int64), self.img, cov)
        self.log["layers"].append(dict(kind="discs", label=label, n=int(n), order=order, radius_um=float(np.median(r) * self.view.um_per_px / self.view.ss)))

    def glow(self, xyz, sigma_um, colours, weight, label: str = ""):
        """Luminous Gaussian splats added in linear light (cells as soft points of light)."""
        px, py, _, _ = self.view.pixels(xyz)
        n = len(px)
        sig = np.broadcast_to(np.asarray(sigma_um, np.float64) * self.view.ss / self.view.um_per_px, (n,)).copy()
        sig = np.maximum(sig, 0.6 * self.view.ss)
        w = np.broadcast_to(np.asarray(weight, np.float64), (n,)).copy()
        _gauss_add(px, py, sig, _colour_array(colours, n), w, self.img)
        self.log["layers"].append(dict(kind="glow", label=label, n=int(n), sigma_um=float(np.median(sig) * self.view.um_per_px / self.view.ss)))

    def max_projection(self, xyz, radius_um, values, cmap, vmin, vmax, log: bool = False, gain: float = 1.0, label: str = ""):
        """Maximum-intensity projection of cell values mapped through a colour map (black = no signal)."""
        px, py, _, _ = self.view.pixels(xyz)
        v = np.asarray(values, np.float64)
        if log:
            t = (np.log10(np.clip(v, vmin, vmax)) - math.log10(vmin)) / (math.log10(vmax) - math.log10(vmin))
        else:
            t = (np.clip(v, vmin, vmax) - vmin) / (vmax - vmin)
        t = np.where(v > vmin, t, 0.0)
        out = np.zeros(self.view.shape, np.float64)
        r = np.full(len(px), radius_um * self.view.ss / self.view.um_per_px)
        _discs_max(px, py, r, t * 0.999 + 1e-6 * (t > 0), out)
        rgb = srgb_to_linear(cmap(np.clip(out, 0, 1))[..., :3]) * gain
        mask = out > 0
        self.img[mask] = rgb[mask]
        self.log["layers"].append(dict(kind="max_projection", label=label, n=int(len(px)), vmin=vmin, vmax=vmax, log=log))

    # ---- lines
    def segments(self, a_xyz, b_xyz, width_um, colours, alpha, order: str = "depth", depth_cue: float = 0.0, label: str = ""):
        pa = self.view.pixels(a_xyz)
        pb = self.view.pixels(b_xyz)
        n = len(pa[0])
        w = np.broadcast_to(np.asarray(width_um, np.float64) * self.view.ss / self.view.um_per_px, (n,)).copy()
        w = np.maximum(w, 0.9 * self.view.ss)          # never thinner than ~1 output pixel
        rgb = _colour_array(colours, n).copy()
        z = (pa[2] + pb[2]) / 2
        if depth_cue > 0:                              # farther segments dimmer, towards the background
            zt = (z - z.min()) / max(np.ptp(z), 1e-9)
            rgb *= (1 - depth_cue * zt)[:, None]
        a = np.broadcast_to(np.asarray(alpha, np.float64), (n,)).copy()
        idx = np.argsort(-z, kind="stable") if order == "depth" else np.arange(n)
        cov = np.zeros(self.view.shape, np.float64)
        _segments_over(pa[0], pa[1], pb[0], pb[1], w, rgb, a, idx.astype(np.int64), self.img, cov)
        self.log["layers"].append(dict(kind="segments", label=label, n=int(n), width_um=float(np.median(w) * self.view.um_per_px / self.view.ss)))

    # ---- output
    def resolve(self, exposure: float = 1.0, bloom: tuple | None = None) -> np.ndarray:
        """Downsample (box filter over the supersample), add bloom (sigma in output pixels, strength, threshold on
        luminance), keep hue where a colour exceeds the display range, and encode to 8-bit sRGB."""
        ss = self.view.ss
        H, W = self.view.height, self.view.width
        img = self.img.reshape(H, ss, W, ss, 3).mean(axis=(1, 3)) * exposure
        if bloom:
            from scipy.ndimage import gaussian_filter
            sigma, strength, threshold = bloom
            lum0 = img @ np.array([0.2126, 0.7152, 0.0722])
            bright = img * np.clip((lum0 - threshold) / max(1 - threshold, 1e-6), 0, None)[..., None]
            img = img + strength * gaussian_filter(bright, sigma=(sigma, sigma, 0), mode="constant")
            self.log["bloom"] = dict(sigma_px=sigma, strength=strength, threshold=threshold)
        lum = img @ np.array([0.2126, 0.7152, 0.0722])
        over = lum > 1.0                               # keep hue where a colour exceeds the display range
        img[over] /= lum[over][:, None]
        return (linear_to_srgb(img) * 255 + 0.5).astype(np.uint8)

    def save(self, path: Path, rgb8: np.ndarray, extra: dict | None = None) -> None:
        from PIL import Image
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb8).save(path, optimize=False)
        meta = {**self.log, **(extra or {})}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8", newline="\n")


def _colour_array(colours, n):
    if isinstance(colours, str):
        return np.broadcast_to(hex_linear(colours), (n, 3)).copy()
    c = np.asarray(colours, np.float64)
    if c.ndim == 1:
        return np.broadcast_to(c, (n, 3)).copy()
    return c


def scale_bar_px(view: View, length_um: float) -> float:
    """Length in output pixels of a scale bar of length_um."""
    return length_um / view.um_per_px
