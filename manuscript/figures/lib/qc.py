"""Figure checks, ported from D:/ulife/tools/paper_figstyle.py (check, simulations, palette distances, source data)
and D:/coch-rythum/scripts/906_figure_qc.py (text geometry, coloured text), adapted to this paper:

* size 180 +/- 0.1 mm wide, at most 170 mm tall; Arial 5-7 pt, panel letters 8 pt bold and in order;
* text overlaps above 0.3 mm, text off the canvas, coloured text (chroma > 0.25), internal identifiers in text, and under
  the Cell profile (v4) development labels such as M0, S1, X1, D7 or "frozen" (reader-facing names only);
* lines 0.25-1 pt; raster tiles at >= 600 ppi (main) or >= 300 ppi (Extended Data);
* white text only inside dark image tiles, with contrast >= 4.5:1 against the tile pixels beneath it;
* image tiles of one group separated by 0.6 +/- 0.05 mm gutters;
* palettes: CIE76 distance >= 10 normally, under protan / deutan / tritan simulation (Machado et al. 2009) and in
  greyscale; colour maps monotonic in CIE L*.
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import numpy as np

from . import figstyle as fs
from .names import BRITISH, BRITISH_RE, DEV_LABELS, INTERNAL, british  # noqa: F401  (shared with the manuscript text check)


TOL_MM = 0.3


def _texts(fig):
    """Texts that are drawn: visible, non-empty, and tick labels only inside the view interval."""
    from matplotlib.text import Text

    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            eps = 1e-9 * max(1.0, abs(hi - lo))
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if not (lo - eps <= tick.get_loc() <= hi + eps):
                    hidden.update({id(tick.label1), id(tick.label2)})
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and id(t) not in hidden]


def _chroma(rgb):
    return max(rgb[:3]) - min(rgb[:3])


def _luminance(rgb):
    lin = np.where(np.asarray(rgb, float) <= 0.04045, np.asarray(rgb, float) / 12.92, ((np.asarray(rgb, float) + 0.055) / 1.055) ** 2.4)
    return lin[..., :3] @ np.array([0.2126, 0.7152, 0.0722])


def _contrast(l1, l2):
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def check(cv: "fs.Canvas", allow_overlap: tuple = (), width: float | None = None, allow_names: tuple = ()) -> list[str]:
    """Every violation in a finished figure (empty when it complies). allow_names: published cell-type names that contain an
    underscore (e.g. aIPg_m4), exempt from the internal-identifier check; the caller verifies them against the cell table.
    Width, height, text sizes and the panel-letter case follow the active style profile (fs.use_cell for v4)."""
    from matplotlib.collections import Collection
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    width = fs.FULL if width is None else width
    fig, issues = cv.fig, []
    w, h = fig.get_size_inches() / fs.MM
    if abs(w - width) > 0.1:
        issues.append(f"width {w:.2f} mm (expected {width})")
    if h > fs.MAX_H + 0.05:
        issues.append(f"height {h:.1f} mm over {fs.MAX_H}")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_mm = fig.dpi * fs.MM
    texts = _texts(fig)
    tile_rects = [(t, t["rect"]) for t in cv.tiles]

    def in_tile(bb):
        x0, x1 = bb.x0 / px_per_mm, bb.x1 / px_per_mm
        y0, y1 = h - bb.y1 / px_per_mm, h - bb.y0 / px_per_mm
        for t, (l, tp, tw, th) in tile_rects:
            if x0 >= l - 0.05 and x1 <= l + tw + 0.05 and y0 >= tp - 0.05 and y1 <= tp + th + 0.05:
                return t, (x0, y0, x1, y1)
        return None, (x0, y0, x1, y1)

    def in_fill(x0, y0, x1, y1):
        for f in getattr(cv, "fills", []):
            l, tp, fw, fh = f["rect"]
            if x0 >= l - 0.05 and x1 <= l + fw + 0.05 and y0 >= tp - 0.05 and y1 <= tp + fh + 0.05:
                return f["fill"]
        return None

    for t in texts:
        s = t.get_text()
        size = t.get_fontsize()
        is_letter = str(t.get_gid()).startswith("letter_")
        if is_letter:
            if abs(size - fs.PT_LETTER) > 1e-6 or t.get_fontweight() not in ("bold", 700):
                issues.append(f"panel letter {s!r} not {fs.PT_LETTER:g} pt bold")
        elif size < fs.PT_MIN - 1e-6 or size > fs.PT_MAX + 1e-6:
            issues.append(f"font size {size} for {s[:30]!r}")
        if t.get_fontname() != fs.FONT:
            issues.append(f"font {t.get_fontname()} for {s[:30]!r}")
        plain = re.sub(r"\$[^$]*\$", "", s)                     # mathtext (subscripts, italic symbols) is exempt
        for name in allow_names:
            plain = plain.replace(name, "")
        if INTERNAL.search(plain):
            issues.append(f"internal identifier in {s[:40]!r}")
        if fs.PROFILE == "cell" and DEV_LABELS.search(plain):
            issues.append(f"development label {DEV_LABELS.search(plain).group(0)!r} in {s[:40]!r}")
        if fs.PROFILE == "cell":
            for word, us in british(plain):
                issues.append(f"British spelling {word!r} in {s[:40]!r} (American: {us})")
        rgba = to_rgba(t.get_color())
        if _chroma(rgba) > 0.25:
            issues.append(f"coloured text {s[:30]!r}")
        lum = _luminance(np.array(rgba[:3]))
        if lum > 0.5:                                 # light text: only inside dark tiles, with contrast
            tile, (x0, y0, x1, y1) = in_tile(t.get_window_extent(renderer))
            fill = in_fill(x0, y0, x1, y1)
            if fill is not None:                      # a registered filled box (chip, bar segment): contrast with its fill
                ratio = _contrast(lum, _luminance(np.array(to_rgba(fill)[:3])))
                if ratio < 4.5:
                    issues.append(f"contrast {ratio:.1f}:1 for {s[:30]!r} on its fill")
            elif tile is None or not tile["dark"]:
                issues.append(f"light text outside a dark tile: {s[:30]!r}")
            else:
                ratio = _tile_contrast(tile, (x0, y0, x1, y1), lum)
                if ratio < 4.5:
                    issues.append(f"contrast {ratio:.1f}:1 for {s[:30]!r} on its tile")
    for ln in fig.findobj(Line2D):
        if not ln.get_visible() or ln.get_linestyle() in ("None", "none", "", " "):
            continue
        lw = ln.get_linewidth()
        if lw > 0 and (lw < fs.LW_HAIR - 1e-6 or lw > 1.0 + 1e-6):
            issues.append(f"line {lw} pt ({ln.get_label()})")
    for c in fig.findobj(Collection):
        if not c.get_visible() or not len(c.get_paths()):
            continue
        lws = np.atleast_1d(c.get_linewidths())
        ec = np.asarray(c.get_edgecolors())
        has_edge = ec.ndim == 2 and len(ec) and np.any(ec[:, 3] > 0)
        if has_edge and np.any((lws > 0) & ((lws < fs.LW_HAIR - 1e-6) | (lws > 1.0 + 1e-6))):
            issues.append(f"collection line {lws.min()}-{lws.max()} pt")
    for p in fig.findobj(Patch):
        if not p.get_visible() or p is fig.patch:
            continue
        ec = p.get_edgecolor()
        if ec is not None and len(ec) == 4 and ec[3] > 0 and p.get_linewidth() > 0 and \
                (p.get_linewidth() < fs.LW_HAIR - 1e-6 or p.get_linewidth() > 1.0 + 1e-6):
            issues.append(f"patch edge {p.get_linewidth()} pt")
    fb = fig.bbox
    tol = TOL_MM * px_per_mm
    boxes = []
    for t in texts:
        bb = t.get_window_extent(renderer)
        if bb.x0 < fb.x0 - tol or bb.x1 > fb.x1 + tol or bb.y0 < fb.y0 - tol or bb.y1 > fb.y1 + tol:
            issues.append(f"outside the canvas: {t.get_text()[:30]!r}")
        if t.get_gid() in allow_overlap or bb.width <= 0 or bb.height <= 0:
            continue
        if abs(t.get_rotation() % 90) > 1e-6:
            continue                                  # oblique labels: axis-aligned boxes overstate overlap
        boxes.append((t, bb))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ow = min(a.x1, b.x1) - max(a.x0, b.x0)
            oh = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ow > tol and oh > tol:
                issues.append(f"overlap: {boxes[i][0].get_text()[:25]!r} / {boxes[j][0].get_text()[:25]!r}")
    if cv.letters != [chr(ord("A" if fs.UPPER else "a") + i) for i in range(len(cv.letters))]:
        issues.append(f"panel letters out of order: {cv.letters}")
    for t in cv.tiles:
        l, tp, tw, th = t["rect"]
        ppi = t["pixels"][0] / (tw * fs.MM)
        if ppi < cv.ppi_min - 0.5:
            issues.append(f"tile {t['label'] or t['group']} at {ppi:.0f} ppi (< {cv.ppi_min})")
    issues += gutters(cv)
    return issues


def _tile_contrast(tile, box_mm, text_lum):
    """Contrast between light text and the brightest 5% of tile pixels under the text box (worst case)."""
    l, tp, tw, th = tile["rect"]
    w_px, h_px = tile["pixels"]
    img = tile["ax"].get_images()[0].get_array()
    x0, y0, x1, y1 = box_mm
    c0, c1 = int((x0 - l) / tw * w_px), int(np.ceil((x1 - l) / tw * w_px))
    r0, r1 = int((y0 - tp) / th * h_px), int(np.ceil((y1 - tp) / th * h_px))
    patch = np.asarray(img[max(r0, 0):max(r1, 1), max(c0, 0):max(c1, 1)], float)
    if patch.size == 0:
        return 21.0
    if patch.max() > 1.0:
        patch = patch / 255.0
    lum = _luminance(patch[..., :3]).ravel()
    bg = float(np.quantile(lum, 0.95))
    return _contrast(text_lum, bg)


def gutters(cv) -> list[str]:
    """Adjacent tiles of one group must be separated by GUTTER +/- 0.05 mm (horizontal and vertical neighbours)."""
    issues = []
    by_group = {}
    for t in cv.tiles:
        if t["group"]:
            by_group.setdefault(t["group"], []).append(t["rect"])
    for g, rects in by_group.items():
        for i, (l1, t1, w1, h1) in enumerate(rects):
            for l2, t2, w2, h2 in rects[i + 1:]:
                v_overlap = min(t1 + h1, t2 + h2) - max(t1, t2)
                h_overlap = min(l1 + w1, l2 + w2) - max(l1, l2)
                if v_overlap > 0.5:                   # side by side
                    gap = max(l2 - (l1 + w1), l1 - (l2 + w2))
                    if gap < 3.0 and abs(gap - fs.GUTTER) > 0.05:
                        issues.append(f"tile gutter {gap:.2f} mm in group {g}")
                if h_overlap > 0.5:                   # stacked
                    gap = max(t2 - (t1 + h1), t1 - (t2 + h2))
                    if gap < 3.0 and abs(gap - fs.GUTTER) > 0.05:
                        issues.append(f"tile gutter {gap:.2f} mm in group {g}")
    return issues


# ---- colour-blind and greyscale simulation (Machado, Oliveira & Fernandes 2009, severity 1.0)

MACHADO = {
    "protanopia": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}
VISIONS = ("normal", "protanopia", "deuteranopia", "tritanopia", "greyscale")


def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate(rgb: np.ndarray, kind: str) -> np.ndarray:
    if kind == "normal":
        return np.asarray(rgb, float)
    if kind == "greyscale":
        y = _lin(rgb) @ np.array([0.2126, 0.7152, 0.0722])
        return np.repeat(_srgb(y)[..., None], 3, axis=-1)
    return _srgb(_lin(rgb) @ MACHADO[kind].T)


def lab(rgb: np.ndarray) -> np.ndarray:
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = _lin(rgb) @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def simulate_png(png: Path, out_dir: Path) -> list[Path]:
    from PIL import Image

    img = np.asarray(Image.open(png).convert("RGB"), float) / 255
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for kind in VISIONS[1:]:
        f = out_dir / f"{png.stem}_{kind}.png"
        Image.fromarray((simulate(img, kind) * 255 + 0.5).astype(np.uint8)).save(f)
        files.append(f)
    return files


def distinguishable(colours: dict, min_delta: float = 10.0, background: str | None = None) -> dict:
    """Smallest CIE76 difference between any two colours (name -> hex), normally and under each simulation; pairs
    below min_delta. With a background, each colour's distance to it is checked as well."""
    from matplotlib.colors import to_rgb

    names = list(colours)
    rgb = np.array([to_rgb(colours[n]) for n in names])
    bg = np.array(to_rgb(background)) if background else None
    out = {}
    for kind in VISIONS:
        L = lab(simulate(rgb, kind))
        pairs = sorted((float(np.linalg.norm(L[i] - L[j])), names[i], names[j])
                       for i in range(len(names)) for j in range(i + 1, len(names)))
        below = [(a, b, round(d, 1)) for d, a, b in pairs if d < min_delta]
        if bg is not None:
            Lb = lab(simulate(bg[None], kind))[0]
            below += [(n, "background", round(float(np.linalg.norm(L[i] - Lb)), 1)) for i, n in enumerate(names)
                      if np.linalg.norm(L[i] - Lb) < min_delta]
        out[kind] = {"min": pairs[0][0] if pairs else float("nan"), "below": below}
    return out


def lightness_monotonic(cmap, n: int = 256, tol: float = 0.0) -> tuple[bool, str]:
    """Whether CIE L* rises (or falls) monotonically along a sequential map, or falls then rises symmetrically
    (dark-centred diverging) / rises then falls (light-centred diverging)."""
    rgb = cmap(np.linspace(0, 1, n))[:, :3]
    L = lab(rgb)[:, 0]
    d = np.diff(L)
    if np.all(d >= -tol) or np.all(d <= tol):
        return True, "monotonic"
    mid = n // 2
    left, right = np.diff(L[:mid]), np.diff(L[mid:])
    if (np.all(left <= tol) and np.all(right >= -tol)) or (np.all(left >= -tol) and np.all(right <= tol)):
        return True, "diverging (two monotonic halves)"
    return False, f"L* reverses ({int((d > tol).sum())} rises, {int((d < -tol).sum())} falls)"


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in keys})
    return path


class SourceData:
    """The numbers a figure plots, panel by panel, with the hashes of the tables they came from."""

    def __init__(self, figure: str):
        self.figure = figure
        self.rows: list[dict] = []
        self.inputs: dict[str, str] = {}

    def add(self, panel: str, series: str, **values) -> None:
        self.rows.append({"panel": panel, "series": series, **values})

    def source(self, *paths: Path) -> None:
        for p in paths:
            p = Path(p)
            key = p.as_posix()
            if p.exists() and key not in self.inputs:
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                self.inputs[key] = h.hexdigest()

    def write(self, out_dir: Path) -> Path:
        return write_rows(Path(out_dir) / f"{self.figure}.csv", self.rows)


def _cell(x):
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "true" if x else "false"
    if isinstance(x, float):
        return repr(x) if np.isfinite(x) else ""
    if isinstance(x, (list, tuple)):
        return ";".join(str(_cell(v)) for v in x)
    return x
