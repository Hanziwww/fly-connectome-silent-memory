"""Figure canvas at Nature specifications, ported from D:/coch-rythum/scripts/lib/figstyle.py (the user's house style).

180 mm wide, at most 170 mm tall; Arial 5-7 pt (6 pt default, 5 pt ticks), panel letters 8 pt bold lowercase; lines
0.25-1 pt; axes placed in millimetres from the top-left corner. Text is black outside image tiles; colour lives in
glyphs and keys. Image tiles (anatomy renders, black background) are placed on the 600-ppi pixel grid, and white
labels with a dark halo are allowed only inside them.

Export: PDF (TrueType fonts, no creation date), SVG (text as text, LF), 600-ppi JPG, and a PNG preview for QC.
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.textpath import TextToPath  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402

MM = 1 / 25.4
FULL, MAX_H = 180.0, 170.0
COL, GAP = 28.5, 1.8                                   # six-column grid: spans 28.5, 58.8, 89.1, 119.4, 149.7, 180 mm
FONT = "Arial"
PT, PT_SMALL, PT_LETTER = 6.0, 5.0, 8.0
LW, LW_THIN, LW_HAIR = 0.5, 0.35, 0.25
TEXT, TEXT2, AXIS, GRID = "#1A1A1A", "#4D4D4D", "#333333", "#D9D9D9"
ON_DARK, TILE_BG = "#FFFFFF", "#000000"
DPI = 600
GUTTER = 0.6                                           # mm between image tiles
P_SYM, Q_SYM, R_SYM = r"$\mathit{P}$", r"$\mathit{q}$", r"$\mathit{r}$"
RHO = "ρ"
MINUS = "−"


PT_MIN, PT_MAX = 5.0, 7.0                              # allowed text sizes (qc.check)
UPPER = False                                          # panel letters in upper case (Cell)
PROFILE = "nature"


def use_cell() -> None:
    """Cell Press profile for the v4 figures: 174 mm full width, at most 225 mm tall; Arial 6.5 pt labels, 6 pt ticks and
    keys (nothing below 6 pt), 9-pt bold upper-case panel letters; a grid of four units (unit, unit_x). Rebinds the
    size defaults of the text helpers, so every caller gets Cell sizes. The frozen v1 builds never call it."""
    global FULL, MAX_H, PT, PT_SMALL, PT_LETTER, PT_MIN, PT_MAX, UPPER, PROFILE
    FULL, MAX_H, PT, PT_SMALL, PT_LETTER = 174.0, 225.0, 6.5, 6.0, 9.0
    PT_MIN, PT_MAX, UPPER, PROFILE = 6.0, 7.0, True, "cell"
    text_width.__defaults__ = (PT, "normal", "normal")
    wrap.__defaults__ = (PT_SMALL,)
    Canvas.text.__defaults__ = (PT,)
    Canvas.key.__defaults__ = (PT_SMALL, 2.4, 0.8, TEXT, None)


def unit(n: int) -> float:
    """Width (mm) of n units of the four-unit grid (Cell profile), gaps included."""
    u = (FULL - 3 * GAP) / 4
    return n * u + (n - 1) * GAP


def unit_x(i: int) -> float:
    """Left edge (mm) of unit i (0-3) of the four-unit grid."""
    return i * ((FULL - 3 * GAP) / 4 + GAP)


def span(n: int) -> float:
    """Width (mm) of n grid columns including the gaps between them."""
    return n * COL + (n - 1) * GAP


def col_x(i: int) -> float:
    """Left edge (mm) of grid column i (0-5)."""
    return i * (COL + GAP)


def snap(v: float, dpi: int = DPI) -> float:
    """Millimetres rounded to the nearest pixel of the export grid."""
    return round(v * MM * dpi) / dpi / MM


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": PT, "axes.titlesize": PT,
        "axes.labelsize": PT, "xtick.labelsize": PT_SMALL, "ytick.labelsize": PT_SMALL, "legend.fontsize": PT_SMALL,
        "axes.linewidth": LW, "xtick.major.width": LW, "ytick.major.width": LW, "xtick.major.size": 2.0,
        "ytick.major.size": 2.0, "xtick.major.pad": 1.0, "ytick.major.pad": 1.0, "xtick.direction": "out",
        "ytick.direction": "out", "xtick.minor.size": 1.1, "ytick.minor.size": 1.1, "xtick.minor.width": LW_THIN,
        "ytick.minor.width": LW_THIN, "xtick.minor.visible": False, "ytick.minor.visible": False, "axes.labelpad": 1.5,
        "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": TEXT,
        "ytick.labelcolor": TEXT, "text.color": TEXT, "axes.labelcolor": TEXT, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "flybrain-figures", "pdf.compression": 6, "savefig.dpi": DPI, "figure.dpi": 150,
        "lines.linewidth": LW, "patch.linewidth": LW, "hatch.linewidth": LW, "lines.solid_capstyle": "butt",
        "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
        "mathtext.sf": FONT, "mathtext.default": "regular", "axes.unicode_minus": True, "image.interpolation": "none",
        "figure.facecolor": "white", "axes.facecolor": "none", "savefig.facecolor": "white", "legend.frameon": False})


class Canvas:
    """A figure of width x height mm whose axes, text and image tiles are placed in mm from the top-left corner."""

    def __init__(self, width: float = FULL, height: float = MAX_H, ppi_min: int = 600):
        setup()
        self.w, self.h = float(width), float(height)
        self.fig = plt.figure(figsize=(self.w * MM, self.h * MM))
        self.letters: list[str] = []
        self.tiles: list[dict] = []                    # image tiles: rect (mm), group, dark background, pixels
        self.fills: list[dict] = []                    # filled boxes that may carry light text: rect (mm), fill colour
        self.ppi_min = ppi_min

    # ---- placement
    def axes(self, left: float, top: float, width: float, height: float, **kw):
        return self.fig.add_axes((left / self.w, 1 - (top + height) / self.h, width / self.w, height / self.h), **kw)

    def letter(self, x: float, y: float, letter: str, title: str | None = None):
        self.letters.append(letter)
        self.fig.text(x / self.w, 1 - y / self.h, letter, fontweight="bold", fontsize=PT_LETTER, va="top", ha="left",
                      gid=f"letter_{letter}")
        if title:
            self.fig.text((x + 3.0) / self.w, 1 - (y + 0.55) / self.h, title, fontsize=PT, va="top", ha="left",
                          fontweight="bold", color=TEXT)

    def text(self, x: float, y: float, s: str, size: float = PT, **kw):
        return self.fig.text(x / self.w, 1 - y / self.h, s, fontsize=size, **kw)

    def tile(self, left: float, top: float, image: np.ndarray, width: float | None = None, height: float | None = None,
             group: str | None = None, dark: bool = True, label: str | None = None):
        """Place an RGB(A) uint8 image with its top-left corner snapped to the pixel grid. Give width or height (mm);
        the other follows the image aspect. Returns the axes (data coordinates = pixel coordinates)."""
        h_px, w_px = image.shape[:2]
        if width is None and height is None:
            width = w_px / DPI / MM
        if width is None:
            width = height * w_px / h_px
        if height is None:
            height = width * h_px / w_px
        left, top = snap(left), snap(top)
        ax = self.axes(left, top, width, height)
        ax.imshow(image, interpolation="none", extent=(0, w_px, h_px, 0), aspect="auto", resample=False)
        ax.set_xlim(0, w_px)
        ax.set_ylim(h_px, 0)
        ax.set_axis_off()
        self.tiles.append(dict(rect=(left, top, width, height), group=group, dark=dark, pixels=(w_px, h_px), ax=ax, label=label))
        return ax

    def mark(self, x: float, y: float, s: float = 6.0, marker: str = "o", fc: str = TEXT, ec: str | None = None,
             lw: float = 0.5) -> None:
        """A legend glyph centred at (x, y) mm; s is the scatter size (points squared) so glyphs match the panels."""
        self.fig.add_artist(Line2D([x / self.w], [1 - y / self.h], transform=self.fig.transFigure, marker=marker,
                                   markersize=s ** 0.5, markerfacecolor=fc, markeredgecolor=ec or fc,
                                   markeredgewidth=lw, linestyle="none"))

    def box(self, x: float, y: float, w: float, h: float, fc: str, ec: str = "none", lw: float = 0.0, **kw) -> None:
        """A rectangle with its top-left corner at (x, y) mm."""
        self.fig.patches.append(Rectangle((x / self.w, 1 - (y + h) / self.h), w / self.w, h / self.h,
                                          transform=self.fig.transFigure, facecolor=fc, edgecolor=ec, linewidth=lw, **kw))

    def line(self, xs, ys, colour: str = TEXT, lw: float = LW, **kw) -> None:
        self.fig.add_artist(Line2D([x / self.w for x in xs], [1 - y / self.h for y in ys], transform=self.fig.transFigure,
                                   color=colour, linewidth=lw, **kw))

    def key(self, x: float, y: float, items, size: float = PT_SMALL, gap: float = 2.4, pad: float = 0.8,
            colour: str = TEXT, effects=None) -> float:
        """One legend row starting at x mm, centred on y mm. items: (glyphs, label) pairs, glyphs a list of dicts
        {"kind": "mark" | "box" | "vline" | "hline", ...}. Labels are placed by their measured width, so a row never
        overlaps itself. Returns the right end of the row (mm)."""
        for glyphs, label in items:
            for g in glyphs:
                kind = g.get("kind", "mark")
                if kind == "mark":
                    d = g.get("s", 6.0) ** 0.5 * 25.4 / 72
                    self.mark(x + d / 2, y + g.get("dy", 0.0), g.get("s", 6.0), g.get("marker", "o"),
                              g.get("fc", TEXT), g.get("ec"), g.get("lw", 0.5))
                    x += d + g.get("space", 0.5)
                elif kind == "box":
                    b = g.get("size", 1.3)
                    self.box(x, y - b / 2, b, b, g.get("fc", TEXT), g.get("ec", "none"), g.get("lw", 0.0))
                    x += b + 0.5
                elif kind == "vline":
                    hh = g.get("h", 2.0)
                    self.line([x + 0.3, x + 0.3], [y - hh / 2, y + hh / 2], g.get("colour", TEXT), g.get("lw", 1.0))
                    x += 0.6 + 0.5
                elif kind == "hline":
                    ww = g.get("w", 2.4)
                    self.line([x, x + ww], [y, y], g.get("colour", TEXT), g.get("lw", 0.9), linestyle=g.get("ls", "-"))
                    if "mark_fc" in g:
                        self.mark(x + ww / 2, y, g.get("s", 9.0), g.get("marker", "o"), g["mark_fc"], g.get("colour"),
                                  g.get("mlw", 0.5))
                    x += ww + 0.5
            if label:
                t = self.text(x + pad - 0.5, y, label, size=size, va="center", ha="left", color=colour)
                if effects:
                    t.set_path_effects(effects)
                x += pad - 0.5 + text_width(label, size)
            x += gap
        return x - gap

    # ---- export
    def save(self, out_dir: Path, name: str, preview_dir: Path | None = None, jpg: bool = True) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = []
        p = out_dir / f"{name}.pdf"
        self.fig.savefig(p, format="pdf", metadata={"CreationDate": None, "ModDate": None, "Creator": "manuscript/figures",
                                                     "Title": name})
        files.append(p)
        buf = io.StringIO()
        self.fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": "manuscript/figures"})
        p = out_dir / f"{name}.svg"
        p.write_text(buf.getvalue().replace("\r\n", "\n"), encoding="utf-8", newline="\n")
        files.append(p)
        if jpg:
            p = out_dir / f"{name}.jpg"
            self.fig.savefig(p, format="jpg", dpi=DPI, pil_kwargs={"quality": 95, "optimize": False})
            files.append(p)
        if preview_dir is not None:
            Path(preview_dir).mkdir(parents=True, exist_ok=True)
            p = Path(preview_dir) / f"{name}.png"
            self.fig.savefig(p, format="png", dpi=300, metadata={"Software": None})
            files.append(p)
        return files

    def close(self) -> None:
        plt.close(self.fig)


_T2P = TextToPath()


def text_width(s: str, size: float = PT, weight: str = "normal", style: str = "normal") -> float:
    """Advance width of s in mm at the given size (Arial, as rendered; mathtext measured as drawn)."""
    w, _, _ = _T2P.get_text_width_height_descent(s, FontProperties(family=FONT, size=size, weight=weight, style=style),
                                                 ismath="$" in s)
    return w * 25.4 / 72


def wrap(text: str, width_mm: float, size: float = PT_SMALL) -> list[str]:
    """Greedy word wrap by measured Arial width (mm); mathtext words are measured as drawn."""
    lines, cur = [], ""
    for word in text.split(" "):
        trial = word if not cur else cur + " " + word
        if cur and text_width(trial, size) > width_mm:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def clean(ax, left: bool = True, bottom: bool = True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.set_yticks([])
    if not bottom:
        ax.set_xticks([])


def blank(ax) -> None:
    ax.set_axis_off()


def halo(width: float = 1.2, colour: str = "white"):
    return [patheffects.withStroke(linewidth=width, foreground=colour)]


def dark_halo(width: float = 1.0):
    """Outline for white text inside black tiles (keeps contrast where it crosses bright pixels)."""
    return [patheffects.withStroke(linewidth=width, foreground=TILE_BG)]


def fits(x_end: float, right: float, what: str) -> None:
    assert x_end <= right + 1e-6, f"{what}: ends at {x_end:.1f} mm, beyond {right:.1f} mm"


def fmt_p(p: float) -> str:
    """P for figure text: two significant figures below 0.1 with trailing zeros kept, two decimals above,
    'x 10^-k' scientific form below 0.001 (exact tests reach far below it)."""
    if p < 0.001:
        m, e = f"{p:.1e}".split("e")
        return f"{m} × 10$^{{{int(e)}}}$".replace("-", MINUS)
    return f"{p:#.2g}" if p < 0.1 else f"{p:.2f}"


def p_text(p: float) -> str:
    """'P = 0.012' with an italic P; '<' never used because exact tests give exact values."""
    return f"{P_SYM} = {fmt_p(p)}"


def num(x: float, d: int = 2, sign: bool = False) -> str:
    """A number with a true minus sign (and '+' when sign)."""
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", MINUS)


def minor(ax, x: bool = True, y: bool = True, n: int = 2) -> None:
    """Unlabelled minor ticks: n subdivisions between major ticks on linear axes, 2-9 x each decade on log axes."""
    for on, axis in ((x, ax.xaxis), (y, ax.yaxis)):
        if not on:
            continue
        if axis.get_scale() == "log":
            axis.set_minor_locator(LogLocator(base=10, subs=tuple(np.arange(2, 10)), numticks=20))
            axis.set_minor_formatter(NullFormatter())
        else:
            axis.set_minor_locator(AutoMinorLocator(n))


def interval(ax, pos: float, est: float, lo: float, hi: float, colour: str = TEXT, orient: str = "v",
             ms: float = 9.0, lw: float = 0.8, marker: str = "o", fc: str | None = None, z: float = 4) -> None:
    """A point estimate with its interval at position pos (orient 'v': vertical interval at x = pos)."""
    if orient == "v":
        ax.plot([pos, pos], [lo, hi], color=colour, lw=lw, solid_capstyle="butt", zorder=z)
        ax.scatter([pos], [est], s=ms, marker=marker, facecolor=fc or colour, edgecolor=colour, linewidth=0.5, zorder=z + 1)
    else:
        ax.plot([lo, hi], [pos, pos], color=colour, lw=lw, solid_capstyle="butt", zorder=z)
        ax.scatter([est], [pos], s=ms, marker=marker, facecolor=fc or colour, edgecolor=colour, linewidth=0.5, zorder=z + 1)


def density(ax, values, colour: str, orient: str = "x", grid=None, bw: float | None = None, log: bool = False,
            alpha: float = 0.35, lw: float = LW_THIN, scale: float = 1.0) -> None:
    """Kernel density drawn as a filled curve along x (curve upwards) or y (curve to the right), max-normalised."""
    from scipy.stats import gaussian_kde
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & ((v > 0) if log else True)]
    if len(v) < 3 or np.ptp(v) == 0:
        return
    t = np.log10(v) if log else v
    k = gaussian_kde(t, bw_method=bw)
    g = np.linspace(t.min() - 0.5 * t.std(), t.max() + 0.5 * t.std(), 200) if grid is None else (
        np.log10(grid) if log else np.asarray(grid, float))
    f = k(g)
    f = scale * f / f.max()
    gx = 10 ** g if log else g
    if orient == "x":
        ax.fill_between(gx, 0, f, color=colour, alpha=alpha, lw=0)
        ax.plot(gx, f, color=colour, lw=lw)
    else:
        ax.fill_betweenx(gx, 0, f, color=colour, alpha=alpha, lw=0)
        ax.plot(f, gx, color=colour, lw=lw)


def half_violin(ax, pos: float, values, colour: str, width: float = 0.35, side: str = "right", log: bool = False,
                alpha: float = 0.3, orient: str = "v") -> None:
    """Distribution as a half violin beside position pos."""
    from scipy.stats import gaussian_kde
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & ((v > 0) if log else True)]
    if len(v) < 10 or np.ptp(v) == 0:
        return
    t = np.log10(v) if log else v
    g = np.linspace(np.percentile(t, 0.5), np.percentile(t, 99.5), 160)
    f = gaussian_kde(t)(g)
    f = width * f / f.max()
    gy = 10 ** g if log else g
    sgn = 1 if side == "right" else -1
    if orient == "v":
        ax.fill_betweenx(gy, pos, pos + sgn * f, color=colour, alpha=alpha, lw=0, zorder=2)
    else:
        ax.fill_between(gy, pos, pos + sgn * f, color=colour, alpha=alpha, lw=0, zorder=2)


def null_hist(ax, null, observed: float, colour: str = TEXT2, obs_colour: str = TEXT, bins: int = 30) -> None:
    """Permutation null as a histogram with the observed value as a vertical line."""
    ax.hist(np.asarray(null, float), bins=bins, color=colour, alpha=0.55, lw=0)
    ax.axvline(observed, color=obs_colour, lw=0.8)


def ref_line(ax, value: float, orient: str = "h", colour: str = GRID, lw: float = LW_THIN, ls=(0, (2, 1.5))) -> None:
    """A reference line (1 for ratios, 0 for differences, chance) behind the data."""
    (ax.axhline if orient == "h" else ax.axvline)(value, color=colour, lw=lw, ls=ls, zorder=0)
