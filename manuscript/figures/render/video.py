"""Supplemental videos (Cell, "Video S1-S4, related to Figure N"): dark CNS frames from per-millisecond cell activity, composed
with Pillow and encoded by ffmpeg (H.264, yuv420p, CRF 18, one thread and fixed flags, so that a rebuild is byte-identical).

A Stage holds one projected view of the unfolded CNS: the neuropil shell and the dim cell bodies are rendered once, each frame
adds the cells active at that moment as Gaussian splats in linear light and resolves with the figures' bloom. Text, boards and
bars are drawn on the 8-bit frame with Arial at sizes that read on a laptop screen.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render import project as rp, scenes as sc

W, H, FPS = 1920, 1080, 30
BLOOM = (2.0, 0.40, 0.22)                  # sigma (output px), strength, luminance threshold: as the dark figure tiles
FONT = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
INK, MUTED, DIM = (236, 236, 236), (160, 160, 160), (96, 96, 96)


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


class Stage:
    """One panel of the frame: the unfolded CNS fitted into a w x h pixel box, the static layers rendered once."""

    def __init__(self, all_xyz, w, h, rotate=False, margin=0.02, shell=0.11, bodies=0.05):
        self.rotate = rotate
        vw, vh = (h, w) if rotate else (w, h)
        self.view = rp.fit_view(all_xyz, vw, vh, unfold=True, margin=margin, ss=1)
        tile = rp.Tile(self.view)
        sc.shell(tile, strength=shell, fill=0.010)
        tile.glow(all_xyz, 1.0, "#5A5A5A", bodies * sc.density_weights(self.view, all_xyz), label="all cell bodies")
        self.base = tile.img.copy()

    def project(self, xyz):
        px, py, _, _ = self.view.pixels(xyz)
        return px, py

    def frame(self, px, py, rgb, weight, sigma_px=1.6):
        """The base plus splats at (px, py) in linear light, resolved to 8-bit sRGB (rotated if the stage is upright)."""
        tile = rp.Tile(self.view)
        tile.img = self.base.copy()
        if len(px):
            sig = np.full(len(px), float(sigma_px))
            rp._gauss_add(np.asarray(px, np.float64), np.asarray(py, np.float64), sig, np.asarray(rgb, np.float64),
                          np.asarray(weight, np.float64), tile.img)
        img = tile.resolve(bloom=BLOOM)
        if self.rotate:
            img = np.ascontiguousarray(np.rot90(img))
        return Image.fromarray(img)


def text(draw, xy, s, size=26, fill=INK, bold=False, anchor="la"):
    draw.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def bar(draw, x, y, w, h, frac, fill, back=(40, 40, 40)):
    draw.rectangle([x, y, x + w, y + h], fill=back)
    if frac > 0:
        draw.rectangle([x, y, x + int(round(w * min(frac, 1.0))), y + h], fill=fill)


def board(draw, x, y, cell, walls, goals, boxes, player, size=10, wall=(70, 70, 70), floor=(22, 22, 22), goal=(120, 120, 120),
          box=(214, 160, 60), box_on_goal=(120, 200, 120), agent=(236, 236, 236), hidden_goals=False):
    """A Sokoban board: cells are indices r * size + c."""
    for i in range(size * size):
        r, c = divmod(i, size)
        x0, y0 = x + c * cell, y + r * cell
        draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=wall if i in walls else floor)
        if i in goals and not hidden_goals:
            m = cell // 3
            draw.rectangle([x0 + m, y0 + m, x0 + cell - 1 - m, y0 + cell - 1 - m], outline=goal, width=max(2, cell // 12))
    for b in boxes:
        r, c = divmod(b, size)
        m = cell // 8
        draw.rectangle([x + c * cell + m, y + r * cell + m, x + (c + 1) * cell - 1 - m, y + (r + 1) * cell - 1 - m],
                       fill=box_on_goal if (b in goals and not hidden_goals) else box)
    r, c = divmod(player, size)
    m = cell // 5
    draw.ellipse([x + c * cell + m, y + r * cell + m, x + (c + 1) * cell - 1 - m, y + (r + 1) * cell - 1 - m], fill=agent)


class Chart:
    """A small live chart on the frame (the videos' real-time metrics): a title above the plot, a y axis with ticks (linear or
    log, or "symlog" for log10(1 + y) so that zero sits on the axis), an x range shared with the animation, optional phase
    bands; series are revealed up to the current x, with a dot and the value at each tip. Coordinates are frame pixels; values
    outside the y range are clipped to it."""

    def __init__(self, box, xlim, ylim, title, yticks, log=False, bands=(), xticks=(), ytick_fmt="{:g}", xlabel=""):
        self.x0, self.y0, self.x1, self.y1 = box
        self.xlim, self.ylim, self.log = xlim, ylim, log
        self.title, self.yticks, self.bands, self.xticks, self.ytick_fmt = title, yticks, bands, xticks, ytick_fmt
        self.xlabel = xlabel

    def px(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        fx = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        if self.log == "symlog":
            lo, hi = np.log10(1 + self.ylim[0]), np.log10(1 + self.ylim[1])
            fy = (np.log10(1 + np.clip(y, *self.ylim)) - lo) / (hi - lo)
        elif self.log:
            lo, hi = np.log10(self.ylim[0]), np.log10(self.ylim[1])
            fy = (np.log10(np.clip(y, self.ylim[0], self.ylim[1])) - lo) / (hi - lo)
        else:
            fy = (np.clip(y, *self.ylim) - self.ylim[0]) / (self.ylim[1] - self.ylim[0])
        return self.x0 + fx * (self.x1 - self.x0), self.y1 - fy * (self.y1 - self.y0)

    def frame(self, draw, x_now=None):
        """Axes, bands, ticks, title and the cursor at x_now."""
        for x_a, x_b, colour in self.bands:
            xa, _ = self.px(x_a, self.ylim[0])
            xb, _ = self.px(x_b, self.ylim[0])
            draw.rectangle([xa, self.y0, xb, self.y1], fill=colour)
        draw.line([self.x0, self.y0, self.x0, self.y1], fill=DIM, width=2)
        draw.line([self.x0, self.y1, self.x1, self.y1], fill=DIM, width=2)
        for t in self.yticks:
            _, y = self.px(self.xlim[0], t)
            draw.line([self.x0 - 6, y, self.x0, y], fill=DIM, width=2)
            lab = self.ytick_fmt(t) if callable(self.ytick_fmt) else self.ytick_fmt.format(t)
            text(draw, (self.x0 - 10, y), lab, 17, fill=MUTED, anchor="rm")
        for t, lab in self.xticks:
            x, _ = self.px(t, self.ylim[0])
            draw.line([x, self.y1, x, self.y1 + 6], fill=DIM, width=2)
            text(draw, (x, self.y1 + 9), lab, 17, fill=MUTED, anchor="ma")
        if self.xlabel:
            text(draw, ((self.x0 + self.x1) / 2, self.y1 + 9), self.xlabel, 17, fill=MUTED, anchor="ma")
        text(draw, (self.x0, self.y0 - 30), self.title, 20, fill=INK, bold=True)
        if x_now is not None:
            x, _ = self.px(x_now, self.ylim[0])
            draw.line([x, self.y0, x, self.y1], fill=(90, 90, 90), width=1)

    def series(self, draw, x, y, colour, x_now, width=3, label_fmt="{:,.0f}", tip=True, dx_label=8):
        """The series revealed up to x_now, with the value at its tip."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        keep = (x <= x_now) & np.isfinite(y)
        if keep.sum() < 1:
            return
        px_, py_ = self.px(x[keep], y[keep])
        pts = list(zip(px_.tolist(), py_.tolist()))
        if len(pts) > 1:
            draw.line(pts, fill=colour, width=width, joint="curve")
        if tip:
            tx, ty = pts[-1]
            draw.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], fill=colour)
            if label_fmt:
                lab = label_fmt.format(float(y[keep][-1])).replace("-", "−")
                box = draw.textbbox((tx + dx_label, ty), lab, font=font(18), anchor="lm")
                draw.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill=(0, 0, 0))
                text(draw, (tx + dx_label, ty), lab, 18, fill=colour, anchor="lm")

    def tips(self, draw, items, gap=20, dx=8):
        """Values at several series' tips, spread apart vertically so that none overlaps: items (x, y, colour, label)."""
        pts = sorted(([float(v) for v in self.px(x, y)] + [col, lab] for x, y, col, lab in items), key=lambda q: q[1])
        for i in range(1, len(pts)):
            pts[i][1] = max(pts[i][1], pts[i - 1][1] + gap)
        for tx, ty, col, lab in pts:
            lab = lab.replace("-", "−")
            box = draw.textbbox((tx + dx, ty), lab, font=font(18), anchor="lm")
            draw.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill=(0, 0, 0))
            text(draw, (tx + dx, ty), lab, 18, fill=col, anchor="lm")

    def keys(self, draw, items):
        """Colour keys right-aligned on the title row: (label, colour, mark) with mark None, "dot" or "dash"."""
        x = self.x1
        for lab, col, mark in reversed(items):
            text(draw, (x, self.y0 - 28), lab, 18, fill=col, anchor="ra")
            x -= font(18).getlength(lab) + 10
            if mark == "dot":
                draw.ellipse([x - 8, self.y0 - 22, x, self.y0 - 14], fill=col)
                x -= 14
            elif mark == "dash":
                for k in range(3):
                    draw.line([x - 34 + 12 * k, self.y0 - 18, x - 27 + 12 * k, self.y0 - 18], fill=col, width=2)
                x -= 40
            x -= 24

    def dashed(self, draw, x, y, colour, width=2, dash=10, gap=7):
        """A reference curve drawn whole, dashed."""
        px_, py_ = self.px(x, y)
        s = np.r_[0.0, np.cumsum(np.hypot(np.diff(px_), np.diff(py_)))]
        for i in range(len(px_) - 1):
            if ((s[i] + s[i + 1]) / 2) % (dash + gap) < dash:
                draw.line([px_[i], py_[i], px_[i + 1], py_[i + 1]], fill=colour, width=width)

    def bars(self, draw, x, stacks, colours, x_now, width_px):
        """Stacked bars at positions x (heights per stack, bottom first), drawn up to x_now."""
        for xi, hs in zip(np.asarray(x, float), stacks):
            if xi > x_now:
                break
            base = 0.0
            for h_, col in zip(hs, colours):
                if h_ <= 0:
                    continue
                xa, ya = self.px(xi, base)
                _, yb = self.px(xi, base + h_)
                draw.rectangle([xa - width_px / 2, yb, xa + width_px / 2, ya], fill=col)
                base += h_


def rgb8(hex_or_rgb):
    """An 8-bit colour from a hex string or a 0-1 RGB triple."""
    if isinstance(hex_or_rgb, str):
        h = hex_or_rgb.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(int(round(255 * float(v))) for v in hex_or_rgb[:3])


class Encoder:
    """Raw RGB frames piped to ffmpeg; deterministic flags (one thread, no timestamps or encoder tags in the stream)."""

    def __init__(self, path: Path, fps=FPS, size=(W, H)):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
               "-threads", "1", "-x264-params", "threads=1:lookahead-threads=1", "-movflags", "+faststart",
               "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact", "-metadata", "encoder=", str(self.path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.frames = 0

    def add(self, img: Image.Image, repeat=1):
        if img.size != self.size:
            raise ValueError(f"frame size {img.size} != {self.size}")
        b = np.asarray(img.convert("RGB"), np.uint8).tobytes()
        for _ in range(repeat):
            self.proc.stdin.write(b)
            self.frames += 1

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}")
        return self.frames
