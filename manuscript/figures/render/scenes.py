"""Reusable scene pieces for the anatomy tiles (glass-shell look chosen at the renderer pilot, 2026-10-18)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from lib import palette as pal
from render import project as rp

FIGURE_DATA = rp.RESULTS / "figure_data"
SKELETON_CACHE = rp.ROOT / "flybrain" / "data" / "morphology" / "skeletons"
SKELETON_DOWNLOADS = rp.ROOT / "manuscript" / "figures" / "assets" / "skeletons"
SHELL = "#8E9BA8"
DORSAL_OBLIQUE = ((0.45, -1.0, -0.25), (1.0, 0.0, 0.0))      # towards the camera, screen-down hint (fly left)
DV_MIN, DV_MAX = 1.0, 20.0


def no_lamina(meta) -> bool:
    """The lamina surfaces are truncated by the EM volume (flat cut faces) and are left out of the shell."""
    return not meta["label"].startswith("LA(")


@lru_cache(maxsize=1)
def regions():
    return rp.load_regions()


@lru_cache(maxsize=1)
def cells() -> pd.DataFrame:
    c = pd.read_csv(FIGURE_DATA / "fd_cells.csv", low_memory=False)
    s = pd.read_csv(FIGURE_DATA / "fd_soma_state.csv")
    return c.merge(s.drop(columns=["body_id"]), on="cell", validate="one_to_one")


def somata(df: pd.DataFrame):
    s = df.dropna(subset=["soma_x_um"])
    return s, s[["soma_x_um", "soma_y_um", "soma_z_um"]].to_numpy()


def shell(tile, strength=0.10, fill=0.012, gamma=2.2, colour=SHELL):
    tile.shell(regions(), colour=colour, strength=strength, gamma=gamma, fill=fill, select=no_lamina)


def background(tile, xyz, alpha=0.30, radius=1.2):
    tile.discs(xyz, radius, pal.TILE["somata"], alpha, order="add", label="all somata")


def readout_class(df: pd.DataFrame) -> pd.Series:
    return df["superclass"].map(pal.READOUT_CLASS).fillna("other")


def readout_arbors(df: pd.DataFrame, base_alpha=0.16, ref_class_size=624):
    """Segments of the cached readout arbors, coloured by readout class. Opacity is scaled so that every neuron
    and every class gets similar ink (cable-length and class-size normalisation)."""
    pts, offs, cnts, body = rp.load_arbors()
    by_body = df.set_index("body_id")
    cls = readout_class(by_body.loc[body]).to_numpy()
    role = by_body.loc[body, "role"].to_numpy()
    idx = np.array([i for i in np.flatnonzero(role == "readout") if cnts[i] > 0])
    cable = np.array([np.linalg.norm(pts[offs[i] + 1:offs[i] + cnts[i]:2] - pts[offs[i]:offs[i] + cnts[i]:2], axis=1).sum() for i in idx])
    n_class = pd.Series(cls[idx]).map(pd.Series(cls[idx]).value_counts()).to_numpy()
    alpha = np.clip(base_alpha * np.sqrt(np.median(cable) / cable) * np.sqrt(ref_class_size / n_class), 0.02, 0.30)
    a, b, col, al = [], [], [], []
    for k, i in enumerate(idx):
        p = pts[offs[i]:offs[i] + cnts[i]].astype(np.float64)
        m = len(p) // 2
        a.append(p[0::2])
        b.append(p[1::2])
        col.append(np.repeat(rp.hex_linear(pal.READOUT_ON_DARK[cls[i]])[None], m, axis=0))
        al.append(np.full(m, alpha[k]))
    counts = pd.Series(cls[idx]).value_counts().to_dict()
    return np.vstack(a), np.vstack(b), np.vstack(col), np.concatenate(al), counts


def skeleton_path(body_id: int) -> Path:
    for folder in (SKELETON_CACHE, SKELETON_DOWNLOADS):
        p = folder / str(int(body_id))
        if p.is_file():
            return p
    raise FileNotFoundError(f"no skeleton for body {body_id}")


def skeleton_segments(body_id: int):
    v, e = rp.load_skeleton(skeleton_path(body_id))
    return v[e[:, 0]], v[e[:, 1]]


def screen_direction(view, vector) -> np.ndarray:
    """Unit 2-D screen direction (x right, y down) of a world direction under the view's rotation."""
    d = np.asarray(vector, float) @ view.R.T
    d = d[:2]
    return d / np.linalg.norm(d)


def signed_colours(dv):
    """Colour (linear RGB) and opacity for cells with |dv| >= DV_MIN on the dark-centred cue map."""
    u = np.clip(np.log10(np.abs(dv) / DV_MIN) / np.log10(DV_MAX / DV_MIN), 0, 1)
    t = 0.5 + np.sign(dv) * (0.26 + 0.24 * u)
    return rp.srgb_to_linear(pal.SIGNED_ON_DARK(t)[:, :3]), 0.75 + 0.25 * u


def density_weights(view, xyz, bin_px: float = 6.0, power: float = 0.5, cap: float = 3.0) -> np.ndarray:
    """Per-point weights proportional to (local projected density)^-power, normalised to median 1 and capped, so
    dense cortex keeps its colour instead of saturating to white while sparse cells stay visible."""
    px, py, _, _ = view.pixels(xyz)
    b = bin_px * view.ss
    ix, iy = (px // b).astype(np.int64), (py // b).astype(np.int64)
    key = ix * 1_000_003 + iy
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    w = cnt[inv].astype(np.float64) ** -power
    w /= np.median(w)
    return np.minimum(w, cap)


def neuron_tile(df: pd.DataFrame, body_id: int, superclass: str, soma_xyz, width_px: int, height_px: int, bloom, margin: float = 0.13):
    """One neuron on the unfolded view, cropped to its skeleton: glass shell, nearby cell bodies, the skeleton in its
    superclass-group colour and its soma. Returns the tile, the 8-bit image and the colour."""
    a, b = skeleton_segments(body_id)
    view = rp.fit_view(np.vstack([a, b]), width_px, height_px, unfold=True, margin=margin, ss=3)
    tile = rp.Tile(view)
    shell(tile, strength=0.12, fill=0.012)
    _, xyz = somata(df)
    c, _ = view.camera(xyz)
    half = np.array([width_px, height_px]) * view.um_per_px / 2 * 1.1
    near = (np.abs(c[:, 0] - view.centre[0]) < half[0]) & (np.abs(c[:, 1] - view.centre[1]) < half[1])
    tile.discs(xyz[near], 1.2, "#3A3A3A", 0.55, order="depth", label="nearby cell bodies")
    colour = pal.ATLAS_ON_DARK[pal.SUPERCLASS_GROUP.get(superclass, "other")]
    tile.segments(a, b, max(0.5, 2.0 * view.um_per_px), colour, 0.95, order="depth", depth_cue=0.25, label="skeleton")
    soma = np.asarray(soma_xyz, np.float64)
    if not np.isnan(soma).any():
        tile.discs(soma[None], 2.4, colour, 1.0, order="given", label="soma")
    return tile, tile.resolve(bloom=bloom), colour


def scale_bar_um(view, max_mm: float = 6.0, choices=(20, 50, 100, 200)) -> int:
    """The longest round scale-bar length that stays within max_mm on the exported tile."""
    from lib import figstyle as fs
    return max(L for L in choices if L / view.um_per_px / (fs.DPI * fs.MM) <= max_mm)
