"""v4 Fig. 7 dark band (panel G), "disease painted on the brain": four unfolded-CNS tiles in the Fig. 1 visual language.

G1  the Aβ42-induced change in Shal mRNA of each cell's mapped type (AD Fly Cell Atlas), on the dark diverging map; cells
    whose type has an atlas counterpart at full brightness, the rest (pan-neuronal value) at half brightness
G2  network activity under the uniform Kv4-loss lesion: each cell's task-rate change at dose 1
G3  the same mean lesion patterned by Shal expression: each cell's share of the lesion (weight, mean 1). At dose 1 the
    patterned lesion's rate map is indistinguishable from G2 (28,604 vs 26,952 cells faster), so the pattern itself is shown;
    its consequence at 1.5-2x is panel H
G4  the transcriptome-derived ageing operator: each cell's log2 old/young factor on its K+ conductances, Kenyon cells at full
    brightness and the rest dimmed; the outline of the Kenyon-cell bodies (convex hull per side, in tile pixels) is stored
    for the build

Data: manuscript/results/plan_v2/fig7_cells.csv (P12, gated); soma positions from figure_data/fd_cells.csv.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_fig7_tiles.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import figstyle as fs, palette as pal  # noqa: E402
from render import project as rp, scenes as sc  # noqa: E402

fs.use_cell()
OUT = FIG / "renders" / "v4" / "fig7"
PLAN = rp.RESULTS / "plan_v2"
BAND_H = 60.0
TILE_W = (fs.FULL - 3 * fs.GUTTER) / 4
TITLE_MM = 9.6                                           # black band for three lines of tile text
BLOOM = (2.8, 0.42, 0.20)
GHOST = "#5A5A5A"
SHAL_MAX = 0.8                                           # |log2 fold change| at full colour
RATE_UNIT, RATE_MAX = 2.5, 100.0                         # one spike per 400-ms trial; |change| at full colour (Hz)
AGE_MAX = 0.25                                           # |log2 factor| at full colour
WEIGHT_MAX = 1.0                                         # |lesion weight - 1| at full colour
OTHER_GAIN = 0.3                                         # G4: cells other than Kenyon cells
PAN_GAIN = 0.5                                           # cells without an atlas type: half brightness


def px(mm):
    return int(np.ceil(mm * fs.MM * fs.DPI))


def base(all_xyz):
    W, H, top = px(TILE_W), px(BAND_H), px(TITLE_MM)
    v0 = rp.fit_view(all_xyz, W, H - top, unfold=True, margin=0.010, ss=3)
    view = rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 3)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.11, fill=0.010)
    tile.glow(all_xyz, 1.0, GHOST, 0.05 * sc.density_weights(view, all_xyz), label="all cell bodies")
    return tile


def signed(tile, xyz, v, vmax, gain, label, unit=None, scale=1.0, power=0.6):
    """Dark diverging glow: hue by sign (blue down, orange up), lightness and weight by |v| / vmax (log in units of `unit`)."""
    a = np.abs(v)
    u = np.clip(np.log1p(a / unit) / np.log1p(vmax / unit), 0, 1) if unit else np.clip(a / vmax, 0, 1)
    order = np.argsort(u, kind="stable")
    t = 0.5 + np.sign(v[order]) * (0.22 + 0.28 * u[order])
    col = rp.srgb_to_linear(pal.SIGNED_ON_DARK(t)[:, :3])
    w = scale * gain[order] * (0.10 + 0.9 * u[order] ** 1.5) * sc.density_weights(tile.view, xyz[order], power=power)
    tile.glow(xyz[order], 1.3, col, w, label=label)


def hull_px(view, xyz):
    """Convex hull of a soma set per side of the midline, in output pixels of the tile."""
    pxs, pys, _, _ = view.pixels(xyz)
    pts = np.column_stack([pxs, pys]) / view.ss
    mid = np.median(pts[:, 0])
    out = []
    for side in (pts[pts[:, 0] < mid], pts[pts[:, 0] >= mid]):
        if len(side) >= 3:
            h = ConvexHull(side)
            out.append(side[h.vertices].round(1).tolist())
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fd = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False)
    _, all_xyz = sc.somata(fd)
    cells = pd.read_csv(PLAN / "fig7_cells.csv").drop(columns=["source_file"])
    s, xyz = sc.somata(fd.merge(cells, on="cell", validate="one_to_one"))
    meta = {}
    # G1: Shal mRNA change under Abeta42 (AD Fly Cell Atlas), by the cell's mapped type
    tile = base(all_xyz)
    typed = s["adfca_tier"].eq("cell type").to_numpy()
    gain = np.where(typed, 1.0, PAN_GAIN)
    signed(tile, xyz, s["shal_log2fc_abeta42"].to_numpy(float), SHAL_MAX, gain, "Shal log2 fold change, Abeta42 vs control")
    img = tile.resolve(bloom=BLOOM)
    tile.save(OUT / "shal.png", img, dict(cells_located=int(len(s)), typed_located=int(typed.sum()), vmax=SHAL_MAX, pan_gain=PAN_GAIN,
                                          scale_um_per_px=tile.view.um_per_px))
    meta["shal"] = dict(typed_located=int(typed.sum()), cells_located=int(len(s)))
    print(f"shal {time.time() - t0:.1f}s", flush=True)
    # G2: task-rate change under the uniform lesion (dose 1)
    d = (s["rate_uniform_dose1_hz"] - s["rate_healthy_hz"]).to_numpy(float)
    on = d != 0
    tile = base(all_xyz)
    signed(tile, xyz[on], d[on], RATE_MAX, np.ones(int(on.sum())), "task-rate change, uniform lesion dose 1", unit=RATE_UNIT)
    img = tile.resolve(bloom=BLOOM)
    tile.save(OUT / "uniform.png", img, dict(changed_located=int(on.sum()), faster_located=int((d > 0).sum()), slower_located=int((d < 0).sum()),
                                             rate_unit_hz=RATE_UNIT, vmax_hz=RATE_MAX, scale_um_per_px=tile.view.um_per_px))
    meta["uniform"] = dict(changed_located=int(on.sum()))
    print(f"uniform {time.time() - t0:.1f}s", flush=True)
    # G3: the Shal pattern of the same mean lesion (each cell's weight)
    tile = base(all_xyz)
    signed(tile, xyz, s["kv4_weight"].to_numpy(float) - 1.0, WEIGHT_MAX, np.ones(len(s)), "lesion weight from Shal expression (mean 1)", scale=0.8, power=0.75)
    img = tile.resolve(bloom=BLOOM)
    tile.save(OUT / "pattern.png", img, dict(cells_located=int(len(s)), vmax=WEIGHT_MAX, scale_um_per_px=tile.view.um_per_px))
    meta["pattern"] = dict(cells_located=int(len(s)))
    print(f"pattern {time.time() - t0:.1f}s", flush=True)
    # G4: the ageing operator, Kenyon cells outlined
    tile = base(all_xyz)
    kc = s["kenyon_cell"].astype(str).str.lower().eq("true").to_numpy()
    signed(tile, xyz, s["ageing_log2_factor"].to_numpy(float), AGE_MAX, np.where(kc, 1.0, OTHER_GAIN), "ageing operator, log2 old/young K+ factor",
           scale=0.7, power=0.8)
    hulls = hull_px(tile.view, xyz[kc])
    img = tile.resolve(bloom=BLOOM)
    tile.save(OUT / "ageing.png", img, dict(kenyon_cells_located=int(kc.sum()), kenyon_hulls_px=hulls, vmax=AGE_MAX,
                                            scale_um_per_px=tile.view.um_per_px))
    meta["ageing"] = dict(kenyon_cells_located=int(kc.sum()))
    print(f"ageing {time.time() - t0:.1f}s", flush=True)
    (OUT / "tiles.json").write_text(json.dumps(dict(band_mm=BAND_H, tile_mm=TILE_W, title_mm=TITLE_MM, shal_max=SHAL_MAX, rate_unit_hz=RATE_UNIT,
                                                    rate_max_hz=RATE_MAX, age_max=AGE_MAX, weight_max=WEIGHT_MAX, pan_gain=PAN_GAIN,
                                                    other_gain=OTHER_GAIN, tiles=meta), indent=1),
                                    encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
