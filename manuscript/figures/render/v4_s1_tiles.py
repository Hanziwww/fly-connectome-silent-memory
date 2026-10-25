"""v4 Figure S1 dark band: where training acts on the connectome, on the unfolded CNS (upright, as Figs 2C and 7G).

A1-A3  each located cell glowing by the mean |change| of its parameter group over the 24 planning maps (fraction of the
       parameter range; magma, log scale) for output gain, input gain and threshold; cells of unchanged groups are the dim ghost
A4     the cells of the 1% most consistently changed groups in their superclass-group colours (as Fig. 1A)

Data: manuscript/results/figure_data/fd_group_change.csv and fd_cells.csv (the frozen parameter-change analysis).
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_s1_tiles.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import figstyle as fs, palette as pal  # noqa: E402
from render import project as rp, scenes as sc  # noqa: E402

fs.use_cell()
OUT = FIG / "renders" / "v4" / "s1"
BAND_H = 50.0
TILE_W = (fs.FULL - 3 * fs.GUTTER) / 4
TITLE_MM = 7.6
BLOOM = (2.6, 0.40, 0.22)
LO, HI = 1e-4, 2e-2                                      # mean |change| as a fraction of the parameter range
PARAMS = [("post_log_gain", "output gain"), ("pre_log_gain", "input gain"), ("threshold_mv", "threshold")]
GHOST = "#5A5A5A"
UNCHANGED = "#2E2E2E"                                    # cells of groups that did not change
DISC_UM = 1.6


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


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fd = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False)
    groups = pd.read_csv(sc.FIGURE_DATA / "fd_group_change.csv").set_index("group")
    s, xyz = sc.somata(fd)
    g = s["parameter_group"].to_numpy()
    meta = []
    for key, lab in PARAMS:
        v = groups[f"mean_normalized_abs_{key}"].reindex(g).to_numpy(float)
        on = v > 0
        u = np.clip((np.log10(np.maximum(v, LO)) - np.log10(LO)) / (np.log10(HI) - np.log10(LO)), 0, 1)
        tile = base(xyz)
        col = rp.srgb_to_linear(pal.MAGNITUDE_ON_DARK(0.22 + 0.78 * u)[:, :3])       # a dense map: one opaque disc per cell, nearest in front
        col[~on] = rp.hex_linear(UNCHANGED)
        tile.discs(xyz, DISC_UM, col, 0.8, order="depth", label=f"mean |change| of {lab}")
        img = tile.resolve(bloom=BLOOM)
        info = dict(parameter=key, label=lab, cells_located=int(len(v)), changed_located=int(on.sum()), lo=LO, hi=HI, scale_um_per_px=tile.view.um_per_px)
        tile.save(OUT / f"{key}.png", img, info)
        meta.append(dict(name=key, **info))
        print(f"{key} {time.time() - t0:.1f}s", flush=True)
    top = groups["top_1pct"].astype(str).str.lower().eq("true").reindex(g).fillna(False).to_numpy(bool)
    grp = s["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other").to_numpy()
    col = rp.srgb_to_linear(np.array([to_rgb(pal.ATLAS_ON_DARK[k]) for k in grp[top]]))
    tile = base(xyz)
    tile.glow(xyz[top], 1.5, col, 0.9 * sc.density_weights(tile.view, xyz[top], power=0.5), label="cells of the most consistently changed groups")
    img = tile.resolve(bloom=BLOOM)
    info = dict(cells_located=int(top.sum()), groups=int(groups["top_1pct"].astype(str).str.lower().eq("true").sum()), scale_um_per_px=tile.view.um_per_px)
    tile.save(OUT / "consistent.png", img, info)
    meta.append(dict(name="consistent", **info))
    (OUT / "tiles.json").write_text(json.dumps(dict(band_mm=BAND_H, tile_mm=TILE_W, title_mm=TITLE_MM, tiles=meta), indent=1), encoding="utf-8", newline="\n")
    print(f"done {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
