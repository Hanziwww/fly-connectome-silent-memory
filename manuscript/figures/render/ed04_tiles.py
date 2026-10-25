"""Extended Data Fig. 4 band: mean |change| of every located cell's parameter group (mean over the 24 maps of the normalized
absolute final-minus-initial change; figure_data/fd_group_change) for each parameter type, on the unfolded CNS of Fig. 4.

Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\ed04_tiles.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import figstyle as fs, palette as pal  # noqa: E402
from render import project as rp, scenes as sc  # noqa: E402

OUT = FIG / "renders" / "ed04"
BAND_H, MAP_W = 56.0, 42.0
BLOOM = (2.5, 0.40, 0.22)
LO, HI, CMAP_LO = 1e-4, 2e-2, 0.10
ZERO = "#3A3A3A"
PARAMS = [("post_log_gain", "output gain"), ("pre_log_gain", "input gain"), ("threshold_mv", "threshold")]


def px(mm):
    return int(round(mm * fs.MM * fs.DPI))


def colour(v):
    t = (np.log10(np.clip(v, LO, HI)) - np.log10(LO)) / (np.log10(HI) - np.log10(LO))
    return pal.MAGNITUDE_ON_DARK(CMAP_LO + (1 - CMAP_LO) * t)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = sc.cells()
    groups = pd.read_csv(sc.FIGURE_DATA / "fd_group_change.csv").set_index("group")
    s, xyz = sc.somata(df)
    g = s["parameter_group"].to_numpy()
    meta = []
    for key, lab in PARAMS:
        v = groups[f"mean_normalized_abs_{key}"].reindex(g).to_numpy(float)
        W, H = px(MAP_W), px(BAND_H)
        view = rp.fit_view(xyz, W, H, unfold=True, margin=0.012, ss=3)
        tile = rp.Tile(view)
        sc.shell(tile, strength=0.11, fill=0.010)
        rgb = rp.srgb_to_linear(colour(v)[:, :3])
        rgb[v <= 0] = rp.hex_linear(ZERO)
        tile.discs(xyz, 1.5, rgb, 0.75, order="depth", label=f"cells coloured by the group mean |change| of {lab}")
        img = tile.resolve(bloom=BLOOM)
        info = dict(parameter=key, label=lab, cells=int(len(v)), zero_cells=int((v <= 0).sum()), lo=LO, hi=HI, cmap_lo=CMAP_LO, zero_colour=ZERO,
                    median_cell_value=float(np.median(v)), scale_um_per_px=view.um_per_px)
        tile.save(OUT / f"{key}.png", img, info)
        meta.append(dict(name=key, **info))
    (OUT / "tiles.json").write_text(json.dumps(dict(band_mm=BAND_H, map_mm=[MAP_W, BAND_H], tiles=meta), indent=1), encoding="utf-8", newline="\n")
    print(f"done {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
