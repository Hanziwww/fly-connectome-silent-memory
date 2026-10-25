"""v4 Figure S7 dark band: the disease's network activity at both doses, in the colour language of Fig. 7G (dark diverging
map of each cell's task-rate change, log in units of one spike per trial), and the C01-vulnerable cells.

A1 the uniform Kv4-loss lesion at dose 0.5; A2 the same at dose 1 (Fig. 7G2); A3 the Shal-patterned lesion of the same mean
at dose 1; A4 the C01-vulnerable cells (the 1% of memory-relevant cells with the largest rate change at dose 1), each glowing
in its cell group's colour over the other memory-relevant cells.

Data: manuscript/results/plan_v2/s7_cells.csv and s7_vulnerable.csv (P12, gated against fig7_cells and sc_annotation_sets);
soma positions from figure_data/fd_cells.csv; the rate scale from renders/v4/fig7/tiles.json.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_s7_tiles.py
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
from render import v4_fig7_tiles as f7  # noqa: E402

fs.use_cell()
OUT = FIG / "renders" / "v4" / "s7"
PLAN = rp.RESULTS / "plan_v2"
F7 = json.loads((FIG / "renders" / "v4" / "fig7" / "tiles.json").read_text(encoding="utf-8"))
TILE_W = (fs.FULL - 3 * fs.GUTTER) / 4
BAND_H = 54.0
TITLE_MM = 9.6
BLOOM = (2.8, 0.42, 0.20)


def px(mm):
    return int(np.ceil(mm * fs.MM * fs.DPI))


def base(all_xyz):
    W, H, top = px(TILE_W), px(BAND_H), px(TITLE_MM)
    v0 = rp.fit_view(all_xyz, W, H - top, unfold=True, margin=0.010, ss=3)
    view = rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 3)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.11, fill=0.010)
    tile.glow(all_xyz, 1.0, f7.GHOST, 0.05 * sc.density_weights(view, all_xyz), label="all cell bodies")
    return tile


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fd = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False)
    _, all_xyz = sc.somata(fd)
    cells = pd.read_csv(PLAN / "s7_cells.csv").drop(columns=["source_file"])
    s, xyz = sc.somata(fd.merge(cells, on="cell", validate="one_to_one"))
    unit, vmax = F7["rate_unit_hz"], F7["rate_max_hz"]
    meta = {}
    for key, col in (("uniform_0p5", "rate_uniform_dose0p5_hz"), ("uniform_1", "rate_uniform_dose1_hz"), ("shal_1", "rate_shal_dose1_hz")):
        d = (s[col] - s["rate_healthy_hz"]).to_numpy(float)
        on = d != 0
        tile = base(all_xyz)
        f7.signed(tile, xyz[on], d[on], vmax, np.ones(int(on.sum())), f"task-rate change, {key}", unit=unit)
        img = tile.resolve(bloom=BLOOM)
        info = dict(changed_located=int(on.sum()), faster_located=int((d > 0).sum()), slower_located=int((d < 0).sum()), rate_unit_hz=unit,
                    vmax_hz=vmax, scale_um_per_px=tile.view.um_per_px)
        tile.save(OUT / f"{key}.png", img, info)
        meta[key] = info
        print(f"{key} {time.time() - t0:.1f}s", flush=True)
    # the vulnerable cells in their cell group's colour
    vu = pd.read_csv(PLAN / "s7_vulnerable.csv")
    v, vxyz = sc.somata(fd.merge(vu[["cell"]], on="cell", validate="one_to_one"))
    grp = v["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other")
    tile = base(all_xyz)
    rgb = rp.srgb_to_linear(np.array([to_rgb(pal.ATLAS_ON_DARK[g]) for g in grp]))
    tile.glow(vxyz, 2.6, rgb, 3.0, label="C01-vulnerable cells")
    img = tile.resolve(bloom=BLOOM)
    info = dict(vulnerable_located=int(len(v)), groups=grp.value_counts().to_dict(), scale_um_per_px=tile.view.um_per_px)
    tile.save(OUT / "vulnerable.png", img, info)
    meta["vulnerable"] = info
    print(f"vulnerable {time.time() - t0:.1f}s", flush=True)
    (OUT / "tiles.json").write_text(json.dumps(dict(tile_mm=[TILE_W, BAND_H], title_mm=TITLE_MM, rate_unit_hz=unit, rate_max_hz=vmax, tiles=meta), indent=1),
                                    encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
