"""v4 Figure S6 dark film strip: the route network at every 10% of removal, random (draw 0 of the graph analysis) and hubs
first (cells in descending order of connections), in the colour language of Fig. 6A (magma by route traffic on Fig. 6A's log
scale, readout cells in cyan). Each strip starts from the intact network.

Data: manuscript/results/plan_v2/s6_routes.csv (P12, gated against fig6_routes and the percolation table); soma positions
from figure_data/fd_cells.csv; the traffic scale from renders/v4/fig6/tiles.json.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_s6_tiles.py
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

fs.use_cell()
OUT = FIG / "renders" / "v4" / "s6"
PLAN = rp.RESULTS / "plan_v2"
F6 = json.loads((FIG / "renders" / "v4" / "fig6" / "tiles.json").read_text(encoding="utf-8"))
LABEL_W = 11.0                                # the strip names, left of the tiles
TILE_W = (fs.FULL - LABEL_W - 10 * fs.GUTTER) / 10
TITLE_MM = 6.0
TILE_H = TILE_W / 0.80 + TITLE_MM             # the unfolded CNS at the aspect of the Fig. 6A tiles
BLOOM = (2.5, 0.40, 0.22)
LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
KEYS = ["intact"] + [f"random_{k}" for k in LEVELS] + [f"hubs_{k}" for k in LEVELS]
READOUT = "#7FD3FF"                           # as Fig. 6A


def px(mm):
    return int(round(mm * fs.MM * fs.DPI))


def base(xyz):
    W, H, top = px(TILE_W), px(TILE_H), px(TITLE_MM)
    v0 = rp.fit_view(xyz, W, H - top, unfold=True, margin=0.010, ss=3)
    view = rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 3)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.11, fill=0.010)
    tile.glow(xyz, 1.0, "#5A5A5A", 0.05 * sc.density_weights(view, xyz), label="all cell bodies")
    return tile


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fd = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False)
    rt = pd.read_csv(PLAN / "s6_routes.csv").drop(columns=["source_file"])
    df = fd.drop(columns=["role"]).merge(rt, on="cell", validate="one_to_one")
    s, xyz = sc.somata(df)
    _, fxyz = sc.somata(fd)
    lo, hi = F6["log10_traffic_range"]
    m_lo, m_hi = F6["magma"]
    ro = s["role"].eq("readout").to_numpy()
    meta = {}
    for k in KEYS:
        t = s[f"traffic_{k}"].to_numpy(float)
        on = t > 0
        u = np.clip((np.log10(t[on]) - lo) / (hi - lo), 0, 1)
        tile = base(fxyz)
        rgb = rp.srgb_to_linear(pal.MAGNITUDE_ON_DARK(m_lo + (m_hi - m_lo) * u)[:, :3])
        w = 0.7 * (0.05 + 0.7 * u ** 2) * sc.density_weights(tile.view, xyz[on], power=0.6)   # 0.7: the smaller tiles saturate sooner
        tile.glow(xyz[on], 1.2, rgb, w, label=f"route cells, {k}")
        tile.glow(xyz[ro], 1.1, READOUT, 0.35 * sc.density_weights(tile.view, xyz[ro], power=0.5), label="readout cells")
        img = tile.resolve(bloom=BLOOM)
        info = dict(map=k, route_cells_located=int(on.sum()), readout_located=int(ro.sum()), log10_traffic_range=[lo, hi],
                    scale_um_per_px=tile.view.um_per_px)
        tile.save(OUT / f"routes_{k}.png", img, info)
        meta[k] = info
        print(f"{k} {time.time() - t0:.1f}s", flush=True)
    (OUT / "tiles.json").write_text(json.dumps(dict(tile_mm=[TILE_W, TILE_H], title_mm=TITLE_MM, label_mm=LABEL_W, levels=LEVELS, keys=KEYS,
                                                    log10_traffic_range=[lo, hi], magma=[m_lo, m_hi], tiles=meta), indent=1),
                                    encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
