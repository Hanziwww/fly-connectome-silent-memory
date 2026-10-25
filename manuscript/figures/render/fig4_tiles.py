"""Fig. 4a: three unfolded-CNS tiles of the cue-related state difference (M0 readout-only, hidden goal, last window
at 160-ms delay and 4 prefix windows), in the Fig. 1 visual language (glass shell, luminous cells, bloom):

* signed voltage difference (cue 1 - cue 0) for cells with |dv| > 1 mV on the dark-centred cue map;
* maximum projection of |dv| (magma, log 0.01-10 mV) over all cells;
* cells that spike in the window.

Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\fig4_tiles.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import figstyle as fs, palette as pal  # noqa: E402
from render import project as rp, scenes as sc  # noqa: E402

OUT = FIG / "renders" / "fig4"
TILE_H = 62.0
TILE_W = 0.703 * TILE_H
BLOOM = (2.5, 0.40, 0.22)
SPIKE = "#FFE7A3"


def px(mm):
    return int(round(mm * fs.MM * fs.DPI))


TITLE_MM = 5.0                     # black strip at the top of each tile for its title


def base(xyz, background_alpha=0.20):
    W, H, top = px(TILE_W), px(TILE_H), px(TITLE_MM)
    v0 = rp.fit_view(xyz, W, H - top, unfold=True, margin=0.010, ss=3)
    view = rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 3)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.11, fill=0.010)
    if background_alpha:
        tile.glow(xyz, 1.0, "#5A5A5A", background_alpha * sc.density_weights(view, xyz), label="all cell bodies")
    return tile


def signed(s, xyz):
    dv = s["dv_m0_mv"].to_numpy()
    keep = np.abs(dv) > sc.DV_MIN
    tile = base(xyz, 0.16)
    rgb, alpha = sc.signed_colours(dv[keep])
    tile.discs(xyz[keep], 1.8, rgb, alpha, order="value", values=np.abs(dv[keep]), label="|dv| > 1 mV")
    img = tile.resolve(bloom=BLOOM)
    tile.save(OUT / "signed.png", img, dict(cells_above=int(keep.sum()), positive=int((dv > sc.DV_MIN).sum()),
                                           negative=int((dv < -sc.DV_MIN).sum()), cells=int(len(dv)), dv_min=sc.DV_MIN, dv_max=sc.DV_MAX,
                                           scale_um_per_px=tile.view.um_per_px))


def magnitude(s, xyz):
    dv = np.abs(s["dv_m0_mv"].to_numpy())
    tile = base(xyz, 0.0)
    tile.max_projection(xyz, 1.7, dv, pal.MAGNITUDE_ON_DARK, 0.01, 10.0, log=True, label="|dv| maximum projection")
    img = tile.resolve(bloom=BLOOM)
    tile.save(OUT / "magnitude.png", img, dict(vmin=0.01, vmax=10.0, cells=int(len(dv)), scale_um_per_px=tile.view.um_per_px))


def spiking(s, xyz):
    sp = s["spikes_m0"].to_numpy() > 0
    tile = base(xyz, 0.26)
    tile.glow(xyz[sp], 1.3, SPIKE, 1.1, label="spiking cells")
    img = tile.resolve(bloom=BLOOM)
    tile.save(OUT / "spiking.png", img, dict(spiking=int(sp.sum()), located=int(len(sp)), scale_um_per_px=tile.view.um_per_px,
                                            colour=SPIKE))


def main():
    t = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = sc.cells()
    s, xyz = sc.somata(df)
    signed(s, xyz)
    magnitude(s, xyz)
    spiking(s, xyz)
    all_dv = df["dv_m0_mv"].to_numpy()
    (OUT / "tiles.json").write_text(json.dumps(dict(tile_mm=[TILE_W, TILE_H], title_mm=TITLE_MM, cells_all=int(len(all_dv)),
                                                    spiking_all=int((df["spikes_m0"] > 0).sum())), indent=1), encoding="utf-8", newline="\n")
    print(f"done {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
