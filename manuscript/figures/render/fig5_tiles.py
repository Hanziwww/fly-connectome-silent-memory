"""Fig. 5a: the two cue states as mirror images (M0 readout-only, last window at 160-ms delay and 4 prefix windows).

The cue difference dv = v(cue 1) - v(cue 0) splits the cells: those more depolarised after cue 0 (dv < -1 mV, lit in
the cue-0 colour) and those more depolarised after cue 1 (dv > +1 mV, cue-1 colour). A whole-state swap exchanges
these two patterns between histories. Same glass-shell look as Figs 1 and 4.

Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\fig5_tiles.py
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

OUT = FIG / "renders" / "fig5"
TILE_H = 47.0
TILE_W = 0.703 * TILE_H
TITLE_MM = 4.4
BLOOM = (2.5, 0.45, 0.20)


def px(mm):
    return int(round(mm * fs.MM * fs.DPI))


def tile_for(xyz):
    W, H, top = px(TILE_W), px(TILE_H), px(TITLE_MM)
    v0 = rp.fit_view(xyz, W, H - top, unfold=True, margin=0.010, ss=3)
    view = rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 3)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.11, fill=0.010)
    tile.glow(xyz, 1.0, "#555555", 0.16 * sc.density_weights(view, xyz), label="all cell bodies")
    return tile


def main():
    t = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = sc.cells()
    s, xyz = sc.somata(df)
    dv = s["dv_m0_mv"].to_numpy()
    meta = {}
    for cue, sign in ((0, -1.0), (1, 1.0)):
        keep = sign * dv > sc.DV_MIN
        mag = np.abs(dv[keep])
        u = np.clip(np.log10(mag / sc.DV_MIN) / np.log10(sc.DV_MAX / sc.DV_MIN), 0, 1)
        tile = tile_for(xyz)
        tile.glow(xyz[keep], 1.2, pal.CUE_ON_DARK[cue], 0.55 + 0.75 * u, label=f"more depolarised after cue {cue}")
        img = tile.resolve(bloom=BLOOM)
        tile.save(OUT / f"cue{cue}.png", img, dict(cells=int(keep.sum()), threshold_mv=sc.DV_MIN, scale_um_per_px=tile.view.um_per_px))
        meta[f"cue{cue}"] = int(keep.sum())
    (OUT / "tiles.json").write_text(json.dumps(dict(tile_mm=[TILE_W, TILE_H], title_mm=TITLE_MM, **meta), indent=1),
                                    encoding="utf-8", newline="\n")
    print(f"done {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
