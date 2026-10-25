"""v4 Figure S2 dark gallery: the cue difference at the decision in five state sets, on the unfolded CNS (as Fig. 5A).

Each cell's membrane-potential difference between the cues (cue 1 - cue 0) in the decision window of the 160-ms-delay,
4-prefix history (figure_data/fd_soma_state), on the dark-centred cue map with log magnitude (0.3-10 mV): scene 1 readout
trained, scene 1 brain and readout trained, scene 2, and scene 3 in each of its two contexts.

Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_s2_tiles.py
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
OUT = FIG / "renders" / "v4" / "s2"
TILE_W = (fs.FULL - 4 * fs.GUTTER) / 5
TILE_H = 46.0
TITLE_MM = 7.4
BLOOM = (2.5, 0.40, 0.22)
DV_LO, DV_HI = 0.3, 10.0
SETS = ("m0", "m0_joint", "s1", "s2_o0", "s2_o180")


def px(mm):
    return int(np.ceil(mm * fs.MM * fs.DPI))


def base(xyz):
    W, H, top = px(TILE_W), px(TILE_H), px(TITLE_MM)
    v0 = rp.fit_view(xyz, W, H - top, unfold=True, margin=0.010, ss=3)
    view = rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 3)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.11, fill=0.010)
    tile.glow(xyz, 1.0, "#5A5A5A", 0.06 * sc.density_weights(view, xyz), label="all cell bodies")
    return tile


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False).merge(
        pd.read_csv(sc.FIGURE_DATA / "fd_soma_state.csv").drop(columns=["body_id"]), on="cell", validate="one_to_one")
    s, xyz = sc.somata(df)
    meta = {}
    for key in SETS:
        dv = s[f"dv_{key}_mv"].to_numpy(float)
        keep = np.abs(dv) > DV_LO
        u = np.clip(np.log10(np.abs(dv[keep]) / DV_LO) / np.log10(DV_HI / DV_LO), 0, 1)
        rgb = rp.srgb_to_linear(pal.SIGNED_ON_DARK(0.5 + np.sign(dv[keep]) * (0.26 + 0.24 * u))[:, :3])
        tile = base(xyz)
        tile.glow(xyz[keep], 1.5, rgb, (0.15 + 1.4 * u ** 2) * sc.density_weights(tile.view, xyz[keep]), label=f"|dv| > {DV_LO} mV")
        img = tile.resolve(bloom=BLOOM)
        info = dict(state_set=key, cells_located=int(keep.sum()), positive=int((dv > DV_LO).sum()), negative=int((dv < -DV_LO).sum()),
                    located=int(len(dv)), dv_lo=DV_LO, dv_hi=DV_HI, scale_um_per_px=tile.view.um_per_px)
        tile.save(OUT / f"decision_{key}.png", img, info)
        meta[key] = info
        print(f"{key} {time.time() - t0:.1f}s", flush=True)
    (OUT / "tiles.json").write_text(json.dumps(dict(tile_mm=[TILE_W, TILE_H], title_mm=TITLE_MM, dv_mv=[DV_LO, DV_HI], sets=list(SETS), tiles=meta),
                                               indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
