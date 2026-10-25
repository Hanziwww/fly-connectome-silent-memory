"""v4 Fig. 5 dark band (panel A): one cue, read out at four times. The membrane-potential difference, cue 1 minus cue 0,
of every cell at the first decision after a 40-, 160-, 320- and 640-ms delay (prefix 4; scene 1, readout trained), on the
unfolded CNS at one scale, in the Fig. 2 visual language (glass shell, luminous somata, bloom, the dark-centred cue map).
40 and 160 ms are trained timings, 320 and 640 ms are not; after 320 ms the strongest cells carry the opposite sign.

Data: manuscript/results/plan_v2/fig5_decision_cells.csv (P12, from the 1-ms records, gated against fu4); soma positions
from figure_data/fd_cells.csv.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_fig5_tiles.py
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
OUT = FIG / "renders" / "v4" / "fig5"
PLAN = rp.RESULTS / "plan_v2"
TILE_W = (fs.FULL - 3 * fs.GUTTER) / 4
TILE_H = TILE_W / 0.703                      # the unfolded CNS at the aspect of the Fig. 2 tiles
TITLE_MM = 7.6                               # black strip at the top of each tile for its two title lines
BLOOM = (2.5, 0.40, 0.22)
DV_LO, DV_HI = 0.3, 10.0                     # mV: the decision-time difference reaches 7-14 mV at the 99.9th percentile
TIMINGS = ((40, 4), (160, 4), (320, 4), (640, 4))


def px(mm):
    return int(round(mm * fs.MM * fs.DPI))


def base(xyz):
    W, H, top = px(TILE_W), px(TILE_H), px(TITLE_MM)
    v0 = rp.fit_view(xyz, W, H - top, unfold=True, margin=0.010, ss=3)
    view = rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 3)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.11, fill=0.010)
    tile.glow(xyz, 1.0, "#5A5A5A", 0.06 * sc.density_weights(view, xyz), label="all cell bodies")
    return tile


def code_colours(dv):
    """The dark-centred cue map on a log scale of |dv| (DV_LO-DV_HI): colour (linear RGB) and log position u in [0, 1]."""
    u = np.clip(np.log10(np.abs(dv) / DV_LO) / np.log10(DV_HI / DV_LO), 0, 1)
    t = 0.5 + np.sign(dv) * (0.26 + 0.24 * u)
    return rp.srgb_to_linear(pal.SIGNED_ON_DARK(t)[:, :3]), u


def decision(s, xyz, d, p):
    dv = s[f"dv_d{d}_p{p}_mv"].to_numpy()
    keep = np.abs(dv) > DV_LO
    tile = base(xyz)
    rgb, u = code_colours(dv[keep])
    w = (0.15 + 1.4 * u ** 2) * sc.density_weights(tile.view, xyz[keep])     # dense cortex keeps its hue
    tile.glow(xyz[keep], 1.5, rgb, w, label=f"|dv| > {DV_LO} mV")
    img = tile.resolve(bloom=BLOOM)
    tile.save(OUT / f"decision_d{d}_p{p}.png", img, dict(cells_located=int(keep.sum()), positive=int((dv > DV_LO).sum()),
                                                         negative=int((dv < -DV_LO).sum()), located=int(len(dv)), dv_lo=DV_LO,
                                                         dv_hi=DV_HI, scale_um_per_px=tile.view.um_per_px))


def main():
    t = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False).merge(
        pd.read_csv(PLAN / "fig5_decision_cells.csv"), on="cell", validate="one_to_one")
    s, xyz = sc.somata(df)
    for d, p in TIMINGS:
        decision(s, xyz, d, p)
        print(f"d{d} p{p} {time.time() - t:.1f}s", flush=True)
    (OUT / "tiles.json").write_text(json.dumps(dict(tile_mm=[TILE_W, TILE_H], title_mm=TITLE_MM, dv_mv=[DV_LO, DV_HI],
                                                    timings=[list(x) for x in TIMINGS]), indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
