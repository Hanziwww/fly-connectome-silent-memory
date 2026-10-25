"""v4 Figure S5 dark gallery: one cue, read at every delay - the decision-time cue difference at all seven delays (prefix 4;
scene 1, readout trained) on the unfolded CNS, in the colour language of Fig. 5A (dark-centred cue map, log |dv| 0.3-10 mV).

Data: manuscript/results/plan_v2/s5_decision_cells.csv (P12, gated against fu4 and fig5_decision_cells); soma positions from fd_cells.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_s5_tiles.py
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
OUT = FIG / "renders" / "v4" / "s5"
PLAN = rp.RESULTS / "plan_v2"
TILE_W = (fs.FULL - 6 * fs.GUTTER) / 7
TILE_H = 38.0
TITLE_MM = 7.6
BLOOM = (2.5, 0.40, 0.22)
DV_LO, DV_HI = 0.3, 10.0
DELAYS = (0, 20, 40, 80, 160, 320, 640)


def px(mm):
    return int(np.ceil(mm * fs.MM * fs.DPI))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False).merge(pd.read_csv(PLAN / "s5_decision_cells.csv"), on="cell", validate="one_to_one")
    s, xyz = sc.somata(df)
    W, H, top = px(TILE_W), px(TILE_H), px(TITLE_MM)
    v0 = rp.fit_view(xyz, W, H - top, unfold=True, margin=0.010, ss=3)
    view = rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 3)
    meta = {}
    for d in DELAYS:
        dv = s[f"dv_d{d}_p4_mv"].to_numpy(float)
        keep = np.abs(dv) > DV_LO
        u = np.clip(np.log10(np.abs(dv[keep]) / DV_LO) / np.log10(DV_HI / DV_LO), 0, 1)
        rgb = rp.srgb_to_linear(pal.SIGNED_ON_DARK(0.5 + np.sign(dv[keep]) * (0.26 + 0.24 * u))[:, :3])
        tile = rp.Tile(view)
        sc.shell(tile, strength=0.11, fill=0.010)
        tile.glow(xyz, 1.0, "#5A5A5A", 0.06 * sc.density_weights(view, xyz), label="all cell bodies")
        tile.glow(xyz[keep], 1.5, rgb, (0.15 + 1.4 * u ** 2) * sc.density_weights(view, xyz[keep]), label=f"|dv| > {DV_LO} mV")
        img = tile.resolve(bloom=BLOOM)
        info = dict(delay_ms=d, prefix_windows=4, cells_located=int(keep.sum()), dv_lo=DV_LO, dv_hi=DV_HI, scale_um_per_px=view.um_per_px)
        tile.save(OUT / f"decision_d{d}.png", img, info)
        meta[str(d)] = info
        print(f"d{d} {time.time() - t0:.1f}s", flush=True)
    (OUT / "tiles.json").write_text(json.dumps(dict(tile_mm=[TILE_W, TILE_H], title_mm=TITLE_MM, delays=list(DELAYS), dv_mv=[DV_LO, DV_HI], tiles=meta), indent=1),
                                    encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
