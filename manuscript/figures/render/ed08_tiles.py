"""Extended Data Fig. 8 band: the signed cue difference of the membrane potential (cue 1 - cue 0; cells with |dv| > 1 mV on
the dark-centred cue map of Fig. 4a) for each of the six distinct state sets (last window at 160-ms delay and 4 prefix windows),
on the unfolded CNS at one shared view.

Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\ed08_tiles.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import figstyle as fs  # noqa: E402
from render import project as rp, scenes as sc  # noqa: E402

OUT = FIG / "renders" / "ed08"
COLS = 6
TILE_W = (fs.FULL - (COLS - 1) * fs.GUTTER) / COLS
TILE_H = 34.0
BLOOM = (2.5, 0.40, 0.22)
CELL_UM = 2.4
SETS = [("m0", "dv_m0_mv"), ("m0_joint", "dv_m0_joint_mv"), ("m0vis_visible", "dv_m0vis_visible_mv"), ("s1", "dv_s1_mv"), ("s2_o0", "dv_s2_o0_mv"),
        ("s2_o180", "dv_s2_o180_mv")]


def px(mm):
    return int(round(mm * fs.MM * fs.DPI))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = sc.cells()
    s, xyz = sc.somata(df)
    view = rp.fit_view(xyz, px(TILE_W), px(TILE_H), unfold=True, margin=0.015, ss=3)
    back = 0.16 * sc.density_weights(view, xyz)
    meta = []
    for key, col in SETS:
        dv = s[col].to_numpy()
        keep = np.abs(dv) > sc.DV_MIN
        tile = rp.Tile(view)
        sc.shell(tile, strength=0.11, fill=0.010)
        tile.glow(xyz, 1.0, "#5A5A5A", back, label="all cell bodies")
        rgb, alpha = sc.signed_colours(dv[keep])
        tile.discs(xyz[keep], CELL_UM, rgb, alpha, order="value", values=np.abs(dv[keep]), label="|dv| > 1 mV")
        img = tile.resolve(bloom=BLOOM)
        all_dv = df[col].to_numpy()
        info = dict(state_key=key, column=col, located=int(len(dv)), cells_above=int(keep.sum()), positive=int((dv > sc.DV_MIN).sum()),
                    negative=int((dv < -sc.DV_MIN).sum()), cells_all=int(len(all_dv)), above_all=int((np.abs(all_dv) > sc.DV_MIN).sum()),
                    dv_min=sc.DV_MIN, dv_max=sc.DV_MAX, cell_radius_um=CELL_UM, scale_um_per_px=view.um_per_px)
        tile.save(OUT / f"{key}.png", img, info)
        meta.append(dict(name=key, **info))
    (OUT / "tiles.json").write_text(json.dumps(dict(tile_mm=[TILE_W, TILE_H], cols=COLS, tiles=meta), indent=1), encoding="utf-8", newline="\n")
    print(f"done {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
