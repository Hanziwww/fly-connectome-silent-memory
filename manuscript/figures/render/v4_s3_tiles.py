"""v4 Figure S3 dark band: the scene-2 cue ensemble (top 1% of cells by |cue difference| at the decision) with its internal
connections beside three of its 20 matched random sets drawn by the same rule, rendered as Fig. 3A (cells coloured by the cue
that depolarises them more in scene 2).

Data: manuscript/results/plan_v2/s3_band_sets.csv (P12), figure_data (fd_cells, fd_soma_state); connections from the connectome
the sets were drawn on (flybrain/cache/W_post_pre.npz).
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_s3_tiles.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import figstyle as fs  # noqa: E402
from render import project as rp, scenes as sc  # noqa: E402
from render import v4_fig3_tiles as f3  # noqa: E402

fs.use_cell()
OUT = FIG / "renders" / "v4" / "s3"
PLAN = rp.RESULTS / "plan_v2"
TILE_W = (fs.FULL - 3 * fs.GUTTER) / 4
BAND_H = 56.0
TITLE_MM = 5.0


def view_for(xyz):
    W, H, top = f3.px(TILE_W), f3.px(BAND_H), f3.px(TITLE_MM)
    v0 = rp.fit_view(xyz, W, H - top, unfold=True, margin=0.010, ss=3)
    return rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 3)


def main():
    t = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = sc.cells()
    if not np.array_equal(df["cell"].to_numpy(), np.arange(len(df))):
        raise RuntimeError("cells not in model order")
    _, xyz_all = sc.somata(df)
    pos = df[["soma_x_um", "soma_y_um", "soma_z_um"]].to_numpy(np.float64)
    dv = df["dv_s1_mv"].to_numpy(np.float64)                     # scene 2 (repository key s1)
    W = sp.load_npz(rp.ROOT / "flybrain" / "cache" / "W_post_pre.npz").tocsr()
    W.eliminate_zeros()
    A = (W != 0).tocsr()
    sets = pd.read_csv(PLAN / "s3_band_sets.csv")
    view = view_for(xyz_all)
    tiles = []
    for name in ["ensemble", "matched 1", "matched 2", "matched 3"]:
        cells = np.sort(sets.loc[sets["set"] == name, "cell"].to_numpy())
        tile, info = f3.set_tile(view, xyz_all, pos, dv, cells, A)
        img = tile.resolve(bloom=f3.BLOOM)
        fname = name.replace(" ", "")
        tile.save(OUT / f"{fname}.png", img, info)
        tiles.append(dict(name=fname, set=name, **info))
        print(f"{name} {time.time() - t:.1f}s", flush=True)
    (OUT / "tiles.json").write_text(json.dumps(dict(band_mm=BAND_H, tile_mm=[TILE_W, BAND_H], title_mm=TITLE_MM, sets=tiles), indent=1),
                                    encoding="utf-8", newline="\n")
    print(f"done {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
