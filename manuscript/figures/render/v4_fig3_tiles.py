"""v4 Fig. 3 dark band (panel A): "which one is the memory circuit?" across 174 mm, in the Fig. 1 visual language.

A1     the scene-1 cue ensemble (top 1% of cells by |cue difference| at the decision; X6/X7) with its internal connections,
       cells coloured by the cue that depolarises them more, on the unfolded CNS;
A2-A4  three of its 20 matched random sets (same cell types, distance from the inputs, degree and readout role), drawn by the
       same rule;
right  one neuron of each of the four cell types with the largest mean |cue difference| (set C of fd_neuron_selection).

Data: manuscript/results/plan_v2/fig3_band_sets.csv (P12), figure_data (fd_cells, fd_soma_state, fd_neuron_selection);
connections from the connectome the X6/X7 sets were drawn on (flybrain/cache/W_post_pre.npz).
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_fig3_tiles.py
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
from lib import figstyle as fs, palette as pal  # noqa: E402
from render import project as rp, scenes as sc  # noqa: E402

fs.use_cell()
OUT = FIG / "renders" / "v4" / "fig3"
PLAN = rp.RESULTS / "plan_v2"
BAND_H = 48.0
TILE_W = 0.703 * BAND_H                      # the unfolded CNS at the aspect of the v1 state tiles
TITLE_MM = 4.6
NEURON_AREA = fs.FULL - 4 * (TILE_W + fs.GUTTER)
NEURON_W, NEURON_H = (NEURON_AREA - fs.GUTTER) / 2, (BAND_H - fs.GUTTER) / 2
BLOOM = (2.5, 0.40, 0.22)
EDGE_ALPHA, EDGE_UM, EDGE_COLOUR = 0.009, 0.40, "#C9D7E6"
CELL_UM, BACK_UM = 2.2, 3.0
NEUTRAL = "#8A8A8A"                          # cells without a cue difference


def px(mm):
    return int(np.ceil(mm * fs.MM * fs.DPI))            # at least 600 ppi at the printed size


def view_for(xyz):
    W, H, top = px(TILE_W), px(BAND_H), px(TITLE_MM)
    v0 = rp.fit_view(xyz, W, H - top, unfold=True, margin=0.010, ss=3)
    return rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 3)


def set_tile(view, xyz_all, pos, dv, cells, A):
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.11, fill=0.010)
    tile.glow(xyz_all, 1.0, "#5A5A5A", 0.08 * sc.density_weights(view, xyz_all), label="all cell bodies")
    sub = A[cells][:, cells].tocoo()
    post, pre = cells[sub.row], cells[sub.col]
    located = ~np.isnan(pos[:, 0])
    ok = located[pre] & located[post]
    tile.segments(pos[pre[ok]], pos[post[ok]], EDGE_UM, EDGE_COLOUR, EDGE_ALPHA, order="given", label="connections within the set")
    c = cells[located[cells]]
    c = c[np.argsort(np.abs(dv[c]), kind="stable")]
    sign = np.sign(dv[c]).astype(int)
    lut = {-1: rp.hex_linear(pal.CUE_ON_DARK[0]), 1: rp.hex_linear(pal.CUE_ON_DARK[1]), 0: rp.hex_linear(NEUTRAL)}
    tile.discs(pos[c], BACK_UM, "#000000", 0.85, order="given", label="dark backing")
    tile.discs(pos[c], CELL_UM, np.array([lut[s] for s in sign]), 1.0, order="given", label="set cells by preferred cue")
    return tile, dict(cells=int(len(cells)), located=int(len(c)), connections=int(sub.nnz), drawn_connections=int(ok.sum()),
                      cue0=int((dv[cells] < 0).sum()), cue1=int((dv[cells] > 0).sum()), no_difference=int((dv[cells] == 0).sum()),
                      scale_um_per_px=view.um_per_px)


def neuron_tiles(df):
    sel = pd.read_csv(sc.FIGURE_DATA / "fd_neuron_selection.csv")
    d = sel[sel["selection_set"] == "C_cue_type"].sort_values(["rank", "body_id"]).drop_duplicates("rank").head(4)
    cells = df.set_index("body_id")
    meta = []
    for _, r in d.iterrows():
        body = int(r["body_id"])
        c = cells.loc[body]
        soma = c[["soma_x_um", "soma_y_um", "soma_z_um"]].to_numpy(np.float64)
        tile, img, colour = sc.neuron_tile(df, body, c["superclass"], soma, px(NEURON_W), px(NEURON_H), BLOOM)
        name = f"type{int(r['rank'])}"
        info = dict(body_id=body, cell=int(c["cell"]), type=r["type"], superclass=c["superclass"], rank=int(r["rank"]) + 1,
                    type_mean_abs_dv_mv=float(r["score"]), scale_um_per_px=tile.view.um_per_px, scale_bar_um=sc.scale_bar_um(tile.view),
                    colour=colour, skeleton=sc.skeleton_path(body).relative_to(rp.ROOT).as_posix())
        tile.save(OUT / f"{name}.png", img, info)
        meta.append(dict(name=name, **info))
    return meta


def main():
    t = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = sc.cells()
    if not np.array_equal(df["cell"].to_numpy(), np.arange(len(df))):
        raise RuntimeError("cells not in model order")
    s, xyz_all = sc.somata(df)
    pos = df[["soma_x_um", "soma_y_um", "soma_z_um"]].to_numpy(np.float64)
    dv = df["dv_m0_mv"].to_numpy(np.float64)
    W = sp.load_npz(rp.ROOT / "flybrain" / "cache" / "W_post_pre.npz").tocsr()
    W.eliminate_zeros()
    A = (W != 0).tocsr()
    sets = pd.read_csv(PLAN / "fig3_band_sets.csv")
    view = view_for(xyz_all)
    tiles = []
    for name in ["ensemble", "matched 1", "matched 2", "matched 3"]:
        cells = np.sort(sets.loc[sets["set"] == name, "cell"].to_numpy())
        tile, info = set_tile(view, xyz_all, pos, dv, cells, A)
        img = tile.resolve(bloom=BLOOM)
        fname = name.replace(" ", "")
        tile.save(OUT / f"{fname}.png", img, info)
        tiles.append(dict(name=fname, set=name, **info))
        print(f"{name} {time.time() - t:.1f}s", flush=True)
    neurons = neuron_tiles(df)
    (OUT / "tiles.json").write_text(json.dumps(dict(band_mm=BAND_H, tile_mm=[TILE_W, BAND_H], title_mm=TITLE_MM, neuron_mm=[NEURON_W, NEURON_H],
                                                    sets=tiles, neurons=neurons), indent=1), encoding="utf-8", newline="\n")
    print(f"done {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
