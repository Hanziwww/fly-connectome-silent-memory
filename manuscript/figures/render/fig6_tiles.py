"""Fig. 6 dark band (panels a and b).

a  the cue ensemble of the M0 readout-only state (hidden goal, 160-ms delay, 4 prefix windows): the 1% of all cells with
   the largest |cue 1 - cue 0| membrane-potential difference, coloured by the cue that depolarises them more, and the
   connections among them (figure_data/fd_ensemble_edges) drawn as straight soma-to-soma lines in the sender's colour,
   over every located cell body, on the unfolded CNS of Fig. 4.
b  one neuron of each of the eight cell types with the largest mean |dv| in that state (set C of fd_neuron_selection).

Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\fig6_tiles.py
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

OUT = FIG / "renders" / "fig6"
BAND_H = 56.0
MAP_W = 42.0
TILE_W = (fs.FULL - MAP_W - 4 * fs.GUTTER) / 4
TILE_H = (BAND_H - fs.GUTTER) / 2
BLOOM = (2.5, 0.40, 0.22)
TOP_FRACTION = 0.01
EDGE_ALPHA, EDGE_UM, EDGE_COLOUR = 0.009, 0.40, "#C9D7E6"
CELL_UM, BACK_UM = 2.5, 3.3


def px(mm):
    return int(round(mm * fs.MM * fs.DPI))


def ensemble(df):
    """Cells of the 1% ensemble (same rule as P6b: top |dv| over all cells, stable order) and their preference."""
    assert np.array_equal(df["cell"].to_numpy(), np.arange(len(df)))
    dv = df["dv_m0_mv"].to_numpy(np.float64)
    k = int(np.ceil(TOP_FRACTION * len(dv)))
    top = np.sort(np.argsort(-np.abs(dv), kind="mergesort")[:k])
    return dv, top


def ensemble_map(df):
    dv, top = ensemble(df)
    edges = pd.read_csv(sc.FIGURE_DATA / "fd_ensemble_edges.csv")
    members = set(top.tolist())
    assert set(edges["pre_cell"]).issubset(members) and set(edges["post_cell"]).issubset(members)
    s, xyz_all = sc.somata(df)
    W, H = px(MAP_W), px(BAND_H)
    view = rp.fit_view(xyz_all, W, H, unfold=True, margin=0.012, ss=3)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.11, fill=0.010)
    tile.glow(xyz_all, 1.0, "#5A5A5A", 0.14 * sc.density_weights(view, xyz_all), label="all cell bodies")
    pos = df[["soma_x_um", "soma_y_um", "soma_z_um"]].to_numpy(np.float64)
    located = ~np.isnan(pos[:, 0])
    pre, post = edges["pre_cell"].to_numpy(), edges["post_cell"].to_numpy()
    ok = located[pre] & located[post]
    tile.segments(pos[pre[ok]], pos[post[ok]], EDGE_UM, EDGE_COLOUR, EDGE_ALPHA, order="given", label="connections within the ensemble")
    cells = top[located[top]]
    pref = (dv[cells] > 0).astype(int)                            # 1: cue 1 depolarises more, 0: cue 0
    cells = cells[np.lexsort((np.abs(dv[cells]), pref))]           # cue-0 cells first, cue-1 cells (fewer) last; larger |dv| later
    pref = (dv[cells] > 0).astype(int)
    tile.discs(pos[cells], BACK_UM, "#000000", 0.85, order="given", label="dark backing")
    ccol = np.array([rp.hex_linear(pal.CUE_ON_DARK[0]), rp.hex_linear(pal.CUE_ON_DARK[1])])[pref]
    tile.discs(pos[cells], CELL_UM, ccol, 1.0, order="given", label="ensemble cells by preferred cue")
    img = tile.resolve(bloom=BLOOM)
    info = dict(cells=int(len(top)), located=int(len(cells)), cue0=int((dv[top] < 0).sum()), cue1=int((dv[top] > 0).sum()),
                connections=int(len(edges)), drawn_connections=int(ok.sum()), min_abs_dv_mv=float(np.abs(dv[top]).min()),
                cells_all=int(len(df)), located_all=int(len(xyz_all)), scale_um_per_px=view.um_per_px, edge_alpha=EDGE_ALPHA,
                edge_width_um=EDGE_UM, edge_colour=EDGE_COLOUR, cell_radius_um=CELL_UM)
    return tile, img, info


def neuron_tiles(df):
    sel = pd.read_csv(sc.FIGURE_DATA / "fd_neuron_selection.csv")
    d = sel[sel["selection_set"] == "C_cue_type"].sort_values(["rank", "body_id"]).drop_duplicates("rank")
    cells = df.set_index("body_id")
    meta = []
    for _, r in d.iterrows():
        body = int(r["body_id"])
        c = cells.loc[body]
        soma = c[["soma_x_um", "soma_y_um", "soma_z_um"]].to_numpy(np.float64)
        tile, img, colour = sc.neuron_tile(df, body, c["superclass"], soma, px(TILE_W), px(TILE_H), BLOOM)
        name = f"type{int(r['rank'])}"
        info = dict(body_id=body, cell=int(c["cell"]), type=r["type"], superclass=c["superclass"], role=c["role"], transmitter=c["transmitter"],
                    rank=int(r["rank"]) + 1, type_mean_abs_dv_mv=float(r["score"]), cell_dv_mv=float(c["dv_m0_mv"]),
                    scale_um_per_px=tile.view.um_per_px, scale_bar_um=sc.scale_bar_um(tile.view), colour=colour,
                    skeleton=sc.skeleton_path(body).relative_to(rp.ROOT).as_posix())
        tile.save(OUT / f"{name}.png", img, info)
        meta.append(dict(name=name, **info))
    return meta


def main():
    t = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = sc.cells()
    tile, img, info = ensemble_map(df)
    tile.save(OUT / "ensemble.png", img, info)
    meta = neuron_tiles(df)
    (OUT / "tiles.json").write_text(json.dumps(dict(band_mm=BAND_H, map_mm=[MAP_W, BAND_H], tile_mm=[TILE_W, TILE_H],
                                                    map=dict(name="ensemble", **info), tiles=meta), indent=1), encoding="utf-8", newline="\n")
    print(f"done {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
