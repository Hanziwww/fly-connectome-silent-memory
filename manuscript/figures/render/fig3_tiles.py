"""Fig. 3 dark band (panels g and h).

g  whole-CNS map of where learning acts: every located cell coloured by the consistency score of its parameter group
   (mean within-map percentile of combined normalized |change| over the 24 maps; figure_data/fd_group_change), on the
   unfolded CNS of Fig. 4. Superclass-level groups (cells without a type) are grey.
h  one neuron of each of the eight most consistently changed type-level groups (set D of figure_data/fd_neuron_selection),
   on the same view, cropped to the neuron with its local context.

Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\fig3_tiles.py [--variants]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import figstyle as fs, palette as pal  # noqa: E402
from render import project as rp, scenes as sc  # noqa: E402

OUT = FIG / "renders" / "fig3"
BAND_H = 56.0
MAP_W = 42.0
TILE_W = (fs.FULL - MAP_W - 4 * fs.GUTTER) / 4
TILE_H = (BAND_H - fs.GUTTER) / 2
BLOOM = (2.5, 0.40, 0.22)
SCORE_LO, SCORE_HI = 0.25, 0.95                     # colour range of the consistency score
CMAP_LO = 0.10                                      # lowest magma level used (dark purple, still visible on black)
GREY = "#4A4A4A"
BAR_UM = (20, 50, 100, 200)


def px(mm):
    return int(round(mm * fs.MM * fs.DPI))


def score_colour(v):
    t = np.clip((np.asarray(v, float) - SCORE_LO) / (SCORE_HI - SCORE_LO), 0, 1)
    return pal.MAGNITUDE_ON_DARK(CMAP_LO + (1 - CMAP_LO) * t)


def groups_table():
    return pd.read_csv(sc.FIGURE_DATA / "fd_group_change.csv").set_index("group")


def featured(df):
    sel = pd.read_csv(sc.FIGURE_DATA / "fd_neuron_selection.csv")
    d = sel[sel["selection_set"] == "D_change_group"].sort_values(["rank", "body_id"]).drop_duplicates("rank")
    return d.merge(df[["body_id", "type", "superclass", "soma_x_um", "soma_y_um", "soma_z_um"]], on="body_id", how="left",
                   suffixes=("", "_cell"))


def learning_map(df, groups, mode="depth"):
    s, xyz = sc.somata(df)
    g = s["parameter_group"].to_numpy()
    v = groups["consistency_score"].reindex(g).to_numpy()
    typed = groups["group_level"].reindex(g).to_numpy() == "type"
    W, H = px(MAP_W), px(BAND_H)
    view = rp.fit_view(xyz, W, H, unfold=True, margin=0.012, ss=3)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.11, fill=0.010)
    rgb = rp.srgb_to_linear(score_colour(v)[:, :3])
    rgb[~typed] = rp.hex_linear(GREY)
    top = groups["top_1pct"].astype(str).str.lower().eq("true").reindex(g).to_numpy()
    if mode == "depth":
        tile.discs(xyz, 1.5, rgb, 0.75, order="depth", label="cells, coloured by group consistency (front first)")
        tile.discs(xyz[top], 1.8, rgb[top], 1.0, order="depth", label="cells of the top-1% groups, drawn on top")
    elif mode == "value":
        tile.discs(xyz, 1.5, rgb, 0.9, order="value", values=np.where(typed, v, -1.0), label="cells, highest consistency on top")
    else:                                          # regional mean projection, then the top-1% groups' cells as points
        ss = view.ss
        p_x, p_y, _, _ = view.pixels(xyz[typed])
        ix, iy = np.floor(p_x / ss).astype(np.int64), np.floor(p_y / ss).astype(np.int64)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        num = np.bincount(iy[ok] * W + ix[ok], weights=v[typed][ok], minlength=W * H).reshape(H, W)
        cnt = np.bincount(iy[ok] * W + ix[ok], minlength=W * H).reshape(H, W).astype(float)
        sig = 4.0 / view.um_per_px
        num_s, cnt_s = gaussian_filter(num, sig), gaussian_filter(cnt, sig)
        mean = num_s / np.maximum(cnt_s, 1e-9)
        alpha = 1 - np.exp(-cnt_s / 0.15)
        col = rp.srgb_to_linear(score_colour(mean)[..., :3])
        up = np.repeat(np.repeat(col * alpha[..., None], ss, 0), ss, 1)
        a_up = np.repeat(np.repeat(alpha, ss, 0), ss, 1)
        tile.img = tile.img * (1 - 0.85 * a_up[..., None]) + 0.85 * up
        tile.log["layers"].append(dict(kind="mean_projection", sigma_um=4.0, n=int(typed.sum())))
        tile.discs(xyz[top], 1.7, rgb[top], 0.95, order="depth", label="cells of the top-1% groups")
    img = tile.resolve(bloom=BLOOM)
    return tile, img, dict(cells=int(len(v)), typed_cells=int(typed.sum()), top1pct_cells=int(top.sum()), mode=mode)


def neuron_tile(df, r, groups):
    body = int(r["body_id"])
    soma = r[["soma_x_um", "soma_y_um", "soma_z_um"]].to_numpy(np.float64)
    tile, img, colour = sc.neuron_tile(df, body, r["superclass"], soma, px(TILE_W), px(TILE_H), BLOOM)
    view = tile.view
    g = groups.loc[int(r["parameter_group"])]
    bar = sc.scale_bar_um(view, choices=BAR_UM)
    info = dict(body_id=body, type=r["type"], superclass=r["superclass"], group=r["label"], parameter_group=int(r["parameter_group"]),
                rank=int(g["consistency_rank"]), score=float(g["consistency_score"]), n_cells=int(g["n_cells"]), membership=g["membership"],
                transmitter=g["transmitter"], scale_um_per_px=view.um_per_px, scale_bar_um=bar, colour=colour,
                skeleton=sc.skeleton_path(body).relative_to(rp.ROOT).as_posix())
    return tile, img, info


def main(variants=False):
    t = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = sc.cells()
    groups = groups_table()
    feat = featured(df)
    feat["parameter_group"] = feat["body_id"].map(df.set_index("body_id")["parameter_group"])
    if variants:
        for mode in ("depth", "value", "mean"):
            tile, img, extra = learning_map(df, groups, mode)
            tile.save(OUT / "_variants" / f"map_{mode}.png", img, extra)
            print("variant", mode, f"{time.time() - t:.1f}s", flush=True)
        return
    tile, img, extra = learning_map(df, groups, "depth")
    view = tile.view
    marks = []
    for _, r in feat.iterrows():
        p_x, p_y, _, _ = view.pixels(np.array([[r["soma_x_um"], r["soma_y_um"], r["soma_z_um"]]]))
        marks.append(dict(rank=int(r["rank"]) + 1, type=r["type"], body_id=int(r["body_id"]), x_px=float(p_x[0] / view.ss),
                          y_px=float(p_y[0] / view.ss)))
    extra.update(score_lo=SCORE_LO, score_hi=SCORE_HI, cmap_lo=CMAP_LO, grey=GREY, scale_um_per_px=view.um_per_px, marks=marks)
    tile.save(OUT / "map.png", img, extra)
    meta = []
    for _, r in feat.iterrows():
        tile, img, info = neuron_tile(df, r, groups)
        name = f"group{int(r['rank'])}"
        tile.save(OUT / f"{name}.png", img, info)
        meta.append(dict(name=name, **info))
    (OUT / "tiles.json").write_text(json.dumps(dict(band_mm=BAND_H, map_mm=[MAP_W, BAND_H], tile_mm=[TILE_W, TILE_H], map=dict(name="map", **extra),
                                                    tiles=meta), indent=1), encoding="utf-8", newline="\n")
    print(f"done {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main(variants="--variants" in sys.argv)
