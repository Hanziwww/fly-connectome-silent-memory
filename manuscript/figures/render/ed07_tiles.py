"""Extended Data Fig. 7 band: the six candidate groups (top row) and their matched control sets (bottom row; set A of
figure_data/fd_neuron_selection), every cell's skeleton on the unfolded CNS of Fig. 4. All twelve tiles share one view,
one line width and one opacity, so candidate and matched sets carry equal visual weight. Skeletons are coloured by
superclass group (the atlas key of Figs. 1, 3 and 6) over a faint glow of every located cell body.

Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\ed07_tiles.py
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

OUT = FIG / "renders" / "ed07"
COLS = 6
TILE_W = (fs.FULL - (COLS - 1) * fs.GUTTER) / COLS
TILE_H = 34.0
BLOOM = (2.5, 0.40, 0.22)
GROUPS = ["learning-g1005", "learning-g5492", "learning-g6673", "memory-g10181", "memory-g6546", "memory-g8561"]
SIDES = [("A_candidate", "candidate"), ("A_matched", "matched")]
LINE_PX, LINE_ALPHA, SOMA_UM = 2.2, 0.92, 3.6


def px(mm):
    return int(round(mm * fs.MM * fs.DPI))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = sc.cells()
    by_body = df.set_index("body_id")
    sel = pd.read_csv(sc.FIGURE_DATA / "fd_neuron_selection.csv")
    _, xyz_all = sc.somata(df)
    view = rp.fit_view(xyz_all, px(TILE_W), px(TILE_H), unfold=True, margin=0.015, ss=3)
    back = 0.10 * sc.density_weights(view, xyz_all)
    width_um = LINE_PX * view.um_per_px
    meta = []
    for g in GROUPS:
        for key, side in SIDES:
            d = sel[(sel["selection_set"] == key) & (sel["label"] == g)].sort_values("body_id")
            assert len(d) > 0 and d["body_id"].is_unique
            tile = rp.Tile(view)
            sc.shell(tile, strength=0.11, fill=0.010)
            tile.glow(xyz_all, 1.0, "#5A5A5A", back, label="all cell bodies")
            a_all, b_all, col_all, somata, soma_col, groups = [], [], [], [], [], []
            for body in d["body_id"].astype(int):
                c = by_body.loc[body]
                grp = pal.SUPERCLASS_GROUP.get(c["superclass"], "other")
                groups.append(grp)
                lin = rp.hex_linear(pal.ATLAS_ON_DARK[grp])
                a, b = sc.skeleton_segments(body)
                a_all.append(a)
                b_all.append(b)
                col_all.append(np.repeat(lin[None], len(a), axis=0))
                s = c[["soma_x_um", "soma_y_um", "soma_z_um"]].to_numpy(np.float64)
                if not np.isnan(s).any():
                    somata.append(s)
                    soma_col.append(lin)
            tile.segments(np.vstack(a_all), np.vstack(b_all), width_um, np.vstack(col_all), LINE_ALPHA, order="depth", depth_cue=0.25,
                          label="skeletons")
            if somata:
                tile.discs(np.array(somata), SOMA_UM, np.array(soma_col), 1.0, order="given", label="somata")
            img = tile.resolve(bloom=BLOOM)
            name = f"{g}_{side}"
            types = sorted(set(map(str, d["type"])))
            info = dict(group=g, side=side, cells=int(len(d)), somata=int(len(somata)), types=types, n_types=len(types),
                        superclass_groups=sorted(set(groups)), body_ids=[int(x) for x in d["body_id"]], scale_um_per_px=view.um_per_px,
                        line_px=LINE_PX, line_alpha=LINE_ALPHA, soma_radius_um=SOMA_UM,
                        skeletons=[sc.skeleton_path(int(x)).relative_to(rp.ROOT).as_posix() for x in d["body_id"]])
            tile.save(OUT / f"{name}.png", img, info)
            meta.append(dict(name=name, **{k: v for k, v in info.items() if k != "skeletons"}))
    (OUT / "tiles.json").write_text(json.dumps(dict(tile_mm=[TILE_W, TILE_H], cols=COLS, groups=GROUPS, tiles=meta), indent=1),
                                    encoding="utf-8", newline="\n")
    print(f"done {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
