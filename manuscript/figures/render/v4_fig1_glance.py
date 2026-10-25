"""v4 Fig. 1E brain tile, "the model at a glance", in the grammar of MaleCNS Fig. 6A: half anatomy, half schematic, on white.
Left half: translucent light neuropils with two cells of the model, as in Fig. 1B: an input cell (LC12, orange) and the giant
fiber (DNp01, blue; a readout cell). Right half: the flat warm-grey brain, the neuropils on the task's visual path white with a
dark outline. The halves meet at the midline; the build draws the flow arrows, chips and labels on top.

Data: the neuropil meshes, the two skeletons and figure_data/fd_cells.csv (soma positions). Writes renders/v4/fig1/glance.png and
glance.json (tile size, which half is anatomical, each neuron's median position and the path neuropils' centroids, in mm).
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_fig1_glance.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import figstyle as fs  # noqa: E402
from render import project as rp, scenes as sc  # noqa: E402

fs.use_cell()
OUT = FIG / "renders" / "v4" / "fig1"
TILE_MM = (41.75, 18.8)
NEURONS = [(51086, "LC12", "#F0A030"), (10001, "DNp01", "#5B8FD0")]          # as the arrows of the build (input, descending)
PATH = ("ME(", "LO(", "LOP(", "PVLP(", "GNG")                                 # the task's visual path, white on the flat half
LABELLED = {"ME": "ME(", "LO": "LO(", "PVLP": "PVLP("}                        # neuropils named on the flat half


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    keep = lambda m: sc.no_lamina(m) and m["dataset"] != "vnc"  # noqa: E731
    regions = [r for r in sc.regions() if keep(r[0])]
    pts = np.vstack([v for _, v, _ in regions])
    px = lambda mm: int(round(mm * fs.MM * fs.DPI))  # noqa: E731
    view = rp.fit_view(pts, px(TILE_MM[0]), px(TILE_MM[1]), unfold=True, margin=0.01, ss=3)
    mm_per_px = TILE_MM[0] / view.width
    left = rp.Tile(view, background="#FFFFFF")
    left.glass(sc.regions(), select=keep, strength=0.16, opacity=0.85, colour="#7D848B")
    right = rp.Tile(view, background="#FFFFFF")
    right.flat(sc.regions(), select=keep, highlight=lambda m: m["label"].startswith(PATH), line_px=4)
    cells = sc.cells().set_index("body_id")
    side = {}
    for body, name, _ in NEURONS:
        a, b = sc.skeleton_segments(body)
        sx, sy, _, _ = view.pixels(np.vstack([a, b]))
        side[name] = [float(np.median(sx) / view.ss * mm_per_px), float(np.median(sy) / view.ss * mm_per_px)]
    on_left = float(np.mean([x for x, _ in side.values()])) < TILE_MM[0] / 2
    for body, name, colour in NEURONS:
        a, b = sc.skeleton_segments(body)
        left.segments(a, b, max(0.5, 1.5 * view.um_per_px), rp.hex_linear(colour), 0.95, order="depth", label=name)
        c = cells.loc[body, ["soma_x_um", "soma_y_um", "soma_z_um"]].to_numpy(float)
        if not np.isnan(c).any():
            left.discs(c[None], 3.2, rp.hex_linear(colour), 1.0, order="given", label=f"{name} soma")
    img_l, img_r = left.resolve(), right.resolve()
    mid = img_l.shape[1] // 2
    img = img_l.copy()
    if on_left:
        img[:, mid:] = img_r[:, mid:]
    else:
        img[:, :mid] = img_r[:, :mid]
    centroids = {}                                         # the named neuropils on the schematic half (mm in the tile)
    for name, prefix in LABELLED.items():
        xy = []
        for meta, v, _ in regions:
            if meta["label"].startswith(prefix):
                x, y, _, _ = view.pixels(v, side=False)
                xy.append(np.c_[x, y] / view.ss * mm_per_px)
        xy = np.vstack(xy)
        half = xy[:, 0] >= TILE_MM[0] / 2 if on_left else xy[:, 0] < TILE_MM[0] / 2
        centroids[name] = [float(v) for v in np.median(xy[half], axis=0)]
    Image.fromarray(img).save(OUT / "glance.png")
    (OUT / "glance.json").write_text(json.dumps(dict(tile_mm=list(TILE_MM), anatomy_on_left=bool(on_left), neurons=side,
                                                     neuropils=centroids, um_per_px=float(view.um_per_px)), indent=1), encoding="utf-8")
    print("glance:", img.shape, "anatomy on the left:", on_left, centroids)


if __name__ == "__main__":
    main()
