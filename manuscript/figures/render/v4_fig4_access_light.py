"""v4 Fig. 4I light tiles, in the grammar of MaleCNS Fig. 4C: for each carrier type with genetic access, the brain filled in light
grey (one outline per neuropil), the type's trace carriers as gold somata, its other cells as pale dots, and one exemplar
carrier (the skeleton of the dark access tile) in near-black. One view for all four types, so the tiles compare at one scale.

Data: plan_v2/fig4_access_types.csv and fig4_cell_expression.csv (P12), soma positions from figure_data, the exemplar bodies of
renders/v4/fig4/access_<type>.json (local skeletons; no downloads). Writes renders/v4/fig4/light_<type>.png and light.json.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_fig4_access_light.py
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
OUT = FIG / "renders" / "v4" / "fig4"
PLAN = rp.RESULTS / "plan_v2"
TILE_MM = (20.2, 10.6)
FILL, LINE = "#E6E3DF", "#B9B3AC"                        # the filled light-grey brain (MaleCNS warm grey, darker outline)
SKELETON, CARRIER, OTHER = "#262626", "#C99400", "#9C9C9C"


def px(mm):
    return int(round(mm * fs.MM * fs.DPI))


def main():
    types = pd.read_csv(PLAN / "fig4_access_types.csv")
    expr = pd.read_csv(PLAN / "fig4_cell_expression.csv", usecols=["cell", "carrier_scene1"], low_memory=False)
    cells = sc.cells().merge(expr, on="cell", how="left")
    keep = lambda m: sc.no_lamina(m) and m["dataset"] != "vnc"  # noqa: E731
    brain = np.vstack([v for m, v, _ in sc.regions() if keep(m)])
    view = rp.fit_view(brain, px(TILE_MM[0]), px(TILE_MM[1]), unfold=True, margin=0.02, ss=3)
    meta = {}
    for _, r in types.iterrows():
        t = r["type"]
        body = int(json.loads((OUT / f"access_{t}.json").read_text(encoding="utf-8"))["body_id"])
        a, b = sc.skeleton_segments(body)
        q = cells[cells["type"] == t]
        xyz = q[["soma_x_um", "soma_y_um", "soma_z_um"]].to_numpy(float)
        located = ~np.isnan(xyz).any(axis=1)
        car = q["carrier_scene1"].astype(str).str.lower().eq("true").to_numpy()
        if int(car.sum()) != int(r["carriers_scene1"]):
            raise RuntimeError(f"{t}: {int(car.sum())} carriers in fig4_cell_expression, fig4_access_types {int(r['carriers_scene1'])}")
        tile = rp.Tile(view, background="#FFFFFF")
        tile.silhouette(sc.regions(), fill=FILL, line=LINE, line_px=2, select=keep)
        tile.discs(xyz[located & ~car], 2.4, rp.hex_linear(OTHER), 0.85, order="depth", label="other cells of the type")
        tile.segments(a, b, max(0.45, 1.1 * view.um_per_px), rp.hex_linear(SKELETON), 0.95, order="depth", label="exemplar carrier")
        tile.discs(xyz[located & car], 3.2, rp.hex_linear(CARRIER), 1.0, order="depth", label="trace carriers")
        img = tile.resolve()
        Image.fromarray(img).save(OUT / f"light_{t}.png")
        meta[t] = dict(body_id=body, cells=int(len(q)), located=int(located.sum()), carriers=int(car.sum()),
                       carriers_located=int((located & car).sum()))
    (OUT / "light.json").write_text(json.dumps(dict(tile_mm=list(TILE_MM), um_per_px=float(view.um_per_px), fill=FILL, line=LINE,
                                                    skeleton=SKELETON, carrier=CARRIER, other=OTHER, types=meta), indent=1),
                                    encoding="utf-8", newline="\n")
    print("light tiles:", meta)


if __name__ == "__main__":
    main()
