"""v4 Figure S4 dark gallery: single-gene in silico stains on the unfolded CNS, in the Fig. 4 visual language.

Each mapped soma glows by the z-score of its atlas unit for one gene (SK, slo, KCNQ, Shab, Shal) or for the adaptation index
(magma; above the atlas mean, z 0-2), with the mapping tier as brightness (cell type 1, hemilineage 0.8, hemilineage pair
0.65, class 0.5); unmapped somata are the dim ghost.

Data: manuscript/results/plan_v2/s4_cell_genes.csv (P12, gated against fig4_cell_expression); soma positions from fd_cells.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_s4_tiles.py
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
OUT = FIG / "renders" / "v4" / "s4"
PLAN = rp.RESULTS / "plan_v2"
TILE_W = (fs.FULL - 5 * fs.GUTTER) / 6
TILE_H = 44.0
TITLE_MM = 5.0
BLOOM = (2.8, 0.42, 0.20)
TIER_GAIN = {"exact": 1.0, "alias": 1.0, "fuzzy": 1.0, "lineage": 0.8, "lineage_pair": 0.65, "class": 0.5}
Z_HI = 2.0
GHOST = "#5A5A5A"
STAINS = [("z_SK", "SK"), ("z_slo", "slo"), ("z_KCNQ", "KCNQ"), ("z_Shab", "Shab"), ("z_Shal", "Shal"), ("adaptation_index", "adaptation index")]


def px(mm):
    return int(np.ceil(mm * fs.MM * fs.DPI))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fd = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False)
    g = pd.read_csv(PLAN / "s4_cell_genes.csv").drop(columns=["source_file"])
    s, xyz = sc.somata(fd.merge(g, on="cell", validate="one_to_one"))
    W, H, top = px(TILE_W), px(TILE_H), px(TITLE_MM)
    v0 = rp.fit_view(xyz, W, H - top, unfold=True, margin=0.010, ss=3)
    view = rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 3)
    gain = s["tier"].map(TIER_GAIN).to_numpy(float)
    meta = {}
    for col, lab in STAINS:
        z = s[col].to_numpy(float)
        mapped = ~np.isnan(z) & ~np.isnan(gain)
        tile = rp.Tile(view)
        sc.shell(tile, strength=0.12, fill=0.010)
        w = sc.density_weights(view, xyz)
        tile.glow(xyz[~mapped], 1.0, GHOST, 0.06 * w[~mapped], label="unmapped cell bodies")
        u = np.clip(z[mapped] / Z_HI, 0, 1)
        order = np.argsort(u, kind="stable")
        col_rgb = rp.srgb_to_linear(pal.MAGNITUDE_ON_DARK(0.15 + 0.85 * u[order])[:, :3])
        tile.glow(xyz[mapped][order], 1.2, col_rgb, (gain[mapped] * (0.06 + 0.85 * u ** 1.3))[order] * w[mapped][order], label=f"{lab}, unit z")
        img = tile.resolve(bloom=BLOOM)
        info = dict(stain=lab, column=col, mapped_located=int(mapped.sum()), above_mean_located=int((z[mapped] > 0).sum()), z_range=[0.0, Z_HI],
                    tier_gain=TIER_GAIN, scale_um_per_px=view.um_per_px)
        name = col.replace("z_", "")
        tile.save(OUT / f"{name}.png", img, info)
        meta[name] = info
        print(f"{lab} {time.time() - t0:.1f}s", flush=True)
    (OUT / "tiles.json").write_text(json.dumps(dict(tile_mm=[TILE_W, TILE_H], title_mm=TITLE_MM, stains=[c.replace("z_", "") for c, _ in STAINS],
                                                    labels={c.replace("z_", ""): l for c, l in STAINS}, tiles=meta), indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
