"""v4 Fig. 3 dark maps above panels F and G, on the unfolded CNS (upright, as Fig. 2C), three tiles per panel.

F  where the cue sits at the decision (scene 1, readout trained): every cell's membrane-potential difference between the
   cues on a log magnitude scale (magma), for all cells, for the readout cells only, and for the six olfactory circuits
   (their cells in grey so that the circuits are visible; magma where they carry a difference)
G  where the connection rules act: the share of each cell's input synapses that the receptor table signs differently from
   the presynaptic transmitter, that come from cells of the same type, and that lie beyond 60 synapses per connection

Data: manuscript/results/plan_v2/fig3_cue_cells.csv and fig3_rule_cells.csv (P12, gated against fig3_cue_energy and the
documented receptor-rule entries); soma positions from figure_data/fd_cells.csv.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_fig3_maps.py
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
OUT = FIG / "renders" / "v4" / "fig3" / "maps"
PLAN = rp.RESULTS / "plan_v2"
BLOOM = (2.5, 0.40, 0.22)
MAP_W = (fs.unit(2) - 2 * fs.GUTTER) / 3                 # three tiles across one two-unit panel
MAP_H = 38.0
TITLE_MM = 9.6                                           # black band for three lines of tile text
DV_LO, DV_HI = 0.1, 10.0                                 # |cue difference| at the decision (mV), log scale
SHARE_LO, SHARE_HI = 0.01, 1.0                           # share of input synapses, log scale
OLFACTORY_GREY = "#8FA3B8"


def px(mm):
    return int(np.ceil(mm * fs.MM * fs.DPI))


def base(all_xyz):
    W, H, top = px(MAP_W), px(MAP_H), px(TITLE_MM)
    v0 = rp.fit_view(all_xyz, W, H - top, unfold=True, margin=0.010, ss=3)
    view = rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 3)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.11, fill=0.010)
    tile.glow(all_xyz, 1.0, "#5A5A5A", 0.05 * sc.density_weights(view, all_xyz), label="all cell bodies")
    return tile


def log_unit(x, lo, hi):
    return np.clip(np.log10(np.maximum(x, lo) / lo) / np.log10(hi / lo), 0, 1)


def magma_glow(tile, xyz, u, label):
    order = np.argsort(u, kind="stable")                  # brightest drawn last
    col = rp.srgb_to_linear(pal.MAGNITUDE_ON_DARK(0.25 + 0.75 * u[order])[:, :3])
    w = (0.06 + 0.8 * u[order] ** 2) * sc.density_weights(tile.view, xyz[order], power=0.7)
    tile.glow(xyz[order], 1.4, col, w, label=label)


def save(tile, name, meta):
    img = tile.resolve(bloom=BLOOM)
    tile.save(OUT / f"{name}.png", img, dict(meta, scale_um_per_px=tile.view.um_per_px))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fd = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False)
    _, all_xyz = sc.somata(fd)
    cue = pd.read_csv(PLAN / "fig3_cue_cells.csv").drop(columns=["superclass", "source_file"])
    rules = pd.read_csv(PLAN / "fig3_rule_cells.csv").drop(columns=["source_file", "input_synapses"])
    s, xyz = sc.somata(fd.merge(cue, on="cell", validate="one_to_one").merge(rules, on="cell", validate="one_to_one"))
    a = s["dv_decision_mv"].abs().to_numpy(float)
    meta = {}
    # F: the cue at the decision
    for name, sel, lab in (("cue_all", np.ones(len(s), bool), "all cells"), ("cue_readout", s["readout_cell"].to_numpy(bool), "readout cells"),
                           ("cue_olfactory", s["olfactory_cell"].to_numpy(bool), "olfactory circuits")):
        tile = base(all_xyz)
        if name == "cue_olfactory":
            tile.glow(xyz[sel], 1.2, OLFACTORY_GREY, 0.35 * sc.density_weights(tile.view, xyz[sel], power=0.4), label="olfactory-circuit cells")
        on = sel & (a > DV_LO)
        magma_glow(tile, xyz[on], log_unit(a[on], DV_LO, DV_HI), f"{lab}: |cue difference| at the decision")
        save(tile, name, dict(cells_located=int(sel.sum()), cells_drawn=int(on.sum()), dv_range_mv=[DV_LO, DV_HI]))
        meta[name] = dict(cells_located=int(sel.sum()), cells_drawn=int(on.sum()))
        print(f"{name} {time.time() - t0:.1f}s", flush=True)
    # G: where the connection rules act
    for name, col in (("rule_receptor", "receptor_sign_share"), ("rule_same_type", "same_type_share"), ("rule_cap", "beyond_cap_share")):
        v = s[col].to_numpy(float)
        on = v > SHARE_LO
        tile = base(all_xyz)
        magma_glow(tile, xyz[on], log_unit(v[on], SHARE_LO, SHARE_HI), f"share of input synapses: {col}")
        save(tile, name, dict(cells_drawn=int(on.sum()), share_range=[SHARE_LO, SHARE_HI]))
        meta[name] = dict(cells_drawn=int(on.sum()))
        print(f"{name} {time.time() - t0:.1f}s", flush=True)
    (OUT / "maps.json").write_text(json.dumps(dict(tile_mm=[MAP_W, MAP_H], title_mm=TITLE_MM, dv_range_mv=[DV_LO, DV_HI],
                                                   share_range=[SHARE_LO, SHARE_HI], olfactory_grey=OLFACTORY_GREY, tiles=meta), indent=1),
                                   encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
