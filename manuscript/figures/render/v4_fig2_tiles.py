"""v4 Fig. 2 dark band (panel D): the silent afterimage on the unfolded CNS, four tiles across 174 mm at one scale, in
the Fig. 1 visual language (glass shell, luminous somata, bloom).

D1     scene 1 (readout trained): where the trace is written, the cells that spike differently after the two cues from
       cue onset to the end of the delay (gold). They all do so during the cue or in the first tens of ms after it; for the
       rest of the delay no cell does.
D2-D4  scenes 1-3 (readout trained): the membrane-potential difference, cue 1 minus cue 0, at the last ms of the delay,
       for cells above 0.01 mV, on the dark-centred cue map (log 0.01-2 mV). It is 99.9-100% from adaptation (fu4).

Data: manuscript/results/plan_v2/fig2_delay_end_cells.csv (P12, from the dense records, gated against fu4); soma
positions from figure_data/fd_cells.csv.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_fig2_tiles.py
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
OUT = FIG / "renders" / "v4" / "fig2"
PLAN = rp.RESULTS / "plan_v2"
TILE_W = (fs.FULL - 3 * fs.GUTTER) / 4
TILE_H = TILE_W / 0.703                      # the unfolded CNS at the aspect of the v1 state tiles
TITLE_MM = 5.0                               # black strip at the top of each tile for its title
BLOOM = (2.5, 0.40, 0.22)
SPIKE = "#FFE7A3"
DV_LO, DV_HI = 0.01, 2.0                     # mV: the delay-end trace peaks near 2.2 mV
TRACES = ("s1_readout", "s2_readout", "s3_readout")


def px(mm):
    return int(round(mm * fs.MM * fs.DPI))


def base(xyz, background_alpha):
    W, H, top = px(TILE_W), px(TILE_H), px(TITLE_MM)
    v0 = rp.fit_view(xyz, W, H - top, unfold=True, margin=0.010, ss=3)
    view = rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 3)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.11, fill=0.010)
    tile.glow(xyz, 1.0, "#5A5A5A", background_alpha * sc.density_weights(view, xyz), label="all cell bodies")
    return tile


def trace_colours(dv):
    """sc.signed_colours on the delay-end range: colour (linear RGB) and the log position u in [0, 1] of |dv|."""
    u = np.clip(np.log10(np.abs(dv) / DV_LO) / np.log10(DV_HI / DV_LO), 0, 1)
    t = 0.5 + np.sign(dv) * (0.26 + 0.24 * u)
    return rp.srgb_to_linear(pal.SIGNED_ON_DARK(t)[:, :3]), u


def write_spikes(s, xyz):
    summary = pd.read_csv(PLAN / "fig2_delay_end_summary.csv").set_index("learner").loc["s1_readout"]
    raster = pd.read_csv(PLAN / "fig2_raster.csv")
    written = np.unique(raster.loc[raster["ms"] <= summary["end_of_delay_ms"], "cell"].to_numpy())
    n = np.isin(s["cell"].to_numpy(), written)
    tile = base(xyz, 0.08)
    tile.glow(xyz[n], 1.6, SPIKE, 1.6, label="cells spiking differently, cue onset to the end of the delay")
    img = tile.resolve(bloom=BLOOM)
    tile.save(OUT / "write.png", img, dict(cells=int(len(written)), cells_located=int(n.sum()), located=int(len(n)), colour=SPIKE,
                                          end_of_delay_ms=int(summary["end_of_delay_ms"]), scale_um_per_px=tile.view.um_per_px))


def trace(s, xyz, key):
    dv = s[f"dv_{key}_mv"].to_numpy()
    keep = np.abs(dv) > DV_LO
    tile = base(xyz, 0.07)
    rgb, u = trace_colours(dv[keep])
    tile.glow(xyz[keep], 1.7, rgb, 1.0 + 1.5 * u, label=f"|dv| > {DV_LO} mV")
    img = tile.resolve(bloom=BLOOM)
    tile.save(OUT / f"trace_{key}.png", img, dict(cells_located=int(keep.sum()), positive=int((dv > DV_LO).sum()),
                                                  negative=int((dv < -DV_LO).sum()), located=int(len(dv)), dv_lo=DV_LO, dv_hi=DV_HI,
                                                  scale_um_per_px=tile.view.um_per_px))


def main():
    t = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False).merge(
        pd.read_csv(PLAN / "fig2_delay_end_cells.csv"), on="cell", validate="one_to_one")
    s, xyz = sc.somata(df)
    write_spikes(s, xyz)
    print(f"write {time.time() - t:.1f}s", flush=True)
    for key in TRACES:
        trace(s, xyz, key)
        print(f"{key} {time.time() - t:.1f}s", flush=True)
    (OUT / "tiles.json").write_text(json.dumps(dict(tile_mm=[TILE_W, TILE_H], title_mm=TITLE_MM, dv_mv=[DV_LO, DV_HI], traces=TRACES),
                                               indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
