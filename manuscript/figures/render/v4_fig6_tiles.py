"""v4 Fig. 6 dark renders (after the user's correction of 2026-10-22: no perturbation-susceptibility content).

A  band, "many routes, one readout": the cells on input-to-readout routes (walks of length 2-3 from the visual input cells
   to the readout cells) in the intact connectome and after random removal of 30, 60 and 90% of the other cells (the seeded
   draws of the graph analysis), glowing by route traffic (magma, log scale, one scale for all four), with the readout cells
   in cyan. The route network thins tenfold; the readout stays reachable.
C  inset, "a single extra spike is absorbed": the five cells given one extra spike and the fibres to their direct
   postsynaptic partners.

Data: manuscript/results/plan_v2/fig6_routes.csv, fig6_routes_summary.csv, fig6_kick_partners.csv (P12, gated against the
graph analysis); soma positions from figure_data/fd_cells.csv.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_fig6_tiles.py
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
OUT = FIG / "renders" / "v4" / "fig6"
PLAN = rp.RESULTS / "plan_v2"
TILE_W = (fs.FULL - 3 * fs.GUTTER) / 4
TILE_H = TILE_W / 0.703                      # the unfolded CNS at the aspect of the Fig. 2 and Fig. 5 tiles
TITLE_MM = 7.6
KICK_W = 30.0
KICK_H = KICK_W / 0.703
BLOOM = (2.5, 0.40, 0.22)
LEVELS = ("0", "30", "60", "90")
MAP_LO, MAP_HI = 0.30, 0.98                  # magma positions for the weakest and strongest route cells
READOUT = "#7FD3FF"
KICK = "#FFE7A3"
PARTNER = "#FFC46B"


def px(mm):
    return int(round(mm * fs.MM * fs.DPI))


def base(xyz, w_mm, h_mm, top_mm):
    """Shell and every soma, the view fitted on all cells (xyz: every located soma)."""
    W, H, top = px(w_mm), px(h_mm), px(top_mm)
    v0 = rp.fit_view(xyz, W, H - top, unfold=True, margin=0.010, ss=3)
    view = rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 3)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.11, fill=0.010)
    tile.glow(xyz, 1.0, "#5A5A5A", 0.05 * sc.density_weights(view, xyz), label="all cell bodies")
    return tile


def routes(df, xyz, all_xyz, lo, hi):
    meta = {}
    for k in LEVELS:
        t = df[f"traffic_{k}"].to_numpy(float)
        on = t > 0
        u = np.clip((np.log10(t[on]) - lo) / (hi - lo), 0, 1)
        tile = base(all_xyz, TILE_W, TILE_H, TITLE_MM)
        rgb = rp.srgb_to_linear(pal.MAGNITUDE_ON_DARK(MAP_LO + (MAP_HI - MAP_LO) * u)[:, :3])
        w = (0.05 + 0.7 * u ** 2) * sc.density_weights(tile.view, xyz[on], power=0.6)
        tile.glow(xyz[on], 1.3, rgb, w, label=f"route cells, {k}% removed")
        ro = df["role"].eq("readout").to_numpy()
        tile.glow(xyz[ro], 1.2, READOUT, 0.35 * sc.density_weights(tile.view, xyz[ro], power=0.5), label="readout cells")
        img = tile.resolve(bloom=BLOOM)
        tile.save(OUT / f"routes_{k}.png", img, dict(removed_percent=int(k), route_cells_located=int(on.sum()), readout_located=int(ro.sum()),
                                                     log10_traffic_range=[lo, hi], scale_um_per_px=tile.view.um_per_px))
        meta[k] = int(on.sum())
    return meta


def kicks(cells, xyz_of):
    kp = pd.read_csv(PLAN / "fig6_kick_partners.csv")
    kicked = sorted(kp["kicked_cell"].unique())
    allxyz = xyz_of(cells["cell"].to_numpy())
    tile = base(allxyz[np.isfinite(allxyz).all(1)], KICK_W, KICK_H, 0.0)
    a, b = [], []
    for c in kicked:
        p = kp[kp["kicked_cell"] == c]["partner_cell"].to_numpy()
        pa, pb = xyz_of(np.full(len(p), c)), xyz_of(p)
        ok = np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
        a.append(pa[ok])
        b.append(pb[ok])
    a, b = np.concatenate(a), np.concatenate(b)
    tile.segments(a, b, 0.6, PARTNER, 0.10, order="given", label="fibres to direct partners")
    tile.glow(b, 1.2, PARTNER, 0.6, label="direct partners")
    k_xyz = xyz_of(np.asarray(kicked))
    tile.glow(k_xyz, 3.0, KICK, 6.0, label="kicked cells")
    img = tile.resolve(bloom=BLOOM)
    pxy = tile.view.pixels(k_xyz)
    ss = tile.view.ss
    tile.save(OUT / "kicks.png", img, dict(kicked=[int(c) for c in kicked], kicked_px=[[float(x / ss), float(y / ss)] for x, y in zip(pxy[0], pxy[1])],
                                           fibres=int(len(a)), scale_um_per_px=tile.view.um_per_px))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("spread_*"):                          # renders of the first Fig. 6 build (class c since the correction)
        f.unlink()
    fd = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False)
    rt = pd.read_csv(PLAN / "fig6_routes.csv").drop(columns=["source_file"])
    df = fd.drop(columns=["role"]).merge(rt, on="cell", validate="one_to_one")
    s, xyz = sc.somata(df)
    fs_, fxyz = sc.somata(fd)
    t_all = np.concatenate([s[f"traffic_{k}"].to_numpy(float) for k in LEVELS])
    t_all = t_all[t_all > 0]
    lo, hi = np.log10(np.percentile(t_all, 5)), np.log10(np.percentile(t_all, 99.9))
    n_on = routes(s, xyz, fxyz, lo, hi)
    print(f"routes {time.time() - t0:.1f}s", flush=True)
    pos = pd.DataFrame(fxyz, index=fs_["cell"].to_numpy())

    def xyz_of(ids):
        return pos.reindex(np.asarray(ids)).to_numpy(float)

    kicks(fs_, xyz_of)
    print(f"kicks {time.time() - t0:.1f}s", flush=True)
    (OUT / "tiles.json").write_text(json.dumps(dict(tile_mm=[TILE_W, TILE_H], title_mm=TITLE_MM, kick_mm=[KICK_W, KICK_H], levels=list(LEVELS),
                                                    log10_traffic_range=[float(lo), float(hi)], magma=[MAP_LO, MAP_HI], route_cells_located=n_on),
                                               indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
