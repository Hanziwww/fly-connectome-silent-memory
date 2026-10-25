"""v4 graphical abstract renders: the trained connectome in the frontal view of the whole CNS (brain from the front, the ventral
nerve cord hung below it), the view readers know from connectome papers, with its silent trace at the end of the delay
glowing as warm light. Tiles at one view and one scale:

learned   the state of learning (healthy): the trace written and held in the healthy brain
shifted   after the disease-like loss of potassium current (dose 1, the Kv4-loss lesion): the trace written and held in the
          lesioned brain, where it lands in other and many more cells; the shell takes the lesion colour of Figure 7
fly_cns   the learned state small, for the fly drawn at the abstract's start (its nervous system seen through the body)

Both traces share one brightness scale (the healthy trace's 5th to 99th percentile of |adaptation difference|, log), so the
tiles compare directly. The trace is gold light laid over the brain rather than added to it: its luminance L sets an opacity
1 - exp(-L / OPACITY), so single cells are soft gold points and overlapping cells saturate to gold instead of mixing with the
shell towards white; the lesion-written trace, about ten times larger, reads as a dense golden field. The tiles carry no scale
bar, key or number: the abstract shows no data items.

Data: manuscript/results/plan_v2/link_cells.csv (P12, gated; adaptation cue difference at the end of the 160-ms delay with
four prefix windows, written and read in the same state); soma positions from figure_data/fd_cells.csv.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_ga_tiles.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from render import project as rp, scenes as sc  # noqa: E402

OUT = FIG / "renders" / "v4" / "ga"
PLAN = rp.RESULTS / "plan_v2"
GA_MM, PPI = 101.6, 300                      # the Cell abstract: one square panel of 1,200 px at 300 dpi
TILE_MM = (30.7, 43.0)                       # each state's render (mm): the frontal CNS is about 0.7 as wide as tall
FLY_MM = (5.0, 7.0)                          # the CNS inside the fly (mm)
GOLD = "#FFC46B"                             # the silent trace
STATES = {
    "learned": dict(column="trace_healthy_written_healthy_read", shell="#9FB0C0"),
    "shifted": dict(column="trace_lesion_written_lesion_read", shell="#A97BD6"),   # pal LESION_STATE #7B3294, brightened
}
BLOOM = (2.4, 0.35, 0.45)
SIGMA_UM, OPACITY, GAIN = 1.35, 0.45, 1.35   # trace splat size; luminance scale of the gold's opacity; the gold's brightness
LUMA = np.array([0.2126, 0.7152, 0.0722])    # linear-light luminance weights


def px(mm):
    return int(round(mm / 25.4 * PPI))


def scale(trace):
    """The shared log scale's ends: the 5th and 99th percentiles of |trace| over the cells that carry one."""
    a = np.abs(trace[np.abs(trace) > 0])
    lo, hi = float(np.quantile(a, 0.05)), float(np.quantile(a, 0.99))
    return lo, hi


def render(name, out, s, xyz, view, lo, hi, shell_strength=0.15, sigma_um=SIGMA_UM, dust=0.05, gold=1.0, bloom=BLOOM):
    """One state's tile: the neuropil shell, the cell bodies faint, and the trace laid over them as gold."""
    spec = STATES[name]
    tr = s[spec["column"]].to_numpy()
    keep = np.abs(tr) > 0
    u = np.clip(np.log10(np.abs(tr[keep]) / lo) / np.log10(hi / lo), 0, 1)
    tile = rp.Tile(view)
    tile.shell(sc.regions(), colour=spec["shell"], strength=shell_strength, gamma=2.2, fill=0.012, select=sc.no_lamina)
    tile.glow(xyz, 1.0, spec["shell"], dust * sc.density_weights(view, xyz), label="all cell bodies")
    light = rp.Tile(view)                                  # the trace on its own layer, then laid over the brain as gold
    light.glow(xyz[keep], sigma_um, GOLD, gold * (0.35 + 1.25 * u), label="the silent trace, |adaptation difference| on the shared log scale")
    alpha = (1.0 - np.exp(-(light.img @ LUMA) / OPACITY))[..., None]
    tile.img = tile.img * (1.0 - alpha) + GAIN * rp.hex_linear(GOLD)[None, None, :] * alpha
    tile.log["layers"] += light.log["layers"] + [dict(kind="gold over", opacity=OPACITY, gain=GAIN, max_alpha=float(alpha.max()))]
    img = tile.resolve(bloom=bloom)
    tile.save(OUT / f"{out}.png", img, dict(trace_cells_located=int(keep.sum()), trace_cells=int((s[spec["column"]] != 0).sum()),
                                            located=int(len(tr)), scale=[lo, hi], gold=GOLD, shell=spec["shell"],
                                            scale_um_per_px=view.um_per_px))


def main():
    t = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("key_*"):                          # the key thumbnails of an earlier design
        old.unlink()
    df = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False).merge(
        pd.read_csv(PLAN / "link_cells.csv"), on="cell", validate="one_to_one")
    s, xyz = sc.somata(df)
    lo, hi = scale(s[STATES["learned"]["column"]].to_numpy())
    view = rp.fit_view(xyz, px(TILE_MM[0]), px(TILE_MM[1]), unfold=True, margin=0.012, ss=3)
    for name in STATES:
        render(name, name, s, xyz, view, lo, hi)
        print(f"{name} {time.time() - t:.1f}s", flush=True)
    small = rp.fit_view(xyz, px(FLY_MM[0]), px(FLY_MM[1]), unfold=True, margin=0.02, ss=4)
    render("learned", "fly_cns", s, xyz, small, lo, hi, shell_strength=0.004, sigma_um=2.2, dust=0.006, gold=0.08,
           bloom=None)                                     # light accumulates over many cells per pixel when small
    (OUT / "tiles.json").write_text(json.dumps(dict(ga_mm=GA_MM, ppi=PPI, tile_mm=TILE_MM, tile_px=[px(TILE_MM[0]), px(TILE_MM[1])],
                                                    fly_mm=FLY_MM, view="unfolded CNS (frontal brain, ventral cord below)",
                                                    scale=[lo, hi], states=list(STATES)),
                                               indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
