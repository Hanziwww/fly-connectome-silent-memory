"""v4 dark filmstrips of one memory trial (scene 1, readout trained; 160-ms delay, 4 prefix windows), for the opening figures.

Fig. 1  activity_<phase>: every cell that spikes in the phase (cue 0: cue, delay, input, decision), glowing in its superclass
        group colour (as Fig. 1A), brighter with more spikes; rendered upright and turned 90 degrees so that the brain is on
        the left, the orientation of Fig. 1B.
Fig. 2  trace_t<ms>: the trace through the trial at five moments (cue end, 40 ms into the delay, delay end, 40 ms of input,
        decision): each cell's adaptation difference (cue 1 minus cue 0) on the dark-centred cue map, and as white sparks the
        cells that spiked differently after the two cues in the 40 ms before the moment.

Data: manuscript/results/plan_v2/trial_cells.csv (P12, from the 1-ms records, gated against fu4 and fig2_delay_end_cells);
soma positions from figure_data/fd_cells.csv.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_trial_tiles.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from PIL import Image

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import figstyle as fs, palette as pal  # noqa: E402
from render import project as rp, scenes as sc  # noqa: E402

fs.use_cell()
OUT = FIG / "renders" / "v4" / "trial"
PLAN = rp.RESULTS / "plan_v2"
BLOOM = (2.5, 0.40, 0.22)
ACT_W = (fs.FULL - 3 * fs.GUTTER) / 4                     # Fig. 1: four landscape tiles across the page
ACT_H = ACT_W * 0.703                                     # the unfolded CNS turned on its side
TRACE_W = (fs.FULL - 4 * fs.GUTTER) / 5                   # Fig. 2: five upright tiles across the page
TRACE_H = TRACE_W / 0.703
TITLE_MM = 10.0                                           # black band above the brain for three lines of tile text
PHASES = ("cue", "delay", "input", "decision")
SNAPSHOTS = (40, 80, 200, 240, 400)
DA_LO, DA_HI = 1e-3, 3.0                                   # adaptation difference, log scale (mV of the adaptation variable)
SPARK = "#FFFFFF"


def px(mm):
    return int(round(mm * fs.MM * fs.DPI))


def base(all_xyz, w_mm, h_mm, top_mm, background=0.05):
    W, H, top = px(w_mm), px(h_mm), px(top_mm)
    v0 = rp.fit_view(all_xyz, W, H - top, unfold=True, margin=0.010, ss=3)
    view = rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 3)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.11, fill=0.010)
    tile.glow(all_xyz, 1.0, "#5A5A5A", background * sc.density_weights(view, all_xyz), label="all cell bodies")
    return tile


def activity(s, xyz, all_xyz):
    groups = s["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other").to_numpy()
    colours = rp.srgb_to_linear(np.array([to_rgb(pal.ATLAS_ON_DARK[g]) for g in groups]))
    meta = {}
    for ph in PHASES:
        n = s[f"spikes_{ph}_cue0"].to_numpy(float)
        on = n > 0
        tile = base(all_xyz, ACT_H, ACT_W, 0.0)             # upright: the tile is ACT_H wide and ACT_W tall, turned afterwards
        w = (0.6 + 0.8 * np.log1p(n[on]) / np.log1p(20)) * sc.density_weights(tile.view, xyz[on], power=0.45)
        tile.glow(xyz[on], 1.5, colours[on], w, label=f"cells spiking in the {ph}")
        img = tile.resolve(bloom=BLOOM)
        img = np.asarray(Image.fromarray(img).transpose(Image.Transpose.ROTATE_90))
        tile.save(OUT / f"activity_{ph}.png", img, dict(phase=ph, cells_spiking_located=int(on.sum()), spikes_located=int(n[on].sum()),
                                                        rotated="90 degrees counter-clockwise (brain left)", scale_um_per_px=tile.view.um_per_px))
        meta[ph] = int(on.sum())
    return meta


def trace(s, xyz, all_xyz):
    meta = {}
    for t in SNAPSHOTS:
        da = s[f"dadapt_t{t}"].to_numpy(float)
        on = np.abs(da) > DA_LO
        u = np.clip(np.log10(np.abs(da[on]) / DA_LO) / np.log10(DA_HI / DA_LO), 0, 1)
        col = rp.srgb_to_linear(pal.SIGNED_ON_DARK(0.5 + np.sign(da[on]) * (0.26 + 0.24 * u))[:, :3])
        tile = base(all_xyz, TRACE_W, TRACE_H, TITLE_MM)
        tile.glow(xyz[on], 1.5, col, (0.25 + 1.3 * u ** 1.5) * sc.density_weights(tile.view, xyz[on], power=0.4), label="adaptation difference")
        sp = s[f"diffspike_w{t}"].to_numpy(float) > 0
        if sp.any():
            tile.glow(xyz[sp], 0.8, SPARK, 1.0, label="spiking differently in the last 40 ms")
        img = tile.resolve(bloom=BLOOM)
        tile.save(OUT / f"trace_t{t}.png", img, dict(ms=t, trace_cells_located=int(on.sum()), spark_cells_located=int(sp.sum()),
                                                     da_range=[DA_LO, DA_HI], scale_um_per_px=tile.view.um_per_px))
        meta[t] = int(on.sum())
    return meta


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fd = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False)
    tc = pd.read_csv(PLAN / "trial_cells.csv").drop(columns=["superclass", "source_file"])
    df = fd.merge(tc, on="cell", validate="one_to_one")
    s, xyz = sc.somata(df)
    _, all_xyz = sc.somata(fd)
    a = activity(s, xyz, all_xyz)
    print(f"activity {time.time() - t0:.1f}s", flush=True)
    b = trace(s, xyz, all_xyz)
    print(f"trace {time.time() - t0:.1f}s", flush=True)
    (OUT / "tiles.json").write_text(json.dumps(dict(activity_mm=[ACT_W, ACT_H], trace_mm=[TRACE_W, TRACE_H], title_mm=TITLE_MM, phases=list(PHASES),
                                                    snapshots_ms=list(SNAPSHOTS), da_range=[DA_LO, DA_HI], activity_located=a, trace_located=b),
                                               indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
