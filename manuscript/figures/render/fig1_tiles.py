"""Fig. 1 hero band: two dark tiles that fill a full-width band with no dead space.

* atlas: the unfolded CNS (brain frontal, VNC ventral) with all 140,024 cell bodies as points of light coloured by
  superclass group (one key for the whole figure);
* interface: the dorsal oblique view with the 4,096 readout arbors (same colours) and the 3,200 input cell bodies
  (white), plus zoomed insets of three exemplar neurons placed in the view's empty quadrants.

Band geometry: tile widths follow the content aspects (0.70 and 1.33) so both tiles share one height.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\fig1_tiles.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import figstyle as fs, palette as pal  # noqa: E402
from render import project as rp, scenes as sc  # noqa: E402

OUT = FIG / "renders" / "fig1"
ASPECT_ATLAS, ASPECT_INTERFACE = 0.703, 1.333
BAND_H = (fs.FULL - fs.GUTTER) / (ASPECT_ATLAS + ASPECT_INTERFACE)          # one height for both tiles
INTERFACE_MM = (fs.FULL - fs.GUTTER - ASPECT_ATLAS * BAND_H, BAND_H)
ATLAS_MM = (ASPECT_ATLAS * BAND_H, BAND_H)
BLOOM = (3.0, 0.40, 0.22)
EXEMPLARS = [  # key, body id, label, colour, inset box in the interface tile (fractions: x0, y0, x1, y1)
    ("input", 51086, "LC12", pal.TILE["input"], (0.355, 0.058, 0.735, 0.448)),
    ("motor", 800381, "i2 MN", pal.ATLAS_ON_DARK["motor & efferent"], (0.755, 0.058, 0.985, 0.448)),
]
HIGHLIGHT = (10001, "DNp01")       # drawn in place in the interface view: it spans brain, neck and VNC


def px(mm):
    return int(round(mm * fs.MM * fs.DPI))


def groups(df):
    return df["superclass"].map(pal.SUPERCLASS_GROUP).fillna("other")


def group_colours(df):
    return np.vstack([rp.hex_linear(pal.ATLAS_ON_DARK[g]) for g in groups(df)])


def atlas():
    df = sc.cells()
    s, xyz = sc.somata(df)
    view = rp.fit_view(xyz, px(ATLAS_MM[0]), px(ATLAS_MM[1]), unfold=True, margin=0.012, ss=2)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.11, fill=0.010)
    tile.glow(xyz, 1.0, group_colours(s), 0.34 * sc.density_weights(view, xyz), label="140,024 cell bodies by superclass group")
    rgb = tile.resolve(bloom=BLOOM)
    counts = groups(s).value_counts().to_dict()
    tile.save(OUT / "atlas.png", rgb, dict(group_counts=counts, somata=int(len(xyz)), cells=int(len(df)),
                                           scale_um_per_px=view.um_per_px))
    return view


def inset(df, body, colour, w_px, h_px):
    """A zoomed render of one neuron in the dorsal oblique view with nearby context."""
    a, b = sc.skeleton_segments(body)
    pts = np.vstack([a, b])
    R = rp.look(*sc.DORSAL_OBLIQUE)
    view = rp.fit_view(pts, w_px, h_px, R=R, margin=0.12, ss=3)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.09, fill=0.010)
    s, xyz = sc.somata(df)
    c, _ = view.camera(xyz)
    half = np.array([w_px, h_px]) * view.um_per_px / 2 * 1.1
    pc, _ = view.camera(pts)
    near = (np.abs(c[:, 0] - view.centre[0]) < half[0]) & (np.abs(c[:, 1] - view.centre[1]) < half[1]) &         (c[:, 2] > pc[:, 2].min() - 12) & (c[:, 2] < pc[:, 2].max() + 12)     # a slab around the neuron, not the whole column
    tile.discs(xyz[near], 1.3, group_colours(s[near]) * 0.40, 0.45, order="depth", label="cell bodies within 12 um of the neuron's depth")
    tile.segments(a, b, 2.4 * view.um_per_px, colour, 0.95, order="depth", depth_cue=0.30, label="skeleton")
    row = df[df["body_id"] == body].iloc[0]
    if not np.isnan(row["soma_x_um"]):
        tile.discs(row[["soma_x_um", "soma_y_um", "soma_z_um"]].to_numpy(float)[None], 2.6, colour, 1.0, order="given", label="soma")
    return tile.resolve(bloom=BLOOM), view, row


def interface():
    df = sc.cells()
    s, xyz = sc.somata(df)
    W, H = px(INTERFACE_MM[0]), px(INTERFACE_MM[1])
    R = rp.look(*sc.DORSAL_OBLIQUE)
    # the quadrants beside the VNC are empty in this view; they hold the exemplar insets
    view = rp.fit_view(xyz, W, H, R=R, margin=0.02, ss=2)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.10, fill=0.010)
    sc.background(tile, xyz, 0.20)
    a, b, col, al, counts = sc.readout_arbors(df)
    tile.segments(a, b, 0.28, col, al, depth_cue=0.6, label="4,096 readout arbors")
    inp = s[s["role"] == "input"][["soma_x_um", "soma_y_um", "soma_z_um"]].to_numpy()
    tile.glow(inp, 1.1, pal.TILE["input"], 0.75, label="3,200 input cell bodies")
    ha, hb = sc.skeleton_segments(HIGHLIGHT[0])
    tile.segments(ha, hb, 1.3, "#FFFFFF", 0.95, order="depth", label=f"{HIGHLIGHT[1]} highlighted")
    rgb = tile.resolve(bloom=BLOOM)
    hp = np.vstack([ha, hb])
    hx, hy, _, _ = view.pixels(hp)
    meta = dict(readout_group_counts=counts, input_cells=int(len(inp)), scale_um_per_px=view.um_per_px, insets=[],
                highlight=dict(body_id=HIGHLIGHT[0], label=HIGHLIGHT[1], skeleton=sc.skeleton_path(HIGHLIGHT[0]).relative_to(rp.ROOT).as_posix(),
                               bbox_px=[float(hx.min() / view.ss), float(hy.min() / view.ss), float(hx.max() / view.ss), float(hy.max() / view.ss)]),
                orientation={k: sc.screen_direction(view, v).round(6).tolist() for k, v in (("A", (0, 0, -1)), ("L", (1, 0, 0)))})
    for key, body, label, colour, (x0, y0, x1, y1) in EXEMPLARS:
        bx0, by0, bx1, by1 = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
        img, iv, row = inset(df, body, colour, bx1 - bx0, by1 - by0)
        rgb[by0:by1, bx0:bx1] = img
        frame = (np.array([0x5A, 0x63, 0x6E], np.uint8))       # thin grey frame around each inset
        rgb[by0:by0 + 2, bx0:bx1] = frame
        rgb[by1 - 2:by1, bx0:bx1] = frame
        rgb[by0:by1, bx0:bx0 + 2] = frame
        rgb[by0:by1, bx1 - 2:bx1] = frame
        meta["insets"].append(dict(key=key, body_id=body, label=label, type=row["type"], superclass=row["superclass"],
                                   role=row["role"], box_px=[bx0, by0, bx1, by1], scale_um_per_px=iv.um_per_px,
                                   skeleton=sc.skeleton_path(body).relative_to(rp.ROOT).as_posix()))
    tile.save(OUT / "interface.png", rgb, meta)
    return view


def main():
    t = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    atlas()
    print(f"atlas {time.time() - t:.1f}s", flush=True)
    interface()
    print(f"interface {time.time() - t:.1f}s", flush=True)
    (OUT / "band.json").write_text(json.dumps(dict(band_height_mm=BAND_H, atlas_mm=ATLAS_MM, interface_mm=INTERFACE_MM), indent=1),
                                   encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
