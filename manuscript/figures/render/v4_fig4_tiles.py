"""v4 Fig. 4 tiles: the transcriptome painted on the connectome, in the Fig. 1 visual language (glass shell, luminous
somata, bloom), and the neuron renders of panel I.

A   the unfolded CNS as an in silico triple stain: three fluorescence-like channel layers added in linear light, SK cyan,
    Shal magenta, slo yellow; each mapped soma glows in a channel where its atlas unit expresses the gene above the atlas
    mean (z clipped to [0, 2], gamma 1.3); mapping tier sets the brightness (cell type 1, hemilineage 0.8, hemilineage pair
    0.65, class 0.5); unmapped somata a dim ghost. A colour-mixing triangle serves as the key.
B   the same view: the scene-1 trace carriers glowing by their unit's SK z on the dark-centred SK map (blue low, orange
    high), unmapped carriers a soft white, every other soma a faint ghost; centroids of the three hemilineages with the
    highest carrier share for their badges.
C   the "Rosetta": the 26,775 adult VNC atlas cells in UMAP space rendered as a luminous field (left), the 21 hemilineage
    units on a spine ordered by SK with their SK swatches (centre) and the MaleCNS VNC in a ventral-oblique view with every
    mapped soma in its unit's colour (right), joined cluster to spine to anatomy by colour-matched fibres of light.
I   one trace-carrying neuron of each of the four carrier types with split-GAL4 lines (fig4_access_types).

Data: manuscript/results/plan_v2 (fig4_cell_expression, fig4_hemilineages, fig4_vnc_atlas_umap, fig4_access_types; P12),
figure_data (fd_cells, fd_soma_state), MaleCNS skeletons.
Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_fig4_tiles.py
"""
import colorsys
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import figstyle as fs, palette as pal  # noqa: E402
from render import project as rp, scenes as sc  # noqa: E402

fs.use_cell()
OUT = FIG / "renders" / "v4" / "fig4"
PLAN = rp.RESULTS / "plan_v2"
BAND_H = 73.0
CNS_W = 0.703 * BAND_H                       # the unfolded CNS at the aspect of the v1 state tiles
ROSETTA_W = fs.FULL - 2 * CNS_W - 2 * fs.GUTTER
TITLE_MM = 5.0
GAL_W, SPINE_W = 31.0, 13.0                  # Rosetta: galaxy | spine | VNC (mm)
VNC_W = ROSETTA_W - GAL_W - SPINE_W
NEURON_W = (fs.unit(2) - 3 * fs.GUTTER) / 4
NEURON_H = 27.0
BLOOM = (3.0, 0.40, 0.22)
STAIN = {"SK": (0.0, 1.0, 1.0), "Shal": (1.0, 0.0, 1.0), "slo": (1.0, 1.0, 0.0)}          # cyan, magenta, yellow
TIER_GAIN = {"exact": 1.0, "alias": 1.0, "fuzzy": 1.0, "lineage": 0.8, "lineage_pair": 0.65, "class": 0.5}
Z_LO, Z_HI, GAMMA = 0.0, 2.0, 1.3                       # a channel lights only where its gene is above the atlas mean
GHOST = "#5A5A5A"
DUST = "#3A4458"
VNC_VIEW = ((0.42, 1.0, -0.22), (0.0, 0.0, 1.0))      # towards the camera (ventral, slightly left and anterior), posterior down
HIGHLIGHT_UNITS = ("0A", "13A/19A", "9B", "14A")       # high and low SK, emphasised on the spine


def px(mm):
    return int(np.ceil(mm * fs.MM * fs.DPI))            # at least 600 ppi at the printed size


def unit_colours(units, xy):
    """Bright hues for black tiles, ordered by the angle of each unit's UMAP centroid so the field reads as a colour wheel;
    neighbours alternate in saturation and value."""
    c = np.array([xy[u] for u in units])
    ang = np.arctan2(c[:, 1] - c[:, 1].mean(), c[:, 0] - c[:, 0].mean())
    order = np.argsort(ang)
    out = {}
    for rank, i in enumerate(order):
        h = (0.58 + rank / len(units)) % 1.0
        s, v = (0.70, 1.0) if rank % 2 else (0.95, 0.92)
        out[units[i]] = "#%02X%02X%02X" % tuple(int(round(255 * x)) for x in colorsys.hsv_to_rgb(h, s, v))
    return out


def cns_view(xyz, width_mm):
    W, H, top = px(width_mm), px(BAND_H), px(TITLE_MM)
    v0 = rp.fit_view(xyz, W, H - top, unfold=True, margin=0.012, ss=2)
    return rp.View(W, H, v0.um_per_px, v0.centre - np.array([0.0, top / 2 * v0.um_per_px, 0.0]), v0.R, True, 2)


def stain(s, xyz, view):
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.12, fill=0.010)
    w = sc.density_weights(view, xyz)
    mapped = s["tier"].ne("unmapped").to_numpy() & s["z_SK"].notna().to_numpy()
    tile.glow(xyz[~mapped], 1.0, GHOST, 0.06 * w[~mapped], label="unmapped cell bodies")
    gain = s.loc[mapped, "tier"].map(TIER_GAIN).to_numpy(float)
    for g, c in STAIN.items():                                   # one fluorescence channel per gene, added in linear light
        u = np.clip((s.loc[mapped, f"z_{g}"].to_numpy(float) - Z_LO) / (Z_HI - Z_LO), 0, 1) ** GAMMA
        tile.glow(xyz[mapped], 1.1, rp.srgb_to_linear(np.array(c)[None])[0], 0.85 * gain * u * w[mapped], label=f"{g} channel")
    img = tile.resolve(bloom=(3.2, 0.45, 0.20))
    tile.save(OUT / "stain.png", img, dict(mapped_located=int(mapped.sum()), located=int(len(s)), z_range=[Z_LO, Z_HI], gamma=GAMMA,
                                          tier_gain=TIER_GAIN, scale_um_per_px=view.um_per_px))


def triangle(size_mm=11.0):
    """The colour-mixing key of the stain: barycentric mixes of the three channels at full strength, on black."""
    n = px(size_mm)
    img = np.zeros((n, n, 3))
    verts = np.array([[0.5, 0.06], [0.06, 0.88], [0.94, 0.88]]) * n          # SK top, Shal lower left, slo lower right
    yy, xx = np.mgrid[0:n, 0:n] + 0.5
    T = np.array([[verts[0, 0] - verts[2, 0], verts[1, 0] - verts[2, 0]], [verts[0, 1] - verts[2, 1], verts[1, 1] - verts[2, 1]]])
    inv = np.linalg.inv(T)
    d = np.stack([xx - verts[2, 0], yy - verts[2, 1]], -1) @ inv.T
    b = np.dstack([d[..., 0], d[..., 1], 1 - d[..., 0] - d[..., 1]])
    inside = (b >= 0).all(-1)
    cols = np.array([STAIN["SK"], STAIN["Shal"], STAIN["slo"]])
    mix = np.clip(b, 0, 1) @ cols
    mix = mix / np.maximum(mix.max(-1, keepdims=True), 1e-9)
    img[inside] = mix[inside]
    Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(OUT / "triangle.png")
    return dict(size_mm=size_mm, vertices_mm=(verts / n * size_mm).round(3).tolist())


def carriers(s, xyz, view, hem):
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.12, fill=0.010)
    w = sc.density_weights(view, xyz)
    car = s["carrier_scene1"].astype(bool).to_numpy()
    tile.glow(xyz[~car], 1.0, GHOST, 0.045 * w[~car], label="other cell bodies")
    mapped = car & s["z_SK"].notna().to_numpy()
    um = car & ~mapped
    tile.glow(xyz[um], 1.2, "#CFCFCF", 0.20, label="unmapped carriers")
    z = s.loc[mapped, "z_SK"].to_numpy(float)
    u = np.clip(np.abs(z) / 2.0, 0, 1)
    t = 0.5 + np.sign(z) * (0.20 + 0.30 * u)
    tile.glow(xyz[mapped], 1.5, rp.srgb_to_linear(pal.SIGNED_ON_DARK(t)[:, :3]), 1.1 + 1.0 * u, label="mapped carriers by SK z")
    img = tile.resolve(bloom=(3.4, 0.50, 0.18))
    top = hem.sort_values("carrier_share", ascending=False).head(3)["unit"].tolist()
    cents = {}
    for unit in top:
        m = car & (s["source"].eq("allen2020vnc") & s["unit"].astype(str).eq(unit)).to_numpy()
        pxs, pys, _, _ = view.pixels(xyz[m])
        cents[unit] = [float(np.median(pxs) / view.ss), float(np.median(pys) / view.ss)]
    tile.save(OUT / "carriers.png", img, dict(carriers_located=int(car.sum()), mapped_located=int(mapped.sum()), unmapped_located=int(um.sum()),
                                             badge_units=top, badge_px=cents, scale_um_per_px=view.um_per_px))


def peak(pxs, pys, shape, bin_px=6, sigma_bins=1.5):
    """The densest point of a cell cloud in pixel space (smoothed 2D histogram), a truthful anchor for a fibre where a
    median would fall between bilateral or segmental clusters."""
    from scipy.ndimage import gaussian_filter
    h, w = shape
    H2, xe, ye = np.histogram2d(pxs, pys, bins=(max(w // bin_px, 1), max(h // bin_px, 1)), range=((0, w), (0, h)))
    H2 = gaussian_filter(H2, sigma_bins)
    i, j = np.unravel_index(np.argmax(H2), H2.shape)
    return [float((xe[i] + xe[i + 1]) / 2), float((ye[j] + ye[j + 1]) / 2)]


def galaxy(d, colours):
    """The atlas UMAP as a luminous field (pseudo-3D points on a plane, the renderer's glow and bloom), turned upright when
    it is wider than tall (UMAP axes carry no meaning) so it fills its column."""
    u1, u2 = d["umap1"].to_numpy(), d["umap2"].to_numpy()
    core = d["unit"].notna().to_numpy()
    lo1, hi1 = np.percentile(u1[core], [0.5, 99.5])
    lo2, hi2 = np.percentile(u2[core], [0.5, 99.5])
    if hi1 - lo1 > hi2 - lo2:                                       # the robust extent of the clustered cells sets the turn
        u1, u2 = -u2, u1
    k = 1000.0 / max(np.ptp(u1[core]), np.ptp(u2[core]))
    xyz = np.c_[u1 * k, -u2 * k, np.zeros(len(d))]
    W, H = px(GAL_W), px(GAL_W)                                      # square: the cluster cloud is about as tall as wide
    q_lo, q_hi = np.percentile(xyz[core][:, :2], 0.5, axis=0), np.percentile(xyz[core][:, :2], 99.5, axis=0)
    box = np.array([[q_lo[0], q_lo[1], 0.0], [q_hi[0], q_hi[1], 0.0]])
    view = rp.fit_view(box, W, H, R=np.eye(3), margin=0.04, ss=2)      # stray atlas cells may fall off the edge
    tile = rp.Tile(view)
    out = d["unit"].isna().to_numpy()
    tile.glow(xyz[out], 2.4, DUST, 0.24, label="atlas cells outside the units")
    for unit, c in colours.items():
        q = (d["unit"] == unit).to_numpy()
        tile.glow(xyz[q], 2.8, rp.hex_linear(c), 0.48, label=f"unit {unit}")
    img = tile.resolve(bloom=(4.2, 0.60, 0.12))
    cents = {}
    for unit in colours:
        q = (d["unit"] == unit).to_numpy()
        pxs, pys, _, _ = view.pixels(xyz[q])
        cents[unit] = peak(pxs / view.ss, pys / view.ss, (H, W))
    return img, cents


def vnc(df, expr, colours):
    m = df.merge(expr[["cell", "source", "unit"]], on="cell")
    v = m[m["cns_part"].eq("vnc") & m["soma_x_um"].notna()]
    xyz = v[["soma_x_um", "soma_y_um", "soma_z_um"]].to_numpy(float)
    R = rp.look(*VNC_VIEW)
    W, H = px(VNC_W), px(BAND_H - TITLE_MM)
    view = rp.fit_view(xyz, W, H, R=R, margin=0.03, ss=2)
    tile = rp.Tile(view)
    sc.shell(tile, strength=0.12, fill=0.010, )
    lin = v["source"].eq("allen2020vnc").to_numpy()
    tile.glow(xyz[~lin], 1.0, GHOST, 0.10 * sc.density_weights(view, xyz[~lin]), label="VNC cell bodies outside the units")
    col = np.array([rp.hex_linear(colours[u]) for u in v.loc[lin, "unit"].astype(str)])
    tile.glow(xyz[lin], 1.1, col, 0.40, label="VNC cell bodies by hemilineage unit")
    img = tile.resolve(bloom=(3.2, 0.45, 0.20))
    left = xyz[:, 0] > np.median(xyz[:, 0])                         # +x is the fly's left: anchor fibres on one side
    cents = {}
    for u in colours:
        q = lin & v["unit"].astype(str).eq(u).to_numpy() & left
        if q.any():
            pxs, pys, _, _ = view.pixels(xyz[q])
            cents[u] = peak(pxs / view.ss, pys / view.ss, (H, W))
    return img, cents, dict(vnc_located=int(len(v)), in_units=int(lin.sum()), um_per_px=view.um_per_px)


def rosetta(df, expr, hem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MPath
    d = pd.read_csv(PLAN / "fig4_vnc_atlas_umap.csv")
    units = hem["unit"].tolist()
    xy = {u: (float(d.loc[d["unit"] == u, "umap1"].median()), float(d.loc[d["unit"] == u, "umap2"].median())) for u in units}
    colours = unit_colours(units, xy)
    gal, gc = galaxy(d, colours)
    vimg, vc, vmeta = vnc(df, expr, colours)
    mm_gal = GAL_W / gal.shape[1]
    mm_vnc = VNC_W / vimg.shape[1]
    gx = {u: (gc[u][0] * mm_gal, TITLE_MM + gc[u][1] * mm_gal) for u in units}
    vx0 = GAL_W + SPINE_W
    vx = {u: (vx0 + vc[u][0] * mm_vnc, TITLE_MM + vc[u][1] * mm_vnc) for u in vc}
    order = hem.sort_values("z_SK", ascending=False)["unit"].tolist()
    top, bot = TITLE_MM + 4.0, BAND_H - 4.0
    row = (bot - top) / (len(order) - 1)
    spine = {u: (GAL_W + SPINE_W / 2, top + i * row) for i, u in enumerate(order)}
    port = SPINE_W / 2 - 1.2                                        # ribbons end 1.2 mm inside the spine column edges
    fig = plt.figure(figsize=(ROSETTA_W / 25.4, BAND_H / 25.4), dpi=fs.DPI, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
    ax.set_xlim(0, ROSETTA_W)
    ax.set_ylim(BAND_H, 0)
    ax.set_axis_off()
    ax.imshow(gal, extent=(0, GAL_W, TITLE_MM + GAL_W, TITLE_MM), interpolation="lanczos", zorder=1)
    ax.imshow(vimg, extent=(vx0, ROSETTA_W, BAND_H, TITLE_MM), interpolation="lanczos", zorder=1)

    def fibre(a, b, colour, strong):
        (x0, y0), (x1, y1) = a, b
        dx = (x1 - x0) * 0.55
        path = MPath([(x0, y0), (x0 + dx, y0), (x1 - dx, y1), (x1, y1)], [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
        layers = ((2.4, 0.06), (1.2, 0.14), (0.45, 0.85)) if strong else ((1.4, 0.05), (0.7, 0.10), (0.28, 0.55))
        for lw, al in layers:                                      # a soft glow under a fine core
            ax.add_patch(PathPatch(path, facecolor="none", edgecolor=colour, lw=lw, alpha=al, capstyle="round", zorder=3))

    for u in units:
        strong = u in HIGHLIGHT_UNITS
        sx, sy = spine[u]
        fibre(gx[u], (sx - port, sy), colours[u], strong)
        if u in vx:
            fibre((sx + port, sy), vx[u], colours[u], strong)
            ax.add_patch(plt.Circle(vx[u], 0.5 if strong else 0.32, facecolor="none", edgecolor="white", lw=0.35, alpha=0.9, zorder=4))
        ax.add_patch(plt.Circle(gx[u], 0.5 if strong else 0.32, facecolor="none", edgecolor="white", lw=0.35, alpha=0.9, zorder=4))
    # SK swatches on the spine (dark-centred SK map, as B)
    sw = {}
    for u in order:
        z = float(hem.loc[hem["unit"] == u, "z_SK"].iloc[0])
        uu = min(abs(z) / 2.0, 1.0)
        c = pal.SIGNED_ON_DARK(0.5 + np.sign(z) * (0.20 + 0.30 * uu))
        sx, sy = spine[u]
        ax.add_patch(Rectangle((sx + port - 2.2, sy - 0.75), 2.2, 1.5, facecolor=c, edgecolor="none", zorder=4))
        sw[u] = [sx + port - 2.2, sy]
    fig.savefig(OUT / "rosetta.png", dpi=fs.DPI, facecolor="black")
    plt.close(fig)
    meta = dict(width_mm=ROSETTA_W, galaxy_mm=GAL_W, spine_mm=SPINE_W, vnc_mm=VNC_W, title_mm=TITLE_MM, unit_colours=colours,
                spine_order=order, spine_mm_xy={u: [spine[u][0] - port + 0.3, spine[u][1]] for u in order}, swatch_mm=sw,
                galaxy_centroids_mm=gx, vnc_centroids_mm=vx, highlight_units=HIGHLIGHT_UNITS, atlas_cells=int(len(d)),
                atlas_in_units=int(d["unit"].notna().sum()), **vmeta)
    (OUT / "rosetta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    return colours


def neuron_tiles(df, expr):
    types = pd.read_csv(PLAN / "fig4_access_types.csv")
    soma = sc.cells().set_index("cell")
    meta = []
    for _, r in types.iterrows():
        cand = df.merge(expr[["cell", "carrier_scene1"]], on="cell")
        cand = cand[(cand["type"] == r["type"]) & cand["carrier_scene1"].astype(bool)].copy()
        cand["absdv"] = soma.loc[cand["cell"], "dv_m0_mv"].abs().to_numpy()
        body = None
        for b in cand.sort_values("absdv", ascending=False)["body_id"]:
            try:
                sc.skeleton_path(int(b))
                body = int(b)
                break
            except FileNotFoundError:
                continue
        if body is None:
            raise RuntimeError(f"no carrier skeleton for {r['type']}")
        c = df.set_index("body_id").loc[body]
        tile, img, colour = sc.neuron_tile(df, body, c["superclass"], c[["soma_x_um", "soma_y_um", "soma_z_um"]].to_numpy(float),
                                           px(NEURON_W), px(NEURON_H), BLOOM)
        name = f"access_{r['type']}"
        info = dict(type=r["type"], body_id=body, driver_lines=r["driver_lines"], scale_um_per_px=tile.view.um_per_px,
                    scale_bar_um=sc.scale_bar_um(tile.view), colour=colour, skeleton=sc.skeleton_path(body).relative_to(rp.ROOT).as_posix())
        tile.save(OUT / f"{name}.png", img, info)
        meta.append(dict(name=name, **info))
    return meta


def main():
    t = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(sc.FIGURE_DATA / "fd_cells.csv", low_memory=False)
    expr = pd.read_csv(PLAN / "fig4_cell_expression.csv", low_memory=False)
    hem = pd.read_csv(PLAN / "fig4_hemilineages.csv")
    m = df.merge(expr.drop(columns=["body_id"]), on="cell", validate="one_to_one")
    s, xyz = sc.somata(m)
    view = cns_view(xyz, CNS_W)
    stain(s, xyz, view)
    tri = triangle()
    print(f"stain {time.time() - t:.1f}s", flush=True)
    carriers(s, xyz, view, hem)
    print(f"carriers {time.time() - t:.1f}s", flush=True)
    colours = rosetta(df, expr, hem)
    print(f"rosetta {time.time() - t:.1f}s", flush=True)
    neurons = neuron_tiles(df, expr)
    (OUT / "tiles.json").write_text(json.dumps(dict(band_mm=BAND_H, cns_mm=CNS_W, rosetta_mm=ROSETTA_W, title_mm=TITLE_MM,
                                                    neuron_mm=[NEURON_W, NEURON_H], unit_colours=colours, highlight_units=HIGHLIGHT_UNITS,
                                                    stain=STAIN, z_range=[Z_LO, Z_HI], gamma=GAMMA, tier_gain=TIER_GAIN, triangle=tri,
                                                    neurons=neurons), indent=1), encoding="utf-8", newline="\n")
    print(f"done {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
