"""v4 light silhouette icons (the MaleCNS grammar of Figs 6A and 6I): flat warm-grey CNS and brain shapes with one outline and
no inner neuropil lines, a lesion site in purple outlined dark. Read by the builds of Fig. 7B (the brain state at learning and
at test) and Fig. 7F (where each disease case acts).

  cns_healthy, cns_lesion   the whole CNS, healthy or under the pan-neuronal Kv4 loss
  cns_split                 left half healthy, right half lesioned (learned healthy, then relearned in the lesion)
  brain_C01 ... brain_C16   the brain, each case's site: pan-neuronal (C01), mushroom bodies (C04, C16), antennal lobes (C13)

Data: the neuropil meshes (render/scenes.py). Run:
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_icons.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import figstyle as fs  # noqa: E402
from render import project as rp, scenes as sc  # noqa: E402

fs.use_cell()
OUT = FIG / "renders" / "v4" / "icons"
HEALTHY, LESION, LINE, LESION_LINE = "#E6E3E0", "#7B3294", "#B5AFA9", "#3F1650"   # MaleCNS warm grey; the lesion purple of Fig. 7
MB = ("CA(", "PED(", "aL(", "a'L(", "bL(", "b'L(", "gL(")
CNS_MM, BRAIN_MM = (4.0, 6.4), (8.0, 4.1)
SITES = {"C01": ("pan-neuronal", lambda m: True), "C04": ("mushroom bodies", lambda m: m["label"].startswith(MB)),
         "C13": ("antennal lobes", lambda m: m["label"].startswith("AL(")), "C16": ("mushroom bodies", lambda m: m["label"].startswith(MB))}


def icon(width_mm, height_mm, brain_only, lesion):
    """One flat silhouette: an outline around the whole shape and around the lesion site, no inner lines. lesion(meta) -> True
    for regions in the lesion colour (they show through the neuropils in front of them)."""
    regions = [r for r in sc.regions() if sc.no_lamina(r[0]) and (not brain_only or r[0]["dataset"] != "vnc")]
    pts = np.vstack([v for _, v, _ in regions])
    px = lambda mm: int(round(mm * fs.MM * fs.DPI))  # noqa: E731
    view = rp.fit_view(pts, px(width_mm), px(height_mm), unfold=True, margin=0.03, ss=3)
    H, W = view.shape
    depth, rid = np.full((H, W), np.inf), np.full((H, W), -1, np.int64)
    depth2, rid2 = np.full((H, W), np.inf), np.full((H, W), -1, np.int64)
    for meta, v, f in regions:
        x, y, z, _ = view.pixels(v, side=meta["dataset"] == "vnc")
        rp._zbuffer(x, y, z, f, 0, depth, rid)
        if lesion(meta):
            rp._zbuffer(x, y, z, f, 1, depth2, rid2)
    rid[rid2 == 1] = 1
    img = np.empty((H, W, 3))
    img[:] = rp.hex_linear("#FFFFFF")
    img[rid == 0] = rp.hex_linear(HEALTHY)
    img[rid == 1] = rp.hex_linear(LESION)
    inside, les = rid >= 0, rid == 1
    outer = np.zeros((H, W), bool)
    outer[:, 1:] |= inside[:, 1:] != inside[:, :-1]
    outer[1:, :] |= inside[1:, :] != inside[:-1, :]
    site = np.zeros((H, W), bool)
    site[:, 1:] |= les[:, 1:] != les[:, :-1]
    site[1:, :] |= les[1:, :] != les[:-1, :]
    whole = bool(les[inside].all())
    img[binary_dilation(outer, iterations=2)] = rp.hex_linear(LESION_LINE if whole else LINE)
    if les.any() and not whole:
        img[binary_dilation(site, iterations=2)] = rp.hex_linear(LESION_LINE)
    img = img.reshape(H // 3, 3, W // 3, 3, 3).mean(axis=(1, 3))
    return (rp.linear_to_srgb(img) * 255 + 0.5).astype(np.uint8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    healthy = icon(*CNS_MM, False, lambda m: False)
    lesion = icon(*CNS_MM, False, lambda m: True)
    split = healthy.copy()
    split[:, healthy.shape[1] // 2:] = lesion[:, healthy.shape[1] // 2:]
    meta = {}
    for name, img, mm in (("cns_healthy", healthy, CNS_MM), ("cns_lesion", lesion, CNS_MM), ("cns_split", split, CNS_MM)):
        Image.fromarray(img).save(OUT / f"{name}.png")
        meta[name] = dict(mm=list(mm))
    for code, (site, sel) in SITES.items():
        Image.fromarray(icon(*BRAIN_MM, True, sel)).save(OUT / f"brain_{code}.png")
        meta[f"brain_{code}"] = dict(mm=list(BRAIN_MM), site=site)
    (OUT / "icons.json").write_text(json.dumps(dict(icons=meta, healthy=HEALTHY, lesion=LESION, line=LINE, lesion_line=LESION_LINE),
                                               indent=1), encoding="utf-8")
    print("icons:", sorted(meta))


if __name__ == "__main__":
    main()
