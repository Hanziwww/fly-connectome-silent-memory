"""v4 Fig. 1 hero band at Cell width: the frozen Fig. 1 tiles (render/fig1_tiles.py: the unfolded atlas of 140,024 glowing
somata and the dorsal-oblique interface with readout arbors, input somata, DNp01 and exemplar insets), re-rendered so that
both tiles share one height across 174 mm. Same scenes, colours and bloom; only the band geometry changes.

Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\render\\v4_fig1_tiles.py
"""
import json
import sys
import time
from pathlib import Path

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import figstyle as fs  # noqa: E402
from render import fig1_tiles as base  # noqa: E402

fs.use_cell()
OUT = FIG / "renders" / "v4" / "fig1"
base.OUT = OUT
base.BAND_H = (fs.FULL - fs.GUTTER) / (base.ASPECT_ATLAS + base.ASPECT_INTERFACE)
base.ATLAS_MM = (base.ASPECT_ATLAS * base.BAND_H, base.BAND_H)
base.INTERFACE_MM = (fs.FULL - fs.GUTTER - base.ATLAS_MM[0], base.BAND_H)


def main():
    t = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    base.atlas()
    print(f"atlas {time.time() - t:.1f}s", flush=True)
    base.interface()
    print(f"interface {time.time() - t:.1f}s", flush=True)
    (OUT / "band.json").write_text(json.dumps(dict(band_height_mm=base.BAND_H, atlas_mm=base.ATLAS_MM,
                                                   interface_mm=base.INTERFACE_MM), indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
