"""Palette checks (CIE76 distances under colour-vision simulations; colour-map lightness). Writes qc/palette.csv."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from lib import palette as pal, qc  # noqa: E402

SETS = {
    "superclass groups on black": ({**pal.ATLAS_ON_DARK, "input": pal.TILE["input"]}, "#000000"),
    "cues on white": ({str(k): v for k, v in pal.CUE.items()}, "#FFFFFF"),
    "cues on black": ({str(k): v for k, v in pal.CUE_ON_DARK.items()}, "#000000"),
    "learners on white": (pal.LEARNER, "#FFFFFF"),
}
MAPS = {"signed on dark": pal.SIGNED_ON_DARK, "magnitude on dark": pal.MAGNITUDE_ON_DARK, "cue diverging on white": pal.diverging_on_white()}


def main():
    rows, failed = [], []
    for name, (cols, bg) in SETS.items():
        res = qc.distinguishable(cols, 10.0, background=bg)
        for vision, r in res.items():
            rows.append(dict(set=name, vision=vision, min_delta_e=round(r["min"], 1), below_10="; ".join(f"{a}~{b} ({d})" for a, b, d in r["below"])))
            if r["below"] and vision != "greyscale":
                failed.append((name, vision, r["below"]))
    for name, cmap in MAPS.items():
        ok, how = qc.lightness_monotonic(cmap, tol=0.05)
        rows.append(dict(set=f"map: {name}", vision="L*", min_delta_e=None, below_10=how))
        if not ok:
            failed.append((name, "L*", how))
    out = qc.write_rows(HERE / "qc" / "palette.csv", rows)
    for r in rows:
        print(f"{r['set']:28s} {r['vision']:12s} {str(r['min_delta_e']):>6s}  {r['below_10']}")
    print(f"written {out}")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
