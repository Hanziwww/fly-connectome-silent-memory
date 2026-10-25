"""Shared helpers for the figure builds: paths, result tables, export, source data and QC."""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

FIG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIG))
from lib import figstyle as fs, qc  # noqa: E402

ROOT = FIG.parents[1]
RESULTS = Path(os.environ.get("FLYBRAIN_RESULTS", ROOT / "manuscript" / "results"))   # preview builds may read another results root
RENDERS = FIG / "renders"
OUT = {"main": FIG / "main", "extended": FIG / "extended"}
QC_DIR = FIG / "qc"
SOURCE = FIG / "source_data"


def use_v4() -> None:
    """The v4 (Cell) figures: Cell style profile; outputs, QC reports and source data under manuscript/figures/v4, so the
    frozen v1 files are never touched."""
    global OUT, QC_DIR, SOURCE
    fs.use_cell()
    base = Path(os.environ.get("FLYBRAIN_V4_OUT", FIG / "v4"))     # preview builds write elsewhere (e.g. v4/preview)
    OUT = {"main": base / "main", "supplement": base / "supplement"}
    QC_DIR = base / "qc"
    SOURCE = base / "source_data"


@lru_cache(maxsize=None)
def table(rel: str, **kw) -> pd.DataFrame:
    """A deliverable table from manuscript/results (read once per build)."""
    return pd.read_csv(RESULTS / rel, low_memory=False, **kw)


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def correction(test_ids) -> str:
    """The multiple-comparison correction ('holm' or 'bh') shared by a set of tests, read from statistical_tests.csv.
    Figure labels are derived from it and never hard-coded (freeze-v1 labelled two BH families 'Holm')."""
    t = table("stats/statistical_tests.csv").set_index("test_id")
    kinds = set(t.loc[list(test_ids), "family_correction"].astype(str))
    if len(kinds) != 1 or not kinds <= {"holm", "bh"}:
        raise ValueError(f"mixed or unknown corrections: {sorted(kinds)}")
    return kinds.pop()


def adjusted_label(test_ids) -> str:
    """'Holm P' or 'BH q' (mathtext italics) for a set of tests that share one correction."""
    return {"holm": f"Holm {fs.P_SYM}", "bh": f"BH {fs.Q_SYM}"}[correction(test_ids)]


def finish(cv: "fs.Canvas", name: str, sd: "qc.SourceData", kind: str = "main", allow_overlap: tuple = (), allow_names: tuple = ()) -> list[str]:
    """Save the figure, write its source data and QC report (check list, colour-vision previews). Returns issues."""
    issues = qc.check(cv, allow_overlap=allow_overlap, allow_names=allow_names)
    files = cv.save(OUT[kind], name, preview_dir=QC_DIR / "previews")
    sd.write(SOURCE)
    preview = QC_DIR / "previews" / f"{name}.png"
    qc.simulate_png(preview, QC_DIR / "vision")
    report = dict(figure=name, issues=issues, files=[f.relative_to(FIG).as_posix() for f in files],
                  inputs=sd.inputs, size_mm=[cv.w, cv.h],
                  tiles=[dict(label=t["label"], rect_mm=[round(v, 3) for v in t["rect"]], pixels=list(t["pixels"]),
                              ppi=round(t["pixels"][0] / (t["rect"][2] * fs.MM), 1)) for t in cv.tiles])
    QC_DIR.mkdir(parents=True, exist_ok=True)
    (QC_DIR / f"{name}_qc.json").write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    print(f"[{name}] {len(issues)} QC issue(s)")
    for i in issues:
        print("   -", i)
    cv.close()
    return issues
