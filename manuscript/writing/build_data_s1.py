r"""Builds Data S1, the source data of the Cell paper's figures, into manuscript/paper/build/supplement/: one .csv per figure
(Figures 1-7 and S1-S7) in Data_S1/, a README.txt with each figure's title, and Data_S1.zip of the folder.

The tables are the figure builds' own source data (manuscript/figures/v4/source_data), unchanged except that the study's
development codes are translated with the names of Tables S2-S4 (planning maps, test families, learners) and British
spellings in column names and labels (the renderer's "colour", "centre") take their American forms. The builder fails if
a code or a British spelling survives in any cell or column name, and a rebuild is byte-identical. Snake-case column
names stay: they are the tables' machine-readable keys.

  .\MIGRATION\python\python.exe -B -X utf8 MIGRATION\launch.py --script manuscript\writing\build_data_s1.py
"""
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_tables_s2_s4 as s2s4  # noqa: E402  (the reader-facing names of Tables S2-S4)
import check_tex  # noqa: E402
from lib.names import BRITISH, BRITISH_RE, DEV_LABELS, british  # noqa: E402

SOURCE = ROOT / "manuscript" / "figures" / "v4" / "source_data"
PAPER = ROOT / "manuscript" / "paper"
OUT = PAPER / "build" / "supplement"
FIGURES = [(f"fig{i}", str(i)) for i in range(1, 8)] + [(f"s{i}", f"S{i}") for i in range(1, 8)]
DATE = (1980, 1, 1, 0, 0, 0)                      # fixed member dates: a rebuild is byte-identical
SERIES = re.compile(r"(\w+?)_((?:remote-dynamic-)?M0-(?:head_only|joint))")
IDENTIFIERS = re.compile(r"remote-dynamic|supp-|difficulty-|head_only|\bnan\b")   # lib.names.INTERNAL without its "_" rule
RENAME = {"run_id": "planning_map"}                # the column of Figure S1D, which names the planning map of each run


def family(code):
    """'F14.SC.kv4 (class pattern)' -> the family's name in Table S2, with the qualifier kept."""
    base, _, rest = code.partition(" ")
    return s2s4.FAMILIES[base][0] + (f" {rest}" if rest else "")


def series(value):
    """'trained_solved_remote-dynamic-M0-joint' -> 'trained solved, scene 1, brain and readout trained'; others unchanged."""
    m = SERIES.fullmatch(value)
    return f"{m.group(1).replace('_', ' ')}, {s2s4.LEARNER[m.group(2)]}" if m else value


def american(text):
    """British spellings in a label or column name in their American form ('colours' -> 'colors', 'centre_um' -> 'center_um')."""
    def fix(m):
        us = BRITISH[m.group(1).lower()]
        return us[0].upper() + us[1:] if m.group(0)[0].isupper() else us
    return BRITISH_RE.sub(fix, text)


TRANSLATE = {"map_id": s2s4.planning_map, "run_id": lambda v: s2s4.planning_map(v.removeprefix("difficulty-")), "family": family,
             "series": series}


def titles():
    """Figure number -> title, from the legends the paper prints (TeX turned into text as the checker does)."""
    ledger = {r["id"]: r for r in csv.DictReader(open(HERE / "number_ledger.csv", encoding="utf-8"))}
    out = {}
    for f in ("legends/figures.tex", "legends/si_figures.tex"):
        tex = re.sub(r"(?<!\\)%.*", "", (PAPER / f).read_text(encoding="utf-8"))
        for m in re.finditer(r"\\deflegend\{(S?\d+)\}\{", tex):
            body, _ = check_tex.group(tex, m.end() - 1)
            out[m.group(1)] = re.sub(r"\s+", " ", check_tex.convert(body, check_tex.Doc(), "Figure legends", ledger, False)).strip()
    return out


def table(name):
    rows = list(csv.reader(open(SOURCE / f"{name}.csv", encoding="utf-8", newline="")))
    header, body = rows[0], rows[1:]
    for col, fn in TRANSLATE.items():
        if col in header:
            i = header.index(col)
            for r in body:
                if i < len(r) and r[i]:
                    r[i] = fn(r[i])
    header = [american(RENAME.get(h, h)) for h in header]
    body = [[american(v) for v in r] for r in body]
    bad = []
    for r in [header] + body:
        for v in r:
            for rx in (IDENTIFIERS, DEV_LABELS):
                bad += [f"{name}: {m.group(0)!r} in {v[:80]!r}" for m in rx.finditer(v)]
            bad += [f"{name}: British spelling {w!r} in {v[:60]!r}" for w, _ in british(v)]
    if bad:
        raise SystemExit("development labels or spelling in Data S1:\n  " + "\n  ".join(bad[:30]))
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows([header] + body)
    return buf.getvalue(), len(body)


def readme(names):
    lines = ["Data S1. Source data for Figures 1–7 and S1–S7", "",
             "The values behind every panel of the paper's figures, one table per figure (.csv, UTF-8). Each row belongs to one",
             "panel (column \"panel\") and one plotted series (column \"series\"); the other columns hold the values that series",
             "plots and are empty where they do not apply. Several values in one cell are separated by semicolons.", ""]
    lines += [f"{f'Figure_{n}.csv':<17}Figure {n}. {names[n]}" for _, n in FIGURES]
    return "\n".join(lines) + "\n"


def main():
    names = titles()
    files = {}
    for src, n in FIGURES:
        text, rows = table(src)
        files[f"Data_S1/Figure_{n}.csv"] = text
        print(f"Figure {n}: {rows} rows")
    files["Data_S1/README.txt"] = readme(names)
    for rel, text in files.items():
        p = OUT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="")
    with zipfile.ZipFile(OUT / "Data_S1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for rel in sorted(files):
            info = zipfile.ZipInfo(rel, DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, files[rel].encode("utf-8"))
    print(f"{(OUT / 'Data_S1.zip').relative_to(ROOT)}: {len(files)} files")


if __name__ == "__main__":
    main()
