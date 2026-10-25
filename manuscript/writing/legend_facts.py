r"""A drafting aid for the figure legends: for every panel of the 14 v4 figures, what its source data records (series, row
counts, and the statistics a legend must state: n, runs, replicates, tests, families, adjusted P, intervals, medians). Reads
manuscript/figures/v4/source_data/*.csv (written by the figure builds); writes manuscript/paper/build/legend_facts.md.

  .\MIGRATION\python\python.exe -B -X utf8 -I manuscript\writing\legend_facts.py [fig1 ...]
"""
import csv
import re
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "manuscript" / "figures" / "v4" / "source_data"
OUT = ROOT / "manuscript" / "paper" / "build" / "legend_facts.md"
FIGS = [f"fig{i}" for i in range(1, 8)] + [f"s{i}" for i in range(1, 8)]
STAT = re.compile(r"(^n$|^n_|_n$|runs?|replicat|learner|test|famil|holm|bh_|_p$|^p_|p_max|p_min|ci|interval|median|mean|range|lo$|hi$|"
                  r"min|max|timings|seeds?|draws?|sets?$|cells$|histories|of$|statistic|equivalence|estimate|sd$|sem)", re.I)


def short(v, n=60):
    v = str(v)
    return v if len(v) <= n else v[:n] + "..."


def facts(name):
    rows = list(csv.DictReader(open(SD / f"{name}.csv", encoding="utf-8")))
    out = [f"## {name}"]
    panels = OrderedDict()
    for r in rows:
        panels.setdefault(r["panel"], OrderedDict()).setdefault(r["series"], []).append(r)
    for p, series in panels.items():
        out.append(f"### {name} {p}")
        for s, rs in series.items():
            cols = [k for k in rs[0] if k not in ("panel", "series") and any(x.get(k) not in ("", None) for x in rs)]
            stat = [k for k in cols if STAT.search(k)]
            line = f"- {s or '(no series)'}: {len(rs)} row(s); columns: {', '.join(cols[:14])}" + (" ..." if len(cols) > 14 else "")
            out.append(line)
            for k in stat[:10]:
                vals = sorted({x[k] for x in rs if x.get(k) not in ("", None)}, key=lambda v: (len(v), v))
                out.append(f"  - {k}: {short('; '.join(vals[:8]), 160)}" + (f" ({len(vals)} distinct)" if len(vals) > 8 else ""))
    return out


def main():
    names = sys.argv[1:] or FIGS
    lines = ["# Legend facts (from the figure source data; a drafting aid, not a deliverable)", ""]
    for n in names:
        lines += facts(n) + [""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(names)} figures -> {OUT.relative_to(ROOT)} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
