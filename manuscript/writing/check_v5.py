"""Runs check_text.py on the drafted parts of manuscript/NARRATIVE_V5_CELL.md (between the draft markers) with the number
ledger, and checks that the Results headings are the built figure titles. Exit code 1 when a check fails.

  python -I manuscript/writing/check_v5.py
"""
import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import check_text as ct  # noqa: E402


def figure_titles():
    out = []
    for i in range(1, 8):
        doc = ast.get_docstring(ast.parse((ROOT / "manuscript/figures/build/v4" / f"fig{i}.py").read_text(encoding="utf-8")))
        m = re.match(r"v4 Fig\. \d \(Cell[^)]*\) - (.+?) \(174 mm", doc.splitlines()[0])
        out.append(m.group(1))
    return out


def main():
    text = (ROOT / "manuscript/NARRATIVE_V5_CELL.md").read_text(encoding="utf-8")
    draft = text.split("<!-- draft:start -->", 1)[1].split("<!-- draft:end -->", 1)[0]
    fails, warns = ct.check(draft, ct.ledger_values(HERE / "number_ledger.csv"))
    heads = ct.parse(draft)["headings"]
    titles = figure_titles()
    if heads != titles:
        fails.append(f"Results headings differ from the figure titles: {[h for h, t in zip(heads, titles) if h != t]}")
    for w in warns:
        print("warn:", w)
    for f in fails:
        print("FAIL:", f)
    print(f"{len(fails)} failure(s), {len(warns)} warning(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
