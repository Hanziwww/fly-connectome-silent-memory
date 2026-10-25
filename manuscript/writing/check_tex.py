r"""Checks the LaTeX manuscript in manuscript/paper against the Cell author guide, the voice guide and the number ledger.
Exit code 1 when a check fails; warnings are printed but do not fail (with --strict, parts not yet drafted and fields left
to fill fail too: the check before submission).

  .\MIGRATION\python\python.exe -B -X utf8 -I manuscript\writing\check_tex.py [--strict] [--report FILE]

It reads main.tex and every file it inputs, turns the TeX into the Markdown form that check_text.py checks (title, Summary,
Highlights, In Brief, Introduction, Results with its headings, Discussion, Limitations of the study, figure legends) and
runs all of check_text's rules on it. Then:
- the Cell word count: every part except STAR Methods, supplemental item legends ("uncounted"), supplemental figure legends
  and the references; it fails at 7,000 and warns above 6,800; the front matter (title, Highlights, In Brief, keywords) is
  reported apart;
- numbers: every \val{ID} exists in the ledger, and every number typed in the prose (STAR Methods included, through the
  ledger's S group) is a ledger value; figure, panel, video, table and citation numbers, the labels "scene N", "cue N" and
  "dose N", \ident{} (software versions, identifiers), formulas and the key resources table (krt) are exempt;
- STAR Methods also passes check_text's voice rules (spelling, internal labels, dropped words, sentence length), outside the
  key resources table and the formulas;
- the Results headings equal the figure titles (the build docstrings);
- call-outs (\fig, \figs, \figr): every one names an existing figure and panel (the source data's panels), every main
  panel is called out in the Results, the main figures are first called out in order, and the supplemental items, videos
  and tables called out exist;
- each main legend says "See also Figure S<N>";
- citations: every \cellcite key is in refs.bib and every refs.bib entry is cited (unless \nocite{*} stands in);
- class (c) terms (the user's rule) appear nowhere but STAR Methods;
- the sections and statements the author guide requires are present.
"""
import argparse
import ast
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT / "manuscript" / "paper"
FIGS = ROOT / "manuscript" / "figures"
sys.path.insert(0, str(HERE))
import check_text as ct  # noqa: E402

WORD_LIMIT, WORD_WARN = 7000, 6800
# Introduction, Discussion and Limitations were deepened at the user's request after Gate C (PLAN.md section 27)
TARGETS = {"Summary": 150, "Introduction": 650, "Results": 3350, "Discussion": 1100, "Limitations of the study": 340,
           "Figure legends": 1450}
# parts the Cell word count includes (STAR Methods, supplemental legends and references are excluded)
COUNTED = ["Summary", "Introduction", "Results", "Discussion", "Limitations of the study", "Resource availability", "Acknowledgments",
           "Author contributions", "Declaration of interests", "Declaration of generative AI", "Supplemental information",
           "Figure legends"]
FRONT = ["Title", "Highlights", "In Brief", "Keywords"]
# parts whose numbers must be ledger values (STAR Methods through the ledger's S group)
NUMBERED = ["Summary", "Highlights", "In Brief", "Introduction", "Results", "Discussion", "Limitations of the study", "Figure legends",
            "Supplemental information", "Supplemental figure legends", "STAR Methods"]
MATH_ENVS = ("equation", "equation*", "align", "align*")
KRT = re.compile(r"\\begin\{krt\}(.*?)\\end\{krt\}", re.S)
CLASS_C = ["susceptib", "ED50", "zero-update", "stagnation", "matched random cells cost"]
LABELS = re.compile(r"\b(?:scenes?|cues?|doses?|learners?|replicates?)\s+[−-]?\d+(?:\.\d+)?(?:\s*(?:,|and|or|to)\s*\d+(?:\.\d+)?)*\b",
                    re.IGNORECASE)
FILES = {  # every file main.tex may input, with its part
    "front/title.tex": "Title", "front/summary.tex": "Summary", "front/highlights.tex": "Highlights", "front/inbrief.tex": "In Brief",
    "front/keywords.tex": "Keywords", "body/introduction.tex": "Introduction", "body/results.tex": "Results",
    "body/discussion.tex": "Discussion", "body/limitations.tex": "Limitations of the study", "back/resource.tex": "Resource availability",
    "back/acknowledgments.tex": "Acknowledgments", "back/contributions.tex": "Author contributions",
    "back/interests.tex": "Declaration of interests", "back/ai.tex": "Declaration of generative AI",
    "back/si_index.tex": "Supplemental information", "legends/figures.tex": "Figure legends", "generated/values.tex": None,
    "legends/si_figures.tex": "Supplemental figure legends",          # Document S1 (si.tex): checked, not counted
    "methods/star.tex": "STAR Methods",                                # checked, not counted
    "generated/table_s1.tex": None,     # Table S1, generated from the screen's record (build_table_s1.py checks its text)
}
NARGS = {"val": 1, "fig": 1, "figs": 2, "figr": 2, "video": 1, "tab": 1, "dataS": 1, "gene": 1, "textit": 1, "emph": 1, "textbf": 1,
         "textsuperscript": 1, "textsubscript": 1, "mbox": 1, "text": 1, "textrm": 1, "cellcite": 1, "cite": 1, "citep": 1, "citet": 1,
         "nocite": 1, "todraft": 1, "tofill": 1, "resultsfigure": 1, "href": 2, "url": 1, "section": 1, "subsection": 1,
         "subsubsection": 1, "deflegend": 3, "setfigfile": 2, "relatedto": 2, "begin": 1, "end": 1, "input": 1, "printlegend": 1,
         "figurepage": 1, "sifigure": 1, "thispagestyle": 1, "cellval": 2, "hyphenpenalty": 0, "ident": 1, "krtgroup": 1}
WORDS = {"ldots": "...", "textendash": "\u2013", "textemdash": "\u2014", "Pv": "P", "times": "\u00d7", "item": "\n- ", "par": "\n\n",
         "noindent": "", "medskip": "\n", "smallskip": "\n", "bigskip": "\n", "clearpage": "", "raggedright": "", "Large": "",
         "large": "", "bfseries": "", "small": "", "footnotesize": "", "quad": " ", "qquad": " ", "linebreak": " ", "newline": " "}


class Doc:
    def __init__(self):
        self.parts = {}            # part -> text (for counting and the voice rules)
        self.numbers = {}          # part -> text with \val replaced by a sentinel (for the number check)
        self.headings = []         # Results headings
        self.callouts = []         # (part, position, item) e.g. ("Results", 12, "2E")
        self.cites = set()
        self.vals = []
        self.todraft = []
        self.tofill = []
        self.nocite_all = False
        self.legends = {}          # figure number -> (title, body)
        self.uncounted = {}        # part -> words excluded from the count


def group(s, i, open_="{", close="}"):
    """The text of the balanced group that starts at s[i] (an opening brace) and the index after it."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == open_:
            depth += 1
        elif s[j] == close:
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError(f"unbalanced {open_} at {i}")


def callout_items(name, args):
    """The figure items a call-out names: \\fig{2E} -> [2E]; \\figs{2B}{2C} -> [2B, 2C]; \\figr{2B}{2D} -> [2B, 2C, 2D]."""
    if name == "fig":
        return [args[0].strip()]
    if name == "figs":
        return [a.strip() for a in args]
    a, b = (x.strip() for x in args)
    ma, mb = re.fullmatch(r"(S?\d+)([A-Z])", a), re.fullmatch(r"(S?\d+)([A-Z])", b)
    if ma and mb and ma.group(1) == mb.group(1):
        return [ma.group(1) + chr(c) for c in range(ord(ma.group(2)), ord(mb.group(2)) + 1)]
    return [a, b]


def convert(s, doc, part, ledger, sentinel):
    """TeX -> plain text for one part (sentinel: \\val as a placeholder that the number check skips)."""
    s = re.sub(r"(?<!\\)%.*", "", s)
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z@]+)\*?", s[i:])
            if not m:
                nxt = s[i + 1:i + 2]
                out.append({"%": "%", "&": "&", "#": "#", "_": "_", "$": "$", ",": " ", " ": " ", "\\": "\n", ";": " ", ":": " "}.get(nxt, ""))
                i += 2
                continue
            name, j = m.group(1), i + m.end()
            if j < len(s) and s[j] == "[":
                _, j = group(s, j, "[", "]")
            args = []
            for _ in range(NARGS.get(name, 0)):
                while j < len(s) and s[j] in " \n":
                    j += 1
                if j < len(s) and s[j] == "{":
                    a, j = group(s, j)
                    args.append(a)
                else:
                    args.append("")
            i = j
            if name == "val":
                doc.vals.append((part, args[0]))
                if sentinel:
                    out.append(" \u27e6v\u27e7 ")
                else:
                    out.append(ledger.get(args[0], {}).get("display", f"?{args[0]}?").replace("^", ""))
            elif name in ("fig", "figs", "figr"):
                items = callout_items(name, args)
                doc.callouts.extend((part, len("".join(out)), it) for it in items)
                word = "Figure" if name == "fig" else "Figures"
                out.append(f"{word} " + (" and ".join(items) if name == "figs" else (items[0] if name == "fig" else f"{items[0]}\u2013{items[-1]}")))
            elif name in ("video", "tab", "dataS"):
                label = {"video": "Video", "tab": "Table", "dataS": "Data"}[name]
                doc.callouts.append((part, len("".join(out)), f"{label} {args[0].strip()}"))
                out.append(f"{label} {args[0]}")
            elif name in ("cellcite", "cite", "citep", "citet", "nocite"):
                keys = [k.strip() for k in args[0].split(",") if k.strip()]
                if name == "nocite" and keys == ["*"]:
                    doc.nocite_all = True
                else:
                    doc.cites.update(keys)
            elif name == "todraft":
                doc.todraft.append((part, args[0]))
            elif name == "tofill":
                doc.tofill.append((part, args[0]))
            elif name == "subsection" and part == "Results":
                h = convert(args[0], doc, part, ledger, False).strip()
                doc.headings.append(h)
                out.append(f"\n\u00a7H {h}\n")
            elif name in ("section", "subsection", "subsubsection"):
                out.append("\n" + convert(args[0], doc, part, ledger, sentinel) + ".\n")
            elif name == "deflegend":
                t = convert(args[1], doc, part, ledger, False).strip()
                body = convert(args[2], doc, part, ledger, sentinel)
                doc.legends[args[0].strip()] = (t, convert(args[2], Doc(), part, ledger, False))
                out.append(f"\nFigure {args[0]}. {t}.\n{body}\n")
            elif name == "begin" and args[0] in MATH_ENVS:      # display formulas: skipped whole
                end = "\\end{" + args[0] + "}"
                k = s.find(end, i)
                i = len(s) if k < 0 else k + len(end)
                out.append(" \u27e6\u27e7 ")
            elif name == "ident":                                 # versions and identifiers: printed, never checked as numbers
                out.append(" \u27e6i\u27e7 " if sentinel else args[0])
            elif name == "begin" and args[0] == "uncounted":
                out.append("\u27eaU\u27eb")
            elif name == "end" and args[0] == "uncounted":
                out.append("\u27ea/U\u27eb")
            elif name in ("begin", "end", "input", "setfigfile", "relatedto", "resultsfigure", "printlegend", "figurepage", "sifigure",
                          "thispagestyle", "cellval"):
                pass
            elif name == "href":
                out.append(convert(args[1], doc, part, ledger, sentinel))
            elif name == "url":
                out.append(args[0])
            elif name in NARGS and NARGS[name] >= 1:
                out.append("".join(convert(a, doc, part, ledger, sentinel) for a in args))
            else:
                out.append(WORDS.get(name, ""))
            continue
        if c == "$":
            j = s.find("$", i + 1)
            m_ = s[i + 1:j] if j > 0 else ""
            if "\\val" in m_:                              # a value inside a formula: check the value, not the symbols
                out.append(re.sub(r"[_^{}]", " ", convert(m_, doc, part, ledger, sentinel)))
            elif "_" in m_ or "\\" in m_:
                out.append(" ⟦⟧ ")              # a formula: no prose, numbers or labels to check
            else:
                out.append(m_)
            i = j + 1 if j > 0 else len(s)
            continue
        if c in "{}":
            i += 1
            continue
        if c == "~":
            out.append(" ")
            i += 1
            continue
        if s.startswith("---", i):
            out.append("\u2014")
            i += 3
            continue
        if s.startswith("--", i):
            out.append("\u2013")
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def inputs(path, seen=None):
    """main.tex's \\input files in order (relative to manuscript/paper)."""
    seen = seen or []
    text = re.sub(r"(?<!\\)%.*", "", (PAPER / path).read_text(encoding="utf-8"))
    for m in re.finditer(r"\\input\{([^}]+)\}", text):
        rel = m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex")
        seen.append(rel)
        inputs(rel, seen)
    return seen


def figure_titles():
    out = {}
    for name in [f"fig{i}" for i in range(1, 8)] + [f"s{i}" for i in range(1, 8)]:
        doc = ast.get_docstring(ast.parse((FIGS / "build" / "v4" / f"{name}.py").read_text(encoding="utf-8")))
        m = re.match(r"v4 (?:Fig\.|Figure) (S?\d)[^-]*- (.+?) \(174 mm", doc.splitlines()[0])
        out[m.group(1)] = m.group(2)
    return out


def panels():
    out = {}
    for name in [f"fig{i}" for i in range(1, 8)] + [f"s{i}" for i in range(1, 8)]:
        rows = csv.DictReader(open(FIGS / "v4" / "source_data" / f"{name}.csv", encoding="utf-8"))
        key = name[3:] if name.startswith("fig") else "S" + name[1:]
        out[key] = sorted({r["panel"] for r in rows if re.fullmatch(r"[A-Z]", r["panel"] or "")})
    return out


def words(text):
    return ct.words(re.sub(r"\u27ea/?U\u27eb", " ", text))


def check(strict=False):
    fails, warns, notes = [], [], []
    ledger = {r["id"]: r for r in csv.DictReader(open(HERE / "number_ledger.csv", encoding="utf-8"))}
    values = ct.ledger_values(HERE / "number_ledger.csv")
    doc = Doc()
    files = inputs("main.tex") + [f for f in inputs("si.tex") if f not in ("generated/values.tex",)]
    for f in files:
        if f not in FILES:
            fails.append(f"main.tex inputs {f}, which the checker does not know (add it to FILES)")
            continue
        part = FILES[f]
        if part is None:
            continue
        tex = re.sub(r"(?<!\\)%.*", "", (PAPER / f).read_text(encoding="utf-8"))
        for block in KRT.findall(tex):              # the key resources table: its citations and fields count; its text is not checked
            convert(block, doc, part, ledger, False)
        tex = KRT.sub(" ", tex)
        doc.parts[part] = doc.parts.get(part, "") + convert(tex, doc, part, ledger, False)
        doc.numbers[part] = doc.numbers.get(part, "") + convert(tex, Doc(), part, ledger, True)
    main = re.sub(r"(?<!\\)%.*", "", (PAPER / "main.tex").read_text(encoding="utf-8"))
    convert(main, doc, "main", ledger, False)
    star_text = doc.parts.get("STAR Methods", "")

    # ---- check_text's rules on the drafted parts, in its Markdown form
    title = re.sub(r"\s+", " ", doc.parts.get("Title", "")).strip()
    title = title.split(" Han Wang")[0].strip()
    md = [f"# {title}"]
    for p in ("Summary", "Highlights", "In Brief", "Introduction"):
        md += [f"## {p}", re.sub(r"^\s*Summary\.|^\s*Highlights\.|^\s*In brief\.", "", doc.parts.get(p, "").strip())]
    res = doc.parts.get("Results", "")
    md += ["## Results", re.sub(r"\n\u00a7H (.+)\n", r"\n### \1\n", res)]
    for p in ("Discussion", "Limitations of the study", "Figure legends", "STAR Methods"):
        md += [f"## {p}", doc.parts.get(p, "")]
    f_, w_ = ct.check("\n".join(md), None)
    fails += f_
    warns += w_

    # ---- the Cell word count
    counted, rows = 0, []
    for p in COUNTED:
        txt = re.sub(r"\u27eaU\u27eb.*?\u27ea/U\u27eb", " ", doc.parts.get(p, ""), flags=re.S)
        n = len(words(txt))
        counted += n
        rows.append((p, n, TARGETS.get(p)))
        if p in TARGETS and n > 1.15 * TARGETS[p] and not any(q == p for q, _ in doc.todraft):
            warns.append(f"{p}: {n} words, over its target of {TARGETS[p]} by more than 15%")
    front = sum(len(words(doc.parts.get(p, ""))) for p in FRONT)
    if counted >= WORD_LIMIT:
        fails.append(f"Cell word count {counted:,} (limit: under {WORD_LIMIT:,})")
    elif counted > WORD_WARN:
        warns.append(f"Cell word count {counted:,} (above {WORD_WARN:,}; limit under {WORD_LIMIT:,})")
    notes.append("Cell word count (main text, back matter and main legends): " + f"{counted:,}; front matter apart: {front:,}; "
                 f"STAR Methods (not counted): {len(words(star_text)):,}")
    notes += [f"  {p:32s} {n:6,d}" + (f"  (target {t:,})" if t else "") for p, n, t in rows]

    # ---- numbers
    for part, vid in doc.vals:
        if vid not in ledger:
            fails.append(f"{part}: \\val{{{vid}}} is not in the ledger")
    for part, txt in doc.numbers.items():
        if part not in NUMBERED:
            continue
        txt = re.sub(r"\u27ea/?U\u27eb", " ", txt)
        txt = re.sub(r"Scale bars?,[^.]*\.", " ", txt)          # drawing facts, recorded in the source data
        txt = LABELS.sub(" ", txt)
        txt = re.sub(r"\b(?:Figures?|Videos?|Tables?|Data)\s+S?\d+[A-Z]?(?:\s*(?:\u2013|-|and|,)\s*S?\d*[A-Z]?)*", " ", txt)
        txt = re.sub(r"https?://\S+|doi\S*|\b10\.\d{4,}/\S+", " ", txt)
        for tok in ct.numbers(txt):
            if not ct.in_ledger(tok, values):
                fails.append(f"{part}: number {tok} is not a ledger value (write it as \\val{{ID}} or add it to the ledger)")

    # ---- headings, call-outs, legends
    titles = figure_titles()
    want = [titles[str(i)] for i in range(1, 8)]
    if doc.headings and doc.headings != want:
        fails.append("Results headings differ from the figure titles: " + "; ".join(f"{h!r} vs {t!r}" for h, t in zip(doc.headings, want) if h != t))
    pan = panels()
    seen_main, first = {}, []
    for part, pos, item in doc.callouts:
        if item.startswith(("Video", "Table", "Data")):
            kind, num = item.split()
            ok = {"Video": {"S1", "S2", "S3", "S4"}, "Table": {"S1", "S2", "S3", "S4"}, "Data": {"S1"}}[kind]
            if num not in ok:
                fails.append(f"{part}: call-out of {item}, which does not exist")
            continue
        m = re.fullmatch(r"(S?\d+)([A-Z]?)", item)
        if not m or m.group(1) not in pan or (m.group(2) and m.group(2) not in pan[m.group(1)]):
            fails.append(f"{part}: call-out of Figure {item}, which does not exist")
            continue
        fig, letter = m.groups()
        if part == "Results" and not fig.startswith("S"):
            seen_main.setdefault(fig, set()).add(letter)
            if fig not in first:
                first.append(fig)
    drafted_results = not any(p == "Results" for p, _ in doc.todraft)
    if drafted_results:
        for fig in map(str, range(1, 8)):
            missing = [p for p in pan[fig] if p not in seen_main.get(fig, set())]
            if missing:
                fails.append(f"Results: Figure {fig} panels never called out: {', '.join(missing)}")
        if first != sorted(first, key=int):
            fails.append(f"Results: main figures first called out out of order: {first}")
    for n in map(str, range(1, 8)):
        if n in doc.legends and f"See also Figure S{n}" not in re.sub(r"\s+", " ", doc.legends[n][1]):
            fails.append(f"Figure {n} legend: no 'See also Figure S{n}'")
        if n in doc.legends and doc.legends[n][0] != titles[n]:
            fails.append(f"Figure {n} legend title differs from the figure's: {doc.legends[n][0]!r}")

    # ---- citations
    bib = set(re.findall(r"^@\w+\{([^,]+),", (PAPER / "refs.bib").read_text(encoding="utf-8"), flags=re.M))
    for k in sorted(doc.cites - bib):
        fails.append(f"citation {k} is not in refs.bib")
    unused = sorted(bib - doc.cites)
    if unused and not doc.nocite_all:
        fails.append(f"refs.bib entries never cited: {', '.join(unused)}")
    elif doc.nocite_all:
        warns.append(r"\nocite{*} lists every reference (skeleton only; remove it once the text cites them)")

    # ---- class (c) terms (the user's rule: not in figures or story; STAR Methods and Table S2 only)
    for part, txt in doc.parts.items():
        if part == "STAR Methods":
            continue
        for term in CLASS_C:
            if re.search(r"\b" + re.escape(term), txt, flags=re.IGNORECASE):
                fails.append(f"{part}: class (c) term {term!r} outside STAR Methods")

    # ---- what the author guide requires
    req = {"Resource availability": ["Lead contact", "Materials availability", "Data and code availability",
                                      "Requests for further information and resources should be directed to and will be fulfilled by the lead contact",
                                      "Any additional information required to reanalyze the data reported in this paper is available from the lead contact upon request"],
           "Declaration of generative AI": ["During the preparation of this work", "take", "full responsibility"]}
    for p, needles in req.items():
        txt = re.sub(r"\s+", " ", doc.parts.get(p, ""))
        if p == "Declaration of generative AI" and not words(txt):
            # left blank at the author's request (2026-10-23): pending, not wrong, until the check before submission
            (fails if strict else warns).append(f"{p}: left blank for now (the guide requires it at submission)")
            continue
        for nd in needles:
            if nd not in txt:
                fails.append(f"{p}: missing the required {nd!r}")
    for p in ("Declaration of interests", "Author contributions", "Acknowledgments"):
        if p not in doc.parts:
            fails.append(f"missing section: {p}")

    # ---- not yet drafted, not yet filled
    pending = {}
    for p, gate in doc.todraft:
        pending[(p, gate)] = pending.get((p, gate), 0) + 1
    for (p, gate), n in pending.items():
        (fails if strict else warns).append(f"{p}: not yet drafted ({gate}; {n} place{'s' if n > 1 else ''})")
    for p, what in doc.tofill:
        (fails if strict else warns).append(f"{p}: to fill: {what}")
    return fails, warns, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--report")
    a = ap.parse_args()
    fails, warns, notes = check(a.strict)
    lines = [*notes, *(f"warn: {w}" for w in warns), *(f"FAIL: {f}" for f in fails), f"{len(fails)} failure(s), {len(warns)} warning(s)"]
    print("\n".join(lines))
    if a.report:
        Path(a.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
