r"""Writes the manuscript's two BibTeX styles, derived from natbib's unsrtnat.bst (as installed with MiKTeX; LPPL, so the
derived files carry new names): manuscript/paper/cellnum.bst (numbered, in citation order: the default) and cellay.bst
(author-year, alphabetical: the switch). Both print references in the Cell form

  Surname, I., Surname, I.J., and Surname, K. (Year). Title. Journal Volume, pages. https://doi.org/...

with titles as given (no case change) and preprints as "Preprint at bioRxiv". The author list itself comes ready-made in the
field authorlist (build_bib.py: at most 10 names, ", and" before the last, "et al." after the tenth); BibTeX's own name
formatting remains only for entries without it and for natbib's labels.

  .\MIGRATION\python\python.exe -B -X utf8 -I manuscript\writing\make_bst.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = Path.home() / "AppData/Local/Programs/MiKTeX/bibtex/bst/natbib/unsrtnat.bst"
OUT = ROOT / "manuscript" / "paper"

FUNCS = {
    # the author list as build_bib.py writes it in the Cell form (field authorlist); BibTeX's own names only for labels
    "format.authors": r"""{ authorlist empty$
    { author empty$
        { "" }
        { author format.names }
      if$
    }
    { authorlist }
  if$
}""",
    "format.names": r"""{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames #10 >
    { #10 'namesleft := }
    { numnames 'namesleft := }
  if$
    { namesleft #0 > }
    { s nameptr "{vv~}{ll}{, f{.}.}{, jj}" format.name$ 't :=
      nameptr #1 >
        { t "others" =
            { ", et~al." * }
            { ", " * t * }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
  numnames #10 >
    { ", et~al." * }
    'skip$
  if$
}""",
    "format.title": r"""{ title empty$
    { "" }
    { title }
  if$
}""",
    "format.doi": r"""{ doi empty$
    { url empty$
        { "" }
        { "\url{" url * "}" * }
      if$
    }
    { "\url{https://doi.org/" doi * "}" * }
  if$
}""",
    "article": r"""{ output.bibitem
  format.authors " " * format.year.paren * "author" output.check
  new.block
  format.title "title" output.check
  new.block
  format.journal.vol.pages "journal" output.check
  new.block
  format.doi output
  fin.entry
}""",
    "book": r"""{ output.bibitem
  author empty$
    { editor format.names ", ed." * }
    { format.authors }
  if$
  " " * format.year.paren * "author" output.check
  new.block
  title publisher empty$
    'skip$
    { " (" * publisher * ")" * }
  if$
  "title" output.check
  new.block
  format.doi output
  fin.entry
}""",
    "incollection": r"""{ output.bibitem
  format.authors " " * format.year.paren * "author" output.check
  new.block
  format.title "title" output.check
  new.block
  "In " booktitle *
  editor empty$
    'skip$
    { ", " * editor format.names * ", eds." * }
  if$
  publisher empty$
    'skip$
    { " (" * publisher * ")" * }
  if$
  pages empty$
    'skip$
    { ", pp. " * pages n.dashify * }
  if$
  output
  new.block
  format.doi output
  fin.entry
}""",
    "inproceedings": r"""{ incollection }""",
    "misc": r"""{ output.bibitem
  format.authors " " * format.year.paren * output
  new.block
  format.title output
  new.block
  howpublished output
  new.block
  format.doi output
  fin.entry
  empty.misc.check
}""",
}

NEW = {
    "numbered": r"""FUNCTION {format.year.paren}
{ year empty$
    { "empty year in " cite$ * warning$ "" }
    { "(" year * ")" * }
  if$
}
""",
    "authoryear": r"""FUNCTION {format.year.paren}
{ year empty$
    { "empty year in " cite$ * warning$ "" }
    { "(" year * extra.label * ")" * }
  if$
}
""",
}
JOURNAL = r"""FUNCTION {format.journal.vol.pages}
{ journal field.or.null
  volume empty$
    'skip$
    { " " * volume * }
  if$
  eid empty$
    { pages empty$
        'skip$
        { ", " * pages n.dashify * }
      if$
    }
    { ", " * eid * }
  if$
}

"""


def build(kind):
    s = SRC.read_text(encoding="latin-1")
    entry = "ENTRY\n  { address\n    author\n"
    if s.count(entry) != 1:
        raise SystemExit("ENTRY block not found")
    s = s.replace(entry, entry + "    authorlist\n")
    for name, body in FUNCS.items():
        pat = re.compile(r"FUNCTION \{" + re.escape(name) + r"\}\n\{.*?\n\}\n", re.S)
        if len(pat.findall(s)) != 1:
            raise SystemExit(f"{name}: expected one definition in {SRC}")
        s = pat.sub(lambda m, body=body, name=name: f"FUNCTION {{{name}}}\n{body}\n", s)
    s = s.replace("FUNCTION {article}\n", NEW[kind] + "\n" + JOURNAL + "FUNCTION {article}\n", 1)
    if kind == "authoryear":                       # keep the first, alphabetical sort: drop the re-sort into citation order
        tail = "ITERATE {bib.sort.order}\n\nSORT\n"
        if s.count(tail) != 1:
            raise SystemExit("bib.sort.order not found")
        s = s.replace(tail, "")
    name = "cellnum" if kind == "numbered" else "cellay"
    head = (f"%% {name}.bst -- the Cell-form reference style of this manuscript ({kind}), generated by\n"
            "%% manuscript/writing/make_bst.py from natbib's unsrtnat.bst (version below; renamed as the LPPL asks).\n")
    (OUT / f"{name}.bst").write_text(head + s, encoding="latin-1", newline="\n")
    print(f"wrote {name}.bst")


if __name__ == "__main__":
    build("numbered")
    build("authoryear")
