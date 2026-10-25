# The *Cell* manuscript (LaTeX)

One source and five builds. Build with `build.py`, never with a bare `xelatex`: this MiKTeX installs missing packages on the fly, and `build.py` runs every TeX program with `--disable-installer`.

```
.\MIGRATION\python\python.exe -B -X utf8 manuscript\paper\build.py all
.\MIGRATION\python\python.exe -B -X utf8 -I manuscript\writing\check_tex.py
```

| Build | File | Form |
|---|---|---|
| `submission` | `build/submission/main-submission.pdf` | 12 pt, double spaced, line numbers; legends after the references, then the figures one per page |
| `submission-text` | `build/submission-text/main-submission-text.pdf` | the same without the figure pages: the manuscript file for *Cell*, which takes each figure as its own file |
| `preprint` | `build/preprint/main-preprint.pdf` | the same text, each figure and legend after its Results section (the guide: the preprint is the submitted version) |
| `si` | `build/si/document_s1.pdf` | Document S1: Figures S1–S7 and Table S1; Arial 10 pt; no title, authors or page numbers; its own references |
| `letter` | `build/letter/cover_letter.pdf` | the cover letter (`letter/cover_letter.tex`): the paper's own claims, its numbers from the ledger, the author's fields marked |

`manuscript/package/build_release.py` assembles the submission package, the preprint and the LaTeX source from these builds
(`release/3_manuscript`).

## Layout

- `main.tex` follows the author guide's order:
  1. title page;
  2. Summary, Highlights, In brief, Keywords;
  3. Introduction, Results, Discussion, Limitations of the study;
  4. Resource availability;
  5. Acknowledgments, Author contributions, Declaration of interests, Declaration of generative AI;
  6. Supplemental information (index and item legends);
  7. STAR Methods;
  8. References;
  9. Figure titles and legends.
- Parts: `front/`, `body/`, `back/`, `methods/`, `legends/` (`figures.tex` main, `si_figures.tex` supplemental), `letter/` (the
  cover letter).
- `cellpress.sty` holds the layout and the text's macros:
  - `\fig{2E}`, `\figs{2B}{2C}`, `\figr{2B}{2D}`, `\video{S2}`, `\tab{S2}`, `\dataS{S1}`;
  - `\val{ID}`, `\gene{SK}`, `\Pv`;
  - `\cellcite{keys}`, written before the punctuation;
  - `\tofill{...}` for fields only the author can fill, and `\todraft{Gate X}`.
  - Package option `authoryear` switches the citation style; numbered is the default.
- `generated/values.tex` comes from `manuscript/writing/number_ledger.csv` (`writing/build_macros.py`, run by `build.py`). Every result in the text is a `\val{ID}`.
- `refs.bib` comes from `citations.csv` and `bib/*.json` (`writing/build_bib.py`). The metadata comes only from the PubMed and bioRxiv tools; each entry keeps its PMID or DOI.
- `cellnum.bst` and `cellay.bst` (`writing/make_bst.py`) derive from natbib's `unsrtnat.bst` and print the *Cell* reference form. biblatex is not used, because it needs the `logreq` package, which is not installed.

## The rules the check holds the text to

The author guide, as the user pasted it on 2026-10-23:
- under 7,000 words, counting the main figure legends but not STAR Methods, supplemental item legends or references;
- at most seven figures and/or tables;
- the Resource availability subsections and statements;
- the generative-AI declaration after the declaration of interests;
- the Document S1 form;
- the sex of the model reported.

Also `manuscript/CELL_VOICE_GUIDE.md` and the number ledger. `check_tex.py --strict` is the check before submission: nothing left to draft or fill.
