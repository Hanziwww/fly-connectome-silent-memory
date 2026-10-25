LaTeX source of "Brain state locks silent memories in a trained fly connectome" (Han Wang et al.)
Built {{DATE}} from commit {{COMMIT}}.

The source compiles with XeLaTeX (fontspec; the Arial font) and BibTeX, from this folder:

  latexmk -pdfxe main.tex             the manuscript: text, STAR Methods, references, legends, then the figures
  latexmk -pdfxe main_text.tex        the same without the figure pages (the manuscript file for Cell)
  latexmk -pdfxe main_preprint.tex    the preprint: each figure with its legend after its Results section
  latexmk -pdfxe si.tex               Document S1: Figures S1-S7 and Table S1
  latexmk -pdfxe letter/cover_letter.tex

The .bbl files are included, so the documents also compile without BibTeX. Files:

  main.tex, si.tex                    the documents; main_text.tex and main_preprint.tex select a build of main.tex
  cellpress.sty                       the layout and the text's macros (\val, \fig, \cellcite, \tofill)
  cellnum.bst, cellay.bst, refs.bib   the reference style (numbered; author-year) and the references
  front/ body/ back/ methods/ legends/ letter/   the parts of the text
  generated/values.tex                every number of the text, from the number ledger of the code release
  generated/table_s1.tex              Table S1, generated from the screen's record
  figures/main/, figures/supplement/  the figures as vector PDFs at print size

Fields marked \tofill{...} print in red and are the author's to complete. The code release holds the tools that generate
values.tex, table_s1.tex and the figures (manuscript/writing, manuscript/figures) and checks the text (check_tex.py).
