# Submission to *Cell*: upload checklist

Built {{DATE}} from commit `{{COMMIT}}` by `manuscript/package/build_release.py paper`. This file is for you, not for upload.

## The files, in the order of Editorial Manager

{{FILE_TABLE}}

Figures go up as separate files; the manuscript file carries the title page, the text, STAR Methods, the references and the
figure titles and legends, without the figure pages. The figure PDFs are the vector originals at print size (174 mm wide);
their rendered images reach {{MIN_PPI}} ppi or more (color-bar gradients and small matrices drawn as pixel blocks aside), and
TIFF copies at 600 ppi are in `4_figures_and_tables/main/` if a TIFF is asked for. `optional/Manuscript_with_figures.pdf` is the whole manuscript in one PDF (legends, then one figure per page), for a
presubmission inquiry or an editor who asks for a single file.

## The submission form

`Submission_form.md` holds the text the form asks for: title, Summary, Highlights, In Brief, keywords, the word count and
the availability statement, with their lengths checked.

## Still to fill (marked in red in the PDFs)

{{TOFILL}}

Fill them in the LaTeX source (`manuscript/paper`), then rebuild: `build_release.py paper figures check`.

## Before you submit

1. Reserve the DOIs of the three Zenodo records (each draft's "Reserve DOI"), and enter them in `back/resource.tex` (two
   places) and in the key resources table (`methods/star.tex`, two rows).
2. Post the preprint on bioRxiv (`../biorxiv/`): the guide requires the preprint to be the version submitted. Enter its DOI
   in the cover letter.
3. Rebuild and run `check_tex.py --strict`, which fails while any field is left to fill or any part is blank.
4. Make the Zenodo records available to the reviewers (a share link, or publish them) and give the link in the cover letter.

## Checks of this build

{{CHECKS}}
