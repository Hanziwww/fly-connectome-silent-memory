# Release packages of "Brain state locks silent memories in a trained fly connectome"

Built {{DATE}} from commit `{{COMMIT}}` by `manuscript/package/build_release.py`. This folder is git-ignored; the builder
rebuilds any part (`code`, `data`, `paper`, `figures`, `check`).

| Folder | What it holds | Where it goes |
|---|---|---|
| `1_code/` | the code release: `{{CODE_DIR}}/` and `{{CODE_ZIP}}` ({{CODE_FILES}} files) | a public repository (GitHub), and Zenodo record C |
| `2_zenodo/A_records_core/` | part 1 of the records: everything the analysis, renders, figure builds and writing scripts read, every run's models, configurations, logs and results, the result tables, figures and videos ({{CORE_SIZE}}) | Zenodo record A (dataset, CC BY 4.0) |
| `2_zenodo/B_records_extended/` | part 2: verification replays, behavior features and dense recordings that no analysis reads ({{EXT_SIZE}}) | Zenodo record B (dataset, CC BY 4.0) |
| `2_zenodo/C_code/` | the code release archive ({{CODE_SIZE}}) | Zenodo record C (software, MIT) |
| `3_manuscript/cell_submission/` | cover letter, manuscript, separate figures, graphical abstract, Document S1, Tables S2–S4, Data S1, Videos S1–S4, the form's text | *Cell* (Editorial Manager) |
| `3_manuscript/biorxiv/` | the preprint and its supplementary files | bioRxiv |
| `3_manuscript/latex_source/` | the LaTeX source with its figures, checked to compile on its own | *Cell* at revision; your archive |
| `4_figures_and_tables/` | the original figures (PDF, SVG, JPG; TIFF at 600 ppi for the main figures), the graphical abstract, Tables S1–S4, Data S1, the videos, with `INDEX.md` | your use; production |

Each Zenodo folder holds what to upload: the archives, `MANIFEST.csv`, `README.md` (the record's description),
`verify_deposit.py` and `zenodo.json` (the metadata for the form, with the fields to fill). Each record stays under
Zenodo's 50 GB limit.

## Still to fill

{{TOFILL}}

## Order of the steps

1. Complete the remaining marked author-affiliation, individual-contribution, and competing-interest fields in `manuscript/paper`, and obtain final approval from all authors. The funding and language-review statements are populated.
2. Create the three Zenodo drafts, reserve their DOIs, and enter them in the paper (two places in `back/resource.tex`, two
   rows of the key resources table) and in each record's description.
3. Rebuild: `build_release.py paper figures check` (and `code` after a commit).
4. Post the preprint on bioRxiv, enter its DOI in the cover letter, and rebuild the letter.
5. Submit to *Cell* (`3_manuscript/cell_submission/README.md`).
6. Upload the Zenodo records (shared with the reviewers, or published), and push the code release as a new repository
   (`git init` inside `1_code/{{CODE_DIR}}/`). Never push this workspace's own history: it holds the board, the planning
   documents and connection details of the GPU hosts.

## Checks of this build

{{CHECKS}}
