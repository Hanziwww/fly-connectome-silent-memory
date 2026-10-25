# Simulation records: verification replays, behavior features and dense recordings (part 2 of 2)

Data of the article *Brain state locks silent memories in a trained fly connectome* by Han Wang et al. ({{CITATION}}).
Part 1 of the records, which holds every file the paper's analyses read, is at Zenodo, {{CORE_DOI}}; the code is at
Zenodo, {{CODE_DOI}}.

This part completes the run folders of part 1 with the files that no analysis of the paper reads:

- `verification.json` of the training runs: an independent process reloaded each trained model and replayed its task,
  recording every decision (Table S3, "Verified by reload");
- `behavior-features/` of the training runs: the features that the readout received at every training update;
- the dense recordings of the registered study's mechanism experiments that the published analysis does not read.

## Archives

{{ARCHIVE_TABLE}}

The paths inside the archives are relative to the top folder of the code release; unpack them there, over part 1.
`MANIFEST.csv` lists every file with its archive, size and SHA-256.

```
python verify_deposit.py . --into <code release folder>
```

checks every file against the manifest while it unpacks. It needs Python 3 and nothing else.

## Notes

Folders keep the study's development names; the code release's README translates them, and Table S3 gives the record of
every run. {{SUBSETS}}

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).
