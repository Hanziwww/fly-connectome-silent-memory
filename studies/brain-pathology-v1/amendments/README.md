# Amendments

Append-only record of changes to the registered protocol, schedule or task lock, written by
`Study.amend` (studykit/registry.py). Files are numbered `NNNN-<slug>.json`. Each one records the
SHA-256 of the one before it, so an edited or removed amendment breaks the chain, and research.py
then refuses to run or verify. Never edit or delete a file here.
