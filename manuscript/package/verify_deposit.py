r"""Checks a downloaded Zenodo record of this study against its MANIFEST.csv and unpacks it. Python 3, standard library only.

  python verify_deposit.py <record folder> [--into <code release folder>] [--check-only]

Every member of every archive is read once: its SHA-256 and size are compared with the manifest while it is written into
the target folder (or only read, with --check-only). Then each path that the record stores once for several identical
files (column same_as) is restored as a copy of its stored path and checked too. The exit code is 1 if a file is missing,
differs or is not in the manifest.
"""
import argparse
import csv
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", help="the folder with MANIFEST.csv and the record's archives")
    ap.add_argument("--into", default=".", help="where to unpack (the top folder of the code release)")
    ap.add_argument("--check-only", action="store_true", help="read and check the archives without unpacking")
    a = ap.parse_args()
    record, into = Path(a.record), Path(a.into)
    rows = list(csv.DictReader(open(record / "MANIFEST.csv", encoding="utf-8", newline="")))
    stored = {r["path"]: r for r in rows if not r["same_as"]}
    bad, done = [], 0
    for archive in sorted({r["archive"] for r in stored.values()}):
        path = record / archive
        if not path.exists():
            bad.append(f"missing archive {archive}")
            continue
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                want = stored.get(info.filename)
                if want is None or want["archive"] != archive:
                    bad.append(f"{archive}: {info.filename} is not in the manifest")
                    continue
                h, n = hashlib.sha256(), 0
                target = into / info.filename
                out = None
                if not a.check_only:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    out = open(target, "wb")
                try:
                    with z.open(info) as f:
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            h.update(chunk)
                            n += len(chunk)
                            if out:
                                out.write(chunk)
                finally:
                    if out:
                        out.close()
                if h.hexdigest() != want["sha256"] or n != int(want["bytes"]):
                    bad.append(f"{archive}: {info.filename} differs from the manifest")
                done += 1
        print(f"{archive}: checked", flush=True)
    missing = [p for p, r in stored.items() if (record / r["archive"]).exists() and not (a.check_only or (into / p).exists())]
    bad += [f"not unpacked: {p}" for p in missing[:20]]
    copies = 0
    if not a.check_only:
        for r in rows:
            if r["same_as"]:
                src, dst = into / r["same_as"], into / r["path"]
                if not src.exists():
                    bad.append(f"{r['path']}: its stored copy {r['same_as']} is not unpacked (another record?)")
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                copies += 1
    print(f"{done} files checked, {copies} copies restored, {len(bad)} problems")
    for b in bad[:50]:
        print("  " + b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
