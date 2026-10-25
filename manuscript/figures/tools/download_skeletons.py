"""Download the selected MaleCNS v1.0 skeletons (approved source; CC BY 4.0) that are not already cached.

Selection: manuscript/results/figure_data/fd_neuron_selection.csv (download = true), fixed before any download.
Source:    https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-precomputed/<bodyId>
Target:    manuscript/figures/assets/skeletons/<bodyId> (a folder holding only these files and receipt.csv).

Every file is checked before it is kept: neuroglancer precomputed layout (8 + 12 nv + 8 ne bytes, edge indices in
range, finite coordinates) and, where the soma is known, a skeleton vertex within 5 um of it. Files are parsed with
numpy only and never executed. The run stops if the total would exceed 40 MB. No credentials are used.

Run: .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\figures\\tools\\download_skeletons.py
"""
import csv
import gzip
import hashlib
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SELECTION = ROOT / "manuscript" / "results" / "figure_data" / "fd_neuron_selection.csv"
CELLS = ROOT / "manuscript" / "results" / "figure_data" / "fd_cells.csv"
TARGET = ROOT / "manuscript" / "figures" / "assets" / "skeletons"
URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-precomputed/{}"
LICENCE = "CC BY 4.0 (MaleCNS v1.0, male-cns.janelia.org)"
CAP_BYTES = 40_000_000
SOMA_TOLERANCE_UM = 5.0


def fetch(url, tries=4):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "flybrain-figures/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                status = r.status
                enc = r.headers.get("Content-Encoding", "")
            if enc == "gzip" or data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            return status, data
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                return 404, b""
            time.sleep(2 ** k)
    raise RuntimeError(f"{url}: {last}")


def validate(data, soma):
    if len(data) < 8:
        raise ValueError("file shorter than its header")
    nv, ne = (int(v) for v in np.frombuffer(data, "<u4", 2))
    if len(data) != 8 + 12 * nv + 8 * ne:
        raise ValueError(f"size {len(data)} != 8 + 12*{nv} + 8*{ne}")
    v = np.frombuffer(data, "<f4", nv * 3, 8).reshape(-1, 3).astype(np.float64) / 1000.0
    e = np.frombuffer(data, "<u4", ne * 2, 8 + 12 * nv)
    if not np.isfinite(v).all() or (ne and int(e.max()) >= nv):
        raise ValueError("non-finite vertex or edge index out of range")
    dist = None
    if soma is not None:
        dist = float(np.linalg.norm(v - soma, axis=1).min())
        if dist > SOMA_TOLERANCE_UM:
            raise ValueError(f"nearest vertex {dist:.2f} um from the soma")
    return nv, ne, dist


def main():
    sel = pd.read_csv(SELECTION)
    want = sorted(set(sel.loc[sel["download"].astype(str).str.lower() == "true", "body_id"].astype(np.int64)))
    cells = pd.read_csv(CELLS, usecols=["body_id", "soma_x_um", "soma_y_um", "soma_z_um"]).set_index("body_id")
    TARGET.mkdir(parents=True, exist_ok=True)
    receipt_p = TARGET / "receipt.csv"
    rows = {}
    if receipt_p.exists():
        for r in csv.DictReader(receipt_p.open(encoding="utf-8")):
            rows[int(r["body_id"])] = r
    total = sum(int(r["bytes"]) for r in rows.values() if r.get("bytes"))
    print(f"{len(want)} skeletons selected for download; {len(rows)} already in the receipt", flush=True)
    for i, b in enumerate(want, 1):
        f = TARGET / str(b)
        if b in rows and f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == rows[b]["sha256"]:
            continue
        url = URL.format(b)
        status, data = fetch(url)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if status != 200 or not data:
            rows[b] = dict(body_id=b, url=url, http_status=status, bytes="", sha256="", n_vertices="", n_edges="",
                           soma_distance_um="", retrieved_utc=stamp, licence=LICENCE, note="not available")
            print(f"  [{i}/{len(want)}] {b}: HTTP {status}", flush=True)
            continue
        if total + len(data) > CAP_BYTES:
            raise SystemExit(f"stopping: the total would exceed {CAP_BYTES / 1e6:.0f} MB")
        soma = None
        if b in cells.index and not pd.isna(cells.at[b, "soma_x_um"]):
            soma = cells.loc[b, ["soma_x_um", "soma_y_um", "soma_z_um"]].to_numpy(np.float64)
        nv, ne, dist = validate(data, soma)
        tmp = f.with_name(f.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(f)
        total += len(data)
        rows[b] = dict(body_id=b, url=url, http_status=status, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                       n_vertices=nv, n_edges=ne, soma_distance_um="" if dist is None else round(dist, 4), retrieved_utc=stamp,
                       licence=LICENCE, note="")
        if i % 20 == 0 or i == len(want):
            print(f"  [{i}/{len(want)}] {total / 1e6:.1f} MB", flush=True)
    keys = ["body_id", "url", "http_status", "bytes", "sha256", "n_vertices", "n_edges", "soma_distance_um", "retrieved_utc", "licence", "note"]
    with receipt_p.open("w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for b in sorted(rows):
            w.writerow({k: rows[b].get(k, "") for k in keys})
    ok = sum(1 for r in rows.values() if str(r.get("http_status")) == "200")
    print(f"done: {ok}/{len(want)} downloaded and verified, {total / 1e6:.1f} MB; receipt {receipt_p}", flush=True)


if __name__ == "__main__":
    main()
