"""Read-only loaders. Each file is read once as bytes, hashed into the manifest, then parsed."""
import io as _io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import manifest


def read_bytes(path):
    path = Path(path)
    data = path.read_bytes()
    manifest.register(path, data)
    return data


def text(path):
    return read_bytes(path).decode('utf-8-sig')


def json_load(path):
    return json.loads(text(path))


def jsonl_load(path):
    return [json.loads(line) for line in text(path).splitlines() if line.strip()]


def csv_load(path, **kwargs):
    kwargs.setdefault('encoding', 'utf-8-sig')
    kwargs.setdefault('keep_default_na', True)
    return pd.read_csv(_io.BytesIO(read_bytes(path)), **kwargs)


def parquet_load(path, columns=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    return pq.read_table(pa.BufferReader(read_bytes(path)), columns=columns).to_pandas()


def npz_load(path, keys=None):
    with np.load(_io.BytesIO(read_bytes(path)), allow_pickle=False) as z:
        return {k: z[k] for k in z.files if keys is None or k in keys}


def torch_load(path):
    import torch
    return torch.load(_io.BytesIO(read_bytes(path)), map_location='cpu', weights_only=True)


def sparse_load(path):
    import scipy.sparse as sp
    return sp.load_npz(_io.BytesIO(read_bytes(path)))
