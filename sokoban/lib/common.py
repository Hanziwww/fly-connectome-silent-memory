"""Small, atomic experiment records and a shared absolute cycle deadline."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
ASSETS = HERE / "artifacts"


class BudgetExpired(RuntimeError):
    pass


class Budget:
    def __init__(self, started: float, seconds: float = 1200):
        if not 0 < seconds <= 1200:
            raise ValueError("A cycle must not exceed the 1200-second safety ceiling")
        self.started = started
        self.seconds = seconds

    def elapsed(self):
        return time.monotonic() - self.started

    def end(self, phase):
        return self.started + self.seconds * {"prepare": 240/1200, "train": 900/1200,
                                             "evaluate": 1140/1200, "save": 1}[phase]

    def check(self, phase, reserve=0):
        if time.monotonic() + reserve >= self.end(phase):
            raise BudgetExpired(f"{phase} budget expired at {self.elapsed():.2f}s")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def save_checkpoint(path, value):
    import torch
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def configure(seed):
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
