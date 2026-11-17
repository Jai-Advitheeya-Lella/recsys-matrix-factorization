"""Shared helpers for the experiment scripts (threads, seeding, JSON I/O)."""

from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path

import numpy as np


def limit_threads(n: int = 4) -> None:
    """Cap BLAS / OpenMP / numba / torch threads (set before heavy imports)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = str(n)
    try:
        import numba

        numba.set_num_threads(n)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:
        pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def hardware_info() -> dict:
    info = {"cpu": platform.processor() or platform.machine(), "python": platform.python_version()}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        info["torch"] = torch.__version__
    except Exception:
        pass
    return info


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if k not in ("per_user", "topk", "users")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_clean(obj), f, indent=2)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def mean_std(records: list[dict]) -> tuple[dict, dict]:
    keys = [k for k, v in records[0].items() if isinstance(v, (int, float))]
    mean = {k: float(np.mean([r[k] for r in records])) for k in keys}
    std = {k: float(np.std([r[k] for r in records])) for k in keys}
    return mean, std
