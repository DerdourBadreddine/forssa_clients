from __future__ import annotations

import json
import os
import random
import re
import hashlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def safe_model_id(model_id: str) -> str:
    """Make a filesystem-safe model id for folder names."""
    s = model_id.strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s


def json_dump(obj: Any, path: Path, *, indent: int = 2) -> None:
    ensure_dir(path.parent)

    def default(o: Any):
        if is_dataclass(o):
            return asdict(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=default)


def json_load(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def reorder_proba_columns(probs: np.ndarray, classes: Iterable[int], canonical: list[int]) -> np.ndarray:
    """Reorder probs columns (aligned to `classes`) into canonical label order."""
    classes = list(map(int, classes))
    index = {c: i for i, c in enumerate(classes)}
    out = np.zeros((probs.shape[0], len(canonical)), dtype=probs.dtype)
    for j, c in enumerate(canonical):
        if c not in index:
            raise ValueError(f"Missing class {c} in classes_ {classes}")
        out[:, j] = probs[:, index[c]]
    return out


def assert_probs_ok(probs: np.ndarray, n_classes: int) -> None:
    if probs.ndim != 2 or probs.shape[1] != n_classes:
        raise ValueError(f"Expected probs shape (n,{n_classes}), got {probs.shape}")
    if not np.isfinite(probs).all():
        raise ValueError("Non-finite probabilities detected")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
