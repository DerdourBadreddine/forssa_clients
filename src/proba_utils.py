from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def canonical_classes(canonical: Sequence[int]) -> np.ndarray:
    return np.asarray([int(x) for x in canonical], dtype=int)


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def reorder_proba_to_canonical(
    proba: np.ndarray,
    model_classes: Iterable,
    canonical: Sequence[int],
) -> np.ndarray:
    """Reorder probability columns into canonical class order.

    Fails fast if model classes don't match canonical set.
    """
    proba = np.asarray(proba)
    if proba.ndim != 2:
        raise ValueError(f"proba must be 2D, got shape={proba.shape}")

    canon = canonical_classes(canonical)

    try:
        cls = np.asarray([int(c) for c in list(model_classes)], dtype=int)
    except Exception as e:
        raise ValueError(f"Could not convert model_classes to int: {list(model_classes)}") from e

    if set(cls.tolist()) != set(canon.tolist()):
        raise ValueError(
            "Model classes do not match canonical classes. "
            f"model={sorted(set(cls.tolist()))} canonical={sorted(set(canon.tolist()))}"
        )

    # map canonical -> index in model output
    idx = [int(np.where(cls == c)[0][0]) for c in canon]
    out = proba[:, idx].astype(np.float32, copy=False)

    if out.shape[1] != len(canon):
        raise ValueError(f"Reordered proba has wrong shape {out.shape}; expected (*, {len(canon)})")

    if np.any(np.isnan(out)) or np.any(np.isinf(out)):
        raise ValueError("NaN/Inf found in probabilities")

    return out


def predict_proba_any(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        return softmax(scores)

    raise ValueError(f"Model {type(model)} has neither predict_proba nor decision_function")


def predict_proba_canonical(model, X, canonical: Sequence[int]) -> np.ndarray:
    if not hasattr(model, "classes_"):
        raise ValueError(f"Model {type(model)} has no classes_; cannot align probabilities")
    proba = predict_proba_any(model, X)
    return reorder_proba_to_canonical(proba, model.classes_, canonical)


def assert_proba_is_canonical(proba: np.ndarray, canonical: Sequence[int]) -> None:
    proba = np.asarray(proba)
    canon = canonical_classes(canonical)
    if proba.ndim != 2 or proba.shape[1] != len(canon):
        raise AssertionError(f"Expected proba shape (n, {len(canon)}), got {proba.shape}")
    if np.any(np.isnan(proba)) or np.any(np.isinf(proba)):
        raise AssertionError("NaN/Inf in proba")
