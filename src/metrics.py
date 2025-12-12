from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def probs_to_labels(probs: np.ndarray, labels: list[int]) -> np.ndarray:
    labels_arr = np.asarray(labels)
    return labels_arr[np.argmax(probs, axis=1)]


def full_report(y_true: np.ndarray, y_pred: np.ndarray, *, labels: list[int] | None = None) -> Tuple[str, np.ndarray]:
    report_str = classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return report_str, cm


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, *, labels: list[int] | None = None) -> Dict[int, float]:
    if labels is None:
        labels = sorted(set(map(int, y_true)) | set(map(int, y_pred)))
    scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {int(lbl): float(score) for lbl, score in zip(labels, scores)}
