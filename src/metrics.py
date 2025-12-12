from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro")


def full_report(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, np.ndarray]:
    report_str = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    return report_str, cm


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, float]:
    labels = sorted(set(y_true) | set(y_pred))
    scores = f1_score(y_true, y_pred, labels=labels, average=None)
    return {int(lbl): float(score) for lbl, score in zip(labels, scores)}
