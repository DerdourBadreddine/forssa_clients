from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config
from .data_io import load_datasets


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro"))


def top_confusions(cm: np.ndarray, labels: List[int], k: int = 10) -> List[Tuple[int, int, int]]:
    items: List[Tuple[int, int, int]] = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j:
                continue
            items.append((a, b, int(cm[i, j])))
    items.sort(key=lambda x: x[2], reverse=True)
    return items[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="OOF error analysis: confusion + examples")
    parser.add_argument("--exp-dir", type=Path, default=None, help="artifacts/experiments/<exp_id> (default: best_experiment.json exp)")
    parser.add_argument("--model", choices=["logreg", "nbsvm", "sgd", "blend"], default="blend")
    parser.add_argument("--k", type=int, default=8, help="number of top confusions")
    args = parser.parse_args()

    exp_dir = args.exp_dir
    if exp_dir is None:
        best_path = config.EXPERIMENTS_DIR / "best_experiment.json"
        best = json.loads(best_path.read_text(encoding="utf-8"))
        exp_dir = Path(best["models"][best["best_model"]]).parent

    y_true = np.load(exp_dir / "y_true.npy")
    oof_path = exp_dir / f"oof_{args.model}.npy"
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing {oof_path}")

    probs = np.load(oof_path)
    y_pred = np.array(config.LABELS)[np.argmax(probs, axis=1)]

    train_df, _, schema = load_datasets()
    text = train_df[schema.text_col].astype(str).tolist()

    from sklearn.metrics import confusion_matrix

    labels = config.LABELS
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    score = macro_f1(y_true, y_pred)

    print("exp_dir:", exp_dir)
    print("model:", args.model)
    print("OOF macro_f1:", round(score, 6))

    pairs = top_confusions(cm, labels, k=args.k)
    print("Top confusions (true->pred: count):")
    for a, b, c in pairs:
        if c <= 0:
            continue
        print(f"{a}->{b}: {c}")

    # Print example texts for top 3 pairs
    for a, b, c in pairs[:3]:
        idx = [i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt == a and yp == b]
        if not idx:
            continue
        print("\nExamples for", f"{a}->{b}")
        for i in idx[:5]:
            print("-", text[i][:200].replace("\n", " "))


if __name__ == "__main__":
    main()
