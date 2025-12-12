from __future__ import annotations

import numpy as np
import pandas as pd
import argparse
from typing import List, Tuple

from . import config
from .data_io import load_datasets
from .metrics import full_report, macro_f1, probs_to_labels
from .utils import assert_probs_ok, json_load


def top_confusions(cm: np.ndarray, labels: List[int], top_k: int = 10) -> List[Tuple[int, int, int]]:
    pairs: List[Tuple[int, int, int]] = []
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            if i == j:
                continue
            c = int(cm[i, j])
            if c > 0:
                pairs.append((t, p, c))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def _load_probs(source: str) -> np.ndarray:
    if source == "blend":
        bw = json_load(config.BLEND_WEIGHTS_PATH)
        weights = bw.get("weights", {})
        blend = None
        for src, w in weights.items():
            p = np.load(config.OOF_DIR / f"{src}.npy")
            assert_probs_ok(p, config.NUM_CLASSES)
            blend = p * float(w) if blend is None else (blend + p * float(w))
        if blend is None:
            raise RuntimeError("blend requested but no weights/sources found")
        return blend
    p = np.load(config.OOF_DIR / f"{source}.npy")
    assert_probs_ok(p, config.NUM_CLASSES)
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Error analysis on OOF predictions")
    parser.add_argument("--source", type=str, default="blend", help="OOF source name (e.g. tfidf_svc, xlm-roberta-base) or 'blend'")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()

    train_df, _, schema = load_datasets(None)
    y_true = train_df[schema.label_col].astype(int).to_numpy()
    text = train_df[schema.text_col].astype(str).tolist()

    probs = _load_probs(args.source)
    y_pred = probs_to_labels(probs, config.LABELS)
    report, cm = full_report(y_true, y_pred, labels=config.LABELS)
    print("OOF macro-F1:", round(macro_f1(y_true, y_pred), 6))
    print("\nClassification report:\n" + report)
    print("Confusion matrix:\n", cm)

    pairs = top_confusions(cm, config.LABELS, top_k=args.top_k)
    print("\nTop confusion pairs (true->pred: count):")
    for t, p, c in pairs:
        print(f"{t}->{p}: {c}")

    conf = probs.max(axis=1)
    wrong = np.where(y_true != y_pred)[0]
    # show most confident mistakes first
    wrong = wrong[np.argsort(-conf[wrong])]
    print("\nSample misclassifications (true, pred, conf):")
    for idx in wrong[: int(args.examples)]:
        print(f"true={int(y_true[idx])} pred={int(y_pred[idx])} conf={float(conf[idx]):.3f} :: {text[idx][:200].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
