from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack

from . import config
from .data_io import DataSchema, load_datasets


def _find_col_case_insensitive(df: pd.DataFrame, name: str) -> str | None:
    if name in df.columns:
        return name
    low = name.lower()
    for c in df.columns:
        if c.lower() == low:
            return c
    return None


def _load_best_experiment() -> Dict:
    p = config.EXPERIMENTS_DIR / "best_experiment.json"
    if not p.exists():
        raise FileNotFoundError("Missing artifacts/experiments/best_experiment.json. Run: python -m src.train")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _predict_tfidf_payload(payload: Dict, test_df: pd.DataFrame) -> np.ndarray:
    schema = DataSchema(**payload["schema"])
    word_vec = payload["word_vectorizer"]
    char_vec = payload["char_vectorizer"]
    model = payload["model"]

    text_col = _find_col_case_insensitive(test_df, schema.text_col)
    if text_col is None:
        raise RuntimeError(f"Expected text column '{schema.text_col}' in test, found {list(test_df.columns)}")

    social_col = None
    if getattr(schema, "social_col", None):
        social_col = _find_col_case_insensitive(test_df, schema.social_col)

    Xw = word_vec.transform(test_df[text_col])
    Xc = char_vec.transform(test_df[text_col])
    X_text = hstack([Xw, Xc])

    social_encoder = payload.get("social_encoder")
    if social_encoder is not None and social_col is not None:
        Xs = social_encoder.transform(test_df[social_col].fillna("").astype(str).str.strip().to_numpy().reshape(-1, 1))
        X = hstack([X_text, Xs])
    else:
        X = X_text

    kind = payload.get("kind")
    if kind == "nbsvm":
        r = payload["r"]
        # r applies only to text features
        X_text_scaled = X_text.multiply(r)
        if social_encoder is not None and social_col is not None:
            X = hstack([X_text_scaled, Xs])
        else:
            X = X_text_scaled

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    # fallback: decision_function -> softmax
    scores = model.decision_function(X)
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission from best experiment artifacts")
    parser.add_argument("--mode", choices=["best", "blend"], default="blend", help="Use best single model or best 2-model blend")
    args = parser.parse_args()

    _, test_df, schema_live = load_datasets()

    best = _load_best_experiment()
    schema = DataSchema(**best["schema"])

    # Align columns from live detection to artifact schema
    # (both are normalized in data_io; we just need the correct text/id columns)
    if schema.text_col not in test_df.columns:
        if _find_col_case_insensitive(test_df, schema.text_col) is None:
            raise RuntimeError(f"Expected text column '{schema.text_col}' in test, found {list(test_df.columns)}")
    if schema.id_col not in test_df.columns:
        if _find_col_case_insensitive(test_df, schema.id_col) is None:
            raise RuntimeError(f"Expected id column '{schema.id_col}' in test, found {list(test_df.columns)}")

    models = best["models"]

    if args.mode == "best":
        model_key = best["best_model"]
        payload = joblib.load(models[model_key])
        probs = _predict_tfidf_payload(payload, test_df)
    else:
        blend = best["best_blend"]
        a, b = blend["models"]
        w = float(blend["w"])
        pa = _predict_tfidf_payload(joblib.load(models[a]), test_df)
        pb = _predict_tfidf_payload(joblib.load(models[b]), test_df)
        probs = w * pa + (1 - w) * pb

    labels = np.array(config.LABELS)
    preds = labels[np.argmax(probs, axis=1)]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    score = float(best.get("best_score", 0.0))
    tag = f"exp_{best['exp']}_{args.mode}"
    out_name = f"submission_{tag}_cv{score:.4f}_{ts}.csv"
    out_path = config.OUTPUT_DIR / out_name

    id_col = _find_col_case_insensitive(test_df, schema.id_col) or schema.id_col
    sub = pd.DataFrame({id_col: test_df[id_col], "Class": preds.astype(int)})
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
