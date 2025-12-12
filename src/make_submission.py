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
from .proba_utils import assert_proba_is_canonical, predict_proba_canonical


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

    proba = predict_proba_canonical(model, X, config.LABELS)
    assert_proba_is_canonical(proba, config.LABELS)
    return proba


def _debug_and_validate_alignment(test_df: pd.DataFrame, sub_df: pd.DataFrame, *, submission_path: Path, debug: bool) -> None:
    if "id" not in test_df.columns:
        raise RuntimeError("test_file.csv must contain column 'id'")
    if "id" not in sub_df.columns or "Class" not in sub_df.columns:
        raise RuntimeError("submission must contain columns: id, Class")

    ids_equal_order = bool(test_df["id"].equals(sub_df["id"]))
    same_id_set = bool(set(test_df["id"]) == set(sub_df["id"]))
    duplicates = int(sub_df["id"].duplicated().sum())
    class_min = int(sub_df["Class"].min()) if len(sub_df) else None
    class_max = int(sub_df["Class"].max()) if len(sub_df) else None

    if debug:
        print(f"submission_path: {submission_path}")
        print(f"ids_equal_order: {ids_equal_order}")
        print(f"same_id_set: {same_id_set}")
        print(f"duplicates: {duplicates}")
        print(f"class_range: {class_min}..{class_max}")
        print(f"rows_test: {len(test_df)}")
        print(f"rows_submission: {len(sub_df)}")

    # Hard fail-fast checks (do NOT write CSV if any fail)
    if len(sub_df) != len(test_df):
        raise RuntimeError(f"Row mismatch: test={len(test_df)} submission={len(sub_df)}")
    if sub_df["id"].isna().any():
        raise RuntimeError("Submission contains NaN ids")
    if sub_df["Class"].isna().any():
        raise RuntimeError("Submission contains NaN Class")
    if duplicates != 0:
        raise RuntimeError(f"Submission contains duplicate ids: {duplicates}")
    if not same_id_set:
        raise RuntimeError("Submission id set differs from test id set")
    if not ids_equal_order:
        raise RuntimeError("Submission id order does not match test_file.csv order")
    if not sub_df["Class"].between(1, 9).all():
        raise RuntimeError("Predictions out of range 1..9")


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission from best experiment artifacts")
    parser.add_argument("--mode", choices=["best", "blend"], default="blend", help="Use best single model or best 2-model blend")
    parser.add_argument("--debug_alignment", action="store_true", help="Print and assert submission/test alignment checks")
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
        blend_models = blend["models"]
        # Backward compatible: either {models:[a,b], w:0.6} or {models:[a,b,c], weights:[...]} 
        if "weights" in blend:
            weights = [float(x) for x in blend["weights"]]
            if len(weights) != len(blend_models):
                raise RuntimeError(f"Blend weights length {len(weights)} != models length {len(blend_models)}")
            probs = None
            for m, w in zip(blend_models, weights):
                pm = _predict_tfidf_payload(joblib.load(models[m]), test_df)
                assert_proba_is_canonical(pm, config.LABELS)
                probs = pm * w if probs is None else (probs + pm * w)
        else:
            a, b = blend_models
            w = float(blend["w"])
            pa = _predict_tfidf_payload(joblib.load(models[a]), test_df)
            pb = _predict_tfidf_payload(joblib.load(models[b]), test_df)
            assert_proba_is_canonical(pa, config.LABELS)
            assert_proba_is_canonical(pb, config.LABELS)
            probs = w * pa + (1 - w) * pb

    # Final proba validation before argmax/blending
    assert_proba_is_canonical(probs, config.LABELS)

    labels = np.array(config.LABELS)
    preds = labels[np.argmax(probs, axis=1)]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    score = float(best.get("best_score", 0.0))
    out_name = f"submission_{args.mode}_{best['exp']}_{ts}.csv"
    out_path = config.OUTPUT_DIR / out_name

    # Kaggle expects the column name exactly "id"
    if "id" not in test_df.columns:
        raise RuntimeError(f"Expected 'id' column in test, found {list(test_df.columns)}")

    # Build submission in EXACT test_file.csv order
    sub = pd.DataFrame({"id": test_df["id"].values, "Class": preds.astype(int)})

    # Validate alignment and fail-fast BEFORE writing
    _debug_and_validate_alignment(test_df, sub, submission_path=out_path, debug=args.debug_alignment)

    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
