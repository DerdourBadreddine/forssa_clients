from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC

from . import config
from .data_io import compute_class_weights, load_datasets, stratified_leakage_safe_split
from .metrics import full_report, macro_f1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_vectorizers() -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    cfg = config.tfidf_config
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=cfg.word_ngrams,
        min_df=cfg.min_df,
        max_features=cfg.max_features_word,
        use_idf=cfg.use_idf,
        smooth_idf=cfg.smooth_idf,
        sublinear_tf=cfg.sublinear_tf,
        norm=cfg.normalize,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=cfg.char_ngrams,
        min_df=cfg.min_df,
        max_features=cfg.max_features_char,
        use_idf=cfg.use_idf,
        smooth_idf=cfg.smooth_idf,
        sublinear_tf=cfg.sublinear_tf,
        norm=cfg.normalize,
    )
    return word_vec, char_vec


def train_and_eval(train_texts, train_labels, val_texts, val_labels, class_weights: Dict[int, float]):
    cfg = config.tfidf_config
    word_vec, char_vec = build_vectorizers()

    word_vec.fit(train_texts)
    char_vec.fit(train_texts)

    X_train = hstack([word_vec.transform(train_texts), char_vec.transform(train_texts)])
    X_val = hstack([word_vec.transform(val_texts), char_vec.transform(val_texts)])

    models = {
        "logreg": LogisticRegression(
            max_iter=cfg.max_iter,
            C=cfg.C,
            solver=cfg.solver,
            n_jobs=cfg.n_jobs,
            class_weight=class_weights,
        ),
        "linearsvc": LinearSVC(class_weight=class_weights),
    }

    results = {}
    for name, model in models.items():
        cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=config.SEED)
        cv_scores = cross_val_score(model, X_train, train_labels, cv=cv, scoring="f1_macro", n_jobs=cfg.n_jobs)
        model.fit(X_train, train_labels)
        val_pred = model.predict(X_val)
        val_f1 = macro_f1(val_labels, val_pred)
        results[name] = {
            "model": model,
            "cv_macro_f1_mean": float(np.mean(cv_scores)),
            "cv_macro_f1_std": float(np.std(cv_scores)),
            "val_macro_f1": float(val_f1),
            "val_pred": val_pred,
        }
    return results, word_vec, char_vec


def save_artifacts(model_name: str, model, word_vec, char_vec, schema_dict: dict, metadata: dict) -> None:
    payload = {
        "model": model,
        "word_vectorizer": word_vec,
        "char_vectorizer": char_vec,
        "schema": schema_dict,
    }
    model_path = config.TFIDF_DIR / f"{model_name}_model.joblib"
    meta_path = config.TFIDF_DIR / f"{model_name}_meta.json"
    joblib.dump(payload, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    # Save a symlink to best model for inference simplicity
    best_link = config.TFIDF_DIR / "best_model.joblib"
    joblib.dump(payload, best_link)


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF baselines")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--test-size", type=float, default=config.tfidf_config.test_size)
    args = parser.parse_args()

    set_seed(config.SEED)
    train_df, _, schema = load_datasets(args.data_dir)
    train_df, val_df = stratified_leakage_safe_split(train_df, schema, test_size=args.test_size, seed=config.SEED)

    class_weights = compute_class_weights(train_df[schema.label_col])

    results, word_vec, char_vec = train_and_eval(
        train_df[schema.text_col].tolist(),
        train_df[schema.label_col].astype(int).values,
        val_df[schema.text_col].tolist(),
        val_df[schema.label_col].astype(int).values,
        class_weights,
    )

    # Pick best by validation macro f1
    best_name = max(results.keys(), key=lambda n: results[n]["val_macro_f1"])
    best = results[best_name]

    report_str, cm = full_report(val_df[schema.label_col].values, best["val_pred"])
    print(f"Best model: {best_name}")
    print(f"CV Macro F1: {best['cv_macro_f1_mean']:.4f} ± {best['cv_macro_f1_std']:.4f}")
    print(f"Validation Macro F1: {best['val_macro_f1']:.4f}")
    print("\nClassification report:\n" + report_str)
    print("Confusion matrix:\n", cm)

    metadata = {
        "best_model": best_name,
        "cv_macro_f1_mean": best["cv_macro_f1_mean"],
        "cv_macro_f1_std": best["cv_macro_f1_std"],
        "val_macro_f1": best["val_macro_f1"],
        "schema": schema.__dict__,
        "class_weights": class_weights,
    }
    save_artifacts(best_name, best["model"], word_vec, char_vec, schema.__dict__, metadata)


if __name__ == "__main__":
    main()
