from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from . import config
from .data_io import build_cv_splits, compute_class_weights, load_datasets
from .metrics import macro_f1, probs_to_labels
from .utils import assert_probs_ok, ensure_dir, json_dump, reorder_proba_columns, set_seed


def build_vectorizers(
    *,
    min_df: int = 2,
    max_features_word: int = 200_000,
    max_features_char: int = 300_000,
) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features_word,
        sublinear_tf=True,
        norm="l2",
        strip_accents=None,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 8),
        min_df=min_df,
        max_features=max_features_char,
        sublinear_tf=True,
        norm="l2",
    )
    return word_vec, char_vec


def _fit_transform(vec_w: TfidfVectorizer, vec_c: TfidfVectorizer, tr_text: np.ndarray, va_text: np.ndarray):
    vec_w.fit(tr_text)
    vec_c.fit(tr_text)
    X_tr = hstack([vec_w.transform(tr_text), vec_c.transform(tr_text)]).tocsr()
    X_va = hstack([vec_w.transform(va_text), vec_c.transform(va_text)]).tocsr()
    return X_tr, X_va


def _transform_test(vec_w: TfidfVectorizer, vec_c: TfidfVectorizer, te_text: np.ndarray):
    return hstack([vec_w.transform(te_text), vec_c.transform(te_text)]).tocsr()


def _train_fold_models(
    X_tr,
    y_tr,
    *,
    seed: int,
) -> Dict[str, object]:
    # 1) LinearSVC + calibration for probabilities
    svc = LinearSVC(class_weight="balanced")
    # Calibrate on train-fold only. (This is inside-fold CV; groups are already contained in fold)
    svc_cal = CalibratedClassifierCV(estimator=svc, cv=3, method="sigmoid")

    # 2) Multinomial LogisticRegression
    logreg = LogisticRegression(
        solver="saga",
        class_weight="balanced",
        max_iter=5000,
        n_jobs=max(1, (config.tfidf_config.n_jobs if hasattr(config, "tfidf_config") else 1)),
        random_state=seed,
        C=4.0,
    )

    svc_cal.fit(X_tr, y_tr)
    logreg.fit(X_tr, y_tr)

    return {"svc_cal": svc_cal, "logreg": logreg}


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF 5-fold StratifiedGroupKFold CV (SVC-calibrated + LogReg)")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features-word", type=int, default=200_000)
    parser.add_argument("--max-features-char", type=int, default=300_000)
    args = parser.parse_args()

    set_seed(args.seed)

    train_df, test_df, schema = load_datasets(args.data_dir)
    y = train_df[schema.label_col].astype(int).to_numpy()
    text = train_df[schema.text_col].astype(str).to_numpy()
    test_text = test_df[schema.text_col].astype(str).to_numpy()

    splits, _ = build_cv_splits(train_df, schema, n_splits=args.folds, seed=args.seed)

    n_train = len(train_df)
    n_test = len(test_df)
    n_classes = config.NUM_CLASSES

    oof = {
        "tfidf_svc": np.zeros((n_train, n_classes), dtype=np.float32),
        "tfidf_logreg": np.zeros((n_train, n_classes), dtype=np.float32),
    }
    test_probs = {
        "tfidf_svc": np.zeros((n_test, n_classes), dtype=np.float32),
        "tfidf_logreg": np.zeros((n_test, n_classes), dtype=np.float32),
    }

    ensure_dir(config.MODELS_DIR / "tfidf")

    for fold, (tr_idx, va_idx) in enumerate(splits):
        tr_text = text[tr_idx]
        va_text = text[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        vec_w, vec_c = build_vectorizers(
            min_df=args.min_df,
            max_features_word=args.max_features_word,
            max_features_char=args.max_features_char,
        )
        X_tr, X_va = _fit_transform(vec_w, vec_c, tr_text, va_text)
        X_te = _transform_test(vec_w, vec_c, test_text)

        models = _train_fold_models(X_tr, y_tr, seed=args.seed)

        # Predict probabilities
        svc_cal = models["svc_cal"]
        p_va = svc_cal.predict_proba(X_va)
        p_te = svc_cal.predict_proba(X_te)
        p_va = reorder_proba_columns(p_va, svc_cal.classes_, config.LABELS)
        p_te = reorder_proba_columns(p_te, svc_cal.classes_, config.LABELS)
        assert_probs_ok(p_va, n_classes)
        assert_probs_ok(p_te, n_classes)
        oof["tfidf_svc"][va_idx] = p_va
        test_probs["tfidf_svc"] += p_te / args.folds

        logreg = models["logreg"]
        p_va = logreg.predict_proba(X_va)
        p_te = logreg.predict_proba(X_te)
        p_va = reorder_proba_columns(p_va, logreg.classes_, config.LABELS)
        p_te = reorder_proba_columns(p_te, logreg.classes_, config.LABELS)
        assert_probs_ok(p_va, n_classes)
        assert_probs_ok(p_te, n_classes)
        oof["tfidf_logreg"][va_idx] = p_va
        test_probs["tfidf_logreg"] += p_te / args.folds

        # Save fold artifacts
        fold_dir = ensure_dir(config.MODELS_DIR / "tfidf" / f"fold_{fold}")
        joblib.dump({"word_vectorizer": vec_w, "char_vectorizer": vec_c}, fold_dir / "vectorizers.joblib")
        joblib.dump({"svc_cal": svc_cal, "logreg": logreg}, fold_dir / "models.joblib")

    # Save OOF + test probs
    for name, arr in oof.items():
        out = config.OOF_DIR / f"{name}.npy"
        ensure_dir(out.parent)
        np.save(out, arr)

    for name, arr in test_probs.items():
        out = config.TESTPROBS_DIR / f"{name}.npy"
        ensure_dir(out.parent)
        np.save(out, arr)

    # Report
    summary = {"schema": schema.__dict__, "folds": args.folds, "seed": args.seed}
    for name, probs in oof.items():
        y_pred = probs_to_labels(probs, config.LABELS)
        summary[f"oof_macro_f1_{name}"] = macro_f1(y, y_pred)

    summary_path = config.OUTPUT_DIR / "tfidf_cv_summary.json"
    json_dump(summary, summary_path)

    print("Saved:")
    print("-", summary_path)
    for name in oof.keys():
        print("-", config.OOF_DIR / f"{name}.npy")
        print("-", config.TESTPROBS_DIR / f"{name}.npy")
    print("OOF macro-F1:")
    for name in ["tfidf_svc", "tfidf_logreg"]:
        print(name, round(summary[f"oof_macro_f1_{name}"], 6))


if __name__ == "__main__":
    main()
