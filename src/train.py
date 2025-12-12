from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import normalize
from sklearn.linear_model import SGDClassifier

from . import config
from .data_io import add_normalized_hash, compute_class_weights, load_datasets
from .exp_utils import ensure_dir, run_id, set_global_seed, write_json, append_jsonl


@dataclass
class TfidfSpec:
    name: str
    word_ngram: tuple[int, int]
    char_ngram: tuple[int, int]
    min_df: int
    max_features_word: int
    max_features_char: int


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def _build_vecs(spec: TfidfSpec) -> tuple[TfidfVectorizer, TfidfVectorizer]:
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=spec.word_ngram,
        min_df=spec.min_df,
        max_features=spec.max_features_word,
        sublinear_tf=True,
        norm="l2",
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=spec.char_ngram,
        min_df=spec.min_df,
        max_features=spec.max_features_char,
        sublinear_tf=True,
        norm="l2",
    )
    return word, char


def _fit_transform(vec_word, vec_char, train_text: List[str], val_text: List[str]) -> tuple[csr_matrix, csr_matrix]:
    vec_word.fit(train_text)
    vec_char.fit(train_text)
    X_tr = hstack([vec_word.transform(train_text), vec_char.transform(train_text)]).tocsr()
    X_va = hstack([vec_word.transform(val_text), vec_char.transform(val_text)]).tocsr()
    return X_tr, X_va


def _nbsvm_features(X: csr_matrix, y: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
    # NB-SVM style log-count ratio scaling (Wang & Manning)
    # Compute per-class log-count ratios and collapse to a single r by averaging.
    # This is a pragmatic variant for multi-class.
    X_bin = X.copy()
    X_bin.data = np.ones_like(X_bin.data)
    n_classes = len(config.LABELS)
    r_all = []
    for cls in config.LABELS:
        mask = y == cls
        p = X_bin[mask].sum(axis=0) + 1
        q = X_bin[~mask].sum(axis=0) + 1
        r = np.log(p / q)
        r_all.append(np.asarray(r).ravel())
    r_mean = np.mean(np.vstack(r_all), axis=0)
    X_scaled = X.multiply(r_mean)
    return X_scaled.tocsr(), r_mean


def _predict_proba_from_sgd(clf: SGDClassifier, X: csr_matrix) -> np.ndarray:
    # SGD log_loss supports predict_proba; guard for versions.
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    scores = clf.decision_function(X)
    # softmax
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def train_oof_tfidf(
    train_text: List[str],
    y: np.ndarray,
    groups: np.ndarray,
    spec: TfidfSpec,
    seeds: List[int],
    n_splits: int,
    out_dir: Path,
) -> Dict[str, Any]:
    n = len(y)
    n_classes = len(config.LABELS)

    # OOF probs per model family
    oof_logreg = np.zeros((n, n_classes), dtype=np.float32)
    oof_sgd = np.zeros((n, n_classes), dtype=np.float32)
    oof_nbsvm = np.zeros((n, n_classes), dtype=np.float32)

    fold_scores: Dict[str, List[float]] = {"logreg": [], "sgd": [], "nbsvm": []}

    for seed in seeds:
        set_global_seed(seed)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(cv.split(np.zeros(n), y, groups)):
            tr_text = [train_text[i] for i in tr_idx]
            va_text = [train_text[i] for i in va_idx]
            y_tr = y[tr_idx]
            y_va = y[va_idx]

            vec_word, vec_char = _build_vecs(spec)
            X_tr, X_va = _fit_transform(vec_word, vec_char, tr_text, va_text)

            # class weights (balanced)
            cw = compute_class_weights(pd.Series(y_tr))
            class_weight = {k: cw[k] for k in config.LABELS}

            # 1) LogReg
            logreg = LogisticRegression(
                C=6.0,
                solver="lbfgs",
                max_iter=3000,
                class_weight=class_weight,
            )
            logreg.fit(X_tr, y_tr)
            p = logreg.predict_proba(X_va)
            oof_logreg[va_idx] += p / len(seeds)
            fold_scores["logreg"].append(macro_f1(y_va, logreg.predict(X_va)))

            # 2) SGD log-loss
            sgd = SGDClassifier(
                loss="log_loss",
                alpha=1e-5,
                penalty="l2",
                max_iter=2000,
                tol=1e-3,
                class_weight=class_weight,
                random_state=seed,
            )
            sgd.fit(X_tr, y_tr)
            p2 = _predict_proba_from_sgd(sgd, X_va)
            oof_sgd[va_idx] += p2 / len(seeds)
            fold_scores["sgd"].append(macro_f1(y_va, sgd.predict(X_va)))

            # 3) NB-SVM-ish: scale then LogReg
            X_tr_nb, r = _nbsvm_features(X_tr, y_tr)
            X_va_nb = X_va.multiply(r)
            nbsvm = LogisticRegression(
                C=4.0,
                solver="lbfgs",
                max_iter=3000,
                class_weight=class_weight,
            )
            nbsvm.fit(X_tr_nb, y_tr)
            p3 = nbsvm.predict_proba(X_va_nb)
            oof_nbsvm[va_idx] += p3 / len(seeds)
            fold_scores["nbsvm"].append(macro_f1(y_va, nbsvm.predict(X_va_nb)))

    # Aggregate OOF scores using argmax
    y_pred_logreg = np.array(config.LABELS)[np.argmax(oof_logreg, axis=1)]
    y_pred_sgd = np.array(config.LABELS)[np.argmax(oof_sgd, axis=1)]
    y_pred_nbsvm = np.array(config.LABELS)[np.argmax(oof_nbsvm, axis=1)]

    oof_scores = {
        "logreg": macro_f1(y, y_pred_logreg),
        "sgd": macro_f1(y, y_pred_sgd),
        "nbsvm": macro_f1(y, y_pred_nbsvm),
    }

    # Simple blend search (grid) between best two
    probs = {
        "logreg": oof_logreg,
        "sgd": oof_sgd,
        "nbsvm": oof_nbsvm,
    }
    ranked = sorted(oof_scores.items(), key=lambda x: x[1], reverse=True)
    a, b = ranked[0][0], ranked[1][0]

    best_blend = {"models": [a, b], "w": 0.5, "macro_f1": -1.0}
    for w in np.linspace(0.0, 1.0, 21):
        blend = w * probs[a] + (1 - w) * probs[b]
        y_pred = np.array(config.LABELS)[np.argmax(blend, axis=1)]
        sc = macro_f1(y, y_pred)
        if sc > best_blend["macro_f1"]:
            best_blend = {"models": [a, b], "w": float(w), "macro_f1": float(sc)}

    # Save OOF arrays
    ensure_dir(out_dir)
    np.save(out_dir / f"oof_{spec.name}_logreg.npy", oof_logreg)
    np.save(out_dir / f"oof_{spec.name}_sgd.npy", oof_sgd)
    np.save(out_dir / f"oof_{spec.name}_nbsvm.npy", oof_nbsvm)

    return {
        "spec": spec.__dict__,
        "seeds": seeds,
        "n_splits": n_splits,
        "fold_scores": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in fold_scores.items()},
        "oof_scores": oof_scores,
        "best_blend": best_blend,
    }


def fit_full_and_save(train_text: List[str], y: np.ndarray, spec: TfidfSpec, out_dir: Path, schema: dict) -> Dict[str, Path]:
    vec_word, vec_char = _build_vecs(spec)
    vec_word.fit(train_text)
    vec_char.fit(train_text)
    X = hstack([vec_word.transform(train_text), vec_char.transform(train_text)]).tocsr()

    cw = compute_class_weights(pd.Series(y))
    class_weight = {k: cw[k] for k in config.LABELS}

    paths: Dict[str, Path] = {}

    # logreg
    logreg = LogisticRegression(C=6.0, solver="lbfgs", max_iter=4000, class_weight=class_weight)
    logreg.fit(X, y)
    payload = {"model": logreg, "word_vectorizer": vec_word, "char_vectorizer": vec_char, "schema": schema, "spec": spec.__dict__, "kind": "logreg"}
    p = out_dir / "tfidf_logreg.joblib"
    joblib.dump(payload, p)
    paths["logreg"] = p

    # sgd
    sgd = SGDClassifier(
        loss="log_loss", alpha=1e-5, penalty="l2", max_iter=4000, tol=1e-3, class_weight=class_weight, random_state=config.SEED
    )
    sgd.fit(X, y)
    payload = {"model": sgd, "word_vectorizer": vec_word, "char_vectorizer": vec_char, "schema": schema, "spec": spec.__dict__, "kind": "sgd"}
    p = out_dir / "tfidf_sgd.joblib"
    joblib.dump(payload, p)
    paths["sgd"] = p

    # nbsvm
    X_nb, r = _nbsvm_features(X, y)
    nbsvm = LogisticRegression(C=4.0, solver="lbfgs", max_iter=4000, class_weight=class_weight)
    nbsvm.fit(X_nb, y)
    payload = {
        "model": nbsvm,
        "word_vectorizer": vec_word,
        "char_vectorizer": vec_char,
        "schema": schema,
        "spec": spec.__dict__,
        "kind": "nbsvm",
        "r": r,
    }
    p = out_dir / "tfidf_nbsvm.joblib"
    joblib.dump(payload, p)
    paths["nbsvm"] = p

    return paths


def main():
    parser = argparse.ArgumentParser(description="Leaderboard-oriented CV training with OOF + blending")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-char", type=int, default=300000)
    parser.add_argument("--max-word", type=int, default=150000)
    parser.add_argument("--word-ngram", type=str, default="1,3")
    parser.add_argument("--char-ngram", type=str, default="3,7")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    word_ng = tuple(int(x) for x in args.word_ngram.split(","))
    char_ng = tuple(int(x) for x in args.char_ngram.split(","))

    exp = run_id("exp")
    out_dir = ensure_dir(config.EXPERIMENTS_DIR / exp)

    train_df, _, schema = load_datasets()
    train_df = add_normalized_hash(train_df, schema.text_col)

    y = train_df[schema.label_col].astype(int).values
    groups = train_df["text_hash"].values
    texts = train_df[schema.text_col].tolist()

    spec = TfidfSpec(
        name=f"w{word_ng[0]}{word_ng[1]}_c{char_ng[0]}{char_ng[1]}_md{args.min_df}",
        word_ngram=word_ng,
        char_ngram=char_ng,
        min_df=args.min_df,
        max_features_word=args.max_word,
        max_features_char=args.max_char,
    )

    results = train_oof_tfidf(texts, y, groups, spec, seeds, args.folds, out_dir)

    # Persist experiment log
    summary = {
        "exp": exp,
        "tfidf_spec": results["spec"],
        "seeds": seeds,
        "folds": args.folds,
        "oof_scores": results["oof_scores"],
        "fold_scores": results["fold_scores"],
        "best_blend": results["best_blend"],
    }
    write_json(out_dir / "summary.json", summary)
    append_jsonl(config.EXPERIMENTS_DIR / "experiments.jsonl", summary)

    # Fit full models for test-time inference
    model_paths = fit_full_and_save(texts, y, spec, out_dir, schema.__dict__)
    write_json(out_dir / "models.json", {k: str(v) for k, v in model_paths.items()})

    # Mark this experiment as current best candidate (by OOF)
    best_model = max(results["oof_scores"].items(), key=lambda x: x[1])[0]
    best_score = float(results["oof_scores"][best_model])
    best_meta = {
        "exp": exp,
        "best_model": best_model,
        "best_score": best_score,
        "best_blend": results["best_blend"],
        "models": {k: str(v) for k, v in model_paths.items()},
        "schema": schema.__dict__,
        "tfidf_spec": results["spec"],
        "seeds": seeds,
        "folds": args.folds,
    }
    write_json(config.EXPERIMENTS_DIR / "best_experiment.json", best_meta)

    print("Experiment:", exp)
    print("OOF scores:", results["oof_scores"])
    print("Best blend:", results["best_blend"])
    print("Best single model:", best_model, "OOF macro F1=", best_score)


if __name__ == "__main__":
    main()
