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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.linear_model import SGDClassifier

from . import config
from .data_io import add_normalized_hash, compute_class_weights, load_datasets
from .exp_utils import ensure_dir, run_id, set_global_seed, write_json, append_jsonl
from .proba_utils import assert_proba_is_canonical, predict_proba_canonical, reorder_proba_to_canonical


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


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


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


def _fit_social_ohe(train_social: List[str], val_social: List[str]) -> tuple[csr_matrix, csr_matrix, OneHotEncoder]:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    Xs_tr = enc.fit_transform(np.array(train_social, dtype=object).reshape(-1, 1)).tocsr()
    Xs_va = enc.transform(np.array(val_social, dtype=object).reshape(-1, 1)).tocsr()
    return Xs_tr, Xs_va, enc


def _metric_fn(name: str):
    if name == "macro_f1":
        return macro_f1
    if name == "accuracy":
        return accuracy
    raise ValueError(f"Unknown metric: {name}")


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


CANONICAL = config.LABELS


def train_oof_tfidf(
    train_text: List[str],
    train_social: List[str] | None,
    y: np.ndarray,
    groups: np.ndarray,
    spec: TfidfSpec,
    seeds: List[int],
    n_splits: int,
    out_dir: Path,
    opt_metric: str,
    blend_trials: int,
) -> Dict[str, Any]:
    n = len(y)
    n_classes = len(config.LABELS)

    # OOF probs per model family (averaged across seeds)
    oof_logreg = np.zeros((n, n_classes), dtype=np.float32)
    oof_sgd = np.zeros((n, n_classes), dtype=np.float32)
    oof_nbsvm = np.zeros((n, n_classes), dtype=np.float32)

    # Per-seed OOF scores for stability reporting
    seed_scores: Dict[str, List[float]] = {"logreg": [], "sgd": [], "nbsvm": []}

    fold_scores: Dict[str, List[float]] = {"logreg": [], "sgd": [], "nbsvm": []}
    metric_fn = _metric_fn(opt_metric)

    for seed in seeds:
        set_global_seed(seed)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        # Seed-specific OOF (not averaged)
        seed_oof_logreg = np.zeros((n, n_classes), dtype=np.float32)
        seed_oof_sgd = np.zeros((n, n_classes), dtype=np.float32)
        seed_oof_nbsvm = np.zeros((n, n_classes), dtype=np.float32)

        for fold, (tr_idx, va_idx) in enumerate(cv.split(np.zeros(n), y, groups)):
            tr_text = [train_text[i] for i in tr_idx]
            va_text = [train_text[i] for i in va_idx]
            tr_social = [train_social[i] for i in tr_idx] if train_social is not None else None
            va_social = [train_social[i] for i in va_idx] if train_social is not None else None
            y_tr = y[tr_idx]
            y_va = y[va_idx]

            vec_word, vec_char = _build_vecs(spec)
            X_tr_text, X_va_text = _fit_transform(vec_word, vec_char, tr_text, va_text)

            social_encoder = None
            if tr_social is not None and va_social is not None:
                Xs_tr, Xs_va, social_encoder = _fit_social_ohe(tr_social, va_social)
                X_tr = hstack([X_tr_text, Xs_tr]).tocsr()
                X_va = hstack([X_va_text, Xs_va]).tocsr()
            else:
                X_tr = X_tr_text
                X_va = X_va_text

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
            p = predict_proba_canonical(logreg, X_va, CANONICAL)
            assert_proba_is_canonical(p, CANONICAL)
            oof_logreg[va_idx] += p / len(seeds)
            seed_oof_logreg[va_idx] = p
            fold_scores["logreg"].append(metric_fn(y_va, logreg.predict(X_va)))

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
            # Use canonical reordering (never assume column order)
            p2_raw = _predict_proba_from_sgd(sgd, X_va)
            # _predict_proba_from_sgd returns columns in sgd.classes_ order
            p2 = reorder_proba_to_canonical(p2_raw, sgd.classes_, CANONICAL)
            assert_proba_is_canonical(p2, CANONICAL)
            oof_sgd[va_idx] += p2 / len(seeds)
            seed_oof_sgd[va_idx] = p2
            fold_scores["sgd"].append(metric_fn(y_va, sgd.predict(X_va)))

            # 3) NB-SVM-ish: scale then LogReg
            X_tr_nb_text, r = _nbsvm_features(X_tr_text, y_tr)
            X_va_nb_text = X_va_text.multiply(r)
            if tr_social is not None and va_social is not None:
                # keep social features unscaled
                X_tr_nb = hstack([X_tr_nb_text, Xs_tr]).tocsr()
                X_va_nb = hstack([X_va_nb_text, Xs_va]).tocsr()
            else:
                X_tr_nb = X_tr_nb_text
                X_va_nb = X_va_nb_text
            nbsvm = LogisticRegression(
                C=4.0,
                solver="lbfgs",
                max_iter=3000,
                class_weight=class_weight,
            )
            nbsvm.fit(X_tr_nb, y_tr)
            p3 = predict_proba_canonical(nbsvm, X_va_nb, CANONICAL)
            assert_proba_is_canonical(p3, CANONICAL)
            oof_nbsvm[va_idx] += p3 / len(seeds)
            seed_oof_nbsvm[va_idx] = p3
            fold_scores["nbsvm"].append(metric_fn(y_va, nbsvm.predict(X_va_nb)))

        # Seed-level stability scores
        seed_scores["logreg"].append(macro_f1(y, np.array(CANONICAL)[np.argmax(seed_oof_logreg, axis=1)]))
        seed_scores["sgd"].append(macro_f1(y, np.array(CANONICAL)[np.argmax(seed_oof_sgd, axis=1)]))
        seed_scores["nbsvm"].append(macro_f1(y, np.array(CANONICAL)[np.argmax(seed_oof_nbsvm, axis=1)]))

    # Aggregate OOF scores using argmax
    y_pred_logreg = np.array(config.LABELS)[np.argmax(oof_logreg, axis=1)]
    y_pred_sgd = np.array(config.LABELS)[np.argmax(oof_sgd, axis=1)]
    y_pred_nbsvm = np.array(config.LABELS)[np.argmax(oof_nbsvm, axis=1)]

    oof_scores = {
        "logreg": metric_fn(y, y_pred_logreg),
        "sgd": metric_fn(y, y_pred_sgd),
        "nbsvm": metric_fn(y, y_pred_nbsvm),
    }

    # Always also compute macro_f1 and accuracy for diagnostics
    oof_scores_diag = {
        "macro_f1": {
            "logreg": macro_f1(y, y_pred_logreg),
            "sgd": macro_f1(y, y_pred_sgd),
            "nbsvm": macro_f1(y, y_pred_nbsvm),
        },
        "accuracy": {
            "logreg": accuracy(y, y_pred_logreg),
            "sgd": accuracy(y, y_pred_sgd),
            "nbsvm": accuracy(y, y_pred_nbsvm),
        },
    }

    # Simple blend search (grid) between best two
    probs = {
        "logreg": oof_logreg,
        "sgd": oof_sgd,
        "nbsvm": oof_nbsvm,
    }
    ranked = sorted(oof_scores.items(), key=lambda x: x[1], reverse=True)

    # 2-model blend search (fine grid)
    a, b = ranked[0][0], ranked[1][0]
    best_blend_2 = {"models": [a, b], "w": 0.5, opt_metric: -1.0}
    for w in np.linspace(0.0, 1.0, 101):
        blend = w * probs[a] + (1 - w) * probs[b]
        y_pred = np.array(config.LABELS)[np.argmax(blend, axis=1)]
        sc = metric_fn(y, y_pred)
        if sc > best_blend_2[opt_metric]:
            best_blend_2 = {"models": [a, b], "w": float(w), opt_metric: float(sc)}

    # 3-model blend search over simplex (deterministic random search)
    m1, m2, m3 = ranked[0][0], ranked[1][0], ranked[2][0]
    best_blend_3 = {"models": [m1, m2, m3], "weights": [1 / 3, 1 / 3, 1 / 3], opt_metric: -1.0}
    rng = np.random.RandomState(config.SEED)
    # Also evaluate a few hand-picked candidates
    candidates = [
        (1 / 3, 1 / 3, 1 / 3),
        (0.5, 0.25, 0.25),
        (0.25, 0.5, 0.25),
        (0.25, 0.25, 0.5),
        (0.7, 0.2, 0.1),
        (0.8, 0.1, 0.1),
    ]

    def _eval_weights(w1: float, w2: float, w3: float) -> float:
        blend = w1 * probs[m1] + w2 * probs[m2] + w3 * probs[m3]
        y_pred = np.array(config.LABELS)[np.argmax(blend, axis=1)]
        return metric_fn(y, y_pred)

    for w1, w2, w3 in candidates:
        sc = _eval_weights(w1, w2, w3)
        if sc > best_blend_3[opt_metric]:
            best_blend_3 = {"models": [m1, m2, m3], "weights": [float(w1), float(w2), float(w3)], opt_metric: float(sc)}

    for _ in range(int(blend_trials)):
        # Dirichlet samples valid simplex weights
        w = rng.dirichlet(alpha=[1.0, 1.0, 1.0])
        sc = _eval_weights(float(w[0]), float(w[1]), float(w[2]))
        if sc > best_blend_3[opt_metric]:
            best_blend_3 = {"models": [m1, m2, m3], "weights": [float(w[0]), float(w[1]), float(w[2])], opt_metric: float(sc)}

    # Choose best blend among 2- and 3-model blends
    best_blend = best_blend_2
    if best_blend_3[opt_metric] > best_blend_2[opt_metric]:
        best_blend = best_blend_3

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
        "oof_scores_diag": oof_scores_diag,
        "best_blend": best_blend,
        "opt_metric": opt_metric,
        "blend_trials": int(blend_trials),
        "seed_scores": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": [float(x) for x in v]} for k, v in seed_scores.items()},
        "oof_probs": {"logreg": oof_logreg, "sgd": oof_sgd, "nbsvm": oof_nbsvm},
    }


def fit_full_and_save(
    train_text: List[str],
    train_social: List[str] | None,
    y: np.ndarray,
    spec: TfidfSpec,
    out_dir: Path,
    schema: dict,
) -> Dict[str, Path]:
    vec_word, vec_char = _build_vecs(spec)
    vec_word.fit(train_text)
    vec_char.fit(train_text)
    X_text = hstack([vec_word.transform(train_text), vec_char.transform(train_text)]).tocsr()

    social_encoder = None
    if train_social is not None:
        social_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        X_social = social_encoder.fit_transform(np.array(train_social, dtype=object).reshape(-1, 1)).tocsr()
        X = hstack([X_text, X_social]).tocsr()
    else:
        X = X_text

    cw = compute_class_weights(pd.Series(y))
    class_weight = {k: cw[k] for k in config.LABELS}

    paths: Dict[str, Path] = {}

    # logreg
    logreg = LogisticRegression(C=6.0, solver="lbfgs", max_iter=4000, class_weight=class_weight)
    logreg.fit(X, y)
    payload = {
        "model": logreg,
        "word_vectorizer": vec_word,
        "char_vectorizer": vec_char,
        "social_encoder": social_encoder,
        "schema": schema,
        "spec": spec.__dict__,
        "kind": "logreg",
    }
    p = out_dir / "tfidf_logreg.joblib"
    joblib.dump(payload, p)
    paths["logreg"] = p

    # sgd
    sgd = SGDClassifier(
        loss="log_loss", alpha=1e-5, penalty="l2", max_iter=4000, tol=1e-3, class_weight=class_weight, random_state=config.SEED
    )
    sgd.fit(X, y)
    payload = {
        "model": sgd,
        "word_vectorizer": vec_word,
        "char_vectorizer": vec_char,
        "social_encoder": social_encoder,
        "schema": schema,
        "spec": spec.__dict__,
        "kind": "sgd",
    }
    p = out_dir / "tfidf_sgd.joblib"
    joblib.dump(payload, p)
    paths["sgd"] = p

    # nbsvm
    X_nb_text, r = _nbsvm_features(X_text, y)
    if train_social is not None:
        X_nb = hstack([X_nb_text, X_social]).tocsr()
    else:
        X_nb = X_nb_text
    nbsvm = LogisticRegression(C=4.0, solver="lbfgs", max_iter=4000, class_weight=class_weight)
    nbsvm.fit(X_nb, y)
    payload = {
        "model": nbsvm,
        "word_vectorizer": vec_word,
        "char_vectorizer": vec_char,
        "social_encoder": social_encoder,
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
    parser.add_argument("--opt-metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--use-social", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--blend-trials", type=int, default=15000, help="Random blend trials for 3-model simplex search")
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

    use_social = False
    social_values: List[str] | None = None
    if args.use_social == "off":
        use_social = False
    elif args.use_social == "on":
        if schema.social_col is None or schema.social_col not in train_df.columns:
            raise RuntimeError("--use-social on, but no social/source column was detected in train.csv")
        use_social = True
    else:
        use_social = schema.social_col is not None and schema.social_col in train_df.columns

    if use_social:
        social_values = train_df[schema.social_col].fillna("").astype(str).str.strip().tolist()

    spec = TfidfSpec(
        name=f"w{word_ng[0]}{word_ng[1]}_c{char_ng[0]}{char_ng[1]}_md{args.min_df}",
        word_ngram=word_ng,
        char_ngram=char_ng,
        min_df=args.min_df,
        max_features_word=args.max_word,
        max_features_char=args.max_char,
    )

    results = train_oof_tfidf(texts, social_values, y, groups, spec, seeds, args.folds, out_dir, args.opt_metric, args.blend_trials)

    # Save y_true and per-model OOF probabilities for downstream blending/analysis
    np.save(out_dir / "y_true.npy", y.astype(int))
    oof_probs = results.get("oof_probs", {})
    for k, v in oof_probs.items():
        np.save(out_dir / f"oof_{k}.npy", v)

    # Persist experiment log
    summary = {
        "exp": exp,
        "tfidf_spec": results["spec"],
        "seeds": seeds,
        "folds": args.folds,
        "opt_metric": args.opt_metric,
        "oof_scores": results["oof_scores"],
        "oof_scores_diag": results.get("oof_scores_diag", {}),
        "fold_scores": results["fold_scores"],
        "best_blend": results["best_blend"],
        "seed_scores": results.get("seed_scores", {}),
        "blend_trials": int(args.blend_trials),
    }
    write_json(out_dir / "summary.json", summary)
    append_jsonl(config.EXPERIMENTS_DIR / "experiments.jsonl", summary)

    # Fit full models for test-time inference
    model_paths = fit_full_and_save(texts, social_values, y, spec, out_dir, schema.__dict__)
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
        "opt_metric": args.opt_metric,
        "use_social": use_social,
    }
    write_json(config.EXPERIMENTS_DIR / "best_experiment.json", best_meta)

    print("Experiment:", exp)
    print("Optimized metric:", args.opt_metric)
    print("Using social feature:", use_social, "(col=", schema.social_col, ")")
    print("OOF scores (optimized):", results["oof_scores"])
    if results.get("oof_scores_diag"):
        print("OOF scores (diagnostic):", results["oof_scores_diag"])
    print("Best blend:", results["best_blend"])
    print("Best single model:", best_model, "OOF", args.opt_metric, "=", best_score)


if __name__ == "__main__":
    main()
