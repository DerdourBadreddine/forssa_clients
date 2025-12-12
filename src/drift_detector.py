from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config
from .data_io import load_datasets
from .utils import ensure_dir, json_dump, set_seed


def _top_features(vectorizer: TfidfVectorizer, clf: LogisticRegression, *, top_k: int = 40) -> list[dict[str, Any]]:
    names = vectorizer.get_feature_names_out()
    coef = np.asarray(clf.coef_).reshape(-1)

    # Positive coef => more predictive of test (is_test=1)
    order = np.argsort(-np.abs(coef))
    feats: list[dict[str, Any]] = []
    for idx in order[:top_k]:
        feats.append({"name": str(names[idx]), "coefficient": float(coef[idx])})
    return feats


def run_drift(*, data_dir: Path | None, seed: int, folds: int) -> dict[str, Any]:
    set_seed(seed)

    train_df, test_df, schema = load_datasets(data_dir)

    tr_text = train_df[schema.text_col].astype(str).to_numpy()
    te_text = test_df[schema.text_col].astype(str).to_numpy()

    X_text = np.concatenate([tr_text, te_text], axis=0)
    y = np.concatenate([np.zeros(len(tr_text), dtype=int), np.ones(len(te_text), dtype=int)], axis=0)

    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=200_000,
        sublinear_tf=True,
        norm="l2",
    )

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)

    for tr_idx, va_idx in skf.split(X_text, y):
        X_tr = vec.fit_transform(X_text[tr_idx])
        X_va = vec.transform(X_text[va_idx])

        clf = LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
        )
        clf.fit(X_tr, y[tr_idx])
        oof[va_idx] = clf.predict_proba(X_va)[:, 1]

    auc = float(roc_auc_score(y, oof))

    # Fit on full data to extract top drifting features
    X_all = vec.fit_transform(X_text)
    clf_all = LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    )
    clf_all.fit(X_all, y)

    report: dict[str, Any] = {
        "auc": auc,
        "top_features": _top_features(vec, clf_all, top_k=40),
        "warnings": [],
        "schema": schema.__dict__,
        "n_train": int(len(tr_text)),
        "n_test": int(len(te_text)),
        "folds": int(folds),
        "seed": int(seed),
    }

    if auc > 0.70:
        report["warnings"].append(
            "Distribution shift likely (AUC > 0.70). Consider transformers (XLM-R) and optionally DAPT (MLM on train+test) before fine-tuning."
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect train/test distribution shift via TF-IDF + LogisticRegression")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    report = run_drift(data_dir=args.data_dir, seed=args.seed, folds=args.folds)

    out_path = config.OUTPUT_DIR / "drift_report.json"
    ensure_dir(out_path.parent)
    json_dump(report, out_path)

    print("Drift AUC:", round(report["auc"], 6))
    if report.get("warnings"):
        print("Warnings:")
        for w in report["warnings"]:
            print("-", w)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
