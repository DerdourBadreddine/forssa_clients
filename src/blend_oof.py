from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import config
from .data_io import load_datasets
from .metrics import macro_f1, probs_to_labels
from .utils import assert_probs_ok, ensure_dir, json_dump


def _available_sources() -> List[str]:
    # Prefer explicit known sources; also allow any .npy in OOF_DIR
    known = ["tfidf_svc", "tfidf_logreg"]
    sources = []
    for s in known:
        if (config.OOF_DIR / f"{s}.npy").exists() and (config.TESTPROBS_DIR / f"{s}.npy").exists():
            sources.append(s)
    for p in config.OOF_DIR.glob("*.npy"):
        name = p.stem
        if name in sources:
            continue
        if (config.TESTPROBS_DIR / f"{name}.npy").exists():
            sources.append(name)
    return sources


def _dirichlet(rng: np.random.RandomState, k: int) -> np.ndarray:
    return rng.dirichlet(np.ones(k, dtype=float))


def main() -> None:
    parser = argparse.ArgumentParser(description="OOF blending weight search optimizing macro-F1")
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    train_df, _, schema = load_datasets(None)
    y = train_df[schema.label_col].astype(int).to_numpy()

    sources = _available_sources()
    if len(sources) < 1:
        raise FileNotFoundError("No OOF sources found in outputs/oof. Run training first.")

    probs_by_src: Dict[str, np.ndarray] = {}
    for s in sources:
        p = np.load(config.OOF_DIR / f"{s}.npy")
        assert_probs_ok(p, config.NUM_CLASSES)
        probs_by_src[s] = p

    rng = np.random.RandomState(args.seed)

    def score_w(w: np.ndarray) -> float:
        blend = np.zeros_like(next(iter(probs_by_src.values())))
        for i, s in enumerate(sources):
            blend += probs_by_src[s] * float(w[i])
        y_pred = probs_to_labels(blend, config.LABELS)
        return macro_f1(y, y_pred)

    # Candidates
    k = len(sources)
    best_w = np.zeros(k, dtype=float)
    best_w[0] = 1.0
    best_score = score_w(best_w)

    # Single models
    for i in range(k):
        w = np.zeros(k, dtype=float)
        w[i] = 1.0
        sc = score_w(w)
        if sc > best_score:
            best_score, best_w = sc, w

    # Random search
    for _ in range(int(args.trials)):
        w = _dirichlet(rng, k)
        sc = score_w(w)
        if sc > best_score:
            best_score, best_w = sc, w

    # Pairwise grid (fast refinement)
    if k >= 2:
        for i in range(k):
            for j in range(i + 1, k):
                for a in np.linspace(0.0, 1.0, 401):
                    w = np.zeros(k, dtype=float)
                    w[i] = float(a)
                    w[j] = float(1 - a)
                    sc = score_w(w)
                    if sc > best_score:
                        best_score, best_w = sc, w

    # Save weights
    weights = {sources[i]: float(best_w[i]) for i in range(k) if best_w[i] > 0}

    out = {
        "sources": sources,
        "weights": weights,
        "oof_macro_f1": float(best_score),
        "trials": int(args.trials),
        "seed": int(args.seed),
        "n_models": int(k),
    }

    # Blend test probabilities
    test_blend = None
    for s, w in weights.items():
        p = np.load(config.TESTPROBS_DIR / f"{s}.npy")
        assert_probs_ok(p, config.NUM_CLASSES)
        test_blend = p * float(w) if test_blend is None else (test_blend + p * float(w))

    assert test_blend is not None

    ensure_dir(config.TESTPROBS_DIR)
    np.save(config.TESTPROBS_DIR / "blend.npy", test_blend)
    json_dump(out, config.BLEND_WEIGHTS_PATH)

    print("Blended OOF macro-F1:", round(best_score, 6))
    print("Saved:")
    print("-", config.BLEND_WEIGHTS_PATH)
    print("-", config.TESTPROBS_DIR / "blend.npy")


if __name__ == "__main__":
    main()
