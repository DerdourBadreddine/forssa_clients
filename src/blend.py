from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import config
from .proba_utils import assert_proba_is_canonical


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Local import to avoid heavyweight deps at module import time
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro"))


def _dirichlet_weights(rng: np.random.RandomState, k: int) -> np.ndarray:
    return rng.dirichlet(alpha=np.ones(k, dtype=float))


def _blend_probs(probs_list: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    if len(probs_list) != len(weights):
        raise ValueError("probs_list and weights must have same length")
    out = None
    for p, w in zip(probs_list, weights):
        out = p * float(w) if out is None else (out + p * float(w))
    assert out is not None
    return out


def _apply_class_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Apply additive bias in log-space then softmax back to probs.

    bias shape: (n_classes,)
    """
    eps = 1e-12
    logp = np.log(np.clip(probs, eps, 1.0))
    logp = logp + bias.reshape(1, -1)
    # softmax
    logp = logp - logp.max(axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class BlendResult:
    models: List[str]
    weights: List[float]
    macro_f1: float
    bias: List[float] | None = None


def search_weights(
    y_true: np.ndarray,
    probs_by_model: Dict[str, np.ndarray],
    *,
    trials: int,
    seed: int,
    allow_2_model: bool = True,
) -> BlendResult:
    """Deterministic random search over simplex weights.

    Returns best macro-F1 argmax after blending.
    """
    model_names = list(probs_by_model.keys())
    probs_list = [probs_by_model[m] for m in model_names]
    for p in probs_list:
        assert_proba_is_canonical(p, config.LABELS)

    rng = np.random.RandomState(seed)

    best = BlendResult(models=model_names, weights=[1.0 / len(model_names)] * len(model_names), macro_f1=-1.0)

    def eval_w(w: np.ndarray) -> float:
        blend = _blend_probs(probs_list, w)
        assert_proba_is_canonical(blend, config.LABELS)
        preds = np.array(config.LABELS)[np.argmax(blend, axis=1)]
        return macro_f1(y_true, preds)

    # Some deterministic candidates
    k = len(model_names)
    base = np.eye(k)
    for i in range(k):
        w = base[i]
        sc = eval_w(w)
        if sc > best.macro_f1:
            best = BlendResult(models=model_names, weights=w.tolist(), macro_f1=sc)

    # Random simplex search
    for _ in range(int(trials)):
        w = _dirichlet_weights(rng, k)
        sc = eval_w(w)
        if sc > best.macro_f1:
            best = BlendResult(models=model_names, weights=w.tolist(), macro_f1=sc)

    if allow_2_model and len(model_names) >= 2:
        # Also try best pair with fine grid
        # Find best 2-model pair under macro-F1
        pair_best = best
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                p1, p2 = probs_by_model[m1], probs_by_model[m2]
                for w in np.linspace(0.0, 1.0, 201):
                    blend = w * p1 + (1 - w) * p2
                    assert_proba_is_canonical(blend, config.LABELS)
                    preds = np.array(config.LABELS)[np.argmax(blend, axis=1)]
                    sc = macro_f1(y_true, preds)
                    if sc > pair_best.macro_f1:
                        pair_best = BlendResult(models=[m1, m2], weights=[float(w), float(1 - w)], macro_f1=sc)
        best = pair_best

    return best


def tune_class_bias(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    trials: int,
    seed: int,
    bias_scale: float = 0.3,
) -> Tuple[np.ndarray, float]:
    """Random-search per-class bias vector to maximize macro-F1.

    bias is added to log-probabilities.
    """
    assert_proba_is_canonical(probs, config.LABELS)
    rng = np.random.RandomState(seed)
    n_classes = probs.shape[1]

    best_bias = np.zeros(n_classes, dtype=np.float32)
    best_score = -1.0

    # Always include zero bias
    candidates = [best_bias]

    for _ in range(int(trials)):
        b = rng.normal(loc=0.0, scale=bias_scale, size=n_classes).astype(np.float32)
        # Keep biases centered (prevents global shift)
        b = b - b.mean()
        candidates.append(b)

    for b in candidates:
        adj = _apply_class_bias(probs, b)
        preds = np.array(config.LABELS)[np.argmax(adj, axis=1)]
        sc = macro_f1(y_true, preds)
        if sc > best_score:
            best_score = sc
            best_bias = b

    return best_bias, float(best_score)


def main() -> None:
    parser = argparse.ArgumentParser(description="OOF blending: weight search + optional per-class bias tuning")
    parser.add_argument("--exp-dir", type=Path, required=True, help="artifacts/experiments/<exp_id>")
    parser.add_argument("--trials", type=int, default=15000)
    parser.add_argument("--bias-trials", type=int, default=8000)
    parser.add_argument("--tune-bias", action="store_true")
    args = parser.parse_args()

    exp_dir = args.exp_dir
    y = np.load(exp_dir / "y_true.npy")

    probs_by_model: Dict[str, np.ndarray] = {}
    for name in ["logreg", "nbsvm", "sgd"]:
        p = exp_dir / f"oof_{name}.npy"
        if p.exists():
            probs_by_model[name] = np.load(p)

    if not probs_by_model:
        raise FileNotFoundError(f"No oof_*.npy found under {exp_dir}")

    best = search_weights(y, probs_by_model, trials=args.trials, seed=config.SEED)

    probs_list = [probs_by_model[m] for m in best.models]
    blend = _blend_probs(probs_list, best.weights)

    meta = {
        "models": best.models,
        "weights": best.weights,
        "macro_f1": best.macro_f1,
        "tune_bias": bool(args.tune_bias),
    }

    if args.tune_bias:
        bias, sc2 = tune_class_bias(y, blend, trials=args.bias_trials, seed=config.SEED)
        meta["bias"] = bias.tolist()
        meta["macro_f1_bias"] = sc2
        blend = _apply_class_bias(blend, bias)

    np.save(exp_dir / "oof_blend.npy", blend)
    with open(exp_dir / "blend.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", exp_dir / "blend.json")


if __name__ == "__main__":
    main()
