from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import config
from .data_io import load_datasets
from .metrics import macro_f1, probs_to_labels
from .utils import assert_probs_ok, json_dump, json_load

def _available_oof_sources() -> List[str]:
    return sorted([p.stem for p in config.OOF_DIR.glob("*.npy")])


def _score_source(y: np.ndarray, probs: np.ndarray) -> float:
    assert_probs_ok(probs, config.NUM_CLASSES)
    pred = probs_to_labels(probs, config.LABELS)
    return macro_f1(y, pred)


def main():
    parser = argparse.ArgumentParser(description="Select best single model or blended ensemble and write outputs/final_config.json")
    args = parser.parse_args()

    train_df, _, schema = load_datasets(None)
    y = train_df[schema.label_col].astype(int).to_numpy()

    sources = _available_oof_sources()
    if not sources:
        raise FileNotFoundError("No OOF files found in outputs/oof. Train models first.")

    scores: Dict[str, float] = {}
    for s in sources:
        p = np.load(config.OOF_DIR / f"{s}.npy")
        scores[s] = _score_source(y, p)

    best_single = max(scores.items(), key=lambda kv: kv[1])

    best_mode = "single"
    best_name = best_single[0]
    best_score = float(best_single[1])
    final_probs_path = str((config.TESTPROBS_DIR / f"{best_name}.npy").as_posix())

    blend_score = None
    blend_weights = None
    if config.BLEND_WEIGHTS_PATH.exists() and (config.TESTPROBS_DIR / "blend.npy").exists():
        bw = json_load(config.BLEND_WEIGHTS_PATH)
        weights = bw.get("weights", {})
        if isinstance(weights, dict) and weights:
            # compute blended OOF score
            blend = None
            for src, w in weights.items():
                if (config.OOF_DIR / f"{src}.npy").exists():
                    p = np.load(config.OOF_DIR / f"{src}.npy")
                    assert_probs_ok(p, config.NUM_CLASSES)
                    blend = p * float(w) if blend is None else (blend + p * float(w))
            if blend is not None:
                blend_score = _score_source(y, blend)
                blend_weights = weights
                if blend_score > best_score:
                    best_mode = "blend"
                    best_name = "blend"
                    best_score = float(blend_score)
                    final_probs_path = str((config.TESTPROBS_DIR / "blend.npy").as_posix())

    final = {
        "mode": best_mode,
        "selected": best_name,
        "selected_oof_macro_f1": best_score,
        "labels": config.LABELS,
        "num_classes": config.NUM_CLASSES,
        "schema": schema.__dict__,
        "probs_path": final_probs_path,
        "all_oof_scores": scores,
    }
    if blend_score is not None:
        final["blend_oof_macro_f1"] = float(blend_score)
    if blend_weights is not None:
        final["blend_weights"] = blend_weights

    json_dump(final, config.FINAL_CONFIG_PATH)
    print("Selected:", best_mode, best_name, "OOF=", round(best_score, 6))
    print("Wrote:", config.FINAL_CONFIG_PATH)


if __name__ == "__main__":
    main()
