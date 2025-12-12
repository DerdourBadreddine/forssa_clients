from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import config
from .data_io import load_datasets
from .metrics import macro_f1, probs_to_labels
from .utils import assert_probs_ok, json_dump, json_load


def _load_validation_choice() -> str | None:
    p = config.OUTPUT_DIR / "validation_report.json"
    if not p.exists():
        return None
    rep = json_load(p)
    chosen = rep.get("chosen_strategy")
    if chosen in {"leakage_safe", "production_like"}:
        return str(chosen)
    return None


def _load_drift_auc() -> float | None:
    p = config.OUTPUT_DIR / "drift_report.json"
    if not p.exists():
        return None
    rep = json_load(p)
    auc = rep.get("auc")
    try:
        return float(auc)
    except Exception:
        return None


def _source_val_strategy(source: str) -> str | None:
    # TF-IDF sources share one summary file.
    if source in {"tfidf_svc", "tfidf_logreg"}:
        p = config.OUTPUT_DIR / "tfidf_cv_summary.json"
        if p.exists():
            rep = json_load(p)
            v = rep.get("val_strategy")
            if v in {"leakage_safe", "production_like"}:
                return str(v)
        return None

    # Transformers store meta per model under outputs/models/<safe_id>/meta.json
    p = config.MODELS_DIR / source / "meta.json"
    if p.exists():
        rep = json_load(p)
        v = rep.get("val_strategy")
        if v in {"leakage_safe", "production_like"}:
            return str(v)
    return None


def _is_transformer_source(source: str) -> bool:
    return (config.MODELS_DIR / source / "meta.json").exists()

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

    chosen_val = _load_validation_choice()
    drift_auc = _load_drift_auc()

    sources = _available_oof_sources()
    if not sources:
        raise FileNotFoundError("No OOF files found in outputs/oof. Train models first.")

    scores: Dict[str, float] = {}
    skipped: Dict[str, str] = {}
    for s in sources:
        src_val = _source_val_strategy(s)
        if chosen_val is not None and src_val is not None and src_val != chosen_val:
            skipped[s] = f"val_strategy mismatch (have {src_val}, need {chosen_val})"
            continue
        if chosen_val is not None and src_val is None:
            skipped[s] = "missing val_strategy metadata"
            continue
        p = np.load(config.OOF_DIR / f"{s}.npy")
        scores[s] = _score_source(y, p)

    if not scores:
        raise FileNotFoundError(
            "No compatible OOF sources found for selection. "
            "Re-run training with --val_strategy auto (or the chosen strategy) to generate aligned OOF files."
        )

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
            # Require all blend sources to be compatible with chosen validation.
            if chosen_val is not None:
                for src in weights.keys():
                    src_val = _source_val_strategy(src)
                    if src_val is None or src_val != chosen_val:
                        weights = {}
                        break
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

    # Drift-aware preference: if shift is likely, prefer best transformer near the top.
    drift_preference_applied = False
    if drift_auc is not None and drift_auc > 0.70:
        transformer_scores = {k: v for k, v in scores.items() if _is_transformer_source(k)}
        if transformer_scores:
            best_tf = max(transformer_scores.items(), key=lambda kv: kv[1])
            # Prefer transformer if it's close to the overall best (within 0.02)
            if float(best_tf[1]) >= float(best_score) - 0.02:
                best_mode = "single"
                best_name = best_tf[0]
                best_score = float(best_tf[1])
                final_probs_path = str((config.TESTPROBS_DIR / f"{best_name}.npy").as_posix())
                drift_preference_applied = True

    final = {
        "mode": best_mode,
        "selected": best_name,
        "selected_oof_macro_f1": best_score,
        "labels": config.LABELS,
        "num_classes": config.NUM_CLASSES,
        "schema": schema.__dict__,
        "probs_path": final_probs_path,
        "all_oof_scores": scores,
        "chosen_validation_strategy": chosen_val,
        "drift_auc": drift_auc,
        "drift_preference_applied": drift_preference_applied,
    }
    if blend_score is not None:
        final["blend_oof_macro_f1"] = float(blend_score)
    if blend_weights is not None:
        final["blend_weights"] = blend_weights
    if skipped:
        final["skipped_sources"] = skipped

    json_dump(final, config.FINAL_CONFIG_PATH)
    print("Selected:", best_mode, best_name, "OOF=", round(best_score, 6))
    if chosen_val is not None:
        print("Using validation strategy:", chosen_val)
    if drift_auc is not None:
        print("Drift AUC:", round(drift_auc, 6))
        if drift_auc > 0.70:
            print("Note: drift likely; consider XLM-R + optional DAPT.")
    print("Wrote:", config.FINAL_CONFIG_PATH)


if __name__ == "__main__":
    main()
