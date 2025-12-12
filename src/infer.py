from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_io import load_datasets
from .utils import assert_probs_ok, json_dump, json_load


def _str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean (true/false)")


def _sanity_check(sub: pd.DataFrame, test_len: int) -> None:
    if list(sub.columns) != ["id", "Class"]:
        raise ValueError("Submission must have EXACT columns: id,Class")
    if len(sub) != test_len:
        raise ValueError(f"Row count mismatch: submission={len(sub)} test={test_len}")
    if sub["id"].isna().any():
        raise ValueError("Submission has NaN ids")
    if sub["Class"].isna().any():
        raise ValueError("Submission has NaN classes")
    if sub["id"].duplicated().any():
        raise ValueError("Submission has duplicate ids")
    if not sub["Class"].between(1, 9).all():
        bad = sub.loc[~sub["Class"].between(1, 9), "Class"].unique().tolist()
        raise ValueError(f"Class out of range 1..9: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference using final ensemble config -> outputs/submission.csv")
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--reject_low_conf", type=_str2bool, default=False)
    parser.add_argument("--reject_threshold", type=float, default=0.35)
    args = parser.parse_args()

    train_df, test_df, schema = load_datasets(None)

    if not config.FINAL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing {config.FINAL_CONFIG_PATH}. Run: python -m src.blend_oof; python -m src.select_best"
        )
    final = json_load(config.FINAL_CONFIG_PATH)
    probs_path = Path(final["probs_path"])
    if not probs_path.is_absolute():
        probs_path = config.PROJECT_ROOT / probs_path
    if not probs_path.exists():
        raise FileNotFoundError(f"Missing probabilities file: {probs_path}")

    probs = np.load(probs_path)
    assert_probs_ok(probs, config.NUM_CLASSES)

    labels = np.asarray(final.get("labels", config.LABELS), dtype=int)
    preds = labels[np.argmax(probs, axis=1)].astype(int)

    reject_stats = None
    if bool(args.reject_low_conf):
        max_prob = np.max(probs, axis=1)
        thr = float(args.reject_threshold)
        if not (0.0 < thr < 1.0):
            raise ValueError("--reject_threshold must be between 0 and 1")

        fallback_class = int(train_df[schema.label_col].value_counts().idxmax())
        if fallback_class not in config.LABELS:
            raise ValueError(f"Fallback class out of range 1..9: {fallback_class}")

        mask = max_prob < thr
        rejected = int(mask.sum())
        preds = preds.copy()
        preds[mask] = fallback_class

        reject_stats = {
            "reject_low_conf": True,
            "threshold": thr,
            "rejected_count": rejected,
            "rejected_ratio": float(rejected / max(1, len(preds))),
            "fallback_class": fallback_class,
        }

    # Kaggle expects column name exactly "id".
    sub = pd.DataFrame({"id": test_df[schema.id_col].to_numpy(), "Class": preds})

    if args.sanity_check:
        _sanity_check(sub, len(test_df))
        print("Sanity-check OK")

    if reject_stats is not None:
        json_dump(reject_stats, config.OUTPUT_DIR / "reject_stats.json")
        print("Rejected low-confidence:", reject_stats["rejected_count"], "(thr=", reject_stats["threshold"], ")")
        print("Wrote:", config.OUTPUT_DIR / "reject_stats.json")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(config.OUTPUT_SUBMISSION, index=False)
    print("Wrote:", config.OUTPUT_SUBMISSION)


if __name__ == "__main__":
    main()
