from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional

from . import config


def run_if_missing(path: Path, cmd: list[str]):
    if path.exists():
        return
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def load_best_tfidf_metric() -> Optional[float]:
    metas = list(config.TFIDF_DIR.glob("*_meta.json"))
    if not metas:
        return None
    best = None
    for mp in metas:
        with open(mp, "r", encoding="utf-8") as f:
            data = json.load(f)
        score = data.get("val_macro_f1")
        if score is None:
            continue
        if best is None or score > best:
            best = score
    return best


def load_transformer_metric() -> Optional[float]:
    meta_path = config.TRANSFORMER_DIR / "best" / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("val_macro_f1")


def main():
    parser = argparse.ArgumentParser(description="Train both tracks and select the best model")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if artifacts already exist")
    args = parser.parse_args()

    if not args.skip_train:
        run_if_missing(config.TFIDF_DIR / "best_model.joblib", ["python", "-m", "src.train_tfidf"])
        run_if_missing(config.TRANSFORMER_DIR / "best" / "meta.json", ["python", "-m", "src.train_transformer"])

    tfidf_score = load_best_tfidf_metric()
    transformer_score = load_transformer_metric()

    if tfidf_score is None and transformer_score is None:
        raise RuntimeError("No models trained. Run without --skip-train.")

    best_name = None
    best_score = -1.0
    if tfidf_score is not None and tfidf_score > best_score:
        best_name = "tfidf"
        best_score = tfidf_score
    if transformer_score is not None and transformer_score > best_score:
        best_name = "transformer"
        best_score = transformer_score

    config.SELECTION_FILE.write_text(best_name, encoding="utf-8")
    print(f"Selected best model: {best_name} (macro F1={best_score:.4f})")


if __name__ == "__main__":
    main()
