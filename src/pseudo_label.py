from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import config
from .data_io import load_datasets
from .metrics import macro_f1, probs_to_labels
from .train_transformer_cv import run_transformer_cv
from .utils import ensure_dir, json_dump, safe_model_id


def _select_pseudo(
    test_probs: np.ndarray,
    *,
    threshold: float,
    cap_per_class: int,
) -> Dict[int, List[int]]:
    """Return indices per class to pseudo-label."""
    conf = test_probs.max(axis=1)
    pred = np.argmax(test_probs, axis=1) + 1  # 1..9

    selected: Dict[int, List[int]] = {c: [] for c in config.LABELS}
    idx = np.argsort(-conf)
    for i in idx:
        if conf[i] < threshold:
            break
        c = int(pred[i])
        if len(selected[c]) >= cap_per_class:
            continue
        selected[c].append(int(i))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="SAFE pseudo-labeling (OFF by default).")
    parser.add_argument("--enable", action="store_true", help="Must be set to actually run.")
    parser.add_argument("--model_id", type=str, default=config.DEFAULT_XLMR)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--cap_per_class", type=int, default=200)
    parser.add_argument("--min_gain", type=float, default=0.005)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--train_bs", type=int, default=8)
    parser.add_argument("--eval_bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--use_dapt", action="store_true")
    args = parser.parse_args()

    if not args.enable:
        print("Pseudo-labeling is OFF by default. Re-run with --enable to execute.")
        return

    safe_id = safe_model_id(args.model_id)
    test_probs_path = config.TESTPROBS_DIR / f"{safe_id}.npy"
    oof_probs_path = config.OOF_DIR / f"{safe_id}.npy"
    if not test_probs_path.exists() or not oof_probs_path.exists():
        raise FileNotFoundError(
            f"Missing transformer probs. Train first: python -m src.train_transformer_cv --model_id {args.model_id}"
        )

    train_df, test_df, schema = load_datasets(None)
    y = train_df[schema.label_col].astype(int).to_numpy()

    base_oof = np.load(oof_probs_path)
    base_pred = probs_to_labels(base_oof, config.LABELS)
    base_score = macro_f1(y, base_pred)

    test_probs = np.load(test_probs_path)

    selected = _select_pseudo(test_probs, threshold=args.threshold, cap_per_class=args.cap_per_class)
    picked = [(c, len(v)) for c, v in selected.items()]
    total = sum(n for _, n in picked)

    if total == 0:
        print("No pseudo-labels selected at threshold", args.threshold)
        return

    # Build pseudo-labeled dataframe
    pseudo_rows = []
    for c, idxs in selected.items():
        for i in idxs:
            pseudo_rows.append((i, c))

    pseudo_idx = [i for i, _ in pseudo_rows]
    pseudo_y = [c for _, c in pseudo_rows]

    pseudo_df = test_df.iloc[pseudo_idx].copy()
    pseudo_df[schema.label_col] = np.asarray(pseudo_y, dtype=int)

    pseudo_dir = ensure_dir(config.OUTPUT_DIR / "pseudo")
    pseudo_path = pseudo_dir / f"pseudo_{safe_id}_thr{args.threshold}.csv"
    pseudo_df.to_csv(pseudo_path, index=False)

    print("Selected pseudo labels (per class):", picked)
    print("Saved pseudo dataset:", pseudo_path)

    # Retrain with pseudo data (safe-gated). We avoid mutating load_datasets(); write augmented CSVs.
    aug_dir = ensure_dir(pseudo_dir / "aug")
    aug_train_path = aug_dir / "train.csv"
    aug_test_path = aug_dir / "test.csv"
    # Save augmented train as (train + pseudo)
    import pandas as pd

    aug_train_df = pd.concat([train_df, pseudo_df], axis=0, ignore_index=True)
    aug_train_df.to_csv(aug_train_path, index=False)
    test_df.to_csv(aug_test_path, index=False)

    run_id = f"{safe_id}__pl_thr{args.threshold}_cap{args.cap_per_class}"

    oof_pl, test_pl, meta = run_transformer_cv(
        model_id=args.model_id,
        run_id=run_id,
        data_dir=aug_dir,
        folds=args.folds,
        seeds=[args.seed] if not args.seeds else [int(x) for x in args.seeds.split(",") if x.strip()],
        max_len=args.max_len,
        train_bs=args.train_bs,
        eval_bs=args.eval_bs,
        lr=args.lr,
        wd=0.01,
        epochs=args.epochs,
        warmup_ratio=0.06,
        grad_accum=2,
        early_stop_patience=2,
        label_smoothing=0.0,
        use_dapt=args.use_dapt,
        loss_kind="weighted_ce",
        focal_gamma=2.0,
        output_root=config.MODELS_DIR,
    )

    # Persist pseudo-run probabilities (for reproducibility). If discarded, we delete them.
    oof_out = config.OOF_DIR / f"{meta['safe_model_id']}.npy"
    test_out = config.TESTPROBS_DIR / f"{meta['safe_model_id']}.npy"
    np.save(oof_out, oof_pl)
    np.save(test_out, test_pl)

    # Compare against original labels only
    y_pred_pl = probs_to_labels(oof_pl[: len(train_df)], config.LABELS)
    pl_score = macro_f1(y, y_pred_pl)

    print("Baseline OOF:", round(base_score, 6))
    print("Pseudo-labeled OOF:", round(pl_score, 6))

    if pl_score >= base_score + args.min_gain:
        out = {
            "base_oof": float(base_score),
            "pseudo_oof": float(pl_score),
            "min_gain": float(args.min_gain),
            "threshold": float(args.threshold),
            "cap_per_class": int(args.cap_per_class),
            "pseudo_path": str(pseudo_path),
            "meta": meta,
        }
        json_dump(out, pseudo_dir / f"pseudo_accept_{safe_id}.json")
        print("ACCEPTED pseudo-labeling (gain >= min_gain).")
    else:
        # Discard: remove produced artifacts to avoid accidental usage.
        try:
            oof_out.unlink(missing_ok=True)
            test_out.unlink(missing_ok=True)
            # Remove model directory (best-effort)
            import shutil

            shutil.rmtree(config.MODELS_DIR / meta["safe_model_id"], ignore_errors=True)
        except Exception:
            pass
        print("DISCARDED pseudo-labeling (gain < min_gain).")


if __name__ == "__main__":
    main()
