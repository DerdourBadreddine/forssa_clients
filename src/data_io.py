from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from . import config
from .text_norm import normalize, normalize_strong
from .utils import stable_hash


@dataclass
class DataSchema:
    id_col: str
    text_col: str
    label_col: Optional[str] = None
    social_col: Optional[str] = None


class SchemaError(Exception):
    """Raised when expected columns are missing."""


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
        for existing in df.columns:
            if existing.lower() == col.lower():
                return existing
    return None


PRODUCTION_GROUP_CANDIDATES = [
    "Réseau Social",
    "reseau_social",
    "channel",
    "source",
    "platform",
    "origine",
    "canal",
]


def _resolve_data_paths(data_dir: Path) -> tuple[Path, Path]:
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file at {train_path}")

    # Kaggle varies; accept both, but prefer required name.
    test_csv = data_dir / "test.csv"
    test_file = data_dir / "test_file.csv"
    if test_csv.exists():
        return train_path, test_csv
    if test_file.exists():
        return train_path, test_file
    raise FileNotFoundError(f"Missing test file at {test_csv} (or legacy {test_file})")


def load_datasets(data_dir: Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, DataSchema]:
    data_dir = data_dir or config.DATA_DIR
    train_path, test_path = _resolve_data_paths(data_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    id_col = _detect_column(train_df, config.CANDIDATE_ID_COLS)
    text_col = _detect_column(train_df, config.CANDIDATE_TEXT_COLS)
    label_col = _detect_column(train_df, config.CANDIDATE_LABEL_COLS)
    social_col = _detect_column(train_df, getattr(config, "CANDIDATE_SOCIAL_COLS", []))

    if id_col is None or text_col is None or label_col is None:
        raise SchemaError(
            f"Could not detect columns. Found: id={id_col}, text={text_col}, label={label_col}. "
            f"Expected id in {config.CANDIDATE_ID_COLS}, text in {config.CANDIDATE_TEXT_COLS}, label in {config.CANDIDATE_LABEL_COLS}"
        )

    test_id_col = _detect_column(test_df, config.CANDIDATE_ID_COLS)
    test_text_col = _detect_column(test_df, config.CANDIDATE_TEXT_COLS)
    test_social_col = _detect_column(test_df, getattr(config, "CANDIDATE_SOCIAL_COLS", []))
    if test_id_col is None or test_text_col is None:
        raise SchemaError("Test file missing required id/text columns.")

    # Clean basic types
    train_df[text_col] = train_df[text_col].fillna("").astype(str)
    train_df[label_col] = train_df[label_col].astype(int)

    test_df[test_text_col] = test_df[test_text_col].fillna("").astype(str)
    if social_col is not None and test_social_col is not None:
        train_df[social_col] = train_df[social_col].fillna("").astype(str).str.strip()
        test_df[test_social_col] = test_df[test_social_col].fillna("").astype(str).str.strip()

    # Normalize text
    train_df[text_col] = train_df[text_col].apply(normalize)
    test_df[test_text_col] = test_df[test_text_col].apply(normalize)

    # Normalize column names to a consistent schema for downstream code
    # (we keep original columns too; schema indicates canonical ones)
    if test_text_col != text_col:
        test_df = test_df.rename(columns={test_text_col: text_col})
    if test_id_col != id_col:
        test_df = test_df.rename(columns={test_id_col: id_col})
    if social_col is not None and test_social_col is not None and test_social_col != social_col:
        test_df = test_df.rename(columns={test_social_col: social_col})

    # Validate label range
    invalid_labels = set(train_df[label_col]) - set(config.LABELS)
    if invalid_labels:
        raise SchemaError(f"Labels out of expected range 1-9: {sorted(invalid_labels)}")

    schema = DataSchema(id_col=id_col, text_col=text_col, label_col=label_col, social_col=social_col)
    return train_df, test_df, schema


def add_normalized_hash(df: pd.DataFrame, text_col: str, hash_col: str = "text_hash") -> pd.DataFrame:
    df = df.copy()
    df[hash_col] = df[text_col].astype(str).apply(stable_hash)
    return df


def add_group_columns(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Add duplicate/near-duplicate grouping columns.

    - exact_group: stable_hash(normalize(text))
    - strong_group: stable_hash(normalize_strong(text))

    Assumes df[text_col] is already normalized via normalize().
    """
    df = df.copy()
    df["exact_group"] = df[text_col].astype(str).apply(stable_hash)
    df["strong_group"] = df[text_col].astype(str).apply(lambda x: stable_hash(normalize_strong(x)))
    return df


def make_groups(df: pd.DataFrame, *, mode: str = "exact") -> np.ndarray:
    mode = (mode or "exact").strip().lower()
    if mode not in {"exact", "strong"}:
        raise ValueError("mode must be 'exact' or 'strong'")
    col = "strong_group" if mode == "strong" else "exact_group"
    if col not in df.columns:
        raise ValueError(f"Missing group column {col}. Call add_group_columns() first.")
    return df[col].astype(str).to_numpy()


def _script_bucket(text: str) -> str:
    # Very lightweight heuristic: arabic-dominant vs latin-dominant vs mixed
    if not text:
        return "mixed"
    arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    latin = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    letters = arabic + latin
    if letters == 0:
        return "mixed"
    ar_ratio = arabic / letters
    la_ratio = latin / letters
    if ar_ratio >= 0.60:
        return "arabic"
    if la_ratio >= 0.60:
        return "latin"
    return "mixed"


def _length_bucket(lengths: np.ndarray) -> np.ndarray:
    # Buckets by quantiles (short/med/long)
    q1 = float(np.quantile(lengths, 0.33))
    q2 = float(np.quantile(lengths, 0.66))
    out = np.zeros_like(lengths, dtype=int)
    out[lengths > q1] = 1
    out[lengths > q2] = 2
    return out


def _enforce_strong_leakage_check(strong_groups: np.ndarray, tr_idx: np.ndarray, va_idx: np.ndarray) -> None:
    tr = set(strong_groups[tr_idx])
    va = set(strong_groups[va_idx])
    inter = tr.intersection(va)
    if inter:
        raise RuntimeError(
            "Leakage detected: near-duplicate normalize_strong(text) appears in both train/valid in a fold. "
            "Re-run with leakage-safe grouping mode strong (make_groups(mode='strong'))."
        )


def build_cv_splits(
    df: pd.DataFrame,
    schema: DataSchema,
    *,
    n_splits: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Backward-compatible leakage-safe CV splits.

    This function is kept for older modules and defaults to the most
    conservative option: leakage-safe strategy with exact duplicate groups.
    It also enforces the near-duplicate (normalize_strong) leakage check.
    """
    splits, groups = build_cv_splits_v2(
        df,
        schema,
        n_splits=n_splits,
        seed=seed,
        split_strategy="leakage_safe",
        group_mode="exact",
        return_stats=False,
    )
    return splits, groups


def build_cv_splits_v2(
    df: pd.DataFrame,
    schema: DataSchema,
    *,
    n_splits: int,
    seed: int,
    split_strategy: str = "leakage_safe",
    group_mode: str = "exact",
    return_stats: bool = False,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, dict[str, Any]] | tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    df = add_group_columns(df, schema.text_col)
    y = df[schema.label_col].astype(int).to_numpy()
    exact_groups = df["exact_group"].astype(str).to_numpy()
    strong_groups = df["strong_group"].astype(str).to_numpy()
    groups = make_groups(df, mode=group_mode)

    split_strategy = (split_strategy or "leakage_safe").strip().lower()
    if split_strategy not in {"leakage_safe", "production_like"}:
        raise ValueError("split_strategy must be one of: leakage_safe, production_like")

    fold_stats: list[dict[str, Any]] = []
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    if split_strategy == "leakage_safe":
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(df, y, groups=groups)):
            _enforce_strong_leakage_check(strong_groups, tr_idx, va_idx)
            splits.append((tr_idx, va_idx))
            tr_exact = set(exact_groups[tr_idx])
            va_exact = set(exact_groups[va_idx])
            tr_strong = set(strong_groups[tr_idx])
            va_strong = set(strong_groups[va_idx])
            fold_stats.append(
                {
                    "fold": fold,
                    "n_train": int(len(tr_idx)),
                    "n_valid": int(len(va_idx)),
                    "n_exact_groups_train": int(len(tr_exact)),
                    "n_exact_groups_valid": int(len(va_exact)),
                    "exact_duplicate_count_train": int(len(tr_idx) - len(tr_exact)),
                    "exact_duplicate_count_valid": int(len(va_idx) - len(va_exact)),
                    "n_strong_groups_train": int(len(tr_strong)),
                    "n_strong_groups_valid": int(len(va_strong)),
                    "strong_duplicate_count_train": int(len(tr_idx) - len(tr_strong)),
                    "strong_duplicate_count_valid": int(len(va_idx) - len(va_strong)),
                }
            )
    else:
        # V2: production-like.
        # We MUST still enforce strong near-duplicate isolation.
        # To avoid leakage, we split at the strong-group level and then expand to rows.
        group_col = _detect_column(df, PRODUCTION_GROUP_CANDIDATES)

        # Build mapping: strong_group -> list[row_idx]
        group_to_rows: dict[str, list[int]] = {}
        for idx, g in enumerate(strong_groups.tolist()):
            group_to_rows.setdefault(g, []).append(int(idx))

        unique_groups = np.asarray(list(group_to_rows.keys()), dtype=object)

        def expand(group_idx: np.ndarray) -> np.ndarray:
            rows: list[int] = []
            for gi in group_idx.tolist():
                rows.extend(group_to_rows[str(unique_groups[int(gi)])])
            return np.asarray(rows, dtype=int)

        # Derive per-group metadata / strat keys
        group_label = np.zeros(len(unique_groups), dtype=int)
        group_meta = None
        meta_conflicts = 0
        if group_col is not None:
            meta_vals = df[group_col].fillna("").astype(str).str.strip().to_numpy()
            group_meta = np.empty(len(unique_groups), dtype=object)
            for i, g in enumerate(unique_groups.tolist()):
                rows = group_to_rows[str(g)]
                y_rows = y[np.asarray(rows, dtype=int)]
                # Representative label: majority within group (usually 1)
                vals, cnts = np.unique(y_rows, return_counts=True)
                group_label[i] = int(vals[int(np.argmax(cnts))])

                m = meta_vals[np.asarray(rows, dtype=int)]
                uniq = set(m.tolist())
                if len(uniq) > 1:
                    meta_conflicts += 1
                # Representative meta: most common
                mv, mc = np.unique(m, return_counts=True)
                group_meta[i] = mv[int(np.argmax(mc))]
        else:
            for i, g in enumerate(unique_groups.tolist()):
                rows = group_to_rows[str(g)]
                y_rows = y[np.asarray(rows, dtype=int)]
                vals, cnts = np.unique(y_rows, return_counts=True)
                group_label[i] = int(vals[int(np.argmax(cnts))])

        if group_col is not None and meta_conflicts == 0:
            # Split by metadata group at strong-group granularity
            gmeta = np.asarray(group_meta, dtype=object)
            unique_meta = len(set(gmeta.tolist()))
            if unique_meta >= n_splits:
                splitter = GroupKFold(n_splits=n_splits)
                iterator = splitter.split(np.zeros(len(unique_groups)), group_label, groups=gmeta)
            else:
                splitter = GroupShuffleSplit(
                    n_splits=n_splits,
                    test_size=min(0.2, 1.0 / max(n_splits, 2)),
                    random_state=seed,
                )
                iterator = splitter.split(np.zeros(len(unique_groups)), group_label, groups=gmeta)

            for fold, (tr_gi, va_gi) in enumerate(iterator):
                tr_idx = expand(tr_gi)
                va_idx = expand(va_gi)
                _enforce_strong_leakage_check(strong_groups, tr_idx, va_idx)
                splits.append((tr_idx, va_idx))
                tr_strong = set(strong_groups[tr_idx])
                va_strong = set(strong_groups[va_idx])
                fold_stats.append(
                    {
                        "fold": fold,
                        "n_train": int(len(tr_idx)),
                        "n_valid": int(len(va_idx)),
                        "group_col": group_col,
                        "meta_conflict_groups": int(meta_conflicts),
                        "n_meta_groups": int(unique_meta),
                        "n_strong_groups_train": int(len(tr_strong)),
                        "n_strong_groups_valid": int(len(va_strong)),
                        "strong_duplicate_count_train": int(len(tr_idx) - len(tr_strong)),
                        "strong_duplicate_count_valid": int(len(va_idx) - len(va_strong)),
                    }
                )
        else:
            # Pseudo-production stratification (label + length bucket + script bucket) at strong-group level.
            text = df[schema.text_col].astype(str).to_numpy()
            lengths = np.asarray([len(t) for t in text], dtype=float)
            len_b = _length_bucket(lengths)
            scr_b = np.asarray([_script_bucket(t) for t in text], dtype=object)
            scr_map = {"arabic": 0, "latin": 1, "mixed": 2}
            scr_i = np.asarray([scr_map.get(s, 2) for s in scr_b], dtype=int)

            group_len = np.zeros(len(unique_groups), dtype=int)
            group_scr = np.zeros(len(unique_groups), dtype=int)
            for i, g in enumerate(unique_groups.tolist()):
                rows = np.asarray(group_to_rows[str(g)], dtype=int)
                # Representative buckets: majority within group
                lv, lc = np.unique(len_b[rows], return_counts=True)
                sv, sc = np.unique(scr_i[rows], return_counts=True)
                group_len[i] = int(lv[int(np.argmax(lc))])
                group_scr[i] = int(sv[int(np.argmax(sc))])

            strat_key = np.asarray(
                [f"{int(lbl)}_{int(lb)}_{int(sb)}" for lbl, lb, sb in zip(group_label, group_len, group_scr)],
                dtype=object,
            )

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold, (tr_gi, va_gi) in enumerate(skf.split(np.zeros(len(unique_groups)), strat_key)):
                tr_idx = expand(tr_gi)
                va_idx = expand(va_gi)
                _enforce_strong_leakage_check(strong_groups, tr_idx, va_idx)
                splits.append((tr_idx, va_idx))
                tr_strong = set(strong_groups[tr_idx])
                va_strong = set(strong_groups[va_idx])
                fold_stats.append(
                    {
                        "fold": fold,
                        "n_train": int(len(tr_idx)),
                        "n_valid": int(len(va_idx)),
                        "pseudo_production": True,
                        "group_col": group_col,
                        "meta_conflict_groups": int(meta_conflicts),
                        "n_strat_keys": int(len(set(strat_key.tolist()))),
                        "n_strong_groups_train": int(len(tr_strong)),
                        "n_strong_groups_valid": int(len(va_strong)),
                        "strong_duplicate_count_train": int(len(tr_idx) - len(tr_strong)),
                        "strong_duplicate_count_valid": int(len(va_idx) - len(va_strong)),
                    }
                )

    stats = {
        "split_strategy": split_strategy,
        "group_mode": group_mode,
        "n_splits": int(n_splits),
        "seed": int(seed),
        "fold_stats": fold_stats,
        "n_rows": int(len(df)),
        "n_exact_groups": int(len(set(exact_groups))),
        "n_strong_groups": int(len(set(strong_groups))),
        "exact_duplicate_count": int(len(df) - len(set(exact_groups))),
        "strong_duplicate_count": int(len(df) - len(set(strong_groups))),
    }

    if return_stats:
        return splits, groups, stats
    return splits, groups


def compute_class_weights(labels: pd.Series) -> dict[int, float]:
    classes = np.array(config.LABELS)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels.values)
    return {int(c): float(w) for c, w in zip(classes, weights)}
