from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from . import config
from .text_norm import normalize


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


def stable_hash(text: str) -> str:
    """Stable hash for grouping duplicates/near-duplicates.

    We hash the *normalized* text to avoid leakage across folds.
    """
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


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


def build_cv_splits(
    df: pd.DataFrame,
    schema: DataSchema,
    *,
    n_splits: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Leakage-safe CV splits.

    - groups are stable_hash(normalized_text)
    - StratifiedGroupKFold
    - Explicit check: no identical normalized text crosses train/valid
    """
    df = add_normalized_hash(df, schema.text_col)
    y = df[schema.label_col].astype(int).to_numpy()
    groups = df["text_hash"].to_numpy()

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    text = df[schema.text_col].astype(str).to_numpy()

    for tr_idx, va_idx in skf.split(df, y, groups=groups):
        # Explicit leakage check (should always pass with group split)
        tr_set = set(text[tr_idx])
        va_set = set(text[va_idx])
        if tr_set.intersection(va_set):
            raise RuntimeError("Leakage detected: identical normalized text appears in both train/valid in a fold")
        splits.append((tr_idx, va_idx))

    return splits, groups


def compute_class_weights(labels: pd.Series) -> dict[int, float]:
    classes = np.array(config.LABELS)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels.values)
    return {int(c): float(w) for c, w in zip(classes, weights)}
