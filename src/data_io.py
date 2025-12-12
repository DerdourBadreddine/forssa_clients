from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from . import config
from .text_norm import normalize_text


@dataclass
class DataSchema:
    id_col: str
    text_col: str
    label_col: Optional[str] = None


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


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def load_datasets(data_dir: Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, DataSchema]:
    data_dir = data_dir or config.DATA_DIR
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test_file.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file at {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    id_col = _detect_column(train_df, config.CANDIDATE_ID_COLS)
    text_col = _detect_column(train_df, config.CANDIDATE_TEXT_COLS)
    label_col = _detect_column(train_df, config.CANDIDATE_LABEL_COLS)

    if id_col is None or text_col is None or label_col is None:
        raise SchemaError(
            f"Could not detect columns. Found: id={id_col}, text={text_col}, label={label_col}. "
            f"Expected id in {config.CANDIDATE_ID_COLS}, text in {config.CANDIDATE_TEXT_COLS}, label in {config.CANDIDATE_LABEL_COLS}"
        )

    # Validate test columns
    if _detect_column(test_df, config.CANDIDATE_ID_COLS) is None or _detect_column(test_df, config.CANDIDATE_TEXT_COLS) is None:
        raise SchemaError("Test file missing required id/text columns.")

    # Clean basic types
    train_df[id_col] = train_df[id_col].astype(str)
    train_df[text_col] = train_df[text_col].astype(str)
    train_df[label_col] = train_df[label_col].astype(int)

    test_id_col = _detect_column(test_df, config.CANDIDATE_ID_COLS)
    test_text_col = _detect_column(test_df, config.CANDIDATE_TEXT_COLS)
    test_df[test_id_col] = test_df[test_id_col].astype(str)
    test_df[test_text_col] = test_df[test_text_col].astype(str)

    # Normalize text
    train_df[text_col] = train_df[text_col].fillna("").apply(normalize_text)
    test_df[test_text_col] = test_df[test_text_col].fillna("").apply(normalize_text)

    # Validate label range
    invalid_labels = set(train_df[label_col]) - set(config.LABELS)
    if invalid_labels:
        raise SchemaError(f"Labels out of expected range 1-9: {sorted(invalid_labels)}")

    schema = DataSchema(id_col=id_col, text_col=text_col, label_col=label_col)
    return train_df, test_df, schema


def add_normalized_hash(df: pd.DataFrame, text_col: str, hash_col: str = "text_hash") -> pd.DataFrame:
    df = df.copy()
    df[hash_col] = df[text_col].apply(_hash_text)
    return df


def stratified_leakage_safe_split(
    df: pd.DataFrame,
    schema: DataSchema,
    test_size: float = 0.15,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df = add_normalized_hash(df, schema.text_col)
    labels = df[schema.label_col]

    # Detect duplicates by hash
    dup_counts = df.groupby("text_hash")[schema.id_col].transform("count")
    has_duplicates = (dup_counts > 1).any()

    if has_duplicates:
        splitter = StratifiedGroupKFold(n_splits=int(1 / test_size), shuffle=True, random_state=seed)
        groups = df["text_hash"]
        for train_idx, val_idx in splitter.split(df, labels, groups):
            train_df = df.iloc[train_idx].drop(columns=["text_hash"])
            val_df = df.iloc[val_idx].drop(columns=["text_hash"])
            return train_df, val_df
        raise RuntimeError("Group split failed.")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_idx, val_idx in splitter.split(df, labels):
        train_df = df.iloc[train_idx].drop(columns=["text_hash"])
        val_df = df.iloc[val_idx].drop(columns=["text_hash"])
        return train_df, val_df
    raise RuntimeError("Stratified split failed.")


def compute_class_weights(labels: pd.Series) -> dict[int, float]:
    classes = np.array(config.LABELS)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels.values)
    return {int(c): float(w) for c, w in zip(classes, weights)}
