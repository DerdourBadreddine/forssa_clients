from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
from scipy.sparse import hstack

from . import config
from .data_io import DataSchema, load_datasets, stratified_leakage_safe_split
from .metrics import full_report


def load_selection() -> str:
    if config.SELECTION_FILE.exists():
        return config.SELECTION_FILE.read_text(encoding="utf-8").strip()
    return "tfidf"


def predict_with_tfidf(train_df: pd.DataFrame, val_df: pd.DataFrame, schema: DataSchema) -> np.ndarray:
    payload = joblib.load(config.TFIDF_DIR / "best_model.joblib")
    model = payload["model"]
    word_vec = payload["word_vectorizer"]
    char_vec = payload["char_vectorizer"]

    X_val_word = word_vec.transform(val_df[schema.text_col])
    X_val_char = char_vec.transform(val_df[schema.text_col])
    X_val = hstack([X_val_word, X_val_char])
    return model.predict(X_val)


def predict_with_transformer(val_df: pd.DataFrame, schema: DataSchema) -> np.ndarray:
    meta_path = config.TRANSFORMER_DIR / "best" / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_DIR / "best")
    model = AutoModelForSequenceClassification.from_pretrained(config.TRANSFORMER_DIR / "best")

    def encode(batch):
        return tokenizer(batch[schema.text_col], truncation=True, padding=False, max_length=config.transformer_config.max_length)

    ds = Dataset.from_pandas(val_df[[schema.text_col, schema.label_col]])
    ds = ds.map(lambda batch: {**encode(batch), "labels": batch[schema.label_col]}, batched=True, remove_columns=ds.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=data_collator)
    outputs = trainer.predict(ds)
    preds = np.argmax(outputs.predictions, axis=-1)
    labels_map = np.array(config.LABELS)
    return labels_map[preds]


def top_confusions(cm: np.ndarray, labels: List[int], top_k: int = 5) -> List[Tuple[int, int, int]]:
    pairs = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                pairs.append((true_label, pred_label, count))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def main():
    train_df, _, schema = load_datasets()
    train_df, val_df = stratified_leakage_safe_split(train_df, schema, seed=config.SEED)

    choice = load_selection()
    if choice == "transformer":
        preds = predict_with_transformer(val_df, schema)
    else:
        preds = predict_with_tfidf(train_df, val_df, schema)

    y_true = val_df[schema.label_col].astype(int).values
    report, cm = full_report(y_true, preds)
    print("Classification report:\n" + report)
    print("Confusion matrix:\n", cm)

    conf_pairs = top_confusions(cm, config.LABELS, top_k=10)
    print("\nTop confusion pairs (true -> pred: count):")
    for t, p, c in conf_pairs:
        print(f"{t} -> {p}: {c}")

    val_df = val_df.copy()
    val_df["pred"] = preds
    misclassified = val_df[val_df[schema.label_col] != val_df["pred"]]
    print("\nSample misclassifications:")
    for _, row in misclassified.head(20).iterrows():
        print(f"id={row[schema.id_col]} true={row[schema.label_col]} pred={row['pred']} text={row[schema.text_col][:200]}")


if __name__ == "__main__":
    main()
