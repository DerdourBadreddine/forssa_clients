from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
from scipy.sparse import hstack

from . import config
from .data_io import DataSchema, load_datasets


def load_selection() -> str:
    if config.SELECTION_FILE.exists():
        return config.SELECTION_FILE.read_text(encoding="utf-8").strip()
    # Default to tfidf if no selection
    return "tfidf"


def predict_tfidf(test_df: pd.DataFrame) -> np.ndarray:
    payload = joblib.load(config.TFIDF_DIR / "best_model.joblib")
    model = payload["model"]
    word_vec = payload["word_vectorizer"]
    char_vec = payload["char_vectorizer"]
    schema_dict = payload.get("schema")
    schema = DataSchema(**schema_dict)

    X_test_word = word_vec.transform(test_df[schema.text_col])
    X_test_char = char_vec.transform(test_df[schema.text_col])
    X_test = hstack([X_test_word, X_test_char])
    preds = model.predict(X_test)
    return preds


def predict_transformer(test_df: pd.DataFrame) -> np.ndarray:
    meta_path = config.TRANSFORMER_DIR / "best" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Transformer artifacts missing. Run train_transformer.py")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    schema = DataSchema(**meta["schema"])

    tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_DIR / "best")
    model = AutoModelForSequenceClassification.from_pretrained(config.TRANSFORMER_DIR / "best")

    def encode(batch):
        return tokenizer(batch[schema.text_col], truncation=True, padding=False, max_length=config.transformer_config.max_length)

    ds = Dataset.from_pandas(test_df[[schema.text_col]])
    ds = ds.map(encode, batched=True, remove_columns=ds.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=data_collator)
    preds_output = trainer.predict(ds)
    pred_ids = np.argmax(preds_output.predictions, axis=-1)
    labels_map = np.array(config.LABELS)
    return labels_map[pred_ids]


def main():
    _, test_df, schema = load_datasets()

    model_choice = load_selection()
    print(f"Using model: {model_choice}")

    if model_choice == "transformer":
        preds = predict_transformer(test_df)
    else:
        preds = predict_tfidf(test_df)

    submission = pd.DataFrame({
        schema.id_col: test_df[schema.id_col],
        "Class": preds.astype(int),
    })
    submission_path = config.OUTPUT_SUBMISSION
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")


if __name__ == "__main__":
    main()
