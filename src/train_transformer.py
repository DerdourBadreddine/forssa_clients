from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

from . import config
from .data_io import compute_class_weights, load_datasets, stratified_leakage_safe_split
from .metrics import macro_f1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def label_mappings() -> tuple[Dict[str, int], Dict[int, str]]:
    label2id = {str(lbl): i for i, lbl in enumerate(config.LABELS)}
    id2label = {i: str(lbl) for i, lbl in enumerate(config.LABELS)}
    return label2id, id2label


def tokenize_function(examples, tokenizer, text_col: str, max_length: int):
    return tokenizer(
        examples[text_col],
        truncation=True,
        padding=False,
        max_length=max_length,
    )


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def prepare_datasets(train_df, val_df, schema, tokenizer):
    label2id, id2label = label_mappings()
    train_ds = Dataset.from_pandas(train_df[[schema.text_col, schema.label_col]])
    val_ds = Dataset.from_pandas(val_df[[schema.text_col, schema.label_col]])

    def encode(batch):
        tokens = tokenize_function(batch, tokenizer, schema.text_col, config.transformer_config.max_length)
        labels = [label2id[str(int(l))] for l in batch[schema.label_col]]
        tokens["labels"] = labels
        return tokens

    train_ds = train_ds.map(encode, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(encode, batched=True, remove_columns=val_ds.column_names)
    return train_ds, val_ds, label2id, id2label


def build_model(label2id, id2label):
    model = AutoModelForSequenceClassification.from_pretrained(
        config.transformer_config.model_name,
        num_labels=config.NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune XLM-R on customer comments")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.TRANSFORMER_DIR)
    args = parser.parse_args()

    set_seed(config.SEED)
    train_df, _, schema = load_datasets(args.data_dir)
    train_df, val_df = stratified_leakage_safe_split(train_df, schema, test_size=0.15, seed=config.SEED)

    tokenizer = AutoTokenizer.from_pretrained(config.transformer_config.model_name)

    train_ds, val_ds, label2id, id2label = prepare_datasets(train_df, val_df, schema, tokenizer)

    class_weights = compute_class_weights(train_df[schema.label_col])
    weight_tensor = torch.tensor([class_weights[int(lbl)] for lbl in config.LABELS], dtype=torch.float)

    model = build_model(label2id, id2label)

    training_args = TrainingArguments(
        output_dir=args.output_dir / "checkpoints",
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=config.transformer_config.learning_rate,
        per_device_train_batch_size=config.transformer_config.train_batch_size,
        per_device_eval_batch_size=config.transformer_config.eval_batch_size,
        num_train_epochs=config.transformer_config.num_train_epochs,
        weight_decay=config.transformer_config.weight_decay,
        logging_steps=config.transformer_config.logging_steps,
        eval_steps=config.transformer_config.eval_steps,
        save_total_limit=config.transformer_config.save_total_limit,
        seed=config.SEED,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        gradient_accumulation_steps=config.transformer_config.gradient_accumulation_steps,
        warmup_ratio=config.transformer_config.warmup_ratio,
        fp16=config.transformer_config.fp16,
        dataloader_num_workers=2,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        labels_map = np.array(config.LABELS)
        y_true = labels_map[labels]
        y_pred = labels_map[preds]
        return {"macro_f1": macro_f1(y_true, y_pred)}

    trainer = WeightedTrainer(
        class_weights=weight_tensor,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.transformer_config.early_stopping_patience)],
    )

    trainer.train()

    # Evaluate on validation set with label mapping back to 1..9
    eval_outputs = trainer.predict(val_ds)
    preds = np.argmax(eval_outputs.predictions, axis=-1)
    labels_map = np.array(config.LABELS)
    y_true = labels_map[eval_outputs.label_ids]
    y_pred = labels_map[preds]
    val_macro_f1 = macro_f1(y_true, y_pred)

    print(f"Validation Macro F1: {val_macro_f1:.4f}")

    # Save model, tokenizer, and metadata
    best_dir = args.output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)

    meta = {
        "val_macro_f1": float(val_macro_f1),
        "label2id": label2id,
        "id2label": id2label,
        "schema": schema.__dict__,
        "config": asdict(config.transformer_config),
    }
    with open(best_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
