from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from . import config
from .data_io import load_datasets
from .utils import ensure_dir, json_dump, safe_model_id, set_seed

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")


def main() -> None:
    parser = argparse.ArgumentParser(description="Domain-adaptive pretraining (MLM) on train+test text")
    parser.add_argument("--model_id", type=str, default=config.DEFAULT_XLMR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=config.DAPT_DIR)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--mlm-prob", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    train_df, test_df, schema = load_datasets(args.data_dir)
    texts = train_df[schema.text_col].astype(str).tolist() + test_df[schema.text_col].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_id)

    ds = Dataset.from_dict({"text": texts})

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_len,
            padding=False,
            return_special_tokens_mask=True,
        )

    ds = ds.map(tok, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob)

    out_dir = ensure_dir(Path(args.output_dir) / safe_model_id(args.model_id))
    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=float(args.epochs),
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        fp16=use_fp16,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    # Save adapted checkpoint
    adapted_dir = ensure_dir(out_dir / "adapted")
    trainer.model.save_pretrained(adapted_dir)
    tokenizer.save_pretrained(adapted_dir)

    meta = {
        "model_id": args.model_id,
        "seed": args.seed,
        "max_len": args.max_len,
        "mlm_prob": args.mlm_prob,
        "epochs": float(args.epochs),
        "n_texts": len(texts),
        "output": str(adapted_dir),
    }
    json_dump(meta, out_dir / "dapt_meta.json")

    print("Saved adapted MLM checkpoint to:")
    print(adapted_dir)


if __name__ == "__main__":
    main()
