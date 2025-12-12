from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from . import config
from .data_io import build_cv_splits, compute_class_weights, load_datasets
from .metrics import macro_f1, probs_to_labels
from .utils import (
    assert_probs_ok,
    ensure_dir,
    json_dump,
    safe_model_id,
    set_seed,
    softmax,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")


@dataclass
class TrainCfg:
    model_id: str
    folds: int
    seeds: List[int]
    max_len: int
    train_bs: int
    eval_bs: int
    lr: float
    wd: float
    epochs: float
    warmup_ratio: float
    grad_accum: int
    early_stop_patience: int
    label_smoothing: float
    use_dapt: bool
    loss: str  # "weighted_ce" or "focal"
    focal_gamma: float


class WeightedOrFocalTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights: Optional[torch.Tensor] = None,
        loss_kind: str = "weighted_ce",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_kind = loss_kind
        self.focal_gamma = focal_gamma
        self.label_smoothing = float(label_smoothing)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        if labels is None:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        import torch.nn.functional as F

        if self.loss_kind == "focal":
            # Focal loss on probabilities
            logp = F.log_softmax(logits, dim=-1)
            p = torch.exp(logp)
            # Gather p_t
            labels_flat = labels.view(-1, 1)
            p_t = p.gather(1, labels_flat).squeeze(1)
            logp_t = logp.gather(1, labels_flat).squeeze(1)
            w = 1.0
            if self.class_weights is not None:
                w = self.class_weights.to(logits.device).gather(0, labels)
            loss = -w * ((1 - p_t) ** self.focal_gamma) * logp_t
            loss = loss.mean()
        else:
            loss = F.cross_entropy(
                logits,
                labels,
                weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
                label_smoothing=self.label_smoothing,
            )

        return (loss, outputs) if return_outputs else loss


def _parse_seeds(s: str, default: int) -> List[int]:
    s = (s or "").strip()
    if not s:
        return [default]
    return [int(x) for x in s.split(",") if x.strip()]


def _maybe_dapt_checkpoint(model_id: str, use_dapt: bool) -> str | Path:
    if not use_dapt:
        return model_id
    dapt_path = config.DAPT_DIR / safe_model_id(model_id) / "adapted"
    if not dapt_path.exists():
        raise FileNotFoundError(
            f"--use_dapt was set but DAPT checkpoint not found at {dapt_path}. "
            f"Run: python -m src.dapt_mlm --model_id {model_id}"
        )
    return dapt_path


def run_transformer_cv(
    *,
    model_id: str,
    run_id: str | None,
    data_dir: Optional[Path],
    folds: int,
    seeds: List[int],
    max_len: int,
    train_bs: int,
    eval_bs: int,
    lr: float,
    wd: float,
    epochs: float,
    warmup_ratio: float,
    grad_accum: int,
    early_stop_patience: int,
    label_smoothing: float,
    use_dapt: bool,
    loss_kind: str,
    focal_gamma: float,
    output_root: Path,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    train_df, test_df, schema = load_datasets(data_dir)
    y_labels = train_df[schema.label_col].astype(int).to_numpy()
    y_internal = (y_labels - 1).astype(int)  # 0..8

    splits, _ = build_cv_splits(train_df, schema, n_splits=folds, seed=seeds[0])

    safe_id = safe_model_id(run_id if run_id else model_id)
    model_root = ensure_dir(output_root / safe_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.save_pretrained(model_root)

    n_train = len(train_df)
    n_test = len(test_df)
    n_classes = config.NUM_CLASSES

    oof_accum = np.zeros((n_train, n_classes), dtype=np.float32)
    test_accum = np.zeros((n_test, n_classes), dtype=np.float32)

    for seed in seeds:
        set_seed(seed)

        oof = np.zeros((n_train, n_classes), dtype=np.float32)
        test_probs = np.zeros((n_test, n_classes), dtype=np.float32)

        for fold, (tr_idx, va_idx) in enumerate(splits):
            tr_df = train_df.iloc[tr_idx]
            va_df = train_df.iloc[va_idx]

            # Class weights computed on fold-train
            cw = compute_class_weights(tr_df[schema.label_col])
            weights = torch.tensor([cw[i] for i in config.LABELS], dtype=torch.float)

            init_ckpt = _maybe_dapt_checkpoint(model_id, use_dapt)
            model = AutoModelForSequenceClassification.from_pretrained(
                init_ckpt,
                num_labels=n_classes,
                label2id={str(i): i - 1 for i in config.LABELS},
                id2label={i - 1: str(i) for i in config.LABELS},
            )

            def encode(batch):
                tok = tokenizer(
                    batch[schema.text_col],
                    truncation=True,
                    max_length=max_len,
                    padding=False,
                )
                tok["labels"] = (np.asarray(batch[schema.label_col], dtype=int) - 1).tolist()
                return tok

            ds_tr = Dataset.from_pandas(tr_df[[schema.text_col, schema.label_col]])
            ds_va = Dataset.from_pandas(va_df[[schema.text_col, schema.label_col]])
            ds_tr = ds_tr.map(encode, batched=True, remove_columns=ds_tr.column_names)
            ds_va = ds_va.map(encode, batched=True, remove_columns=ds_va.column_names)

            def encode_test(batch):
                return tokenizer(
                    batch[schema.text_col],
                    truncation=True,
                    max_length=max_len,
                    padding=False,
                )

            ds_te = Dataset.from_pandas(test_df[[schema.text_col]])
            ds_te = ds_te.map(encode_test, batched=True, remove_columns=ds_te.column_names)

            use_fp16 = torch.cuda.is_available()

            fold_dir = ensure_dir(model_root / f"fold_{fold}" / f"seed_{seed}")
            args = TrainingArguments(
                output_dir=str(fold_dir / "checkpoints"),
                per_device_train_batch_size=train_bs,
                per_device_eval_batch_size=eval_bs,
                gradient_accumulation_steps=grad_accum,
                num_train_epochs=float(epochs),
                learning_rate=lr,
                warmup_ratio=warmup_ratio,
                weight_decay=wd,
                lr_scheduler_type="cosine",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                load_best_model_at_end=True,
                metric_for_best_model="macro_f1",
                greater_is_better=True,
                fp16=use_fp16,
                seed=seed,
                report_to=[],
                logging_steps=50,
                dataloader_num_workers=2,
            )

            collator = DataCollatorWithPadding(tokenizer=tokenizer)

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                probs = softmax(np.asarray(logits))
                preds_internal = np.argmax(probs, axis=1)
                y_true = (np.asarray(labels) + 1).astype(int)
                y_pred = (preds_internal + 1).astype(int)
                return {"macro_f1": macro_f1(y_true, y_pred)}

            trainer = WeightedOrFocalTrainer(
                model=model,
                args=args,
                train_dataset=ds_tr,
                eval_dataset=ds_va,
                tokenizer=tokenizer,
                data_collator=collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stop_patience)],
                class_weights=weights,
                loss_kind=loss_kind,
                focal_gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )

            trainer.train()

            # Save best fold model
            trainer.model.save_pretrained(fold_dir)

            # Val probabilities
            out_va = trainer.predict(ds_va)
            probs_va = softmax(np.asarray(out_va.predictions))
            assert_probs_ok(probs_va, n_classes)
            oof[va_idx] = probs_va

            # Test probabilities
            out_te = trainer.predict(ds_te)
            probs_te = softmax(np.asarray(out_te.predictions))
            assert_probs_ok(probs_te, n_classes)
            test_probs += probs_te / folds

        oof_accum += oof / len(seeds)
        test_accum += test_probs / len(seeds)

    meta = {
        "model_id": model_id,
        "safe_model_id": safe_id,
        "run_id": run_id,
        "schema": schema.__dict__,
        "folds": folds,
        "seeds": seeds,
        "use_dapt": bool(use_dapt),
        "max_len": max_len,
        "train_bs": train_bs,
        "eval_bs": eval_bs,
        "lr": lr,
        "wd": wd,
        "epochs": float(epochs),
        "warmup_ratio": warmup_ratio,
        "grad_accum": grad_accum,
        "early_stop_patience": early_stop_patience,
        "label_smoothing": float(label_smoothing),
        "loss": loss_kind,
        "focal_gamma": float(focal_gamma),
    }

    # OOF metric
    y_pred = probs_to_labels(oof_accum, config.LABELS)
    meta["oof_macro_f1"] = macro_f1(y_labels, y_pred)

    return oof_accum, test_accum, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer CV fine-tuning with StratifiedGroupKFold")
    parser.add_argument("--model_id", type=str, default=config.DEFAULT_XLMR)
    parser.add_argument("--run_id", type=str, default="", help="Optional name for outputs (does not affect HF loading)")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--train_bs", type=int, default=8)
    parser.add_argument("--eval_bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_dapt", action="store_true")
    parser.add_argument("--loss", choices=["weighted_ce", "focal"], default="weighted_ce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--output_root", type=Path, default=config.MODELS_DIR)
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds, args.seed)

    oof, test_probs, meta = run_transformer_cv(
        model_id=args.model_id,
        run_id=args.run_id.strip() or None,
        data_dir=args.data_dir,
        folds=args.folds,
        seeds=seeds,
        max_len=args.max_len,
        train_bs=args.train_bs,
        eval_bs=args.eval_bs,
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        grad_accum=args.grad_accum,
        early_stop_patience=args.early_stop_patience,
        label_smoothing=args.label_smoothing,
        use_dapt=args.use_dapt,
        loss_kind=args.loss,
        focal_gamma=args.focal_gamma,
        output_root=args.output_root,
    )

    safe_id = meta["safe_model_id"]
    np.save(config.OOF_DIR / f"{safe_id}.npy", oof)
    np.save(config.TESTPROBS_DIR / f"{safe_id}.npy", test_probs)
    json_dump(meta, args.output_root / safe_id / "meta.json")

    print("Saved:")
    print("-", config.OOF_DIR / f"{safe_id}.npy")
    print("-", config.TESTPROBS_DIR / f"{safe_id}.npy")
    print("-", args.output_root / safe_id / "meta.json")
    print("OOF macro-F1:", round(meta["oof_macro_f1"], 6))


if __name__ == "__main__":
    main()
