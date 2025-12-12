from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Global seed for reproducibility
SEED: int = 42

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_CANDIDATES = [
    PROJECT_ROOT / "data" / "forsa-client-satisfaction",  # requested
    PROJECT_ROOT / "data" / "forsa-clients-satisfaction",  # legacy / existing workspace
]


def resolve_data_dir() -> Path:
    for p in DATA_DIR_CANDIDATES:
        if p.exists():
            return p
    # Default to requested path (even if missing) for clearer errors.
    return DATA_DIR_CANDIDATES[0]


DATA_DIR = resolve_data_dir()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TFIDF_DIR = ARTIFACTS_DIR / "tfidf"
TRANSFORMER_DIR = ARTIFACTS_DIR / "transformer"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_SUBMISSION = OUTPUT_DIR / "submission.csv"
SELECTION_FILE = ARTIFACTS_DIR / "selected_model.txt"

# New, competition-grade outputs (requested)
OOF_DIR = OUTPUT_DIR / "oof"
TESTPROBS_DIR = OUTPUT_DIR / "testprobs"
MODELS_DIR = OUTPUT_DIR / "models"
DAPT_DIR = OUTPUT_DIR / "dapt"
BLEND_WEIGHTS_PATH = OUTPUT_DIR / "blend_weights.json"
FINAL_CONFIG_PATH = OUTPUT_DIR / "final_config.json"

# Columns to try to auto-detect
CANDIDATE_ID_COLS = ["id", "ID"]
CANDIDATE_TEXT_COLS = [
    "comment",
    "text",
    "message",
    "content",
    "review",
    "body",
    "Commentaire client",  # dataset-specific french column
    "Commentaire_client",
]
CANDIDATE_LABEL_COLS = ["Class", "label", "target", "labels"]
if "class" not in [c.lower() for c in CANDIDATE_LABEL_COLS]:
    CANDIDATE_LABEL_COLS.append("class")

# Optional auxiliary feature columns (categorical)
CANDIDATE_SOCIAL_COLS = [
    "Réseau Social",
    "Reseau Social",
    "reseau social",
    "social",
    "source",
]

LABELS = list(range(1, 10))  # 1..9
NUM_CLASSES = len(LABELS)

# Transformers internal ids: 0..8; submission labels: 1..9
LABEL2ID = {str(lbl): i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: str(lbl) for i, lbl in enumerate(LABELS)}

# Default transformer IDs
DEFAULT_XLMR = "xlm-roberta-base"
DEFAULT_DZIRIBERT_PLACEHOLDER = "CHANGE_ME_DZIRIBERT_MODEL_ID"

# TF-IDF settings
@dataclass
class TfidfConfig:
    min_df: int = 2
    max_features_word: int = 120000
    max_features_char: int = 250000
    word_ngrams: tuple[int, int] = (1, 3)
    char_ngrams: tuple[int, int] = (3, 7)
    use_idf: bool = True
    smooth_idf: bool = True
    sublinear_tf: bool = True
    normalize: str = "l2"
    solver: str = "lbfgs"
    C: float = 6.0
    max_iter: int = 2000
    n_jobs: int = max(os.cpu_count() - 1, 1)
    cv_folds: int = 5
    test_size: float = 0.15


# Transformer settings
@dataclass
class TransformerConfig:
    model_name: str = "xlm-roberta-base"
    max_length: int = 256
    train_batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 5
    warmup_ratio: float = 0.06
    logging_steps: int = 50
    eval_steps: int = 100
    save_total_limit: int = 1
    gradient_accumulation_steps: int = 2
    early_stopping_patience: int = 2
    label_smoothing: float = 0.0
    fp16: bool = True  # will be auto-disabled if CUDA is unavailable


tfidf_config = TfidfConfig()
transformer_config = TransformerConfig()

# Ensure directories exist
for path in [
    ARTIFACTS_DIR,
    TFIDF_DIR,
    TRANSFORMER_DIR,
    EXPERIMENTS_DIR,
    OUTPUT_DIR,
    OOF_DIR,
    TESTPROBS_DIR,
    MODELS_DIR,
    DAPT_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
