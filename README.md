# FORSA 2025 – Clients Satisfaction (9-class Macro‑F1)

Competition-grade, reproducible pipeline for multilingual/noisy Algerian customer comments (Darija/Arabizi/Arabic/French/English) → labels 1..9.

This repo implements:

- Leakage-safe CV (duplicate/near-duplicate defense via normalized-text hashing + StratifiedGroupKFold)
- Strong TF‑IDF baselines (char_wb + word) with CV OOF/test probabilities
- Transformer CV fine-tuning (XLM‑R / DeBERTa / DziriBERT via user-provided model id)
- Optional DAPT (MLM) on train+test text
- OOF blending weight search optimizing Macro‑F1
- Bulletproof inference to `outputs/submission.csv`

## Data

Required paths:

- `data/forsa-client-satisfaction/train.csv`
- `data/forsa-client-satisfaction/test.csv`

Notes:

- The code also accepts legacy `test_file.csv` if present.
- Column auto-detection:
  - id: `id` or `ID`
  - text: `comment`, `text`, `message`, `content`, `Commentaire client`, `Commentaire_client`
  - label: `Class`, `label`, `target`, `class`

## Colab (GPU) – exact commands

```bash
from google.colab import drive
drive.mount('/content/drive')
%cd /content
!git clone <YOUR_REPO_URL>
%cd /content/<YOUR_REPO_FOLDER>

# Put train.csv + test.csv under:
# data/forsa-client-satisfaction/

!pip uninstall -y peft accelerate
# If you previously installed conflicting versions, restart once:
# Runtime -> Restart runtime

!pip install -r requirements.txt --no-cache-dir

# 1) TF‑IDF CV
!python -m src.drift_detector

# 2) TF‑IDF CV (auto validation selection)
!python -m src.train_tfidf_cv --val_strategy auto --folds 5 --seed 42

# 3) Transformer CV (XLM‑R) (auto validation selection)
!python -m src.train_transformer_cv --model_id xlm-roberta-base --val_strategy auto --folds 5 --seed 42

# 3) Transformer CV (DziriBERT) – YOU MUST set the real HF model id
!python -m src.train_transformer_cv --model_id CHANGE_ME_DZIRIBERT_MODEL_ID --folds 5 --seed 42

# 4) Optional DAPT (MLM) then re-train transformer with --use_dapt
!python -m src.dapt_mlm --model_id xlm-roberta-base
!python -m src.train_transformer_cv --model_id xlm-roberta-base --use_dapt --folds 5 --seed 42

# 5) Blend OOF + build blended test probs
!python -m src.blend_oof

# 6) Select best final approach (best single vs blend)
!python -m src.select_best

# 7) Inference (writes outputs/submission.csv)
!python -m src.infer --sanity-check

# Optional: error analysis on OOF
!python -m src.error_analysis --source blend
```

## Outputs (reproducibility)

- `outputs/oof/*.npy`: OOF probabilities per model (shape `(n_train, 9)`)
- `outputs/testprobs/*.npy`: test probabilities per model (shape `(n_test, 9)`)
- `outputs/models/<model_id>/fold_k/seed_s/`: transformer checkpoints
- `outputs/models/tfidf/fold_k/`: TF‑IDF vectorizers + fold models
- `outputs/dapt/<model_id>/adapted/`: DAPT MLM checkpoint (optional)
- `outputs/blend_weights.json`: best blending weights
- `outputs/final_config.json`: chosen final ensemble config
- `outputs/submission.csv`: final Kaggle submission (`id,Class`)

## Packaging for “model + code submission”

```bash
zip -r artifacts_bundle.zip outputs src requirements.txt README.md
```
