# Multilingual Customer Comments Classification

End-to-end pipeline for the Kaggle multilingual Algerian customer comments challenge (9 classes). Handles noisy mixed-language text (Darija/Arabizi/Arabic/French/English) with both TF-IDF baselines and a transformer (XLM-R).

## Repo layout

- `data/forsa-clients-satisfaction/` — expected CSVs (`train.csv`, `test_file.csv`).
- `src/` — code (config, preprocessing, training, inference, analysis).
- `artifacts/` — saved models and vectorizers (created after training).
- `outputs/` — submission.csv written here.
- `scripts/` — helper script for Colab runs.

## Quickstart

### Colab (GPU)

```bash
# 1) Mount Drive and clone (adjust folder if you prefer a different path)
from google.colab import drive
drive.mount('/content/drive')
%cd /content
!git clone https://github.com/DerdourBadreddine/forssa_clients.git
%cd /content/forssa_clients

# 2) Place data
# Ensure data/forsa-clients-satisfaction/train.csv and test_file.csv exist.
# If your text column is "Commentaire client", the code auto-detects it.

# 3) Install deps
!pip install -r requirements.txt

# 4) Train TF-IDF (fast)
!python -m src.train_tfidf

# 5) Train transformer (GPU, fp16 auto-enabled)
!python -m src.train_transformer --report_to none

# 6) Select best and infer
!python -m src.select_best
!python -m src.infer

# Submission written to outputs/submission.csv
```

### Local

```bash
pip install -r requirements.txt
python -m src.train_tfidf
python -m src.train_transformer
python -m src.select_best
python -m src.infer
```

Error analysis:

```bash
python -m src.error_analysis
```

## Key design points

- Deterministic seeding via `config.SEED`.
- Robust column auto-detection for `id`, `text`, `label` with clear errors if missing.
- Text normalization: URL/user placeholders, hashtag preservation, whitespace/spacing cleanup, elongation reduction, light Arabic normalization, emoji preserved as `<EMOJI>`.
- Leakage-resistant split: stratified; if duplicate normalized texts exist, switches to group-based split so identical texts do not cross train/val.
- TF-IDF track: char_wb 3–6 and word 1–2 n-grams; Logistic Regression + Linear SVC with class weights, 5-fold CV, best saved.
- Transformer track: XLM-R fine-tuning with class-weighted loss, early stopping on macro F1; saves model + tokenizer + metadata.

## Outputs & artifacts

- TF-IDF artifacts: `artifacts/tfidf/best_model.joblib` (+ meta JSONs).
- Transformer artifacts: `artifacts/transformer/best/` (HF format + meta.json).
- Selected best flag: `artifacts/selected_model.txt` ("tfidf" or "transformer").
- Submission: `outputs/submission.csv`.

## Tips

- If you modify hyperparameters, update `src/config.py`.
- To skip retraining when re-running selection: `python -m src.select_best --skip-train`.
- Ensure labels remain in [1..9]; the transformer internally maps to 0..8 but maps back on inference.
