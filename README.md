# Multilingual Customer Comments Classification

End-to-end pipeline for the Kaggle multilingual Algerian customer comments challenge (9 classes). Handles noisy mixed-language text (Darija/Arabizi/Arabic/French/English) with both TF-IDF baselines and a transformer (XLM-R).

## Repo layout
- `data/forsa-clients-satisfaction/` — expected CSVs (`train.csv`, `test_file.csv`).
- `src/` — code (config, preprocessing, training, inference, analysis).
- `artifacts/` — saved models and vectorizers (created after training).
- `outputs/` — submission.csv written here.
- `scripts/` — helper script for Colab runs.

## Quickstart (local)
1) Install deps (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
2) Train TF-IDF baselines:
   ```bash
   python -m src.train_tfidf
   ```
3) Train transformer (GPU optional; will fall back to CPU):
   ```bash
   python -m src.train_transformer
   ```
4) Select best model (by validation macro F1) and record choice:
   ```bash
   python -m src.select_best
   ```
5) Produce submission.csv:
   ```bash
   python -m src.infer
   ```
   Output written to `outputs/submission.csv` with columns `id,Class`.
6) Run error analysis on the current best model:
   ```bash
   python -m src.error_analysis
   ```

## Colab notes
- Upload this repo or clone it, place data under `data/forsa-clients-satisfaction/`.
- Run `pip install -r requirements.txt`.
- Execute steps 2–5. GPU acceleration is automatic if available (transformer uses PyTorch/Accelerate).

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
