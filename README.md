# Favorita Store Sales Forecasting (Kaggle) — Time Series Pipeline

End-to-end forecasting pipeline based on Kaggle’s **Store Sales – Time Series Forecasting** (Corporación Favorita) dataset.
This repo is structured for a strong Master’s application portfolio: **clean code**, **time-based validation**, **feature engineering**, **model persistence**, and **reproducible outputs**.

---

## What this project does

- Loads Kaggle data (`train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `sample_submission.csv`)
- Builds a unified table via joins (stores/oil/holidays)
- Engineers forecasting features:
  - calendar/time features (day-of-week, month, week-of-year, etc.)
  - lag features per `(store_nbr, family)`
  - rolling mean/std features (leakage-safe using shift)
- Performs **rolling backtesting** (time-based folds)
- Trains a model (LightGBM preferred; sklearn fallback)
- Produces:
  - `outputs/submissions/submission.csv` (Kaggle format)
  - `outputs/models/*.joblib` (saved model)
  - `outputs/plots/*.png` (key EDA plots)
  - optional cached features in `data/processed/*.parquet`

---

## Folder structure

```text
Favorita_Store_Sales_Analysis/
  main.py
  requirements.txt
  readme.md
  .gitignore
  src/
    __init__.py
    paths.py
    data.py
    split.py
    features.py
    metrics.py
    model.py
    viz.py
    utils.py
  data/
    raw/          # Kaggle CSV files (NOT committed)
    processed/    # optional feature caches
  outputs/
    plots/
    models/
    submissions/
