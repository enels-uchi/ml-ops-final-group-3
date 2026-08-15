# Diabetes 30-Day Readmission Prediction — MLOps Final Project

Predicting hospital readmission risk for diabetic patients using the UCI/Kaggle "Diabetes 130-US Hospitals for Years 1999-2008" dataset.

## Project Structure

```
ml-ops-final-group-3/
├── data/
│   ├── raw/              # diabetic_data.csv (not in git — see Setup)
│   └── processed/        # cleaned + feature-engineered CSVs (DVC-tracked)
├── feast_repo/            # Feast feature store repo
├── metrics/                # model evaluation metrics (JSON)
├── models/                 # saved model artifacts (pickle)
├── notebooks/               # main analysis notebook
├── src/                    # pipeline scripts (see Pipeline Orchestration)
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── feast_registry.py
│   ├── train_baseline.py
│   ├── train_automl.py        # runs under venv-automl, invoked via subprocess
│   ├── progress.py            # console heartbeat for long-running steps
│   └── pipeline_flow.py       # Prefect flow wiring the above together
├── requirements.txt
└── README.md
```

## Setup

**Python version:** developed and tested on Python 3.14 with pinned package versions below. Should also work on Python 3.9–3.12.

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=ml-ops-final-projet
```

**Get the data:** download `diabetic_data.csv` from the [UCI/Kaggle dataset page](https://www.kaggle.com/datasets/brandao/diabetes) and place it at `data/raw/diabetic_data.csv` (this file is excluded from git via `.gitignore` due to size).

## Reproducing the Pipeline

Open `notebooks/final-group3-en.ipynb`, select the `ml-ops-final-projet` kernel, and run cells top to bottom. The notebook covers, in order:

1. **Setup** — imports, Git/DVC/Feast initialization checks
2. **Data ingestion & raw inspection** — load `diabetic_data.csv`, structure/missingness review
3. **EDA** — distributions, target variable analysis, correlation review
4. **Data cleaning** — see in-notebook markdown for full documented decisions (dropped columns, recoded missing-value placeholders, removed hospice/expired encounters, deduplicated to one encounter per patient)
5. **Feature engineering** — binary readmission target, ICD-9 diagnosis grouping, medical specialty bucketing, binary race feature, ordinal age encoding
6. **Feast feature store** — features registered and retrievable via `feast_repo/feature_repo/`
7. **Train/test split & baseline model** — 80/20 stratified split, XGBoost baseline classifier
8. **Model evaluation** — F1, AUC-ROC, confusion matrix, feature importance

## Data Version Control (DVC)

Processed data files (`data/processed/*.csv`) are tracked via DVC — `.dvc` pointer files are committed to git, but no shared remote is configured yet. To regenerate the processed data, re-run the notebook's cleaning and feature engineering sections rather than `dvc pull`.

## Feature Store (Feast)

Feast is used to register and retrieve engineered features (`feast_repo/feature_repo/feature_definitions.py` defines the `diabetes_features_v1` feature view, keyed by `encounter_id`). Run `feast apply` (already scripted in the notebook) after any change to the feature schema.

## Pipeline Orchestration (Prefect)

The full pipeline — data cleaning, feature engineering, Feast registration, baseline training, and AutoML — is orchestrated end to end via a Prefect flow (`src/pipeline_flow.py`), rather than requiring someone to manually run notebook cells in order.

**Run it:**
```bash
source venv/bin/activate
python src/pipeline_flow.py
```

**What it does**, as a DAG of tasks:
1. `clean_data` — raw CSV → `data/processed/diabetic_data_clean.csv`
2. `engineer_features` → `data/processed/diabetic_data_features.csv`
3. `register_feast_features` — writes the feature view, materializes the parquet, runs `feast apply`
4. `train_baseline` — trains the XGBoost baseline, saves `models/model_baseline.pkl` + `metrics/metrics_baseline.json`
5. `train_automl` — runs the PyCaret AutoML search and registers the winning model to MLflow

Steps 1–4 run directly in the main `venv`. Step 5 is invoked as a **subprocess using `venv-automl`'s Python interpreter**, since PyCaret doesn't support the main environment's Python 3.13 — the orchestrator itself doesn't need PyCaret installed, it just shells out to the other environment for that one step. Long-running steps (the AutoML search and hyperparameter tuning) print a periodic elapsed-time heartbeat (`src/progress.py`) so the pipeline doesn't appear to hang during multi-minute stretches.

**Retries:** the first four tasks retry once automatically on failure; the AutoML step does not auto-retry (a failed multi-minute run should be investigated, not silently rerun).

**Note:** every run registers a new MLflow model version and re-points the `champion` alias at it — intentional, since each pipeline execution is a new trained candidate, not an overwrite.

## Current Status

- [x] Environment setup, data ingestion, EDA
- [x] Data cleaning (documented decisions in notebook)
- [x] Feature engineering (target, diagnosis grouping, specialty bucketing, race/age encoding)
- [x] Feast feature store setup and retrieval
- [x] Train/test split, XGBoost baseline model
- [x] Pipeline orchestration (Prefect)
- [ ] Experiment tracking (MLflow)
- [ ] Model containerization & deployment (Docker + FastAPI/Flask/BentoML)
- [ ] Production monitoring & drift simulation (EvidentlyAI)

## Environment Notes

Package versions are pinned in `requirements.txt` to avoid known compatibility issues between `pandas`, `pyarrow`, and `pathspec` on newer Python releases. If reinstalling, use `pip install -r requirements.txt` as-is rather than letting pip resolve to the latest versions.