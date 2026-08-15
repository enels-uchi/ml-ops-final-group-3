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
├── notebooks/
│   ├── final-group3-en.ipynb   # cleaning, Feast, DVC, XGBoost baseline
│   └── automl_pycaret.ipynb    # PyCaret AutoML + MLflow tracking (separate env, see Setup)
├── outputs/                # AutoML leaderboard CSVs (gitignored)
├── src/                    # (reserved for future pipeline scripts)
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

**AutoML environment (separate):** `notebooks/automl_pycaret.ipynb` uses PyCaret, which doesn't yet support Python 3.13. It runs in a second, dedicated environment on Python 3.11, kept fully separate from the main `venv`/`requirements.txt`:

```bash
python3.11 -m venv venv-automl
source venv-automl/bin/activate
pip install --upgrade pip
pip install pycaret mlflow seaborn xgboost catboost
python -m ipykernel install --user --name=ml-ops-automl --display-name "Python (ml-ops-automl)"
```

Use the `ml-ops-final-projet` kernel for `notebooks/final-group3-en.ipynb`, and `ml-ops-automl` for `notebooks/automl_pycaret.ipynb`.

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

## Experiment Tracking & AutoML (MLflow + PyCaret)

`notebooks/automl_pycaret.ipynb` runs an automated algorithm search against the same engineered features and target (`readmitted_binary`) as the baseline, using PyCaret's classification AutoML and local MLflow tracking.

**What it does:**
- Runs `compare_models()` across 16 classification algorithms (including XGBoost and CatBoost) with 10-fold stratified cross-validation, logging every candidate to a local MLflow tracking server (`mlflow.db`, SQLite-backed).
- Attempts hyperparameter tuning on the leader (CatBoost) via a 25-iteration random search. Note: CatBoost's tuning hit a known compatibility bug between its sklearn wrapper's `n_estimators`/`iterations` parameter aliasing and scikit-learn's `clone()` validation — worked around with a custom hyperparameter grid that tunes `iterations` directly. Tuning did not beat CatBoost's default hyperparameters (PyCaret's `choose_better` returned the untuned model) — see `metrics/metrics_automl.json`'s `tuning_note`.
- Registers the final model to MLflow's Model Registry (`diabetes-readmission-catboost`) with a `champion` alias and a `semantic_version: 1.0.0` tag (MLflow's native registry versions are plain integers; semantic versioning is layered on as a tag).
- Exports the final model and metrics to the main repo (`models/model_automl.pkl`, `metrics/metrics_automl.json`), mirroring the baseline's convention.

**Result vs. baseline:** AutoML's CatBoost (AUC 0.657, F1 0.432) performs comparably to — not clearly better than — the XGBoost baseline (AUC 0.648, F1 0.436); every algorithm tried clustered in the same 0.63–0.66 AUC range regardless of tuning, suggesting the current feature set is closer to its practical ceiling than the choice of algorithm. Feature importance also diverged notably between the two models (XGBoost's baseline was dominated by `number_inpatient`; CatBoost's ranking is led by `num_lab_procedures` and is far more evenly distributed) — a reminder that single-model feature importance shouldn't be read as ground truth.

**View the tracking dashboard:**
```bash
source venv-automl/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

## Current Status

- [x] Environment setup, data ingestion, EDA
- [x] Data cleaning (documented decisions in notebook)
- [x] Feature engineering (target, diagnosis grouping, specialty bucketing, race/age encoding)
- [x] Feast feature store setup and retrieval
- [x] Train/test split, XGBoost baseline model
- [x] Experiment tracking (MLflow)
- [ ] Pipeline orchestration (Airflow/Prefect)
- [ ] Model containerization & deployment (Docker + FastAPI/Flask/BentoML)
- [ ] Production monitoring & drift simulation (EvidentlyAI)

## Environment Notes

Package versions are pinned in `requirements.txt` to avoid known compatibility issues between `pandas`, `pyarrow`, and `pathspec` on newer Python releases. If reinstalling, use `pip install -r requirements.txt` as-is rather than letting pip resolve to the latest versions.