# Diabetes 30-Day Readmission Prediction — MLOps Final Project

Predicting hospital readmission risk for diabetic patients using the UCI/Kaggle "Diabetes 130-US Hospitals for Years 1999-2008" dataset.

## Project Structure

```
ml-ops-final-group-3/
├── data/
│   ├── raw/              # diabetic_data.csv (not in git — see Setup)
│   └── processed/        # cleaned + feature-engineered CSVs (DVC-tracked)
├── eda/                    # exploratory data analysis (see EDA section)
│   ├── figures/            # seaborn/matplotlib charts (PNG)
│   ├── EDA_SUMMARY.md      # written EDA summary
│   └── eda_report.html     # Evidently data-quality report
├── feast_repo/            # Feast feature store repo
├── metrics/                # model evaluation metrics (JSON)
├── models/                 # saved model artifacts (pickle) + feature_columns.json
├── monitoring/             # deployment monitoring (see Production Monitoring)
│   ├── baseline_validation.py  # clean test set through API, validate vs baseline metrics
│   ├── drift_simulation.py     # generate corrupted/drifted test datasets
│   ├── anomaly_verification.py # send drifted data, catch + alert on drift
│   ├── _client.py              # shared API client + path bootstrap
│   ├── drifted/                # simulated drifted CSVs (generated)
│   └── reports/                # Evidently HTML dashboards + DRIFT_SUMMARY.md (generated)
├── notebooks/
│   ├── final-group3-en.ipynb   # cleaning, Feast, DVC, XGBoost baseline
│   └── automl_pycaret.ipynb    # PyCaret AutoML + MLflow tracking (separate env, see Setup)
├── outputs/                # AutoML leaderboard CSVs (gitignored)
├── src/                    # pipeline scripts (see Pipeline Orchestration)
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── feast_registry.py
│   ├── train_baseline.py
│   ├── train_automl.py        # runs under venv-automl, invoked via subprocess
│   ├── progress.py            # console heartbeat for long-running steps
│   ├── pipeline_flow.py       # Prefect flow wiring the above together
│   ├── prepare_data.py        # materialize features CSV + clean train/test split
│   ├── serve_api.py           # FastAPI serving app (see Model Serving API)
│   ├── monitoring.py          # EvidentlyAI drift/performance report helpers
│   └── eda.py                 # exploratory data analysis generator (see EDA)
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
```
Then open http://127.0.0.1:5000.

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

## Exploratory Data Analysis (EDA)

Beyond the EDA in `notebooks/final-group3-en.ipynb`, a scripted, model-centric EDA
(`src/eda.py`) profiles the 16 features the models actually use plus the `readmitted_binary`
target, and writes reproducible artifacts to `eda/`.

**Run it:**
```bash
python src/eda.py    # from the repo root
```

**Outputs:**
- `eda/figures/*.png` — target balance, numeric distributions, categorical distributions,
  correlation heatmap, and readmission rate by age / diagnosis / prior inpatient visits.
- `eda/EDA_SUMMARY.md` — written summary: shape, target balance (40.7% readmitted), missingness,
  numeric stats, and the strongest target correlations.
- `eda/eda_report.html` — Evidently `DataQualityPreset` report (same classic look as the
  monitoring dashboards) for interactive per-column exploration.

**Key takeaways:** the target is mildly imbalanced (~41% positive); linear correlations with the
target are all weak (top is `number_inpatient` at +0.15), which explains the models' modest
~0.65 AUC — readmission is driven by many small signals rather than one dominant feature.
Readmission rate rises with age, with prior inpatient visits (0.38 → 0.83 across 0→5+ visits),
and is highest for diabetes/respiratory/circulatory primary diagnoses.

## Model Serving API (FastAPI)

The baseline XGBoost model is served over HTTP by a FastAPI app (`src/serve_api.py`). It loads
`models/model_baseline.pkl`, reproduces the exact train-time preprocessing (the 16 raw features →
`get_dummies` → align to the model's 36 one-hot columns), and exposes:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Model + version status |
| `/predict` | POST | `{"records":[{...16 raw features...}]}` → predictions + probabilities |
| `/metrics` | GET | Prometheus metrics (serving counters + drift gauges) |
| `/internal/drift` | POST | Monitoring scripts push drift results so they surface on `/metrics` |

**macOS note:** the model is XGBoost, which needs the OpenMP runtime — `brew install libomp` once.

**Run it** (from the repo root):
```bash
brew install libomp                      # one-time, macOS only
python src/prepare_data.py               # materialize data/processed/diabetic_data_features.csv from the Feast parquet
uvicorn src.serve_api:app --port 8000
```
Quick check: `curl localhost:8000/health` and `curl localhost:8000/metrics`.

## Production Monitoring & Drift Simulation (EvidentlyAI + Prometheus)

A three-stage monitoring workflow (`monitoring/`) validates the deployed API and stress-tests it
against simulated drift. **Start the API first**, then run in order:

**1. Baseline validation** — `python monitoring/baseline_validation.py`
Pushes the clean test set through `/predict` and asserts the live metrics reproduce
`metrics/metrics_baseline.json` (F1 0.436 / AUC 0.648, exact match), then writes a combined
Evidently report of *clean train vs clean test* (`monitoring/reports/00_baseline.html`) to
confirm **no drift** — the healthy reference.

**2. Drift simulation** — `python monitoring/drift_simulation.py`
Writes four artificially corrupted copies of the test set to `monitoring/drifted/`:
`out_of_bounds` (impossible numeric magnitudes), `column_swap` (values wired to wrong fields),
`schema_change` (unseen categorical values), `dist_shift` (older/sicker population).

**3. Anomaly verification** — `python monitoring/anomaly_verification.py`
Sends each drifted set to the API, writes a combined Evidently report vs the clean reference
(`monitoring/reports/<scenario>.html`), pushes the result to the API's Prometheus gauges
(`data_drift_share`, `drifted_columns_total`, `drift_detected`), and **fires alerts** when columns
drift or the predicted-positive ratio swings. All four scenarios are caught. Output: one HTML
dashboard per scenario plus a consolidated `monitoring/reports/DRIFT_SUMMARY.md`.

**Report layout:** each report is a single classic Evidently dashboard (rendered via
`evidently.legacy`) combining **Dataset Drift** (summary tiles) → **Data Drift Summary** (per-column
table with reference-vs-current distributions, stat test, and drift score — including the model's
`prediction` and `readmitted_binary` target rows) → **Classification Model Quality** (current vs
reference, with confusion matrix). This mirrors the standard Evidently monitoring report layout.

**Dashboards:** Evidently reports are self-contained HTML in `monitoring/reports/` — open them in a
browser. Live serving/drift metrics are on `GET /metrics` (Prometheus format), ready to be scraped
by Prometheus/Grafana if desired. Monitoring config is self-contained (no Docker required).

## Current Status

- [x] Environment setup, data ingestion, EDA
- [x] Data cleaning (documented decisions in notebook)
- [x] Feature engineering (target, diagnosis grouping, specialty bucketing, race/age encoding)
- [x] Feast feature store setup and retrieval
- [x] Train/test split, XGBoost baseline model
- [x] Experiment tracking & AutoML (MLflow + PyCaret) — see AutoML section
- [x] Pipeline orchestration (Prefect)
- [x] Model serving API (FastAPI) — see Model Serving API section
- [x] Production monitoring & drift simulation (EvidentlyAI + Prometheus) — see Monitoring section

## Environment Notes

Package versions are pinned in `requirements.txt` to avoid known compatibility issues between `pandas`, `pyarrow`, and `pathspec` on newer Python releases. If reinstalling, use `pip install -r requirements.txt` as-is rather than letting pip resolve to the latest versions.