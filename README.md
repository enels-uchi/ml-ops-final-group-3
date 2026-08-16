# Diabetes 30-Day Readmission Prediction — MLOps Final Project

Predicting hospital readmission risk for diabetic patients using the UCI/Kaggle "Diabetes 130-US Hospitals for Years 1999-2008" dataset.

## Quickstart — run the deployed service

**Only Docker is required.** The trained model is baked into the image, so you do not need the
dataset, a Python environment, or any training run to see the deployment working.

```bash
git clone https://github.com/enels-uchi/ml-ops-final-group-3.git
cd ml-ops-final-group-3
docker compose up --build -d          # first build ~3-5 min; subsequent starts ~5s
```

That starts three containers. Give them ~25 seconds, then:

| Service | URL | What it is |
|---|---|---|
| **API** | http://localhost:8000/docs | interactive Swagger UI — send a patient, get a risk score |
| **Grafana** | http://localhost:3000/d/readmission-monitoring | monitoring dashboard (no login needed) |
| **Prometheus** | http://localhost:9090/alerts | alert rules and their state |

**Get a prediction** — expand `POST /predict` in Swagger, click *Try it out*, and paste the
contents of [deploy/demo_request.json](deploy/demo_request.json). Or from a terminal:

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d @deploy/demo_request.json
```

**Verify the deployment** (needs Python, but only the standard library):

```bash
python deploy/smoke_test.py           # expects: 14/14 checks passed
```

**Stop it:** `docker compose down -v`

> **Behind a corporate/university proxy?** If the build fails with
> `CERTIFICATE_VERIFY_FAILED`, TLS inspection is breaking pip. Rebuild with
> `PIP_TRUSTED_HOSTS=1 docker compose up --build -d` (`set PIP_TRUSTED_HOSTS=1` first on cmd.exe).

Everything else — the monitoring workflow, the drift simulation, and retraining from raw data —
needs the Python environment described under [Setup](#setup). Jump to:
[Containerized Deployment](#containerized-deployment-docker--docker-compose) ·
[Monitoring & Drift](#offline-monitoring--drift-simulation-evidentlyai--prometheus) ·
[Known Limitations](#known-limitations)

## Problem Statement

A high proportion of diabetic patients are readmitted to hospital within 30 days of discharge —
a signal that they were released before being fully stabilised, at real cost to both the patient
and the health system. We predict that risk **at discharge**, as a binary classification over
~70,000 encounters from 130 US hospitals (1999–2008), so care teams can target follow-up at the
patients most likely to come back.

**Target:** `readmitted_binary` — 1 if readmitted within 30 days, else 0. ~41% positive.

## Evaluation Metric

**Primary: F1 and AUC-ROC. Tiebreaker: recall.**

The target is mildly imbalanced (~41% positive), so accuracy is uninformative — a model that
never predicts readmission still scores ~59%. F1 balances the precision/recall trade-off on the
positive class, and AUC-ROC measures ranking quality independent of the 0.5 threshold, which
matters because the operating threshold is a clinical decision, not a modelling one.

Recall breaks ties because **the error costs are asymmetric**: a false negative is a patient sent
home who is readmitted — a missed intervention, and the expensive outcome. A false positive is an
unnecessary follow-up appointment. Under-triaging is the costlier mistake, so between two
otherwise comparable models we prefer the one that misses fewer at-risk patients.

Both models land around 0.65 AUC. That ceiling is a property of the data, not the search: linear
correlations with the target are all weak (strongest is `number_inpatient` at +0.15), so
readmission here is driven by many small signals rather than one dominant feature. See
[eda/EDA_SUMMARY.md](eda/EDA_SUMMARY.md).

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
│   ├── metrics_baseline.json    # XGBoost held-out test metrics
│   ├── metrics_automl.json      # AutoML leaderboard (10-fold CV means) + registry coordinates
│   └── metrics_deployment.json  # deployed champion's certified metrics (see Baseline Validation)
├── models/                 # saved model artifacts
│   ├── model_baseline.pkl       # XGBoost baseline
│   ├── model_automl.pkl         # PyCaret champion pipeline (needs PyCaret to load)
│   ├── champion/                # champion flattened for serving (CatBoost .cbm + JSON transforms)
│   └── drift_reference.csv      # clean-training sample for in-container drift detection
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
├── screenshots/            # MLflow tracking + registry evidence (see AutoML section)
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
│   ├── drift_detector.py      # in-container Evidently monitor (background thread)
│   ├── build_drift_reference.py  # generates models/drift_reference.csv for the above
│   ├── export_champion.py     # flattens the PyCaret champion into models/champion/
│   ├── champion_preprocess.py # pure-pandas replay of the champion's preprocessing
│   ├── monitoring.py          # EvidentlyAI drift/performance report helpers (online + offline)
│   └── eda.py                 # exploratory data analysis generator (see EDA)
├── deploy/                 # containerized deployment + observability stack
│   ├── smoke_test.py           # post-deploy verification of the running API
│   ├── demo_traffic.py         # continuous traffic generator for a live drift demo
│   ├── demo_request.json       # sample /predict payload
│   ├── prometheus/             # scrape config + drift alert rules
│   └── grafana/                # provisioned datasource + monitoring dashboard
├── Dockerfile              # inference container (FastAPI + uvicorn + baked-in model)
├── docker-compose.yml      # api + prometheus + grafana stack
├── .dockerignore
├── requirements.txt        # full training/monitoring environment
├── requirements-serve.txt  # slim serving-only dependencies (used by the Dockerfile)
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

**Registry evidence.** `mlflow.db` is gitignored, so the registry itself isn't browsable from a
clone (see Known Limitations). These committed screenshots are the record:

| Screenshot | Shows |
|---|---|
| [screenshots/training_runs.png](screenshots/training_runs.png) | the MLflow experiment with every AutoML candidate logged |
| [screenshots/model_overview.png](screenshots/model_overview.png) | the registered model, its params and metrics |
| [screenshots/model_registery_versioning.png](screenshots/model_registery_versioning.png) | `diabetes-readmission-catboost` v1 with the `champion` alias and `semantic_version: 1.0.0` tag |

The same registry coordinates are also machine-readable in `metrics/metrics_automl.json`,
`models/champion/metadata.json`, and on the live API's `GET /health`.

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
6. `export_champion` — flattens the registered champion into `models/champion/` for serving
   (self-verifying: fails the run rather than exporting an artifact that disagrees with the model)
7. `build_drift_reference` — refreshes `models/drift_reference.csv`, the clean-training sample the
   container's online drift monitor compares live traffic against

Steps 6–7 exist so a retrain leaves the deployment artifacts consistent with the new model rather
than silently stale.

Steps 1–4 and 7 run directly in the main `venv`. Steps 5–6 are invoked as **subprocesses using `venv-automl`'s Python interpreter**, since PyCaret doesn't support the main environment's Python 3.13 — the orchestrator itself doesn't need PyCaret installed, it just shells out to the other environment for that one step. Long-running steps (the AutoML search and hyperparameter tuning) print a periodic elapsed-time heartbeat (`src/progress.py`) so the pipeline doesn't appear to hang during multi-minute stretches.

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

The registered AutoML champion is served over HTTP by a FastAPI app (`src/serve_api.py`).
Requests always carry the **16 raw clinical features**; the app reproduces the served model's own
training-time preprocessing internally, so serving can never silently diverge from training.
It exposes:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Model + version status |
| `/predict` | POST | `{"records":[{...16 raw features...}]}` → predictions + probabilities |
| `/drift` | GET | Live self-detected drift verdict, including per-column p-values |
| `/metrics` | GET | Prometheus metrics (serving counters + drift gauges) |
| `/internal/drift` | POST | Monitoring scripts push offline Evidently results onto `/metrics` |

### Online drift detection (EvidentlyAI, in-container)

EvidentlyAI runs **inside the serving container**, not just as an offline script. The API keeps a
rolling window of the raw feature rows it has served and runs Evidently's `DataDriftPreset`
against a 5,000-row sample of the clean training split baked into the image
(`models/drift_reference.csv`, generated by `src/build_drift_reference.py`). The deployed service
therefore detects its own drift, and Prometheus alerts fire on live traffic with nobody running
an analysis script.

Both the online monitor (`src/drift_detector.py`) and the offline reports go through the same
helper in `src/monitoring.py`, so live numbers and committed reports come from an identical
Evidently configuration — Evidently's own per-column stat-test selection and its 0.5
dataset-level threshold.

**Off the critical path.** An Evidently report over a few thousand rows takes ~0.5s — cheap, but
far too slow to sit inside a `/predict` call that completes in ~8ms. So `/predict` only appends
to the buffer; when enough new rows arrive, a single background thread runs the analysis and
publishes to `data_drift_share` / `drifted_columns_total` / `drift_detected`. Measured p95
serving latency is unchanged at ~10ms. If an analysis is still running, later batches fold into
the next window rather than queueing.

| Variable | Default | Purpose |
|---|---|---|
| `DRIFT_WINDOW` | 3000 | rolling window of served rows |
| `DRIFT_MIN_ROWS` | 600 | recompute every N new rows |
| `DRIFT_MIN_WINDOW` | 2000 | rows required before the first verdict |

`DRIFT_MIN_WINDOW` exists because Evidently selects its stat test by sample size and switches
below ~1000 rows; verdicts from a barely-filled buffer flickered a false positive on genuinely
clean traffic. Waiting for a substantial window costs a few seconds of `"warming up"` on
`GET /drift` and removes the flicker entirely.

`GET /drift` returns the current verdict with the per-column stat test and drift score Evidently
used, so the numbers on the dashboard are auditable. If `models/drift_reference.csv` is missing
the API still serves normally and logs that online detection is off.

The offline workflow below is unchanged and remains the deeper analysis — it produces the
per-column HTML reports and the model-quality comparison, which the in-container monitor
deliberately skips.

**macOS note:** the model is XGBoost, which needs the OpenMP runtime — `brew install libomp` once.
(The container handles this itself by installing `libgomp1`.)

**Run it locally, without Docker** (from the repo root):
```bash
brew install libomp                      # one-time, macOS only
python src/prepare_data.py               # materialize data/processed/diabetic_data_features.csv from the Feast parquet
uvicorn src.serve_api:app --port 8000
```
Quick check: `curl localhost:8000/health` and `curl localhost:8000/metrics`.

**Which model is served** is set by environment variable, so the same image can serve a local
pickle or pull the registered champion straight out of the MLflow Model Registry:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/model_baseline.pkl` | local pickle to load |
| `MLFLOW_MODEL_URI` | *(unset)* | e.g. `models:/diabetes-readmission-catboost@champion` — takes precedence when set |
| `MLFLOW_TRACKING_URI` | *(unset)* | registry backend, e.g. `sqlite:///mlflow.db` |
| `MODEL_VERSION` | `baseline-1.0.0` | version string on `/health` and the `model_info` metric |
| `REQUEST_LOG` | `monitoring/logs/requests.csv` | where served rows are appended |

### The served model is the AutoML champion

The container serves **CatBoost** — the model PyCaret's AutoML search selected and the one
registered in MLflow as `diabetes-readmission-catboost` v1, alias `champion`, semantic version
1.0.0. `GET /health` echoes those registry details so the deployment is traceable to the
registry entry.

It is **not** loaded from `models/model_automl.pkl` directly, because that file is a PyCaret
pipeline: unpickling it requires PyCaret importable, PyCaret does not support the container's
Python 3.12, and its fitted transformers are scikit-learn 1.4 / category_encoders objects that
are unsafe to unpickle under the container's scikit-learn 1.9.

Instead `src/export_champion.py` reads that pipeline and flattens it into `models/champion/`:

| File | What it is |
|---|---|
| `catboost_model.cbm` | the CatBoost model in CatBoost's own version-stable binary format |
| `preprocessing.json` | the pipeline's four fitted transforms as plain parameters |
| `metadata.json` | registry coordinates + the verification result |

`src/champion_preprocess.py` replays those four transforms — mean imputation, most-frequent
imputation, ordinal encoding of `gender`, one-hot encoding of the two remaining categoricals —
in pure pandas, producing the same 35 columns CatBoost was fitted on. The container therefore
needs only pandas + catboost.

**The export is verified, not assumed.** `export_champion.py` scores all 69,987 rows through both
the original pipeline and the replay and refuses to write the artifact unless the probabilities
agree to within 1e-9. The current export reproduces the pipeline **exactly** — maximum difference
`0.000e+00`.

Re-generate it (needs the PyCaret environment, Python 3.11) with:

```bash
python src/export_champion.py
```

Set `MODEL_KIND=baseline` to serve the XGBoost baseline from the same image instead.

## Containerized Deployment (Docker + Docker Compose)

The model is packaged as a self-contained inference image and deployed alongside its monitoring
stack. Nothing is configured by hand — the Prometheus scrape config, the alert rules, and the
Grafana datasource and dashboard are all provisioned from files in `deploy/`.

**Start the whole stack:**
```bash
docker compose up --build -d
```

| Service | URL | What it is |
|---|---|---|
| `api` | http://localhost:8000/docs | FastAPI inference service (interactive Swagger UI) |
| `prometheus` | http://localhost:9090/alerts | scrapes `/metrics` every 5s, evaluates the drift alert rules |
| `grafana` | http://localhost:3000/d/readmission-monitoring | "Model Serving & Drift Monitoring" dashboard (anonymous viewing enabled; `admin`/`admin` to edit) |

**Verify the deployment:**
```bash
python deploy/smoke_test.py
```
Standard-library only, so it also runs inside the container
(`docker compose exec api python deploy/smoke_test.py`). It checks `/health`, that a full record,
a sparse record and an empty batch all score correctly, that the high-risk record actually scores
above the typical one, and that every Prometheus series the dashboard depends on is exported.
Exit code 0/1 makes it usable as a CI gate.

**Get a prediction:**
```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "records": [{
    "time_in_hospital": 13, "num_lab_procedures": 68, "num_procedures": 2,
    "num_medications": 28, "number_outpatient": 0, "number_emergency": 0,
    "number_inpatient": 0, "number_diagnoses": 8, "admission_type_id": 2,
    "discharge_disposition_id": 1, "admission_source_id": 4,
    "diag_1_category": "Circulatory", "medical_specialty_grouped": "Missing",
    "race_white": 1, "gender": "Female", "age_ordinal": 8
  }]
}'
```

**Image notes:**
- Base `python:3.12-slim`, ~1.38 GB. `requirements-serve.txt` is a strict subset of
  `requirements.txt` (no jupyter/dvc/feast/matplotlib/seaborn) and uses `xgboost-cpu` rather than
  `xgboost`, dropping ~250 MB of CUDA/NCCL wheels an inference-only service never touches.
  Evidently and CatBoost are the two heavyweights, and both are deliberate: Evidently runs
  in-container for online drift detection, and CatBoost is the served champion. A serving-only
  image without them builds at ~519 MB; +Evidently is ~993 MB; +CatBoost is the current ~1.38 GB.
- Runs as a non-root user (`appuser`, uid 10001) with a `HEALTHCHECK` on `/health`.
- The model is **baked into the image**, so the container is a reproducible unit — the same image
  digest always serves the same weights.
- Behind a TLS-inspecting proxy (corporate/campus network), pip can't verify PyPI's certificate.
  Build with `PIP_TRUSTED_HOSTS=1 docker compose up --build` in that case; the default build keeps
  full certificate verification.

**Stop it:** `docker compose down` (add `-v` to also wipe the Prometheus/Grafana volumes).

### Live demo traffic

The monitoring scripts send one large burst and exit, which leaves the Grafana panels a flat
line with a step in it. `deploy/demo_traffic.py` instead drives steady traffic and *gradually*
corrupts it, so drift arrives as a moving curve — what it would actually look like in production:

```bash
python monitoring/drift_simulation.py     # once, to create the drifted CSVs
python deploy/demo_traffic.py             # ~5 min: healthy -> ramp -> fully drifted
```

It runs in three phases (clean traffic near the 0.244 baseline → a linear 0–100% blend of
drifted rows → fully corrupted) and prints the predicted-positive ratio per batch with an inline
bar, so the trend is visible in the terminal as well as on the dashboard. `--scenario
out_of_bounds` makes the ratio collapse toward 0.00 instead of climbing to 0.95;
`--healthy-only` holds steady clean traffic; `--duration N` shortens the run; Ctrl-C stops early.

Note it moves the *serving* metrics (throughput, latency, predicted-positive ratio). The
`data_drift_share` / `drifted_columns_total` gauges are Evidently's verdict, so run
`monitoring/anomaly_verification.py` to move those.

## Offline Monitoring & Drift Simulation (EvidentlyAI + Prometheus)

A three-stage monitoring workflow (`monitoring/`) validates the deployed API and stress-tests it
against simulated drift. **Start the API first** (`docker compose up -d`, or uvicorn locally),
then run the three scripts on the host in order. They stay on the host rather than in the
container because they import Evidently/scikit-learn and write HTML reports into the working
tree, while the image deliberately carries serving dependencies only. Point them at the
container with `API_URL` (defaults to `http://127.0.0.1:8000`, which is already correct for the
compose stack):

```bash
python src/prepare_data.py               # once, to materialize the features CSV
API_URL=http://localhost:8000 python monitoring/baseline_validation.py
python monitoring/drift_simulation.py
API_URL=http://localhost:8000 python monitoring/anomaly_verification.py
```

**1. Baseline validation** — `python monitoring/baseline_validation.py`
Pushes the clean test set through `/predict` and asserts the live metrics reproduce the deployed
model's certified figures exactly, then writes a combined Evidently report of *clean train vs
clean test* (`monitoring/reports/00_baseline.html`) to confirm **no drift** — the healthy
reference.

Which file it compares against follows the deployed model, read from `/health`:

| Deployed | Reference | Values |
|---|---|---|
| `baseline` (XGBoost) | `metrics/metrics_baseline.json` | F1 0.436 / AUC 0.648 |
| `champion` (CatBoost) | `metrics/metrics_deployment.json` | F1 0.509 / AUC 0.736 |

⚠️ **The champion's deployment figures are not comparable to `metrics_automl.json`** (F1 0.4323 /
AUC 0.6574). Those AutoML numbers are PyCaret's 10-fold cross-validation means over its own
training split, whereas these are a single evaluation on `train_baseline.py`'s held-out split.
PyCaret partitions the data itself, so the two splits do not coincide and part of this test set
was seen during the champion's training — meaning these figures are optimistic as a
generalisation estimate. They exist to certify **deployment integrity**: the container must
reproduce, to three decimals, the numbers the model was certified at. The same caveat is written
into `metrics_deployment.json` itself.

Re-certify after changing the deployed model with
`python monitoring/baseline_validation.py --record`.

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
browser. The scripts also run standalone against a local uvicorn process (no Docker required).

**Live dashboard & automated alerting.** When the compose stack is running, the drift results the
scripts push to `POST /internal/drift` are scraped by Prometheus within 5s and land on the Grafana
dashboard at http://localhost:3000/d/readmission-monitoring — drift status, drifted-column count,
drift share, predicted-positive ratio against the 0.244 clean baseline, throughput and p50/p95
latency. Prometheus evaluates `deploy/prometheus/alert_rules.yml` over the same series
(http://localhost:9090/alerts):

| Alert | Condition | Catches |
|---|---|---|
| `DriftedColumnsDetected` | `drifted_columns_total >= 3` | all four drift scenarios |
| `DataDriftShareHigh` | `data_drift_share > 0.15` | `column_swap`, `dist_shift`, `out_of_bounds`, `schema_change` (0.167–0.278) |
| `DatasetDriftDetected` | `drift_detected == 1` | Evidently's dataset-level verdict (≥50% of columns) |
| `PredictionDistributionShift` | ratio outside `[0.15, 0.40]` | `dist_shift` (0.921) and `out_of_bounds` (0.059) |
| `InferenceAPIDown` | `up == 0` for 15s | container crash / unreachable service |
| `HighPredictionLatency` | p95 > 2s | serving degradation |

Input drift and output drift are deliberately separate signals: `column_swap` and `schema_change`
corrupt the inputs without moving the model's output distribution much, so they trip the
column-drift alerts but not the prediction-shift one. **Verified end to end** against the
containerized API — all four scenarios reproduce the numbers in `DRIFT_SUMMARY.md` exactly, and
`DriftedColumnsDetected` + `DataDriftShareHigh` were observed firing in Prometheus.

## Current Status

- [x] Environment setup, data ingestion, EDA
- [x] Data cleaning (documented decisions in notebook)
- [x] Feature engineering (target, diagnosis grouping, specialty bucketing, race/age encoding)
- [x] Feast feature store setup and retrieval
- [x] Train/test split, XGBoost baseline model
- [x] Experiment tracking & AutoML (MLflow + PyCaret) — see AutoML section
- [x] Model Registry entry with semantic versioning (`diabetes-readmission-catboost` v1 / 1.0.0)
- [x] Pipeline orchestration (Prefect), including deployment-artifact regeneration
- [x] Model serving API (FastAPI) serving the **registered champion** — see Model Serving API
- [x] Containerization & deployment (Docker + Docker Compose) — see Containerized Deployment section
- [x] Production monitoring & drift simulation (EvidentlyAI + Prometheus + Grafana) — see Monitoring section

## Known Limitations

Recorded deliberately rather than left for a reader to discover.

**1. The AutoML champion's split does not match the baseline's.** `src/train_baseline.py` splits
with `train_test_split(..., random_state=42)`; PyCaret's `setup(train_size=0.8, session_id=42)`
partitions the data itself, and the two partitions do not coincide. Part of the baseline's test
set was therefore inside the champion's training data, which is why the champion scores 0.736 AUC
on that split against its recorded 0.657. The assignment asks for the test set to stay isolated
until production validation — that holds for the XGBoost baseline, but **not** for the CatBoost
champion. `metrics/metrics_deployment.json` exists to make the distinction explicit: it certifies
deployment integrity (the container reproduces its certified numbers exactly), not generalisation.
The fix is to have the AutoML step consume the same pre-split data as the baseline.

**2. `metrics_automl.json` reports cross-validation means, not held-out scores.** They come from
`leaderboard.loc['catboost']`, i.e. PyCaret's 10-fold CV over its training split, while
`metrics_baseline.json` is a single held-out evaluation. The 0.657-vs-0.648 AUC comparison behind
the champion selection is therefore not strictly like-for-like.

**3. The MLflow registry is not reproducible from a clone.** `mlflow.db` and `mlruns/` are
gitignored, so the registry exists only on the machine that ran the AutoML step. The registry
*coordinates* are committed (in `metrics_automl.json`, `models/champion/metadata.json`, and on
`GET /health`) and `screenshots/` holds the registry UI as evidence, but a grader cannot browse
the live registry without re-running the pipeline. A shared tracking backend, or committing the
SQLite file, would close this.

**4. DVC has no remote.** `.dvc` pointer files are committed but there is no configured storage,
so `dvc pull` will not work — processed data has to be regenerated by re-running the pipeline.

**5. Orchestration is Prefect, not Airflow.** The proposal named Airflow; Prefect was used
instead. The assignment permits "Airflow, Prefect, or an equivalent workflow platform".

## Assignment Requirements Coverage

| Requirement | Where |
|---|---|
| Dataset with a clear target | `readmitted_binary`, ~70k encounters — Problem Statement above |
| Evaluation metric aligned to constraints | Evaluation Metric section above |
| Train/test split, test isolated | `src/train_baseline.py` 80/20 stratified — see Limitation 1 |
| Automated orchestrator pipeline | Prefect, `src/pipeline_flow.py` (7 stages) |
| Experiment tracking + model logging | MLflow + PyCaret AutoML across 16 algorithms |
| Model Registry with semantic versioning | `diabetes-readmission-catboost` v1, alias `champion`, tag `semantic_version: 1.0.0` — evidence in `screenshots/` |
| Package the registered model, deploy it | Docker + FastAPI serving the champion — Containerized Deployment |
| Accepts test inputs, returns real-time predictions | `POST /predict`, Swagger at `/docs` |
| Monitoring framework / dashboard | EvidentlyAI (online + offline) + Prometheus + Grafana |
| Baseline validation against monitoring baseline | `monitoring/baseline_validation.py` |
| Drift simulation (corrupt the test set) | `monitoring/drift_simulation.py` — 4 scenarios |
| Anomaly verification + alerting documented | `monitoring/anomaly_verification.py`, `monitoring/reports/DRIFT_SUMMARY.md`, Prometheus alert rules |
| Fully commented source code | all modules carry module- and function-level docstrings |
| Comprehensive README | this file |
| Dependency configuration | `requirements.txt`, `requirements-serve.txt`, `Dockerfile` |

## Environment Notes

Package versions are pinned in `requirements.txt` to avoid known compatibility issues between `pandas`, `pyarrow`, and `pathspec` on newer Python releases. If reinstalling, use `pip install -r requirements.txt` as-is rather than letting pip resolve to the latest versions.