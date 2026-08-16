"""FastAPI serving app for the diabetes readmission model.

Endpoints
---------
GET  /health              -> model/version status
POST /predict             -> {records:[{...16 raw features...}]} -> predictions + probabilities
GET  /metrics             -> Prometheus exposition (serving + drift gauges)
POST /internal/drift      -> monitoring scripts push drift results here so they surface on /metrics

Run:  venv/bin/uvicorn src.serve_api:app --port 8000    (from the repo root)
      docker compose up --build                        (containerized, see deploy/ + README)

Requests always carry the 16 RAW features; the app reproduces the model's own training-time
preprocessing so serving can never silently diverge from training.

Two model kinds are supported, selected by MODEL_KIND:

  champion (default)  The AutoML/MLflow-registered CatBoost — `diabetes-readmission-catboost`
                      v1, alias `champion`, semantic version 1.0.0. Loaded from the flattened
                      export in models/champion/ (see src/export_champion.py), which replays the
                      PyCaret pipeline's 4 transforms into the 35 columns CatBoost was fitted on.
  baseline            The XGBoost baseline pickle, whose 16 raw features are expanded to 36
                      one-hot columns by get_dummies + align, mirroring src/train_baseline.py.

    MODEL_KIND        "champion" | "baseline"          (default: champion)
    CHAMPION_DIR      exported champion directory      (default: models/champion)
    MODEL_PATH        baseline pickle path             (default: models/model_baseline.pkl)
    MLFLOW_MODEL_URI  registry/run URI, baseline kind  (e.g. models:/name@champion)
    MLFLOW_TRACKING_URI  tracking backend              (e.g. sqlite:///mlflow.db)
    MODEL_VERSION     version string on /health and the Prometheus model_info label
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_baseline import FEATURE_COLS, CATEGORICAL_COLS  # single source of truth
from drift_detector import DriftDetector

# Which model this container serves.
#   "champion" -> the AutoML/MLflow-registered CatBoost, exported by src/export_champion.py
#   "baseline" -> the XGBoost baseline pickle
MODEL_KIND = os.environ.get("MODEL_KIND", "champion").strip().lower()
CHAMPION_DIR = os.environ.get("CHAMPION_DIR", "models/champion")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_baseline.pkl")
MLFLOW_MODEL_URI = os.environ.get("MLFLOW_MODEL_URI", "").strip()
FEATURE_COLUMNS_PATH = os.environ.get("FEATURE_COLUMNS_PATH", "models/feature_columns.json")
REQUEST_LOG = os.environ.get("REQUEST_LOG", "monitoring/logs/requests.csv")
_DEFAULT_VERSION = "catboost-champion-1.0.0" if MODEL_KIND == "champion" else "baseline-1.0.0"
MODEL_VERSION = os.environ.get("MODEL_VERSION", _DEFAULT_VERSION)

# Online drift detection: reference sample baked into the image by src/build_drift_reference.py.
DRIFT_REFERENCE_PATH = os.environ.get("DRIFT_REFERENCE_PATH", "models/drift_reference.csv")
# Window size is the main knob on responsiveness: drift only becomes visible once the window has
# largely flushed to new data, so 3000 rows detects a shift roughly twice as fast as 5000 while
# still being far more than the K-S / chi-square tests need to be reliable.
DRIFT_WINDOW = int(os.environ.get("DRIFT_WINDOW", "3000"))       # rolling window of served rows
DRIFT_MIN_ROWS = int(os.environ.get("DRIFT_MIN_ROWS", "600"))    # recompute every N new rows
DRIFT_MIN_WINDOW = int(os.environ.get("DRIFT_MIN_WINDOW", "2000"))  # rows needed before 1st verdict

# ---- Prometheus metrics ----------------------------------------------------------------
PREDICTIONS_TOTAL = Counter("predictions_total", "Total number of predicted rows")
PREDICT_REQUESTS = Counter("predict_requests_total", "Total /predict requests")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latency of /predict calls")
PREDICTED_POSITIVE_RATIO = Gauge(
    "predicted_positive_ratio", "Share of positive (readmit) predictions in the last batch"
)
# Drift gauges, updated by the monitoring/anomaly_verification scripts via POST /internal/drift.
DATA_DRIFT_SHARE = Gauge("data_drift_share", "Share of columns flagged as drifted (0-1)")
DRIFTED_COLUMNS_TOTAL = Gauge("drifted_columns_total", "Number of columns flagged as drifted")
DRIFT_DETECTED = Gauge("drift_detected", "1 if dataset-level drift detected, else 0")
# Labelled constant-1 gauge: the conventional Prometheus way to expose build/version metadata,
# so Grafana can show which model version produced the numbers on the rest of the dashboard.
MODEL_INFO = Gauge("model_info", "Served model metadata (always 1)", ["model_version", "source"])


def _load_from_registry(uri: str):
    """Load the registered champion from the MLflow Model Registry.

    Deliberately uses the *flavour-native* loader rather than `mlflow.pyfunc.load_model`: pyfunc
    wraps the estimator behind a bare `.predict()`, but /predict needs `predict_proba` for the
    probability output. `mlflow.sklearn.load_model` returns the real estimator; it also handles
    the sklearn-compatible XGBoost/CatBoost wrappers, which is every model this project registers.
    """
    import mlflow  # imported lazily so the default (local pickle) path needs no mlflow install

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    print(f"[serve_api] Loading registered model from MLflow: {uri}")
    return mlflow.sklearn.load_model(uri)


def _load_champion():
    """Load the exported AutoML champion: CatBoost model + its replayed preprocessing.

    `src/export_champion.py` flattens the registered PyCaret pipeline into a CatBoost binary plus
    a JSON description of its four transforms, verified to reproduce the pipeline's probabilities
    exactly. That lets the container serve the registered champion without PyCaret, which cannot
    run on this image's Python version anyway.
    """
    from catboost import CatBoostClassifier  # lazy: only the champion path needs catboost
    from champion_preprocess import ChampionPreprocessor

    directory = Path(CHAMPION_DIR)
    if not directory.exists():
        raise RuntimeError(
            f"MODEL_KIND=champion but {directory} is missing. Generate it with "
            f"`python src/export_champion.py` in the PyCaret environment."
        )
    model = CatBoostClassifier()
    model.load_model(str(directory / "catboost_model.cbm"))
    preprocessor = ChampionPreprocessor.from_dir(str(directory))
    with open(directory / "metadata.json") as f:
        metadata = json.load(f)
    registry = metadata.get("mlflow_registry", {})
    print(f"[serve_api] Loading AutoML champion from {directory}: "
          f"{registry.get('model_name', 'catboost')} v{registry.get('version', '?')} "
          f"(semantic {registry.get('semantic_version', '?')})")
    return model, preprocessor, metadata


def _load_model():
    """Resolve the model to serve: MLflow registry if configured, else the local pickle."""
    if MLFLOW_MODEL_URI:
        return _load_from_registry(MLFLOW_MODEL_URI)
    print(f"[serve_api] Loading local model: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _expected_columns(model) -> list:
    """The 36 one-hot columns the model expects, in order. Persist them for reference.

    Where the column names live depends on the flavour: XGBoost keeps them on the booster,
    plain sklearn estimators expose `feature_names_in_`, and anything else falls back to the
    committed feature_columns.json — so swapping MODEL_PATH/MLFLOW_MODEL_URI for a different
    algorithm doesn't break startup.
    """
    cols = []
    if hasattr(model, "get_booster"):
        cols = list(model.get_booster().feature_names or [])
    if not cols and hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
    if not cols and Path(FEATURE_COLUMNS_PATH).exists():
        with open(FEATURE_COLUMNS_PATH) as f:
            cols = json.load(f)
    if not cols:
        raise RuntimeError(
            f"Could not determine the model's expected feature columns, and no fallback exists "
            f"at {FEATURE_COLUMNS_PATH}."
        )
    try:
        Path(FEATURE_COLUMNS_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(FEATURE_COLUMNS_PATH, "w") as f:
            json.dump(cols, f, indent=2)
    except OSError as e:
        # Read-only model volume is fine — persisting the column list is a convenience, not a
        # requirement, and the in-memory copy is what actually drives preprocessing.
        print(f"[serve_api] warn: could not persist {FEATURE_COLUMNS_PATH}: {e}")
    return cols


def raw_frame(records: list) -> pd.DataFrame:
    """Records -> a frame of exactly the 16 raw FEATURE_COLS, unknown fields dropped.

    Split out from preprocess() because drift detection has to run on the *raw* feature space
    (that's what the reference sample and the Evidently reports use), not the one-hot expansion.
    """
    df = pd.DataFrame(records)
    # Keep only known raw features; tolerate missing ones (filled as NaN then dummy-encoded).
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[FEATURE_COLS]


def preprocess(records: list, expected_cols: list) -> pd.DataFrame:
    """Raw feature records -> the exact 36-column one-hot frame the model was trained on."""
    df = pd.get_dummies(raw_frame(records), columns=CATEGORICAL_COLS, dtype=int)
    # Align to training columns: add missing (unseen categories) as 0, drop extras, fix order.
    df = df.reindex(columns=expected_cols, fill_value=0)
    # A numeric feature absent from *every* record in the batch comes through as an all-<NA>
    # object column, which XGBoost refuses ("dtypes for data must be int, float, bool or
    # category"). Coerce those back to numeric so the NA stays a real missing value the booster
    # can route on, instead of a 500. (A column missing from only *some* records is already
    # numeric-with-NaN, so this is a no-op for it.)
    object_cols = df.columns[df.dtypes == "object"]
    for col in object_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


class PredictRequest(BaseModel):
    records: list  # list of dicts, each with (a subset of) the 16 raw feature columns


app = FastAPI(title="Diabetes Readmission Model API", version=MODEL_VERSION)

# Both model kinds expose predict_proba over a frame of engineered columns; they differ only in
# how the 16 raw features get turned into that frame, so PREPROCESS captures the difference and
# /predict stays identical for either.
CHAMPION_METADATA = None
if MODEL_KIND == "champion":
    MODEL, _CHAMPION_PREPROCESSOR, CHAMPION_METADATA = _load_champion()
    EXPECTED_COLS = _CHAMPION_PREPROCESSOR.output_columns
    MODEL_SOURCE = CHAMPION_DIR

    def PREPROCESS(records: list) -> pd.DataFrame:
        return _CHAMPION_PREPROCESSOR.transform(raw_frame(records))
else:
    MODEL = _load_model()
    EXPECTED_COLS = _expected_columns(MODEL)
    MODEL_SOURCE = MLFLOW_MODEL_URI or MODEL_PATH

    def PREPROCESS(records: list) -> pd.DataFrame:
        return preprocess(records, EXPECTED_COLS)

MODEL_INFO.labels(model_version=MODEL_VERSION, source=MODEL_SOURCE).set(1)


def _load_drift_detector():
    """Build the online detector, or None if the reference sample isn't available.

    Missing reference is not fatal: the API still serves predictions and still reports the
    serving metrics, it just can't self-diagnose input drift. The monitoring scripts pushing to
    /internal/drift remain available in that case.
    """
    if not Path(DRIFT_REFERENCE_PATH).exists():
        print(f"[serve_api] warn: no drift reference at {DRIFT_REFERENCE_PATH} — "
              f"online drift detection disabled")
        return None
    reference = pd.read_csv(DRIFT_REFERENCE_PATH)
    print(f"[serve_api] online Evidently drift detection on: {len(reference)} reference rows, "
          f"window={DRIFT_WINDOW}, recompute every {DRIFT_MIN_ROWS} rows")
    detector = DriftDetector(
        reference=reference,
        window=DRIFT_WINDOW,
        min_new_rows=DRIFT_MIN_ROWS,
        min_window=DRIFT_MIN_WINDOW,
    )
    detector.on_result(_publish_drift)
    return detector


def _publish_drift(result: dict) -> None:
    """Publish a completed Evidently verdict to the Prometheus gauges. Called off-thread."""
    DATA_DRIFT_SHARE.set(result["drift_share"])
    DRIFTED_COLUMNS_TOTAL.set(result["n_drifted"])
    DRIFT_DETECTED.set(1 if result["dataset_drift"] else 0)
    if result["n_drifted"]:
        print(f"[serve_api] Evidently: {result['n_drifted']}/{result['n_columns']} columns "
              f"drifted ({result['drift_share']:.2f}) -> {', '.join(result['drifted_columns'])}")


DRIFT_DETECTOR = _load_drift_detector()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": type(MODEL).__name__,
        "model_version": MODEL_VERSION,
        "model_source": MODEL_SOURCE,
        "model_kind": MODEL_KIND,
        "n_features": len(EXPECTED_COLS),
        "mlflow_registry": (CHAMPION_METADATA or {}).get("mlflow_registry"),
        "online_drift_detection": DRIFT_DETECTOR is not None,
    }


@app.get("/drift")
def drift_status():
    """Current online drift verdict, including the per-column p-values behind it.

    /metrics carries the same headline numbers for Prometheus; this endpoint exists so a human
    (or the demo) can see *which* columns drifted without opening an Evidently report.
    """
    if DRIFT_DETECTOR is None:
        return {"enabled": False, "reason": f"no reference sample at {DRIFT_REFERENCE_PATH}"}
    if DRIFT_DETECTOR.last_result is None:
        return {
            "enabled": True,
            "status": "warming up",
            "rows_needed": DRIFT_DETECTOR.min_new_rows,
        }
    r = DRIFT_DETECTOR.last_result
    return {
        "enabled": True,
        "detector": "evidently",
        "status": "drift detected" if r["n_drifted"] else "healthy",
        "drifted_columns": r["drifted_columns"],
        "n_drifted": r["n_drifted"],
        "n_columns": r["n_columns"],
        "drift_share": round(r["drift_share"], 4),
        "dataset_drift": r["dataset_drift"],
        "window_rows": r["n_current_rows"],
        # Evidently picks the test per column (K-S, chi-square, Z, Wasserstein...) based on the
        # column type and sample size; surfacing both makes the verdict auditable.
        "stat_tests": r["stat_tests"],
        "drift_scores": r["drift_scores"],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    start = time.perf_counter()
    # Short-circuit the empty batch rather than handing a 0-row frame to the booster: it's a
    # legitimate call (a monitoring run with nothing to send) and there is nothing to score.
    if not req.records:
        PREDICT_REQUESTS.inc()
        return {"n": 0, "predictions": [], "probabilities": [], "predicted_positive_ratio": None}

    X = PREPROCESS(req.records)
    proba = MODEL.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    PREDICT_REQUESTS.inc()
    PREDICTIONS_TOTAL.inc(len(preds))
    # Observe latency before the drift check so the histogram measures inference, not monitoring.
    PREDICTION_LATENCY.observe(time.perf_counter() - start)
    if len(preds):
        PREDICTED_POSITIVE_RATIO.set(float(preds.mean()))

    _check_drift(req.records)
    _log_requests(X)
    return {
        "n": int(len(preds)),
        "predictions": preds.tolist(),
        "probabilities": [round(float(p), 4) for p in proba],
        "predicted_positive_ratio": float(preds.mean()) if len(preds) else None,
    }


class DriftUpdate(BaseModel):
    scenario: str
    drift_detected: bool
    drift_share: float
    n_drifted: int


@app.post("/internal/drift")
def push_drift(update: DriftUpdate):
    """Monitoring scripts push drift results so the latest state shows up on /metrics."""
    DATA_DRIFT_SHARE.set(update.drift_share)
    DRIFTED_COLUMNS_TOTAL.set(update.n_drifted)
    DRIFT_DETECTED.set(1 if update.drift_detected else 0)
    return {"ok": True, "scenario": update.scenario}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _check_drift(records: list) -> None:
    """Hand the served batch to the online Evidently monitor.

    Returns immediately — the Evidently run happens on a worker thread and publishes its verdict
    through the _publish_drift callback. Wrapped defensively for the same reason as
    _log_requests: monitoring is a side effect of serving, and a failure in it must never turn a
    successful prediction into a 500.
    """
    if DRIFT_DETECTOR is None:
        return
    try:
        DRIFT_DETECTOR.add_batch(raw_frame(records))
    except Exception as e:  # noqa: BLE001 - monitoring must never break serving
        print(f"[serve_api] warn: drift check failed: {e}")


def _log_requests(X: pd.DataFrame) -> None:
    """Append served rows to the request log for post-hoc drift analysis.

    Wrapped in a try/except on purpose: in a container the log path may be a read-only or
    unmounted volume, and losing an audit line must never turn a healthy prediction into a 500.
    """
    try:
        Path(REQUEST_LOG).parent.mkdir(parents=True, exist_ok=True)
        header = not Path(REQUEST_LOG).exists()
        X.to_csv(REQUEST_LOG, mode="a", header=header, index=False)
    except OSError as e:
        print(f"[serve_api] warn: could not write request log {REQUEST_LOG}: {e}")
