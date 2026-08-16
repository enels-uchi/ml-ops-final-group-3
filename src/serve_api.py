"""FastAPI serving app for the XGBoost baseline readmission model.

Endpoints
---------
GET  /health              -> model/version status
POST /predict             -> {records:[{...16 raw features...}]} -> predictions + probabilities
GET  /metrics             -> Prometheus exposition (serving + drift gauges)
POST /internal/drift      -> monitoring scripts push drift results here so they surface on /metrics

Run:  venv/bin/uvicorn src.serve_api:app --port 8000    (from the repo root)

The model was trained on 36 one-hot columns (see models/model_baseline.pkl booster feature
names). Requests carry the 16 RAW features; this app reproduces the exact get_dummies + align
used at train time (src/train_baseline.py) so serving preprocessing matches training.
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

MODEL_PATH = "models/model_baseline.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.json"
REQUEST_LOG = "monitoring/logs/requests.csv"
MODEL_VERSION = "baseline-1.0.0"

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


def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _expected_columns(model) -> list:
    """The 36 one-hot columns the model expects, in order. Persist them for reference."""
    cols = list(model.get_booster().feature_names)
    Path(FEATURE_COLUMNS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(cols, f, indent=2)
    return cols


def preprocess(records: list, expected_cols: list) -> pd.DataFrame:
    """Raw feature records -> the exact 36-column one-hot frame the model was trained on."""
    df = pd.DataFrame(records)
    # Keep only known raw features; tolerate missing ones (filled as NaN then dummy-encoded).
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[FEATURE_COLS]
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, dtype=int)
    # Align to training columns: add missing (unseen categories) as 0, drop extras, fix order.
    df = df.reindex(columns=expected_cols, fill_value=0)
    return df


class PredictRequest(BaseModel):
    records: list  # list of dicts, each with (a subset of) the 16 raw feature columns


app = FastAPI(title="Diabetes Readmission Model API", version=MODEL_VERSION)

MODEL = _load_model()
EXPECTED_COLS = _expected_columns(MODEL)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": type(MODEL).__name__,
        "model_version": MODEL_VERSION,
        "n_features": len(EXPECTED_COLS),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    start = time.perf_counter()
    X = preprocess(req.records, EXPECTED_COLS)
    proba = MODEL.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    PREDICT_REQUESTS.inc()
    PREDICTIONS_TOTAL.inc(len(preds))
    PREDICTION_LATENCY.observe(time.perf_counter() - start)
    if len(preds):
        PREDICTED_POSITIVE_RATIO.set(float(preds.mean()))

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


def _log_requests(X: pd.DataFrame) -> None:
    Path(REQUEST_LOG).parent.mkdir(parents=True, exist_ok=True)
    header = not Path(REQUEST_LOG).exists()
    X.to_csv(REQUEST_LOG, mode="a", header=header, index=False)
