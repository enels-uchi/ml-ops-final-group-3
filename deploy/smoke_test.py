"""Post-deploy smoke test for the containerized inference API.

Verifies the deployment is actually serving before anyone runs the (much slower) monitoring
workflow against it:

  1. /health responds and reports a loaded model
  2. /predict returns one prediction + probability per record, for a full record AND for a
     record with fields missing (the API is supposed to tolerate that via reindex/fill)
  3. /metrics exposes every gauge and counter the Prometheus scrape config and the Grafana
     dashboard depend on

Uses only the standard library on purpose, so it runs from any environment — including
`docker compose exec api python deploy/smoke_test.py` inside the slim serving image.

Run:
    python deploy/smoke_test.py                      # defaults to http://localhost:8000
    API_URL=http://localhost:8000 python deploy/smoke_test.py

Exit code 0 = deployment healthy, 1 = something is wrong (usable as a CI gate).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

# Output contains non-ASCII (em dashes). On Windows the console defaults to cp1252, which
# mangles or raises on them — same guard as monitoring/_client.py.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
STARTUP_TIMEOUT_S = 90  # cold start = container boot + pandas/xgboost import + model unpickle

# A realistic encounter, using the same raw feature space the model was trained on
# (16 features; see FEATURE_COLS in src/train_baseline.py).
FULL_RECORD = {
    "time_in_hospital": 13,
    "num_lab_procedures": 68,
    "num_procedures": 2,
    "num_medications": 28,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 8,
    "admission_type_id": 2,
    "discharge_disposition_id": 1,
    "admission_source_id": 4,
    "diag_1_category": "Circulatory",
    "medical_specialty_grouped": "Missing",
    "race_white": 1,
    "gender": "Female",
    "age_ordinal": 8,
}

# A high-risk-looking encounter: many prior inpatient visits is the strongest single predictor
# in the baseline model, so this one should score noticeably higher than FULL_RECORD.
HIGH_RISK_RECORD = dict(
    FULL_RECORD,
    number_inpatient=6,
    number_emergency=3,
    time_in_hospital=9,
    diag_1_category="Diabetes",
    medical_specialty_grouped="InternalMedicine",
)

# Deliberately incomplete: the API fills unknown raw features with NA and unseen one-hot
# columns with 0, so this must still score rather than error.
PARTIAL_RECORD = {
    "time_in_hospital": 4,
    "num_medications": 11,
    "number_inpatient": 1,
    "gender": "Male",
}

# Every series the alert rules and the Grafana dashboard query.
REQUIRED_METRICS = [
    "predictions_total",
    "predict_requests_total",
    "prediction_latency_seconds_bucket",
    "predicted_positive_ratio",
    "data_drift_share",
    "drifted_columns_total",
    "drift_detected",
    "model_info",
]

PASS, FAIL = "[ OK ]", "[FAIL]"


def _get(path: str, timeout: int = 15) -> str:
    with urllib.request.urlopen(f"{API_URL}{path}", timeout=timeout) as r:
        return r.read().decode()


def _post_json(path: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        f"{API_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_for_health() -> dict:
    """Poll /health until the app finishes loading the model, or give up."""
    deadline = time.time() + STARTUP_TIMEOUT_S
    last_err = None
    while time.time() < deadline:
        try:
            return json.loads(_get("/health", timeout=5))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(2)
    raise SystemExit(f"{FAIL} /health never came up within {STARTUP_TIMEOUT_S}s: {last_err}")


def check(condition: bool, label: str, detail: str = "") -> bool:
    print(f"{PASS if condition else FAIL} {label}{(' — ' + detail) if detail else ''}")
    return condition


def main() -> int:
    print(f"Smoke-testing {API_URL}\n")
    results = []

    # --- 1. health -----------------------------------------------------------------------
    health = wait_for_health()
    results.append(check(health.get("status") == "ok", "/health status ok", json.dumps(health)))
    results.append(check(bool(health.get("model")), "model loaded", health.get("model", "none")))
    # Feature count depends on which model is served: the CatBoost champion's PyCaret
    # preprocessing yields 35 columns, the XGBoost baseline's get_dummies yields 36.
    expected_features = {"champion": 35, "baseline": 36}.get(health.get("model_kind"))
    results.append(
        check(
            health.get("n_features") == expected_features,
            f"{expected_features} engineered features ({health.get('model_kind')})",
            str(health.get("n_features")),
        )
    )
    if health.get("model_kind") == "champion":
        registry = health.get("mlflow_registry") or {}
        results.append(
            check(
                registry.get("alias") == "champion" and bool(registry.get("model_name")),
                "serving the MLflow-registered champion",
                f"{registry.get('model_name')} v{registry.get('version')} "
                f"(semantic {registry.get('semantic_version')})",
            )
        )

    # --- 2. predictions ------------------------------------------------------------------
    batch = [FULL_RECORD, HIGH_RISK_RECORD, PARTIAL_RECORD]
    body = _post_json("/predict", {"records": batch})
    results.append(check(body.get("n") == len(batch), "/predict returns one row per record"))
    results.append(
        check(
            len(body.get("predictions", [])) == len(batch)
            and len(body.get("probabilities", [])) == len(batch),
            "predictions + probabilities aligned to the batch",
        )
    )
    probs = body.get("probabilities", [])
    results.append(
        check(all(0.0 <= p <= 1.0 for p in probs), "probabilities within [0, 1]", str(probs))
    )
    results.append(
        check(
            all(p in (0, 1) for p in body.get("predictions", [])),
            "predictions are binary",
            str(body.get("predictions")),
        )
    )
    if len(probs) >= 2:
        # Sanity check that the model is discriminating, not returning a constant.
        results.append(
            check(
                probs[1] > probs[0],
                "high-risk record scores above the typical record",
                f"{probs[1]} > {probs[0]}",
            )
        )

    # Regression guard: a batch where several numeric features are missing from EVERY record.
    # Those columns arrive as all-<NA> object dtype, which XGBoost rejects unless preprocess()
    # coerces them back to numeric — this used to 500.
    sparse = _post_json("/predict", {"records": [PARTIAL_RECORD, PARTIAL_RECORD]})
    results.append(
        check(sparse.get("n") == 2, "batch with columns missing from every record still scores")
    )

    # An empty batch is a legitimate edge case (a monitoring run with nothing to send).
    empty = _post_json("/predict", {"records": []})
    results.append(check(empty.get("n") == 0, "empty batch handled without error"))

    # --- 3. metrics ----------------------------------------------------------------------
    metrics_text = _get("/metrics")
    missing = [m for m in REQUIRED_METRICS if m not in metrics_text]
    results.append(
        check(not missing, "all Prometheus series exported", f"missing: {missing}" if missing else "")
    )

    # --- 4. online drift detection ---------------------------------------------------------
    # The container should be able to diagnose its own inputs without the monitoring scripts.
    drift = json.loads(_get("/drift"))
    results.append(check(drift.get("enabled") is True, "online drift detection enabled",
                         drift.get("reason", "")))
    # A cold container hasn't served enough rows to have a verdict yet — "warming up" is correct
    # here, not a failure. Only assert the endpoint reports a coherent state.
    results.append(
        check(
            drift.get("status") in {"warming up", "healthy", "drift detected"},
            "drift endpoint reports a valid state",
            str(drift.get("status")),
        )
    )

    # --- summary -------------------------------------------------------------------------
    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    if passed != total:
        print("Deployment smoke test FAILED.")
        return 1
    print("Deployment smoke test PASSED — the API is serving and observable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
