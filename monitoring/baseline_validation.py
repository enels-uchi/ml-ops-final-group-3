"""Baseline validation: pass the CLEAN test set through the deployed API and validate.

Two checks:
  1. Performance parity — predictions from the live API reproduce the training-time metrics in
     metrics/metrics_baseline.json (F1/AUC/precision/recall within tolerance).
  2. Monitoring baseline — an Evidently data-drift report of (reference = clean train) vs
     (current = clean test) shows NO dataset drift. This is the healthy reference the
     anomaly-verification step is compared against.

Prereq: the API must be running (venv/bin/uvicorn src.serve_api:app --port 8000).
Run:    venv/bin/python monitoring/baseline_validation.py
"""
import json
import warnings

from _client import api_health, predict_frame, push_drift  # noqa: E402 (path bootstrap)
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from prepare_data import load_clean_split
from monitoring import data_drift_report

warnings.filterwarnings("ignore")

METRICS_BASELINE = "metrics/metrics_baseline.json"
TOLERANCE = 0.02  # absolute tolerance vs. recorded training metrics
REPORT = "monitoring/reports/00_baseline.html"


def main() -> int:
    print(f"[baseline] API health: {api_health()}")

    X_train, X_test, y_train, y_test = load_clean_split()
    print(f"[baseline] clean test set: {X_test.shape}")

    # ---- 1. Performance parity via the live API -----------------------------------------
    preds, probas = predict_frame(X_test)
    live = {
        "f1_score": round(f1_score(y_test, preds), 3),
        "auc_roc": round(roc_auc_score(y_test, probas), 3),
        "precision": round(precision_score(y_test, preds), 3),
        "recall": round(recall_score(y_test, preds), 3),
    }
    with open(METRICS_BASELINE) as f:
        recorded = json.load(f)

    print("\n[baseline] Live API vs recorded training metrics:")
    parity_ok = True
    for k, live_v in live.items():
        rec_v = recorded[k]
        delta = abs(live_v - rec_v)
        ok = delta <= TOLERANCE
        parity_ok &= ok
        print(f"  {k:10s} live={live_v:.3f}  recorded={rec_v:.3f}  Δ={delta:.3f}  {'OK' if ok else 'MISMATCH'}")

    # ---- 2. Monitoring baseline: one combined drift + model-quality report ----------------
    # Reference = clean train, current = clean test; both carry target + prediction so the
    # single report shows Dataset Drift, the Data Drift Summary table, and Model Quality.
    ref = X_train.copy()
    ref[y_train.name] = y_train.values
    ref_preds, _ = predict_frame(X_train)
    ref["prediction"] = ref_preds
    cur = X_test.copy()
    cur[y_test.name] = y_test.values
    cur["prediction"] = preds

    drift = data_drift_report(ref, cur, REPORT, performance=True)
    print(
        f"\n[baseline] Drift (clean train vs clean test): drifted_cols={drift['n_drifted']}/"
        f"{drift['n_columns']} share={drift['drift_share']:.3f} dataset_drift={drift['dataset_drift']}"
    )
    print(f"[baseline] Combined drift + performance report -> {drift['html']}")
    push_drift("baseline_clean", drift["dataset_drift"], drift["drift_share"], drift["n_drifted"])

    no_drift_ok = not drift["dataset_drift"]
    print("\n" + "=" * 60)
    print(f"  Performance parity : {'PASS' if parity_ok else 'FAIL'}")
    print(f"  Clean = no drift   : {'PASS' if no_drift_ok else 'FAIL'}")
    print("=" * 60)
    return 0 if (parity_ok and no_drift_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
