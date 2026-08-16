"""Baseline validation: pass the CLEAN test set through the deployed API and validate.

Two checks:
  1. Performance parity — predictions from the live API reproduce the certified metrics for
     whichever model is deployed (F1/AUC/precision/recall within tolerance).
  2. Monitoring baseline — an Evidently data-drift report of (reference = clean train) vs
     (current = clean test) shows NO dataset drift. This is the healthy reference the
     anomaly-verification step is compared against.

**Which metrics parity compares against depends on the deployed model**, read from /health:

  baseline   metrics/metrics_baseline.json    — XGBoost's true held-out test-set scores, produced
                                                by src/train_baseline.py on this exact split.
  champion   metrics/metrics_deployment.json  — the CatBoost champion's scores measured on this
                                                split at deployment time, via `--record`.

The champion needs its own file because metrics_automl.json is NOT comparable to a held-out
evaluation: those numbers come from `leaderboard.loc['catboost']`, i.e. PyCaret's 10-fold
cross-validation means over its own training split. PyCaret also partitions the data itself, so
its split does not coincide with src/train_baseline.py's — which is why the champion scores
markedly higher here than its recorded CV figures. See the `caveat` field written into
metrics_deployment.json, and the README section on the served model.

What parity therefore certifies for the champion is deployment integrity — the container
reproduces the exact numbers the model was certified at — not a like-for-like comparison against
the AutoML leaderboard.

Prereq: the API must be running (docker compose up -d, or uvicorn src.serve_api:app --port 8000).
Run:    python monitoring/baseline_validation.py
        python monitoring/baseline_validation.py --record   # (re)certify the deployed champion
"""
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

from _client import api_health, predict_frame, push_drift  # noqa: E402 (path bootstrap)
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from prepare_data import load_clean_split
from monitoring import data_drift_report

warnings.filterwarnings("ignore")

METRICS_BASELINE = "metrics/metrics_baseline.json"
METRICS_DEPLOYMENT = "metrics/metrics_deployment.json"
TOLERANCE = 0.02  # absolute tolerance vs. recorded metrics
REPORT = "monitoring/reports/00_baseline.html"

CHAMPION_CAVEAT = (
    "Measured by monitoring/baseline_validation.py --record on the src/train_baseline.py "
    "80/20 split. NOT comparable to metrics/metrics_automl.json, whose figures are PyCaret's "
    "10-fold cross-validation means over its own independently-partitioned training split; "
    "because that partition differs from this one, part of this test set was seen during the "
    "champion's training, so these numbers are optimistic as a generalisation estimate. They "
    "serve as a deployment-integrity reference: the container must reproduce them exactly."
)


def _record_deployment_metrics(live: dict, health: dict, n_rows: int) -> None:
    payload = {
        "model": health.get("model", "unknown"),
        "model_version": health.get("model_version"),
        "model_kind": health.get("model_kind"),
        "mlflow_registry": health.get("mlflow_registry"),
        "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "evaluated_on": f"clean held-out test split ({n_rows} rows)",
        **live,
        "caveat": CHAMPION_CAVEAT,
    }
    Path(METRICS_DEPLOYMENT).parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DEPLOYMENT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[baseline] recorded deployment metrics -> {METRICS_DEPLOYMENT}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="store_true",
                    help="write the live metrics as the deployed model's certified reference")
    args = ap.parse_args()

    health = api_health()
    print(f"[baseline] API health: {health}")
    model_kind = health.get("model_kind", "baseline")

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

    reference_path = METRICS_BASELINE if model_kind == "baseline" else METRICS_DEPLOYMENT
    if args.record and model_kind != "baseline":
        _record_deployment_metrics(live, health, len(X_test))
    if not Path(reference_path).exists():
        raise SystemExit(
            f"No certified metrics at {reference_path} for model_kind={model_kind}. "
            f"Run `python monitoring/baseline_validation.py --record` once to certify the "
            f"currently deployed model."
        )
    with open(reference_path) as f:
        recorded = json.load(f)

    print(f"\n[baseline] Live API vs certified metrics ({reference_path}):")
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
