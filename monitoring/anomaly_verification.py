"""Anomaly verification: send drifted data to the API and confirm monitoring catches it.

For each drifted dataset produced by drift_simulation.py:
  * POST it to /predict (observe how the model's predicted-positive ratio moves vs the clean baseline)
  * run an Evidently data-drift report (reference = clean train, current = drifted)
  * push the drift result to the API so it surfaces on /metrics
  * fire an ALERT when drift is detected (any drifted column, or dataset-level drift, or a large
    swing in predicted-positive ratio)
  * write a per-scenario HTML report and a consolidated DRIFT_SUMMARY.md

Prereqs: API running + `python monitoring/drift_simulation.py` already run.
Run:     venv/bin/python monitoring/anomaly_verification.py
"""
import glob
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from _client import api_health, predict_frame, push_drift
from prepare_data import load_clean_split
from monitoring import data_drift_report

warnings.filterwarnings("ignore")

DRIFTED_DIR = "monitoring/drifted"
REPORTS_DIR = "monitoring/reports"
SUMMARY_MD = "monitoring/reports/DRIFT_SUMMARY.md"
POS_RATIO_ALERT_DELTA = 0.10  # alert if predicted-positive ratio moves > 10pp vs clean baseline


def _clean_baseline_pos_ratio(X_test):
    preds, _ = predict_frame(X_test)
    return float(np.mean(preds))


def main():
    print(f"[anomaly] API health: {api_health()}")
    X_train, X_test, y_train, _ = load_clean_split()

    # Reference frame (built once): clean train features + target + model predictions. Reused as
    # the reference for every scenario's combined drift + model-quality report.
    ref = X_train.copy()
    ref[y_train.name] = y_train.values
    ref_preds, _ = predict_frame(X_train)
    ref["prediction"] = ref_preds

    baseline_pos = _clean_baseline_pos_ratio(X_test)
    print(f"[anomaly] clean baseline predicted-positive ratio: {baseline_pos:.3f}\n")

    drifted_files = sorted(glob.glob(os.path.join(DRIFTED_DIR, "*.csv")))
    if not drifted_files:
        raise SystemExit("No drifted datasets found. Run monitoring/drift_simulation.py first.")

    rows = []
    for path in drifted_files:
        scenario = os.path.splitext(os.path.basename(path))[0]
        drifted = pd.read_csv(path)

        # 1. Model behaviour on drifted data
        preds, _ = predict_frame(drifted)
        pos_ratio = float(np.mean(preds))
        pos_delta = abs(pos_ratio - baseline_pos)

        # 2. Combined drift + model-quality report: clean reference vs drifted current.
        cur = drifted.copy()
        cur["prediction"] = preds
        out_html = os.path.join(REPORTS_DIR, f"{scenario}.html")
        drift = data_drift_report(ref, cur, out_html, performance=True)

        # 3. Surface on /metrics
        push_drift(scenario, drift["dataset_drift"], drift["drift_share"], drift["n_drifted"])

        # 4. Alerting logic
        alerts = []
        if drift["dataset_drift"]:
            alerts.append("DATASET_DRIFT")
        if drift["n_drifted"] >= 1:
            alerts.append(f"{drift['n_drifted']}_COLS_DRIFTED")
        if pos_delta > POS_RATIO_ALERT_DELTA:
            alerts.append(f"PRED_SHIFT_{pos_delta:.2f}")
        fired = bool(alerts)

        banner = "🚨 ALERT" if fired else "✅ ok"
        print(
            f"[anomaly] {scenario:14s} {banner}  drifted={drift['n_drifted']}/{drift['n_columns']} "
            f"share={drift['drift_share']:.2f} dataset_drift={drift['dataset_drift']} "
            f"pos_ratio={pos_ratio:.3f} (Δ{pos_delta:+.3f})  -> {out_html}"
        )

        rows.append(
            {
                "scenario": scenario,
                "n_drifted": drift["n_drifted"],
                "n_columns": drift["n_columns"],
                "drift_share": round(drift["drift_share"], 3),
                "dataset_drift": drift["dataset_drift"],
                "pos_ratio": round(pos_ratio, 3),
                "pos_delta": round(pos_delta, 3),
                "alerts": ", ".join(alerts) if alerts else "-",
                "report": os.path.basename(out_html),
            }
        )

    _write_summary(rows, baseline_pos)
    caught = sum(1 for r in rows if r["alerts"] != "-")
    print(f"\n[anomaly] {caught}/{len(rows)} drift scenarios raised alerts. Summary -> {SUMMARY_MD}")
    return 0 if caught == len(rows) else 1


def _write_summary(rows, baseline_pos):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    lines = [
        "# Drift & Anomaly Verification Summary",
        "",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"Reference: clean training split. Clean baseline predicted-positive ratio: "
        f"**{baseline_pos:.3f}**.",
        "Each scenario below was corrupted by `drift_simulation.py`, sent to the deployed API, "
        "and analysed with EvidentlyAI.",
        "",
        "| Scenario | Drifted cols | Drift share | Dataset drift | Pred+ ratio (Δ) | Alerts | Report |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['scenario']}` | {r['n_drifted']}/{r['n_columns']} | {r['drift_share']} | "
            f"{r['dataset_drift']} | {r['pos_ratio']} ({r['pos_delta']:+}) | {r['alerts']} | "
            f"[html]({r['report']}) |"
        )
    lines += [
        "",
        "## What each scenario injects",
        "- **out_of_bounds** — numeric features pushed to impossible magnitudes "
        "(`time_in_hospital=9999`, lab procedures ×50, etc.).",
        "- **column_swap** — feature values wired into the wrong columns.",
        "- **schema_change** — unseen categorical values (`gender=UNKNOWN`, "
        "`diag_1_category=ZZZ`, a brand-new specialty).",
        "- **dist_shift** — realistic population shift: older, longer-staying, sicker patients.",
        "",
        "## How the monitor catches it",
        "EvidentlyAI compares each column's distribution against the clean reference (K-S test for "
        "numeric, chi-square for categorical). A column is flagged when its p-value < 0.05; the "
        "dataset is flagged when the share of drifted columns ≥ 0.5. The API also exposes the live "
        "state on Prometheus `/metrics` (`data_drift_share`, `drifted_columns_total`, "
        "`drift_detected`) and the predicted-positive ratio, so a monitoring system (or Grafana "
        "scraping `/metrics`) can alert automatically.",
    ]
    with open(SUMMARY_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
