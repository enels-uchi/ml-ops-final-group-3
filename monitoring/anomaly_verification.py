"""Anomaly verification: send drifted data to the API and confirm monitoring catches it.

For each drifted dataset produced by drift_simulation.py:
  * POST it to /predict (observe how the model's predicted-positive ratio moves vs the clean baseline)
  * run an Evidently data-drift report (reference = clean train, current = drifted)
  * push the drift result to the API so it surfaces on /metrics
  * fire an ALERT when drift is detected (any drifted column, or dataset-level drift, or a large
    swing in predicted-positive ratio)
  * write a per-scenario HTML report and a consolidated DRIFT_SUMMARY.md

Prereqs: API running + `python monitoring/drift_simulation.py` already run.
Run:     venv/bin/python monitoring/anomaly_verification.py            # full canonical run
         python monitoring/anomaly_verification.py --scenario dist_shift --sample 5000

The full run scores ~126k rows and builds four Evidently reports, which takes several minutes —
fine as the recorded deliverable, too slow to run live in a demo. `--scenario` limits it to one
corruption and `--sample` subsamples the reference/current frames, together bringing a run down
to seconds. Sampling stays above Evidently's 1000-row cutoff so it still uses the same K-S /
chi-square tests as the full run.

Either flag marks the run as a *demo* run: the HTML goes to monitoring/reports/demo/ and the
canonical DRIFT_SUMMARY.md is left untouched, so a quick rehearsal can never overwrite the
committed deliverable with sampled numbers.
"""
import argparse
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


def _parse_args():
    ap = argparse.ArgumentParser(description="Verify the deployed model's monitoring catches drift.")
    ap.add_argument("--scenario", default=None,
                    help="run a single scenario (e.g. dist_shift) instead of all four")
    ap.add_argument("--sample", type=int, default=0,
                    help="subsample reference/current to N rows for a fast run (0 = full data)")
    return ap.parse_args()


def main():
    args = _parse_args()
    # Any narrowing flag means this is a rehearsal, not the canonical deliverable run.
    demo_mode = bool(args.scenario or args.sample)
    reports_dir = os.path.join(REPORTS_DIR, "demo") if demo_mode else REPORTS_DIR
    os.makedirs(reports_dir, exist_ok=True)

    print(f"[anomaly] API health: {api_health()}")
    X_train, X_test, y_train, _ = load_clean_split()

    if args.sample:
        # random_state fixed so repeated demo runs give the same numbers on camera.
        n_ref = min(args.sample, len(X_train))
        idx = X_train.sample(n_ref, random_state=42).index
        X_train, y_train = X_train.loc[idx], y_train.loc[idx]
        X_test = X_test.sample(min(args.sample, len(X_test)), random_state=42)
        print(f"[anomaly] DEMO MODE: sampled to {n_ref} reference / {len(X_test)} current rows")

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
    if args.scenario:
        drifted_files = [p for p in drifted_files
                         if os.path.splitext(os.path.basename(p))[0] == args.scenario]
        if not drifted_files:
            available = sorted(os.path.splitext(os.path.basename(p))[0]
                               for p in glob.glob(os.path.join(DRIFTED_DIR, "*.csv")))
            raise SystemExit(f"No scenario named '{args.scenario}'. Available: {available}")

    rows = []
    for path in drifted_files:
        scenario = os.path.splitext(os.path.basename(path))[0]
        drifted = pd.read_csv(path)
        if args.sample:
            drifted = drifted.sample(min(args.sample, len(drifted)), random_state=42)

        # 1. Model behaviour on drifted data
        preds, _ = predict_frame(drifted)
        pos_ratio = float(np.mean(preds))
        pos_delta = abs(pos_ratio - baseline_pos)

        # 2. Combined drift + model-quality report: clean reference vs drifted current.
        cur = drifted.copy()
        cur["prediction"] = preds
        out_html = os.path.join(reports_dir, f"{scenario}.html")
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

    caught = sum(1 for r in rows if r["alerts"] != "-")
    if demo_mode:
        # Never let a sampled/partial rehearsal overwrite the committed summary.
        print(f"\n[anomaly] {caught}/{len(rows)} scenario(s) raised alerts. "
              f"Demo run — report in {reports_dir}/, DRIFT_SUMMARY.md left untouched.")
    else:
        _write_summary(rows, baseline_pos)
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
    # encoding is explicit because the summary contains non-ASCII (the Δ column header): on
    # Windows open() defaults to cp1252 and the write dies with UnicodeEncodeError.
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
