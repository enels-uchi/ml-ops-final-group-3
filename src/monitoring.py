"""Evidently monitoring core (classic dashboard via evidently.legacy).

Uses Evidently's classic Report API (`evidently.legacy`) so the generated HTML matches the
familiar Evidently dashboard layout: a "Data Drift" summary banner ("Drift is detected for
X% of columns (N out of M)") over a per-column table (Column | Type | Reference/Current
Distribution | Stat Test | Drift Score), plus a Data Distribution section.

Provides reusable helpers to:
  * describe the feature schema to Evidently (`build_column_mapping`)
  * generate a data-drift report (reference vs current) and parse a compact result
  * generate a classification-performance report (needs prediction + target columns)

Drift result shape returned by `data_drift_report`:
    {"dataset_drift": bool, "drift_share": float, "n_drifted": int, "n_columns": int, "html": str}

`dataset_drift` is Evidently's own dataset-level verdict (drift when the share of drifted
columns crosses its default 0.5 threshold). `n_drifted >= 1` (any single column drifted) is
used by the anomaly-verification script as a more sensitive alerting trigger.
"""
import os
import sys
from pathlib import Path

import pandas as pd
from evidently.legacy.metric_preset import ClassificationPreset, DataDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_baseline import FEATURE_COLS, CATEGORICAL_COLS, TARGET_COL

REPORTS_DIR = "monitoring/reports"

# Raw-space feature typing for Evidently (the 16 model inputs, pre one-hot).
NUMERICAL_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]


def build_column_mapping(with_target: bool = False) -> ColumnMapping:
    """ColumnMapping over the raw feature space; optionally include target/prediction for perf."""
    cm = ColumnMapping()
    cm.numerical_features = list(NUMERICAL_COLS)
    cm.categorical_features = list(CATEGORICAL_COLS)
    if with_target:
        cm.target = TARGET_COL
        cm.prediction = "prediction"
    else:
        cm.target = None
        cm.prediction = None
    return cm


def _parse_drift(report: Report) -> dict:
    """Pull dataset_drift / counts / share out of the classic DataDrift result."""
    res = report.as_dict()["metrics"][0]["result"]
    drift_by_cols = res.get("drift_by_columns", {})
    return {
        "dataset_drift": bool(res.get("dataset_drift", False)),
        "n_drifted": int(res.get("number_of_drifted_columns", 0)),
        "drift_share": float(res.get("share_of_drifted_columns", 0.0)),
        "n_columns": int(res.get("number_of_columns", len(drift_by_cols) or len(FEATURE_COLS))),
    }


def data_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    out_html: str,
    performance: bool = False,
) -> dict:
    """Build a classic Evidently report over the raw features; save HTML; return parsed drift.

    When ``performance=True`` the report combines DataDriftPreset + ClassificationPreset into a
    single dashboard (Dataset Drift tiles -> Data Drift Summary table with reference/current
    distributions -> Model Quality current vs reference), matching the shared example layout. In
    that mode both frames must also carry ``TARGET_COL`` and a ``prediction`` column, which are
    added to the drift table too (so prediction/target drift shows alongside the features).
    """
    cols = list(FEATURE_COLS) + ([TARGET_COL, "prediction"] if performance else [])
    cm = build_column_mapping(with_target=performance)

    metrics = [DataDriftPreset()]
    if performance:
        metrics.append(ClassificationPreset())

    report = Report(metrics=metrics)
    report.run(reference_data=reference_df[cols], current_data=current_df[cols], column_mapping=cm)

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(out_html)

    parsed = _parse_drift(report)
    parsed["html"] = out_html
    return parsed
