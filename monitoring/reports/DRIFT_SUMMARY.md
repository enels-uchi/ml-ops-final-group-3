# Drift & Anomaly Verification Summary

_Generated 2026-08-16 16:23_

Reference: clean training split. Clean baseline predicted-positive ratio: **0.244**.
Each scenario below was corrupted by `drift_simulation.py`, sent to the deployed API, and analysed with EvidentlyAI.

| Scenario | Drifted cols | Drift share | Dataset drift | Pred+ ratio (Δ) | Alerts | Report |
|---|---|---|---|---|---|---|
| `column_swap` | 4/18 | 0.222 | False | 0.161 (+0.083) | 4_COLS_DRIFTED | [html](column_swap.html) |
| `dist_shift` | 5/18 | 0.278 | False | 0.921 (+0.678) | 5_COLS_DRIFTED, PRED_SHIFT_0.68 | [html](dist_shift.html) |
| `out_of_bounds` | 5/18 | 0.278 | False | 0.059 (+0.185) | 5_COLS_DRIFTED, PRED_SHIFT_0.19 | [html](out_of_bounds.html) |
| `schema_change` | 3/18 | 0.167 | False | 0.205 (+0.039) | 3_COLS_DRIFTED | [html](schema_change.html) |

## What each scenario injects
- **out_of_bounds** — numeric features pushed to impossible magnitudes (`time_in_hospital=9999`, lab procedures ×50, etc.).
- **column_swap** — feature values wired into the wrong columns.
- **schema_change** — unseen categorical values (`gender=UNKNOWN`, `diag_1_category=ZZZ`, a brand-new specialty).
- **dist_shift** — realistic population shift: older, longer-staying, sicker patients.

## How the monitor catches it
EvidentlyAI compares each column's distribution against the clean reference (K-S test for numeric, chi-square for categorical). A column is flagged when its p-value < 0.05; the dataset is flagged when the share of drifted columns ≥ 0.5. The API also exposes the live state on Prometheus `/metrics` (`data_drift_share`, `drifted_columns_total`, `drift_detected`) and the predicted-positive ratio, so a monitoring system (or Grafana scraping `/metrics`) can alert automatically.
