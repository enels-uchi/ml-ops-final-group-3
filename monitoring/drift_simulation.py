"""Drift / stress-test simulation.

Takes the CLEAN test set and produces artificially corrupted copies under monitoring/drifted/,
covering the anomaly classes called for in the assignment:

  1. out_of_bounds   - inject extreme, physically-impossible numeric values
  2. column_swap     - swap two feature columns (wrong data wired to wrong field)
  3. schema_change   - inject unseen categorical values + a shifted categorical mix
  4. dist_shift      - shift a distribution (age skewed old, hospital stays inflated)

Each corrupted set keeps the same schema/columns (incl. the target) so it can be sent to the
API and compared to the clean reference by anomaly_verification.py.

Run: venv/bin/python monitoring/drift_simulation.py
"""
import os

import numpy as np

from _client import REPO_ROOT  # noqa: F401  (triggers path bootstrap + chdir to repo root)
from prepare_data import load_clean_split

DRIFTED_DIR = "monitoring/drifted"
RANDOM_STATE = 42


def _base_frame():
    """Clean test features + target as one frame (the thing we corrupt)."""
    _, X_test, _, y_test = load_clean_split()
    df = X_test.copy()
    df[y_test.name] = y_test.values
    return df


def make_out_of_bounds(df, rng):
    """Blow numeric ranges far past anything seen in training."""
    out = df.copy()
    out["num_lab_procedures"] = out["num_lab_procedures"] * 50 + 5000
    out["time_in_hospital"] = 9999
    out["num_medications"] = rng.integers(500, 1000, size=len(out))
    out["number_diagnoses"] = out["number_diagnoses"] + 900
    return out


def make_column_swap(df, rng):
    """Wire values into the wrong fields (num_medications <-> num_lab_procedures, etc.)."""
    out = df.copy()
    out["num_medications"], out["num_lab_procedures"] = (
        df["num_lab_procedures"].values,
        df["num_medications"].values,
    )
    out["num_procedures"], out["time_in_hospital"] = (
        df["time_in_hospital"].values,
        df["num_procedures"].values,
    )
    return out


def make_schema_change(df, rng):
    """Inject unseen categorical values and flip the categorical mix."""
    out = df.copy()
    n = len(out)
    # Unseen categories the model/encoder never saw at train time.
    unknown_gender = rng.random(n) < 0.4
    out.loc[unknown_gender, "gender"] = "UNKNOWN"
    out["diag_1_category"] = rng.choice(["ZZZ", "NEW_CODE", "Unmapped"], size=n)
    out["medical_specialty_grouped"] = "BrandNewSpecialty"
    return out


def make_dist_shift(df, rng):
    """No new values, but a real distribution shift: population skews old + sicker."""
    out = df.copy()
    # Push age toward the oldest bins, hospital stays and inpatient counts upward.
    out["age_ordinal"] = np.clip(out["age_ordinal"] + rng.integers(2, 5, size=len(out)), 0, 9)
    out["time_in_hospital"] = np.clip(out["time_in_hospital"] + rng.integers(4, 10, size=len(out)), 1, 14)
    out["number_inpatient"] = out["number_inpatient"] + rng.integers(2, 6, size=len(out))
    out["number_emergency"] = out["number_emergency"] + rng.integers(1, 4, size=len(out))
    return out


SCENARIOS = {
    "out_of_bounds": make_out_of_bounds,
    "column_swap": make_column_swap,
    "schema_change": make_schema_change,
    "dist_shift": make_dist_shift,
}


def main():
    os.makedirs(DRIFTED_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    df = _base_frame()
    print(f"[drift_sim] clean test base: {df.shape}")
    for name, fn in SCENARIOS.items():
        corrupted = fn(df, rng)
        path = os.path.join(DRIFTED_DIR, f"{name}.csv")
        corrupted.to_csv(path, index=False)
        print(f"[drift_sim] {name:14s} -> {path}  ({corrupted.shape[0]} rows)")
    print(f"[drift_sim] Done. {len(SCENARIOS)} drifted datasets written to {DRIFTED_DIR}/")


if __name__ == "__main__":
    main()
