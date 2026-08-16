"""Data prep for serving & monitoring.

The processed CSVs are DVC-tracked (pointer files only, no remote configured), but the full
engineered feature set is available in the Feast parquet. This module:

1. Materializes `data/processed/diabetic_data_features.csv` from the Feast parquet so the
   serving API and monitoring scripts share one on-disk source (matching the DVC pointer path).
2. Exposes `load_clean_split()` which reproduces *exactly* the 80/20 stratified split used in
   `src/train_baseline.py` (same FEATURE_COLS, RANDOM_STATE), returning the RAW feature space
   (before one-hot encoding). The clean test split is the "golden"/reference dataset for
   monitoring; the API performs one-hot encoding internally at predict time.
"""
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Make sibling modules in src/ importable whether launched directly or via uvicorn (src.serve_api).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the canonical column/split definitions from the training script (single source of truth).
from train_baseline import FEATURE_COLS, TARGET_COL, RANDOM_STATE, TEST_SIZE

PARQUET_PATH = "feast_repo/feature_repo/data/diabetes_features.parquet"
FEATURES_CSV = "data/processed/diabetic_data_features.csv"


def materialize_features_csv(parquet_path: str = PARQUET_PATH, out_csv: str = FEATURES_CSV) -> str:
    """Write the engineered features CSV from the Feast parquet if it isn't already present."""
    out = Path(out_csv)
    if out.exists():
        return str(out)
    df = pd.read_parquet(parquet_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[prepare_data] Wrote {df.shape[0]} rows -> {out}")
    return str(out)


def load_features() -> pd.DataFrame:
    """Load the engineered features frame (materializing the CSV from parquet if needed)."""
    if not Path(FEATURES_CSV).exists():
        materialize_features_csv()
    return pd.read_csv(FEATURES_CSV)


def load_clean_split():
    """Return the RAW-feature train/test split matching src/train_baseline.py.

    Returns
    -------
    X_train, X_test : pd.DataFrame  (columns = FEATURE_COLS, raw values, pre one-hot)
    y_train, y_test : pd.Series     (TARGET_COL)
    """
    df = load_features()
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    materialize_features_csv()
    Xtr, Xte, ytr, yte = load_clean_split()
    print(f"[prepare_data] train={Xtr.shape} test={Xte.shape} | positive rate test={yte.mean():.3f}")
