"""Generate the drift reference sample that gets baked into the serving image.

The online detector in src/drift_detector.py needs something to compare live traffic against.
That reference has to be the *clean training split* — the same reference the offline Evidently
reports use — so the live numbers and the offline reports tell the same story.

The full training split is 56k rows, which is both unnecessary for a two-sample test and too
bulky to ship in a container. A fixed random sample is statistically sufficient (K-S and
chi-square are comparing distributions, not counting rows) and keeps the artifact small.

Run once, and commit the output — the Dockerfile copies models/ into the image:

    python src/build_drift_reference.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_data import load_clean_split
from train_baseline import FEATURE_COLS

OUT_PATH = "models/drift_reference.csv"
N_ROWS = 5000
RANDOM_STATE = 42  # fixed so the artifact is reproducible and diffs stay empty across rebuilds


def build(out_path: str = OUT_PATH, n_rows: int = N_ROWS) -> str:
    X_train, _, _, _ = load_clean_split()
    sample = X_train.sample(min(n_rows, len(X_train)), random_state=RANDOM_STATE)
    sample = sample[FEATURE_COLS]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sample.to_csv(out_path, index=False)
    print(f"[drift_ref] {len(sample)} rows x {sample.shape[1]} cols -> {out_path}")
    print(f"[drift_ref] size: {os.path.getsize(out_path) / 1024:.0f} KB")
    return out_path


if __name__ == "__main__":
    build()
