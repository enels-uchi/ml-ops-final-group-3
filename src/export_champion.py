"""Export the registered AutoML champion into a container-friendly serving artifact.

The champion (`models/model_automl.pkl`) is a PyCaret pipeline. Serving it as-is would mean
installing PyCaret in the inference image, which is impossible: PyCaret does not support the
container's Python 3.12, and its dependency tree dwarfs the model. Worse, the pipeline's fitted
transformers are `scikit-learn` 1.4 / `category_encoders` objects, and unpickling those under the
container's scikit-learn 1.9 is not safe across that version gap.

So instead of shipping the objects, this script ships their *fitted parameters* as plain JSON and
the CatBoost model in CatBoost's own version-stable binary format. `src/champion_preprocess.py`
replays the identical transform using nothing but pandas. The container then needs only
pandas + catboost — no PyCaret, no sklearn, no category_encoders.

This is an export, not a retrain: it reads `models/model_automl.pkl` read-only and never touches
the AutoML pipeline or the MLflow registry.

**The export is only written if it reproduces the pipeline exactly.** The script scores the full
engineered dataset through both the original pipeline and the replay, and aborts unless the
predicted probabilities agree to within 1e-9.

Run in an environment that has PyCaret (the AutoML env, Python 3.11):

    python src/export_champion.py
"""
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_baseline import FEATURE_COLS  # noqa: E402

SOURCE_PKL = "models/model_automl.pkl"
OUT_DIR = Path("models/champion")
TOLERANCE = 1e-9


def _ordinal_map(transformer) -> dict:
    """category -> integer code, from a category_encoders OrdinalEncoder mapping."""
    entry = transformer.mapping[0]
    series = entry["mapping"]
    return {
        str(k): int(v)
        for k, v in series.items()
        if not (isinstance(k, float) and np.isnan(k)) and int(v) >= 0
    }


def _onehot_spec(transformer) -> dict:
    """Per-column ordered category list, reconstructed from the one-hot output names."""
    spec = {}
    for name in transformer.feature_names_out_:
        for col in ("diag_1_category", "medical_specialty_grouped"):
            if name.startswith(col + "_"):
                spec.setdefault(col, []).append(name[len(col) + 1:])
                break
    return spec


def export() -> dict:
    print(f"[export] loading {SOURCE_PKL}")
    pipe = joblib.load(SOURCE_PKL)
    steps = dict(pipe.steps)

    num_step = steps["numerical_imputer"]
    cat_step = steps["categorical_imputer"]
    ord_step = steps["ordinal_encoding"]
    ohe_step = steps["onehot_encoding"]
    model = pipe.steps[-1][1]

    numeric_cols = list(num_step.include)
    categorical_cols = list(cat_step.include)

    preprocessing = {
        # SimpleImputer(strategy="mean") over the numeric features.
        "numeric_columns": numeric_cols,
        "numeric_fill": {
            c: float(v) for c, v in zip(numeric_cols, np.asarray(num_step.transformer.statistics_))
        },
        # SimpleImputer(strategy="most_frequent") over the categoricals.
        "categorical_columns": categorical_cols,
        "categorical_fill": {
            c: str(v) for c, v in zip(categorical_cols, np.asarray(cat_step.transformer.statistics_))
        },
        # gender -> integer code. Unseen categories encode to -1, matching handle_unknown="value".
        "ordinal_column": list(ord_step.include)[0],
        "ordinal_map": _ordinal_map(ord_step.transformer),
        # One-hot expansion; an unseen category yields an all-zero row for that block.
        "onehot": _onehot_spec(ohe_step.transformer),
        # The exact column order the CatBoost model was fitted on.
        "output_columns": list(pipe[:-1].transform(pd.DataFrame([{c: np.nan for c in FEATURE_COLS}])).columns),
    }

    # ---- verify the replay before writing anything ------------------------------------------
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from champion_preprocess import ChampionPreprocessor
    from prepare_data import load_features

    df = load_features()
    X_raw = df[FEATURE_COLS]
    print(f"[export] verifying replay against the pipeline on {len(X_raw)} rows...")

    expected = pipe.predict_proba(X_raw)[:, 1]
    replayed_frame = ChampionPreprocessor(preprocessing).transform(X_raw)
    actual = model.predict_proba(replayed_frame)[:, 1]

    max_diff = float(np.max(np.abs(expected - actual)))
    print(f"[export] max probability difference: {max_diff:.3e}")
    if max_diff > TOLERANCE:
        raise SystemExit(
            f"ABORT: replay does not reproduce the pipeline (max diff {max_diff:.3e} > {TOLERANCE}). "
            f"Nothing was written."
        )
    print("[export] replay is exact.")

    # ---- write the artifact ------------------------------------------------------------------
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    model.save_model(str(OUT_DIR / "catboost_model.cbm"))
    with open(OUT_DIR / "preprocessing.json", "w") as f:
        json.dump(preprocessing, f, indent=2)

    automl_metrics = json.load(open("metrics/metrics_automl.json"))
    metadata = {
        "model": "CatBoost Classifier (PyCaret AutoML champion)",
        "exported_from": SOURCE_PKL,
        "mlflow_registry": automl_metrics.get("mlflow_registry", {}),
        "n_features_in": len(FEATURE_COLS),
        "n_features_out": len(preprocessing["output_columns"]),
        "verified_max_prob_diff": max_diff,
        "note": (
            "Parameters extracted from the PyCaret pipeline and replayed by "
            "src/champion_preprocess.py; verified to reproduce pipeline.predict_proba exactly."
        ),
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    total_kb = sum(p.stat().st_size for p in OUT_DIR.iterdir()) / 1024
    print(f"[export] wrote {OUT_DIR}/ ({total_kb:.0f} KB): "
          f"{', '.join(sorted(p.name for p in OUT_DIR.iterdir()))}")
    return metadata


if __name__ == "__main__":
    export()
