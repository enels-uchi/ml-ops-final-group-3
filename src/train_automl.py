"""AutoML training via PyCaret: engineered features -> registered MLflow model +
models/model_automl.pkl + metrics/metrics_automl.json.

Must run under venv-automl's Python (PyCaret doesn't support the main venv's
Python 3.13) -- invoked as a subprocess from the orchestration flow, not imported.
"""
import json
import os
import random
import warnings

from progress import heartbeat

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from pycaret.classification import (
    compare_models, create_model, pull, save_model,
    setup as pycaret_setup, tune_model,
)

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
FEATURES_PATH = "data/processed/diabetic_data_features.csv"
MODEL_OUTPUT_PATH = "models/model_automl"
METRICS_PATH = "metrics/metrics_automl.json"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "diabetes-readmission-automl-pycaret"
REGISTERED_MODEL_NAME = "diabetes-readmission-catboost"
SEMANTIC_VERSION = "1.0.0"

FEATURE_COLS = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'admission_type_id',
    'discharge_disposition_id', 'admission_source_id',
    'diag_1_category', 'medical_specialty_grouped', 'race_white',
    'gender', 'age_ordinal',
]
NUMERIC_FEATURES = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'admission_type_id',
    'discharge_disposition_id', 'admission_source_id',
    'race_white', 'age_ordinal',
]
CATEGORICAL_FEATURES = ['diag_1_category', 'medical_specialty_grouped', 'gender']
TARGET_COL = 'readmitted_binary'

# CatBoost's sklearn wrapper aliases 'n_estimators' to its native 'iterations' param,
# which breaks scikit-learn's clone() during hyperparameter search. Tuning 'iterations'
# directly avoids the bug entirely.
CATBOOST_TUNE_GRID = {
    'iterations': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200,
                   220, 240, 260, 280, 300],
    'depth': list(range(1, 11)),
    'eta': [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    'l2_leaf_reg': [1, 2, 3, 4, 5, 8, 9, 10, 20, 30, 100, 200],
    'random_strength': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'border_count': [254],
}


def train_automl(features_path: str = FEATURES_PATH) -> dict:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df_features = pd.read_csv(features_path)
    df_model = df_features[FEATURE_COLS + [TARGET_COL]].copy()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    pycaret_setup(
        data=df_model,
        target=TARGET_COL,
        train_size=0.8,
        session_id=RANDOM_SEED,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        fold=10,
        fold_strategy="stratifiedkfold",
        log_experiment=False,
        experiment_name=EXPERIMENT_NAME,
        log_plots=False,
        verbose=False,
    )

    print("[train_automl] Running compare_models() across the algorithm pool...")
    with heartbeat("compare_models"):
        compare_models(sort="AUC")
    leaderboard = pull()
    print(f"[train_automl] Evaluated {leaderboard.shape[0]} candidates. "
          f"Leader: {leaderboard.iloc[0]['Model']}")

    catboost_model = create_model('catboost', verbose=False)
    with heartbeat("tune_model"):
        tuned_model = tune_model(
            catboost_model, optimize="AUC", custom_grid=CATBOOST_TUNE_GRID, n_iter=25,
            verbose=False,
        )

    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    save_model(tuned_model, MODEL_OUTPUT_PATH)
    print(f"[train_automl] Saved {MODEL_OUTPUT_PATH}.pkl")

    row = leaderboard.loc['catboost']
    metrics = {
        "model": "CatBoost Classifier (PyCaret AutoML, default hyperparameters)",
        "f1_score": round(float(row['F1']), 4),
        "auc_roc": round(float(row['AUC']), 4),
        "precision": round(float(row['Prec.']), 4),
        "recall": round(float(row['Recall']), 4),
        "accuracy": round(float(row['Accuracy']), 4),
    }

    with mlflow.start_run(run_name="pycaret_catboost_final") as run:
        mlflow.set_tags({"platform": "PyCaret", "target": TARGET_COL, "stage": "final"})
        mlflow.log_param("model_type", "CatBoostClassifier")
        for key in ("f1_score", "auc_roc", "precision", "recall", "accuracy"):
            mlflow.log_metric(key, metrics[key])
        mlflow.sklearn.log_model(tuned_model, artifact_path="model", serialization_format="cloudpickle")
        registered = mlflow.register_model(f"runs:/{run.info.run_id}/model", REGISTERED_MODEL_NAME)

    client = MlflowClient()
    client.set_model_version_tag(REGISTERED_MODEL_NAME, registered.version, "semantic_version", SEMANTIC_VERSION)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", registered.version)

    metrics["mlflow_registry"] = {
        "model_name": REGISTERED_MODEL_NAME,
        "version": int(registered.version),
        "semantic_version": SEMANTIC_VERSION,
        "alias": "champion",
        "tracking_uri": MLFLOW_TRACKING_URI,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train_automl] Saved {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    train_automl()
