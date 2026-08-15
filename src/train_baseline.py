"""Baseline model training: engineered features -> models/model_baseline.pkl + metrics/metrics_baseline.json"""
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

FEATURES_PATH = "data/processed/diabetic_data_features.csv"
MODELS_DIR = "models"
METRICS_DIR = "metrics"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLS = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'admission_type_id',
    'discharge_disposition_id', 'admission_source_id',
    'diag_1_category', 'medical_specialty_grouped', 'race_white',
    'gender', 'age_ordinal',
]
CATEGORICAL_COLS = ['diag_1_category', 'medical_specialty_grouped', 'gender']
TARGET_COL = 'readmitted_binary'


def train_baseline(
    features_path: str = FEATURES_PATH,
    models_dir: str = MODELS_DIR,
    metrics_dir: str = METRICS_DIR,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> dict:
    df_features = pd.read_csv(features_path)

    X = df_features[FEATURE_COLS]
    y = df_features[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLS, dtype=int)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_COLS, dtype=int)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    print(f"[train_baseline] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    model = XGBClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)

    Path(models_dir).mkdir(exist_ok=True)
    Path(metrics_dir).mkdir(exist_ok=True)

    with open(f"{models_dir}/model_baseline.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"[train_baseline] Saved {models_dir}/model_baseline.pkl")

    metrics = {
        "model": "XGBoost baseline (default params)",
        "f1_score": round(f1_score(y_test, y_pred), 3),
        "auc_roc": round(roc_auc_score(y_test, y_pred_proba), 3),
        "precision": round(report["1"]["precision"], 3),
        "recall": round(report["1"]["recall"], 3),
        "accuracy": round(report["accuracy"], 3),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
    }

    with open(f"{metrics_dir}/metrics_baseline.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train_baseline] Saved {metrics_dir}/metrics_baseline.json")
    print(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    train_baseline()
