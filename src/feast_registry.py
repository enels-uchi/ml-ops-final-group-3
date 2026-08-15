"""Feast feature store registration: write feature view definitions, materialize parquet, apply."""
import subprocess

import pandas as pd

FEATURES_PATH = "data/processed/diabetic_data_features.csv"
FEAST_REPO_PATH = "feast_repo/feature_repo"
FEATURE_DEFINITIONS_PATH = f"{FEAST_REPO_PATH}/feature_definitions.py"
PARQUET_PATH = f"{FEAST_REPO_PATH}/data/diabetes_features.parquet"

FEATURE_DEFINITIONS = '''
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64, String
from datetime import timedelta

encounter = Entity(name="encounter", join_keys=["encounter_id"])

diabetes_source = FileSource(
    path="data/diabetes_features.parquet",
    timestamp_field="event_timestamp",
)

diabetes_features_v1 = FeatureView(
    name="diabetes_features_v1",
    entities=[encounter],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="time_in_hospital", dtype=Int64),
        Field(name="num_lab_procedures", dtype=Int64),
        Field(name="num_procedures", dtype=Int64),
        Field(name="num_medications", dtype=Int64),
        Field(name="number_outpatient", dtype=Int64),
        Field(name="number_emergency", dtype=Int64),
        Field(name="number_inpatient", dtype=Int64),
        Field(name="number_diagnoses", dtype=Int64),
        Field(name="admission_type_id", dtype=Int64),
        Field(name="discharge_disposition_id", dtype=Int64),
        Field(name="admission_source_id", dtype=Int64),
        Field(name="diag_1_category", dtype=String),
        Field(name="medical_specialty_grouped", dtype=String),
        Field(name="race_white", dtype=Int64),
        Field(name="gender", dtype=String),
        Field(name="age_ordinal", dtype=Int64),
    ],
    source=diabetes_source,
)
'''


def register_features(
    features_path: str = FEATURES_PATH,
    feast_repo_path: str = FEAST_REPO_PATH,
) -> None:
    with open(FEATURE_DEFINITIONS_PATH, "w") as f:
        f.write(FEATURE_DEFINITIONS)
    print(f"[feast_registry] Wrote {FEATURE_DEFINITIONS_PATH}")

    df_features = pd.read_csv(features_path)
    df_features['event_timestamp'] = pd.Timestamp.now()
    df_features.to_parquet(PARQUET_PATH, index=False, engine="pyarrow")
    print(f"[feast_registry] Saved {df_features.shape[0]} rows to {PARQUET_PATH}")

    result = subprocess.run(
        ["feast", "apply"],
        cwd=feast_repo_path,
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"feast apply failed:\n{result.stderr}")


if __name__ == "__main__":
    register_features()
