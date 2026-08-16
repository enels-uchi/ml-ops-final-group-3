"""Prefect flow: orchestrates the full diabetes-readmission pipeline end to end.

Clean -> features -> Feast -> baseline -> AutoML -> deployment artifacts.

The last stage matters for reproducibility: retraining changes the model, so the artifacts the
serving container is built from have to be regenerated in the same run. Otherwise a fresh
pipeline run leaves models/champion/ and models/drift_reference.csv describing the *previous*
model, and the container would serve stale weights against a stale drift reference.

Run from the repo root: python src/pipeline_flow.py
"""
import subprocess

from prefect import flow, task

from data_cleaning import clean_data
from feature_engineering import engineer_features
from feast_registry import register_features
from train_baseline import train_baseline
from build_drift_reference import build as build_drift_reference

AUTOML_VENV_PYTHON = "venv-automl/bin/python"
AUTOML_SCRIPT = "src/train_automl.py"
EXPORT_CHAMPION_SCRIPT = "src/export_champion.py"


@task(name="clean_data", retries=1)
def clean_data_task():
    return clean_data()


@task(name="engineer_features", retries=1)
def engineer_features_task(_clean_dep=None):
    return engineer_features()


@task(name="register_feast_features", retries=1)
def feast_task(_features_dep=None):
    return register_features()


@task(name="train_baseline", retries=1)
def train_baseline_task(_features_dep=None):
    return train_baseline()


@task(name="train_automl", retries=0)
def train_automl_task(_features_dep=None):
    """Runs in venv-automl (PyCaret/CatBoost/MLflow) via subprocess -- the main
    pipeline environment (Python 3.13) can't import PyCaret directly. Output
    streams live (not captured) so long steps show progress instead of hanging."""
    result = subprocess.run([AUTOML_VENV_PYTHON, AUTOML_SCRIPT])
    if result.returncode != 0:
        raise RuntimeError(f"train_automl.py failed with exit code {result.returncode}")


@task(name="export_champion", retries=0)
def export_champion_task(_automl_dep=None):
    """Flatten the freshly-registered PyCaret champion into models/champion/ for serving.

    Runs in venv-automl for the same reason train_automl does: reading the pipeline requires
    PyCaret. The script self-verifies (it refuses to write unless the flattened artifact
    reproduces the pipeline's probabilities to 1e-9), so a non-zero exit here means the export
    genuinely disagreed with the model and must not be deployed.
    """
    result = subprocess.run([AUTOML_VENV_PYTHON, EXPORT_CHAMPION_SCRIPT])
    if result.returncode != 0:
        raise RuntimeError(f"export_champion.py failed with exit code {result.returncode}")


@task(name="build_drift_reference", retries=1)
def build_drift_reference_task(_features_dep=None):
    """Refresh the clean-training sample the container's online drift monitor compares against."""
    return build_drift_reference()


@flow(name="diabetes-readmission-pipeline")
def diabetes_pipeline():
    cleaned = clean_data_task()
    features = engineer_features_task(cleaned)
    feast_task(features)
    train_baseline_task(features)
    automl = train_automl_task(features)
    # Deployment artifacts last: both depend on the models trained above.
    export_champion_task(automl)
    build_drift_reference_task(features)


if __name__ == "__main__":
    diabetes_pipeline()
