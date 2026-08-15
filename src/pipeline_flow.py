"""Prefect flow: orchestrates the full diabetes-readmission pipeline end to end.

Run from the repo root: python src/pipeline_flow.py
"""
import subprocess

from prefect import flow, task

from data_cleaning import clean_data
from feature_engineering import engineer_features
from feast_registry import register_features
from train_baseline import train_baseline

AUTOML_VENV_PYTHON = "venv-automl/bin/python"
AUTOML_SCRIPT = "src/train_automl.py"


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


@flow(name="diabetes-readmission-pipeline")
def diabetes_pipeline():
    cleaned = clean_data_task()
    features = engineer_features_task(cleaned)
    feast_task(features)
    train_baseline_task(features)
    train_automl_task(features)


if __name__ == "__main__":
    diabetes_pipeline()
