"""Shared helpers for the monitoring scripts: repo-root path bootstrap + API client.

Every monitoring script imports this first so that:
  * the repo root is on sys.path (relative data paths like data/processed/... resolve, and
    `src` package imports work), and
  * `src/` is importable for prepare_data / monitoring modules.
"""
import os
import sys

import requests

# These scripts print non-ASCII (Δ deltas, 🚨 alert markers). On Windows the console defaults to
# cp1252 and every such print raises UnicodeEncodeError mid-run, so force UTF-8 on stdout/stderr.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Repo root = parent of this file's directory (monitoring/).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# Run relative to the repo root so data/ and monitoring/ paths resolve consistently.
os.chdir(REPO_ROOT)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")


def api_health() -> dict:
    return requests.get(f"{API_URL}/health", timeout=10).json()


def predict_frame(df, batch_size: int = 2000) -> "tuple[list, list]":
    """POST a raw-feature DataFrame to /predict in batches; return (predictions, probabilities)."""
    preds, probas = [], []
    records = df.to_dict(orient="records")
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        resp = requests.post(f"{API_URL}/predict", json={"records": batch}, timeout=120)
        resp.raise_for_status()
        body = resp.json()
        preds.extend(body["predictions"])
        probas.extend(body["probabilities"])
    return preds, probas


def push_drift(scenario: str, drift_detected: bool, drift_share: float, n_drifted: int) -> None:
    """Push a drift result to the API so it surfaces on /metrics. Best-effort."""
    try:
        requests.post(
            f"{API_URL}/internal/drift",
            json={
                "scenario": scenario,
                "drift_detected": bool(drift_detected),
                "drift_share": float(drift_share),
                "n_drifted": int(n_drifted),
            },
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"[warn] could not push drift metrics: {e}")
