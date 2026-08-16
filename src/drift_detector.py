"""Online drift detection for the serving container, powered by EvidentlyAI.

EvidentlyAI is this project's drift-detection framework. It runs in two places:

  * **offline** — `monitoring/anomaly_verification.py` builds the full per-column HTML reports
    committed under `monitoring/reports/`, the deep-dive artifacts.
  * **online, here** — the same Evidently analysis runs *inside the serving container*, against
    a rolling window of the rows the API has actually served. The deployed service therefore
    detects its own drift, with nobody running a script, and publishes the verdict straight to
    the Prometheus gauges.

Both paths call `src/monitoring.py`, so the live numbers and the offline reports come from
exactly the same Evidently configuration (DataDriftPreset over the 16 raw features, Evidently's
own per-column stat-test selection and its 0.5 dataset-level threshold).

**Why a background thread.** An Evidently report over a few thousand rows takes roughly half a
second. That's cheap in absolute terms but far too expensive to sit inside a `/predict` call
that otherwise completes in tens of milliseconds. So `add_batch` only ever appends to a buffer
and, when enough new rows have arrived, hands a snapshot to a single worker thread. Serving
latency is untouched; the gauges lag real time by about a second, which is irrelevant next to
Prometheus's 5-second scrape interval.

Only one analysis runs at a time — if traffic outruns the analysis, later batches are folded
into the next window rather than queued, which is the right behaviour for a monitor.
"""
import os
import sys
import threading
from collections import deque
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monitoring import drift_verdict


class DriftDetector:
    """Rolling-window Evidently drift monitor for the live serving path."""

    def __init__(
        self,
        reference: pd.DataFrame,
        window: int = 3000,
        min_new_rows: int = 600,
        min_window: int = 2000,
    ):
        self.reference = reference
        self.window = window
        self.min_new_rows = min_new_rows
        # Don't publish a verdict from a barely-filled buffer. Evidently selects its stat test
        # by sample size and switches below ~1000 rows, and a few hundred rows against a 5000-row
        # reference produced a flickering false positive on genuinely clean traffic. Waiting for
        # a substantial window costs a few seconds of "warming up" and removes the flicker.
        self.min_window = min_window

        # maxlen gives the rolling window for free — the oldest rows fall off automatically.
        self._buffer = deque(maxlen=window)
        self._rows_since_last_check = 0
        self._lock = threading.Lock()
        self._analysing = False
        self.last_result: Optional[dict] = None
        self.last_error: Optional[str] = None

    def add_batch(self, raw: pd.DataFrame) -> None:
        """Record a served batch, and kick off an Evidently run if one is due.

        Returns immediately — the analysis happens on a worker thread and lands in
        `last_result`. The caller publishes gauges from the `on_result` callback instead.
        """
        with self._lock:
            self._buffer.extend(raw.to_dict("records"))
            self._rows_since_last_check += len(raw)

            due = (
                self._rows_since_last_check >= self.min_new_rows
                and len(self._buffer) >= self.min_window
                and not self._analysing      # skip rather than queue; the next batch will retry
            )
            if not due:
                return
            self._rows_since_last_check = 0
            self._analysing = True
            snapshot = pd.DataFrame(list(self._buffer))

        threading.Thread(
            target=self._run_analysis, args=(snapshot,), daemon=True, name="drift-analysis"
        ).start()

    def _run_analysis(self, current: pd.DataFrame) -> None:
        try:
            result = drift_verdict(self.reference, current)
            result["n_current_rows"] = len(current)
            with self._lock:
                self.last_result = result
                self.last_error = None
            if self._on_result:
                self._on_result(result)
        except Exception as e:  # noqa: BLE001 - a monitoring failure must not kill the thread
            with self._lock:
                self.last_error = str(e)
            print(f"[drift] Evidently analysis failed: {e}")
        finally:
            with self._lock:
                self._analysing = False

    _on_result = None

    def on_result(self, callback) -> None:
        """Register the callback that publishes a completed verdict (used for Prometheus gauges)."""
        self._on_result = callback
