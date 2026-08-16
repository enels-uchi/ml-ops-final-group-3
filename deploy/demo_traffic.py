"""Continuous traffic generator for a live monitoring demo.

The monitoring scripts in monitoring/ send one big burst and exit, which makes the Grafana
panels a flat line with a step in it — technically correct, hard to present. This drives steady
traffic through the deployed API and then *gradually* corrupts it, so the dashboard shows drift
arriving as a moving curve rather than a jump.

Three phases:

  1. HEALTHY  — clean test rows only. Predicted-positive ratio holds near the 0.244 baseline.
  2. RAMP     — clean rows are progressively swapped for drifted ones (0% -> 100%). The ratio
                climbs smoothly, which is what "concept drift creeping in" actually looks like.
  3. DRIFTED  — fully corrupted traffic. Ratio pinned at its extreme, alerts firing.

Run it with the Grafana dashboard open on a 5s refresh / "Last 15 minutes" window:

    python deploy/demo_traffic.py                    # ~5 min, dist_shift (ratio climbs to ~0.95)
    python deploy/demo_traffic.py --scenario out_of_bounds   # ratio collapses to ~0.00 instead
    python deploy/demo_traffic.py --healthy-only     # steady clean traffic, no drift
    python deploy/demo_traffic.py --duration 120     # shorter run

Ctrl-C stops cleanly at any point. Requires the drifted CSVs, so run
`python monitoring/drift_simulation.py` first if monitoring/drifted/ is empty.
"""
import argparse
import random
import sys
import time
from pathlib import Path

# _client handles the repo-root path bootstrap, the UTF-8 console fix, and API_URL.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "monitoring"))
from _client import API_URL, api_health, predict_frame  # noqa: E402

import pandas as pd  # noqa: E402

from prepare_data import load_clean_split  # noqa: E402
from train_baseline import FEATURE_COLS  # noqa: E402

DRIFTED_DIR = Path("monitoring/drifted")
BATCH_SIZE = 300          # small enough to keep each request well under a second
INTERVAL_S = 3.0          # one batch every 3s -> a visible point on a 5s-scrape dashboard


def _load_drifted(scenario: str) -> pd.DataFrame:
    path = DRIFTED_DIR / f"{scenario}.csv"
    if not path.exists():
        available = sorted(p.stem for p in DRIFTED_DIR.glob("*.csv"))
        raise SystemExit(
            f"No drifted dataset at {path}.\n"
            f"Run `python monitoring/drift_simulation.py` first."
            + (f" Available: {available}" if available else "")
        )
    return pd.read_csv(path)[FEATURE_COLS]


def _mixed_batch(clean: pd.DataFrame, drifted: pd.DataFrame, drift_fraction: float) -> pd.DataFrame:
    """A batch of BATCH_SIZE rows, `drift_fraction` of them drawn from the corrupted set."""
    n_drift = int(round(BATCH_SIZE * drift_fraction))
    n_clean = BATCH_SIZE - n_drift
    parts = []
    if n_clean:
        parts.append(clean.sample(n_clean, replace=True))
    if n_drift:
        parts.append(drifted.sample(n_drift, replace=True))
    return pd.concat(parts).sample(frac=1)  # shuffle so the batch isn't ordered clean-then-drift


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="dist_shift",
                    help="drifted dataset to ramp into (default: dist_shift)")
    ap.add_argument("--duration", type=int, default=300,
                    help="total run time in seconds (default: 300)")
    ap.add_argument("--healthy-only", action="store_true",
                    help="stay in phase 1 — clean traffic for the whole run")
    args = ap.parse_args()

    print(f"[demo] API: {API_URL}")
    print(f"[demo] health: {api_health()}\n")

    X_train, X_test, _, _ = load_clean_split()
    clean = X_test
    drifted = None if args.healthy_only else _load_drifted(args.scenario)

    # Split the run into thirds: healthy, ramping, fully drifted.
    third = args.duration / 3
    started = time.time()
    batch_no = 0

    print(f"{'elapsed':>8}  {'phase':<9} {'drift%':>7}  {'pred+ ratio':>12}")
    print("-" * 44)
    try:
        while (elapsed := time.time() - started) < args.duration:
            if args.healthy_only or elapsed < third:
                phase, frac = "HEALTHY", 0.0
            elif elapsed < 2 * third:
                # Linear 0 -> 1 across the middle third.
                phase, frac = "RAMP", (elapsed - third) / third
            else:
                phase, frac = "DRIFTED", 1.0

            batch = clean.sample(BATCH_SIZE, replace=True) if frac == 0.0 else _mixed_batch(clean, drifted, frac)
            preds, _ = predict_frame(batch, batch_size=BATCH_SIZE)
            ratio = sum(preds) / len(preds) if preds else 0.0
            batch_no += 1

            # A crude inline bar so the trend is visible in the terminal too, not just Grafana.
            bar = "#" * int(ratio * 30)
            print(f"{elapsed:7.0f}s  {phase:<9} {frac * 100:6.0f}%  {ratio:11.3f}  {bar}")

            time.sleep(INTERVAL_S)
    except KeyboardInterrupt:
        print("\n[demo] stopped by user")

    print(f"\n[demo] sent {batch_no} batches ({batch_no * BATCH_SIZE:,} rows) over "
          f"{time.time() - started:.0f}s")
    print("[demo] The API detects drift on this traffic itself — check GET /drift for the")
    print("[demo] per-column verdict, or watch the gauges on the Grafana dashboard.")
    print("[demo] monitoring/anomaly_verification.py remains the deeper Evidently analysis.")
    return 0


if __name__ == "__main__":
    random.seed(42)
    sys.exit(main())
