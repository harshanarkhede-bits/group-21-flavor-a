from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.monitoring import DRIFT_REPORT_PATH, load_baseline_rmse, simulate_data_drift

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "engineered_taxi.csv"

def run_drift_simulation() -> dict:
    """Simulate shift in demand patterns and compare monitored RMSE against the baseline."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed data file not found. Run the preprocessing pipeline first to create engineered_taxi.csv."
        )

    df = pd.read_csv(DATA_PATH)
    drifted_df = simulate_data_drift(df)

    baseline_rmse = load_baseline_rmse()
    drifted_mean = float(drifted_df["trip_duration"].mean())
    original_mean = float(df["trip_duration"].mean())
    drift_pct = (drifted_mean - original_mean) / original_mean if original_mean else 0.0

    report = {
        "baseline_rmse": baseline_rmse,
        "original_mean_trip_duration": original_mean,
        "drifted_mean_trip_duration": drifted_mean,
        "mean_increase_pct": drift_pct,
        "simulation_note": "During busy evening and weekend periods, trip durations were increased to simulate operational drift.",
    }

    with open(DRIFT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    print(json.dumps(run_drift_simulation(), indent=2))
