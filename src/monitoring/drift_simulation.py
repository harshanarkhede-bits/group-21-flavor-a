from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import config
from src.monitoring.drift_detector import calculate_statistical_drift

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def simulate_operational_drift(
    df: pd.DataFrame,
    duration_multiplier: float = 1.45,
    rush_hour_shift: bool = True,
    weather_anomaly: bool = True
) -> pd.DataFrame:
    """
    Simulate operational data drift by:
    1. Inflating evening rush hour and weekend trip durations.
    2. Simulating adverse weather / traffic jams inflating distance times.
    """
    drifted = df.copy()

    if rush_hour_shift and "pickup_hour" in drifted.columns:
        rush_mask = (drifted["pickup_hour"].between(16, 20)) | (drifted.get("is_weekend", 0) == 1)
        if config.target_column in drifted.columns:
            drifted.loc[rush_mask, config.target_column] = (
                drifted.loc[rush_mask, config.target_column] * duration_multiplier
            )

    if weather_anomaly and "trip_distance_km" in drifted.columns:
        if "traffic_level" in drifted.columns:
            drifted["traffic_level"] = np.clip(drifted["traffic_level"] + 1, 0, 2)
        if config.target_column in drifted.columns:
            drifted[config.target_column] = drifted[config.target_column] * 1.15

    return drifted


def run_drift_simulation_experiment(
    data_path: Path | str | None = None, output_path: Path | str | None = None
) -> Dict[str, Any]:
    """Run drift simulation against baseline data and evaluate statistical divergence."""
    path = Path(data_path) if data_path else config.test_data_path
    if not path.exists():
        raise FileNotFoundError(f"Test data not found at: {path}")

    logger.info(f"Loading baseline dataset for drift simulation: {path}")
    base_df = pd.read_csv(path)

    drifted_df = simulate_operational_drift(base_df, duration_multiplier=1.50)

    # Calculate statistical drift between baseline and drifted
    drift_results = calculate_statistical_drift(base_df, drifted_df)

    out_file = Path(output_path) if output_path else config.monitoring_dir / "simulated_drift_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "simulation": "Evening Rush Hour Surge & Severe Weather Simulation",
        "original_mean_eta": float(round(base_df[config.target_column].mean(), 2)) if config.target_column in base_df.columns else None,
        "drifted_mean_eta": float(round(drifted_df[config.target_column].mean(), 2)) if config.target_column in drifted_df.columns else None,
        "dataset_drift_detected": drift_results["dataset_drift_detected"],
        "drift_share": drift_results["drift_share"],
        "drifted_features": drift_results["drifted_features"],
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Simulated Drift experiment complete. Output saved to: {out_file}")
    logger.info(f"Drift Results: {summary}")
    return summary


if __name__ == "__main__":
    run_drift_simulation_experiment()
