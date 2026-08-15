from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT_DIR = Path(__file__).resolve().parent.parent
MONITORING_DIR = ROOT_DIR / "monitoring"
PREDICTION_LOG_PATH = MONITORING_DIR / "prediction_log.csv"
DRIFT_REPORT_PATH = MONITORING_DIR / "drift_report.json"
BASELINE_METRICS_PATH = ROOT_DIR / "models" / "trip_duration_model_metrics.json"


def compute_monitoring_metrics(predictions_df: pd.DataFrame) -> dict:
    """Compute MAE and RMSE for a monitoring batch of predictions."""
    if predictions_df.empty:
        raise ValueError("Prediction log is empty.")

    required = {"predicted_trip_duration_seconds", "actual_trip_duration_seconds"}
    missing = required.difference(predictions_df.columns)
    if missing:
        raise ValueError(f"Prediction log missing required columns: {sorted(missing)}")

    mae = mean_absolute_error(
        predictions_df["actual_trip_duration_seconds"],
        predictions_df["predicted_trip_duration_seconds"],
    )
    rmse = np.sqrt(
        mean_squared_error(
            predictions_df["actual_trip_duration_seconds"],
            predictions_df["predicted_trip_duration_seconds"],
        )
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "num_records": int(len(predictions_df)),
    }


def should_retrain(current_rmse: float, baseline_rmse: float, threshold_pct: float = 0.15) -> bool:
    """Retrain if the model error rises beyond the accepted threshold."""
    if baseline_rmse <= 0:
        raise ValueError("Baseline RMSE must be positive.")
    return (current_rmse - baseline_rmse) / baseline_rmse > threshold_pct


def simulate_data_drift(df: pd.DataFrame, drift_factor: float = 1.35) -> pd.DataFrame:
    """Create a drifted version of the data by inflating trip durations during busy periods."""
    drifted = df.copy()
    shift_mask = (
        (drifted["hour_of_day"].between(15, 20))
        | (drifted["is_weekend"] == 1)
    )
    drifted.loc[shift_mask, "trip_duration"] = (
        drifted.loc[shift_mask, "trip_duration"] * drift_factor
    )
    return drifted


def load_baseline_rmse() -> float:
    if not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Baseline metrics file not found. Train the model first to generate trip_duration_model_metrics.json."
        )

    with open(BASELINE_METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return float(metrics["best_metrics"]["rmse"])


def monitor_and_trigger_retraining(predictions_df: pd.DataFrame, threshold_pct: float = 0.15) -> dict:
    """Evaluate the model against monitored predictions and decide whether a retraining is needed."""
    metrics = compute_monitoring_metrics(predictions_df)
    baseline_rmse = load_baseline_rmse()
    retrain = should_retrain(metrics["rmse"], baseline_rmse, threshold_pct)

    report = {
        "baseline_rmse": baseline_rmse,
        "current_rmse": metrics["rmse"],
        "threshold_pct": threshold_pct,
        "triggered_retraining": retrain,
        "mae": metrics["mae"],
        "num_records": metrics["num_records"],
    }

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def create_sample_prediction_log() -> pd.DataFrame:
    """Generate sample prediction data to simulate monitoring and drift checks."""
    return pd.DataFrame(
        {
            "predicted_trip_duration_seconds": [350.0, 420.0, 480.0, 530.0],
            "actual_trip_duration_seconds": [320.0, 440.0, 500.0, 610.0],
        }
    )


if __name__ == "__main__":
    prediction_log = create_sample_prediction_log()
    result = monitor_and_trigger_retraining(prediction_log)
    print(json.dumps(result, indent=2))
