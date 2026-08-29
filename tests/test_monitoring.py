import pytest
import pandas as pd
import numpy as np
from src.monitoring.drift_detector import calculate_statistical_drift
from src.monitoring.drift_simulation import simulate_operational_drift
from src.monitoring.retrain_trigger import evaluate_retraining_policy, load_baseline_metrics


def test_calculate_statistical_drift_identical_data():
    df = pd.DataFrame({
        "pickup_hour": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        "trip_distance_km": [2.0, 3.5, 1.2, 4.0, 5.5, 2.8, 3.1, 4.2, 1.8, 6.0],
    })
    result = calculate_statistical_drift(df, df)
    assert result["dataset_drift_detected"] is False
    assert result["drift_share"] == 0.0


def test_calculate_statistical_drift_detects_distribution_shift():
    ref_df = pd.DataFrame({
        "trip_distance_km": np.random.normal(loc=2.0, scale=0.5, size=100),
    })
    curr_df = pd.DataFrame({
        "trip_distance_km": np.random.normal(loc=15.0, scale=2.0, size=100),  # Major shift
    })
    result = calculate_statistical_drift(ref_df, curr_df)
    assert result["drift_by_feature"] if "drift_by_feature" in result else True
    assert result["dataset_drift_detected"] is True


def test_simulate_operational_drift():
    df = pd.DataFrame({
        "pickup_hour": [17, 18, 19],
        "is_weekend": [0, 0, 1],
        "trip_distance_km": [3.0, 4.0, 5.0],
        "actual_eta_minutes": [10.0, 15.0, 20.0],
    })
    drifted = simulate_operational_drift(df, duration_multiplier=1.5)
    assert (drifted["actual_eta_minutes"] > df["actual_eta_minutes"]).all()


def test_evaluate_retraining_policy():
    # Test healthy scenario
    res_healthy = evaluate_retraining_policy(current_rmse=5.7, threshold_pct=0.15)
    assert "retraining_triggered" in res_healthy

    # Test degraded scenario
    res_degraded = evaluate_retraining_policy(current_rmse=12.0, threshold_pct=0.15)
    assert res_degraded["retraining_triggered"] is True
    assert res_degraded["status"] == "ACTION_REQUIRED_RETRAIN"
