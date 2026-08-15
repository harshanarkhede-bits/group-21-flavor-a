import pandas as pd

from src.monitoring import compute_monitoring_metrics, should_retrain, simulate_data_drift


def test_compute_monitoring_metrics_returns_expected_fields():
    predictions = pd.DataFrame(
        {
            "predicted_trip_duration_seconds": [200.0, 250.0, 300.0],
            "actual_trip_duration_seconds": [180.0, 270.0, 330.0],
        }
    )

    metrics = compute_monitoring_metrics(predictions)

    assert set(metrics.keys()) == {"mae", "rmse", "num_records"}
    assert metrics["num_records"] == 3
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0


def test_simulate_data_drift_increases_trip_times():
    df = pd.DataFrame(
        {
            "distance_km": [1.0, 2.0, 3.0],
            "hour_of_day": [15, 18, 20],
            "day_of_week": [0, 1, 2],
            "is_weekend": [0, 0, 1],
            "trip_duration": [200, 300, 400],
        }
    )

    drifted = simulate_data_drift(df)

    assert drifted["trip_duration"].gt(df["trip_duration"]).all()
    assert drifted["distance_km"].equals(df["distance_km"])


def test_should_retrain_detects_threshold_crossing():
    baseline_rmse = 400.0
    current_rmse = 520.0
    threshold_pct = 0.15

    assert should_retrain(current_rmse, baseline_rmse, threshold_pct) is True
    assert should_retrain(410.0, baseline_rmse, threshold_pct) is False
