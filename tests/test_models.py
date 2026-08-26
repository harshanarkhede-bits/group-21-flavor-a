import pytest
import numpy as np
import pandas as pd
from src.models.train import compute_metrics


def test_compute_metrics():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, 31.0, 39.0])
    metrics = compute_metrics(y_true, y_pred)

    assert "r2" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert metrics["mae"] == 1.0
    assert metrics["r2"] > 0.95
