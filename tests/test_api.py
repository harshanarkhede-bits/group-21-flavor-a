import pytest
from fastapi.testclient import TestClient
from src.serving.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "feature_count" in data


def test_predict_valid_request():
    payload = {
        "pickup_location": "Upper West Side",
        "drop_location": "Harlem",
        "pickup_date": "2026-08-27",
        "pickup_time": "17:30",
        "passenger_count": 1,
        "surge_multiplier": 1.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["eta_minutes"] > 0
    assert data["calculated_distance_km"] > 0
    assert "estimated_traffic_level" in data


def test_predict_invalid_location():
    payload = {
        "pickup_location": "NonExistentCity",
        "drop_location": "Harlem",
        "pickup_date": "2026-08-27",
        "pickup_time": "17:30",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_date_format():
    payload = {
        "pickup_location": "Upper West Side",
        "drop_location": "Harlem",
        "pickup_date": "27-08-2026",  # invalid format
        "pickup_time": "17:30",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch_trips():
    payload = {
        "trips": [
            {
                "pickup_location": "Upper West Side",
                "drop_location": "Harlem",
                "pickup_date": "2026-08-27",
                "pickup_time": "17:30",
            },
            {
                "pickup_location": "Chelsea",
                "drop_location": "South Slope",
                "pickup_date": "2026-08-27",
                "pickup_time": "08:15",
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["total_trips"] == 2
    assert len(data["predictions"]) == 2


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
