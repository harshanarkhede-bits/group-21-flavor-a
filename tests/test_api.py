from fastapi.testclient import TestClient

from src.predict import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload():
    payload = {
        "passenger_count": 1,
        "distance_km": 5.2,
        "hour_of_day": 17,
        "day_of_week": 2,
        "is_weekend": 0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"predicted_trip_duration_seconds", "predicted_trip_duration_minutes"}
    assert data["predicted_trip_duration_seconds"] > 0
    assert data["predicted_trip_duration_minutes"] > 0


def test_predict_rejects_invalid_payload():
    invalid_payload = {
        "passenger_count": 1,
        "distance_km": 0,
        "hour_of_day": 17,
        "day_of_week": 2,
        "is_weekend": 0,
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
