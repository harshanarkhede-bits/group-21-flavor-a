import csv
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "trip_duration_model.joblib"
MONITORING_DIR = ROOT_DIR / "monitoring"
PREDICTION_LOG_PATH = MONITORING_DIR / "prediction_log.csv"
FEATURE_COLUMNS = [
    "passenger_count",
    "distance_km",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

app = FastAPI(title="Ride ETA Prediction API", version="1.0.0")


class PredictionRequest(BaseModel):
    passenger_count: int = Field(..., ge=1)
    distance_km: float = Field(..., gt=0)
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    predicted_trip_duration_seconds: float
    predicted_trip_duration_minutes: float


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run training first using python src/models/train_model.py"
        )
    return joblib.load(MODEL_PATH)


def predict_trip_duration(payload: dict) -> float:
    model = load_model()
    feature_row = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    prediction = model.predict(feature_row)[0]
    return float(prediction)


def log_prediction(payload: dict, predicted_seconds: float) -> None:
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = PREDICTION_LOG_PATH.exists()

    with open(PREDICTION_LOG_PATH, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "timestamp",
                "passenger_count",
                "distance_km",
                "hour_of_day",
                "day_of_week",
                "is_weekend",
                "predicted_trip_duration_seconds",
                "actual_trip_duration_seconds",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                "passenger_count": payload["passenger_count"],
                "distance_km": payload["distance_km"],
                "hour_of_day": payload["hour_of_day"],
                "day_of_week": payload["day_of_week"],
                "is_weekend": payload["is_weekend"],
                "predicted_trip_duration_seconds": round(predicted_seconds, 2),
                "actual_trip_duration_seconds": "",
            }
        )


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    payload = {
        "passenger_count": request.passenger_count,
        "distance_km": request.distance_km,
        "hour_of_day": request.hour_of_day,
        "day_of_week": request.day_of_week,
        "is_weekend": request.is_weekend,
    }

    predicted_seconds = predict_trip_duration(payload)
    log_prediction(payload, predicted_seconds)
    return {
        "predicted_trip_duration_seconds": round(predicted_seconds, 2),
        "predicted_trip_duration_minutes": round(predicted_seconds / 60, 2),
    }


if __name__ == "__main__":
    uvicorn.run("src.predict:app", host="0.0.0.0", port=8000, reload=True)
