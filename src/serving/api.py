from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Prometheus metrics
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
    HAS_PROMETHEUS = True
    PREDICTION_REQUESTS = Counter(
        "eta_prediction_requests_total",
        "Total number of ETA prediction requests",
        ["status", "endpoint"]
    )
    PREDICTION_LATENCY = Histogram(
        "eta_prediction_latency_seconds",
        "Latency of ETA prediction requests in seconds",
        ["endpoint"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
    )
    PREDICTED_ETA_HISTOGRAM = Histogram(
        "eta_predicted_minutes",
        "Distribution of predicted ETA in minutes",
        buckets=[5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0]
    )
except ImportError:
    HAS_PROMETHEUS = False

from src.config import config
from src.serving.locations import ALLOWED_LOCATIONS, calculate_haversine_distance_km

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title=config.project_name,
    description="Enterprise MLOps REST API for Ride and Delivery ETA Prediction.",
    version=config.project_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Artifact Loaders
# ------------------------------------------------------------
model: Any = None
encoder: Any = None
feature_columns: List[str] = []
model_metadata: Dict[str, Any] = {}


def load_artifacts() -> bool:
    global model, encoder, feature_columns, model_metadata
    try:
        if config.model_path.exists():
            model = joblib.load(config.model_path)
            logger.info(f"Loaded model from: {config.model_path}")
        if config.encoder_path.exists():
            encoder = joblib.load(config.encoder_path)
            logger.info(f"Loaded encoder from: {config.encoder_path}")
        if config.feature_columns_path.exists():
            with open(config.feature_columns_path, "r", encoding="utf-8") as f:
                feature_columns = json.load(f)
            logger.info(f"Loaded {len(feature_columns)} feature columns.")
        if config.metadata_path.exists():
            with open(config.metadata_path, "r", encoding="utf-8") as f:
                model_metadata = json.load(f)
        return model is not None and encoder is not None and len(feature_columns) > 0
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        return False


# Attempt initial load
load_artifacts()


# ------------------------------------------------------------
# Schemas
# ------------------------------------------------------------
class ETAPredictionRequest(BaseModel):
    pickup_location: str = Field(..., json_schema_extra={"example": "Upper West Side"})
    drop_location: str = Field(..., json_schema_extra={"example": "Harlem"})
    pickup_date: str = Field(..., json_schema_extra={"example": "2026-08-27"}, description="YYYY-MM-DD format")
    pickup_time: str = Field(..., json_schema_extra={"example": "17:30"}, description="HH:MM format")
    passenger_count: Optional[int] = Field(default=1, ge=1, le=8)
    surge_multiplier: Optional[float] = Field(default=1.0, ge=1.0, le=5.0)

    @field_validator("pickup_location", "drop_location")
    @classmethod
    def validate_locations(cls, val: str) -> str:
        val = val.strip()
        if val not in ALLOWED_LOCATIONS:
            raise ValueError(
                f"Invalid location '{val}'. Allowed locations: {', '.join(ALLOWED_LOCATIONS)}"
            )
        return val

    @field_validator("pickup_date")
    @classmethod
    def validate_date_format(cls, val: str) -> str:
        try:
            datetime.strptime(val.strip(), "%Y-%m-%d")
        except ValueError:
            raise ValueError("pickup_date must be in YYYY-MM-DD format.")
        return val.strip()

    @field_validator("pickup_time")
    @classmethod
    def validate_time_format(cls, val: str) -> str:
        try:
            datetime.strptime(val.strip(), "%H:%M")
        except ValueError:
            raise ValueError("pickup_time must be in HH:MM format.")
        return val.strip()


class ETAPredictionResponse(BaseModel):
    success: bool = True
    eta_minutes: float
    eta_seconds: float
    calculated_distance_km: float
    estimated_traffic_level: str
    pickup_location: str
    drop_location: str
    pickup_date: str
    pickup_time: str
    timestamp: str


class ETABatchPredictionRequest(BaseModel):
    trips: List[ETAPredictionRequest]


class ETABatchPredictionResponse(BaseModel):
    success: bool = True
    total_trips: int
    predictions: List[ETAPredictionResponse]


# ------------------------------------------------------------
# Feature Transformation & Prediction Logic
# ------------------------------------------------------------
def estimate_traffic_level(pickup_hour: int, is_weekend: int) -> str:
    """Deterministic backend rule for traffic category based on time."""
    if is_weekend:
        return "Medium" if 11 <= pickup_hour <= 18 else "Low"
    if (7 <= pickup_hour <= 10) or (16 <= pickup_hour <= 19):
        return "High"
    if 11 <= pickup_hour <= 15:
        return "Medium"
    return "Low"


def build_feature_vector(req: ETAPredictionRequest) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Transform user request into encoded model-ready features."""
    dt = datetime.strptime(f"{req.pickup_date} {req.pickup_time}", "%Y-%m-%d %H:%M")
    pickup_hour = dt.hour
    pickup_minute = dt.minute
    month = dt.month
    day = dt.day
    day_of_year = dt.timetuple().tm_yday
    weekday = dt.strftime("%A")
    is_weekend = int(dt.weekday() >= 5)

    if month in [12, 1, 2]:
        season = "Winter"
    elif month in [3, 4, 5]:
        season = "Spring"
    elif month in [6, 7, 8]:
        season = "Summer"
    else:
        season = "Fall"

    distance_km = calculate_haversine_distance_km(req.pickup_location, req.drop_location)
    traffic_str = estimate_traffic_level(pickup_hour, is_weekend)
    traffic_ordinal = config.traffic_mapping.get(traffic_str, 1)

    raw_features = {
        "pickup_hour": pickup_hour,
        "pickup_minute": pickup_minute,
        "month": month,
        "day": day,
        "day_of_year": day_of_year,
        "weekday": weekday,
        "is_weekend": is_weekend,
        "season": season,
        "pickup_location": req.pickup_location,
        "drop_location": req.drop_location,
        "trip_distance_km": distance_km,
        "traffic_level": traffic_ordinal,
        "passenger_count": req.passenger_count or config.default_passenger_count,
        "surge_multiplier": req.surge_multiplier or config.default_surge_multiplier,
    }

    df_raw = pd.DataFrame([raw_features])
    cat_cols = config.categorical_columns

    # OneHotEncoding
    encoded_vals = encoder.transform(df_raw[cat_cols])
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    df_encoded = pd.DataFrame(encoded_vals, columns=encoded_cols)

    df_numeric = df_raw.drop(columns=cat_cols)
    model_input = pd.concat([df_numeric.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

    # Ensure all expected columns are aligned
    for col in feature_columns:
        if col not in model_input.columns:
            model_input[col] = 0
    model_input = model_input[feature_columns]

    extra_meta = {
        "calculated_distance_km": distance_km,
        "estimated_traffic_level": traffic_str,
    }

    return model_input, extra_meta


def log_prediction_async(req: ETAPredictionRequest, eta_minutes: float, meta: Dict[str, Any]) -> None:
    """Log prediction for drift monitoring."""
    try:
        config.monitoring_dir.mkdir(parents=True, exist_ok=True)
        file_exists = config.prediction_log_path.exists()
        with open(config.prediction_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "pickup_location",
                    "drop_location",
                    "pickup_date",
                    "pickup_time",
                    "distance_km",
                    "traffic_level",
                    "predicted_eta_minutes",
                    "actual_eta_minutes",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "pickup_location": req.pickup_location,
                    "drop_location": req.drop_location,
                    "pickup_date": req.pickup_date,
                    "pickup_time": req.pickup_time,
                    "distance_km": meta["calculated_distance_km"],
                    "traffic_level": meta["estimated_traffic_level"],
                    "predicted_eta_minutes": round(eta_minutes, 2),
                    "actual_eta_minutes": "",
                }
            )
    except Exception as e:
        logger.warning(f"Failed to log prediction: {e}")


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/", tags=["Info"])
def root():
    return {
        "service": config.project_name,
        "version": config.project_version,
        "docs_url": "/docs",
        "health_url": "/health",
        "model_info_url": "/model-info",
        "metrics_url": "/metrics",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Liveness probe."""
    return {"status": "healthy", "service": config.project_name}


@app.get("/ready", tags=["Health"])
def readiness_check():
    """Readiness probe checking if model and encoder artifacts are loaded."""
    is_ready = model is not None and encoder is not None and len(feature_columns) > 0
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded or missing. Run model training first.",
        )
    return {
        "status": "ready",
        "model_class": type(model).__name__,
        "features_loaded": len(feature_columns),
    }


@app.get("/model-info", tags=["Metadata"])
def get_model_info():
    """Return active model metadata and evaluation statistics."""
    return {
        "model_name": model_metadata.get("model", "XGBoost"),
        "model_class": model_metadata.get("best_model_class", type(model).__name__ if model else "None"),
        "test_r2": model_metadata.get("test_r2", None),
        "test_rmse": model_metadata.get("test_rmse", None),
        "test_mae": model_metadata.get("test_mae", None),
        "test_mape": model_metadata.get("test_mape", None),
        "feature_count": len(feature_columns),
        "all_comparisons": model_metadata.get("all_model_comparisons", {}),
    }


@app.post("/predict", response_model=ETAPredictionResponse, tags=["Inference"])
def predict_single_trip(request: ETAPredictionRequest):
    """Predict ETA in minutes for a single trip."""
    start_time = time.time()
    if model is None or encoder is None:
        if not load_artifacts():
            if HAS_PROMETHEUS:
                PREDICTION_REQUESTS.labels(status="error", endpoint="/predict").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifacts not found. Please run training pipeline.",
            )

    try:
        model_input, meta = build_feature_vector(request)
        prediction = model.predict(model_input)[0]
        eta_minutes = max(float(prediction), 0.5)
        eta_seconds = eta_minutes * 60.0

        # Log prediction asynchronously for monitoring
        log_prediction_async(request, eta_minutes, meta)

        duration = time.time() - start_time
        if HAS_PROMETHEUS:
            PREDICTION_REQUESTS.labels(status="success", endpoint="/predict").inc()
            PREDICTION_LATENCY.labels(endpoint="/predict").observe(duration)
            PREDICTED_ETA_HISTOGRAM.observe(eta_minutes)

        return ETAPredictionResponse(
            success=True,
            eta_minutes=round(eta_minutes, 2),
            eta_seconds=round(eta_seconds, 1),
            calculated_distance_km=meta["calculated_distance_km"],
            estimated_traffic_level=meta["estimated_traffic_level"],
            pickup_location=request.pickup_location,
            drop_location=request.drop_location,
            pickup_date=request.pickup_date,
            pickup_time=request.pickup_time,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        if HAS_PROMETHEUS:
            PREDICTION_REQUESTS.labels(status="error", endpoint="/predict").inc()
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post("/predict/batch", response_model=ETABatchPredictionResponse, tags=["Inference"])
def predict_batch_trips(batch_req: ETABatchPredictionRequest):
    """Batch inference endpoint for multiple trip requests."""
    start_time = time.time()
    if model is None or encoder is None:
        if not load_artifacts():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifacts not found.",
            )

    results: List[ETAPredictionResponse] = []
    for req in batch_req.trips:
        model_input, meta = build_feature_vector(req)
        pred = model.predict(model_input)[0]
        eta_min = max(float(pred), 0.5)
        log_prediction_async(req, eta_min, meta)
        results.append(
            ETAPredictionResponse(
                success=True,
                eta_minutes=round(eta_min, 2),
                eta_seconds=round(eta_min * 60.0, 1),
                calculated_distance_km=meta["calculated_distance_km"],
                estimated_traffic_level=meta["estimated_traffic_level"],
                pickup_location=req.pickup_location,
                drop_location=req.drop_location,
                pickup_date=req.pickup_date,
                pickup_time=req.pickup_time,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )

    duration = time.time() - start_time
    if HAS_PROMETHEUS:
        PREDICTION_REQUESTS.labels(status="success", endpoint="/predict/batch").inc()
        PREDICTION_LATENCY.labels(endpoint="/predict/batch").observe(duration)

    return ETABatchPredictionResponse(
        success=True,
        total_trips=len(results),
        predictions=results,
    )


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics scraping endpoint."""
    if not HAS_PROMETHEUS:
        return Response(content="Prometheus client not installed", media_type="text/plain")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.api:app", host=config.serving_host, port=config.serving_port, reload=True)