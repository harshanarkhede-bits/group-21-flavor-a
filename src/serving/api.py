# ============================================================
# ETA PREDICTION - SERVING API
# ============================================================
#
# Purpose:
#   - Accept user inputs from UI
#   - Validate inputs
#   - Perform backend feature engineering
#   - Load trained model and encoder
#   - Generate ETA prediction
#   - Return prediction as JSON
#
# User provides ONLY:
#   1. Pickup location
#   2. Drop location
#   3. Pickup date
#   4. Pickup time
#
# No external APIs are used.
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_STORE = PROJECT_ROOT / "model_store"

MODEL_PATH = MODEL_STORE / "eta_model.pkl"
ENCODER_PATH = MODEL_STORE / "eta_encoder.pkl"
FEATURE_COLUMNS_PATH = MODEL_STORE / "feature_columns.json"
METADATA_PATH = MODEL_STORE / "model_metadata.json"

from src.serving.locations import (
    ALLOWED_LOCATIONS,
    LOCATION_COORDINATES
)
# ============================================================
# 3. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# 4. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ETA Prediction API",
    description=(
        "API for predicting delivery/ride ETA "
        "using pickup location, drop location, "
        "pickup date and pickup time."
    ),
    version="1.0.0"
)



# ============================================================
# 7. MODEL ARTIFACT VALIDATION
# ============================================================

REQUIRED_ARTIFACTS = [
    MODEL_PATH,
    ENCODER_PATH,
    FEATURE_COLUMNS_PATH
]


for artifact in REQUIRED_ARTIFACTS:

    if not artifact.exists():

        raise FileNotFoundError(
            f"Required model artifact not found: {artifact}"
        )


# ============================================================
# 8. LOAD MODEL ARTIFACTS
# ============================================================

logger.info("Loading model artifacts...")

model = joblib.load(
    MODEL_PATH
)

encoder = joblib.load(
    ENCODER_PATH
)

with open(
    FEATURE_COLUMNS_PATH,
    "r"
) as file:

    FEATURE_COLUMNS = json.load(file)


# Optional metadata
MODEL_METADATA = {}

if METADATA_PATH.exists():

    with open(
        METADATA_PATH,
        "r"
    ) as file:

        MODEL_METADATA = json.load(file)


logger.info("Model artifacts loaded successfully.")


# ============================================================
# 9. REQUEST SCHEMA
# ============================================================

class ETAPredictionRequest(BaseModel):

    pickup_location: str = Field(
        ...,
        description="Pickup location"
    )

    drop_location: str = Field(
        ...,
        description="Drop location"
    )

    pickup_date: str = Field(
        ...,
        description="Pickup date in YYYY-MM-DD format"
    )

    pickup_time: str = Field(
        ...,
        description="Pickup time in HH:MM format"
    )

    # --------------------------------------------------------
    # Validate pickup location
    # --------------------------------------------------------

    @field_validator("pickup_location")
    @classmethod
    def validate_pickup_location(cls, value):

        value = value.strip()

        if value not in ALLOWED_LOCATIONS:

            raise ValueError(
                "Invalid pickup location. "
                "Please select a location from the allowed list."
            )

        return value

    # --------------------------------------------------------
    # Validate drop location
    # --------------------------------------------------------

    @field_validator("drop_location")
    @classmethod
    def validate_drop_location(cls, value):

        value = value.strip()

        if value not in ALLOWED_LOCATIONS:

            raise ValueError(
                "Invalid drop location. "
                "Please select a location from the allowed list."
            )

        return value

    # --------------------------------------------------------
    # Validate date
    # --------------------------------------------------------

    @field_validator("pickup_date")
    @classmethod
    def validate_date(cls, value):

        try:

            parsed_date = datetime.strptime(
                value,
                "%Y-%m-%d"
            ).date()

        except ValueError:

            raise ValueError(
                "pickup_date must be in YYYY-MM-DD format."
            )

        return parsed_date.isoformat()

    # --------------------------------------------------------
    # Validate time
    # --------------------------------------------------------

    @field_validator("pickup_time")
    @classmethod
    def validate_time(cls, value):

        try:

            parsed_time = datetime.strptime(
                value,
                "%H:%M"
            ).time()

        except ValueError:

            raise ValueError(
                "pickup_time must be in HH:MM format."
            )

        return parsed_time.strftime("%H:%M")


# ============================================================
# 10. HAVERSINE DISTANCE FUNCTION
# ============================================================

def calculate_distance_km(
    pickup_location: str,
    drop_location: str
) -> float:

    """
    Calculate approximate straight-line distance
    between pickup and drop locations.

    No external API is used.
    """

    pickup_lat, pickup_lon = (
        LOCATION_COORDINATES[pickup_location]
    )

    drop_lat, drop_lon = (
        LOCATION_COORDINATES[drop_location]
    )

    # Convert degrees to radians
    lat1 = np.radians(pickup_lat)
    lon1 = np.radians(pickup_lon)

    lat2 = np.radians(drop_lat)
    lon2 = np.radians(drop_lon)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # Haversine formula
    a = (
        np.sin(delta_lat / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    earth_radius_km = 6371.0

    distance = (
        earth_radius_km * c
    )

    return round(
        float(distance),
        3
    )


# ============================================================
# 11. TIME / DATE FEATURE ENGINEERING
# ============================================================

def create_datetime_features(
    pickup_date: str,
    pickup_time: str
) -> dict:

    """
    Convert user-provided date/time into
    model-compatible features.
    """

    dt = datetime.strptime(
        f"{pickup_date} {pickup_time}",
        "%Y-%m-%d %H:%M"
    )

    pickup_hour = dt.hour

    pickup_minute = dt.minute

    month = dt.month

    day = dt.day

    day_of_year = (
        dt.timetuple().tm_yday
    )

    weekday = dt.strftime(
        "%A"
    )

    is_weekend = int(
        dt.weekday() >= 5
    )

    # --------------------------------------------------------
    # Season
    # --------------------------------------------------------

    if month in [12, 1, 2]:

        season = "Winter"

    elif month in [3, 4, 5]:

        season = "Spring"

    elif month in [6, 7, 8]:

        season = "Summer"

    else:

        season = "Fall"


    return {

        "pickup_hour": pickup_hour,

        "pickup_minute": pickup_minute,

        "month": month,

        "day": day,

        "day_of_year": day_of_year,

        "weekday": weekday,

        "is_weekend": is_weekend,

        "season": season
    }


# ============================================================
# 12. BACKEND TRAFFIC ESTIMATION
# ============================================================

def estimate_traffic_level(
    pickup_hour: int,
    is_weekend: int
) -> str:

    """
    Estimate traffic category from time.

    This is NOT real-time traffic.

    It is a deterministic backend assumption designed
    so the UI does not need to ask the user for traffic.
    """

    # Weekend
    if is_weekend:

        if 11 <= pickup_hour <= 18:

            return "Medium"

        return "Low"

    # Weekday morning rush
    if 7 <= pickup_hour <= 10:

        return "High"

    # Weekday evening rush
    if 16 <= pickup_hour <= 19:

        return "High"

    # Midday
    if 11 <= pickup_hour <= 15:

        return "Medium"

    # Night
    return "Low"


# ============================================================
# 13. INTERNAL DEFAULTS
# ============================================================

def create_backend_features(
    request: ETAPredictionRequest
) -> dict:

    """
    Create all features that are not directly entered
    by the user.
    """

    datetime_features = (
        create_datetime_features(
            request.pickup_date,
            request.pickup_time
        )
    )

    distance = calculate_distance_km(
        request.pickup_location,
        request.drop_location
    )

    traffic_level = estimate_traffic_level(
        datetime_features["pickup_hour"],
        datetime_features["is_weekend"]
    )

    # --------------------------------------------------------
    # Backend assumptions
    # --------------------------------------------------------
    #
    # The UI does not ask for passenger count or surge.
    # Therefore we use fixed baseline values.
    #
    # These should eventually be replaced by learned
    # historical defaults if you retrain the model.
    # --------------------------------------------------------

    passenger_count = 1

    surge_multiplier = 1.0


    features = {

        "pickup_location":
            request.pickup_location,

        "drop_location":
            request.drop_location,

        "pickup_hour":
            datetime_features["pickup_hour"],

        "pickup_minute":
            datetime_features["pickup_minute"],

        "weekday":
            datetime_features["weekday"],

        "is_weekend":
            datetime_features["is_weekend"],

        "season":
            datetime_features["season"],

        "month":
            datetime_features["month"],

        "day":
            datetime_features["day"],

        "day_of_year":
            datetime_features["day_of_year"],

        "trip_distance_km":
            distance,

        "traffic_level":
            traffic_level,

        "passenger_count":
            passenger_count,

        "surge_multiplier":
            surge_multiplier
    }

    return features


# ============================================================
# 14. PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(
    features: dict
) -> pd.DataFrame:

    """
    Convert backend-generated features into exactly the
    format expected by the trained model.
    """

    raw_df = pd.DataFrame(
        [features]
    )

    # --------------------------------------------------------
    # Traffic encoding
    # --------------------------------------------------------

    traffic_map = {

        "Low": 0,

        "Medium": 1,

        "High": 2
    }

    raw_df["traffic_level"] = (
        raw_df["traffic_level"]
        .map(traffic_map)
    )


    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    categorical_columns = [

        "pickup_location",

        "drop_location",

        "weekday",

        "season"
    ]


    # --------------------------------------------------------
    # Encode categories
    # --------------------------------------------------------

    encoded = pd.DataFrame(

        encoder.transform(
            raw_df[
                categorical_columns
            ]
        ),

        columns=encoder.get_feature_names_out(
            categorical_columns
        )
    )


    # Remove original categorical columns
    raw_df = raw_df.drop(
        columns=categorical_columns
    )


    # Combine numerical + encoded features
    model_input = pd.concat(
        [
            raw_df.reset_index(drop=True),

            encoded.reset_index(drop=True)
        ],

        axis=1
    )


    # --------------------------------------------------------
    # Ensure exact feature order
    # --------------------------------------------------------

    for column in FEATURE_COLUMNS:

        if column not in model_input.columns:

            model_input[column] = 0


    # Remove unexpected columns
    model_input = model_input[
        FEATURE_COLUMNS
    ]


    return model_input


# ============================================================
# 15. PREDICTION FUNCTION
# ============================================================

def predict_eta(
    request: ETAPredictionRequest
) -> dict:

    """
    Main prediction workflow.
    """

    # Create backend features
    features = create_backend_features(
        request
    )

    logger.info(
        "Prediction request: %s -> %s",
        request.pickup_location,
        request.drop_location
    )

    # Prepare model input
    model_input = prepare_model_input(
        features
    )

    # Predict
    prediction = model.predict(
        model_input
    )

    eta_minutes = float(
        prediction[0]
    )

    # Prevent negative ETA
    eta_minutes = max(
        eta_minutes,
        0.0
    )

    return {

        "eta_minutes":
            round(
                eta_minutes,
                2
            ),

        "pickup_location":
            request.pickup_location,

        "drop_location":
            request.drop_location,

        "pickup_date":
            request.pickup_date,

        "pickup_time":
            request.pickup_time,

        "calculated_distance_km":
            features["trip_distance_km"],

        "estimated_traffic_level":
            features["traffic_level"]
    }


# ============================================================
# 16. HEALTH CHECK
# ============================================================

@app.get(
    "/health"
)
def health_check():

    """
    Health endpoint for deployment/monitoring.
    """

    return {

        "status": "healthy",

        "service": "ETA Prediction API",

        "model_loaded": model is not None,

        "encoder_loaded": encoder is not None,

        "feature_count":
            len(FEATURE_COLUMNS)
    }


# ============================================================
# 17. MODEL INFORMATION
# ============================================================

@app.get(
    "/model-info"
)
def model_info():

    """
    Return model metadata.
    """

    return {

        "model":
            MODEL_METADATA.get(
                "model",
                "unknown"
            ),

        "test_r2":
            MODEL_METADATA.get(
                "test_r2"
            ),

        "test_mae":
            MODEL_METADATA.get(
                "test_mae"
            ),

        "test_rmse":
            MODEL_METADATA.get(
                "test_rmse"
            ),

        "feature_count":
            len(FEATURE_COLUMNS)
    }


# ============================================================
# 18. ETA PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict"
)
def predict(
    request: ETAPredictionRequest
):

    """
    Main ETA prediction endpoint.
    """

    try:

        result = predict_eta(
            request
        )

        return {

            "success": True,

            "prediction": result
        }

    except Exception as error:

        logger.exception(
            "Prediction failed."
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Unable to generate ETA prediction."
            )
        )