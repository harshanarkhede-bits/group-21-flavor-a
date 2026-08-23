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
#   - Log every successful prediction for monitoring
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

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from monitoring.prediction_logger import log_prediction

from src.serving.locations import (
    ALLOWED_LOCATIONS,
    LOCATION_COORDINATES
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# api.py is located inside:
#
# group-21-flavor-a/
#     src/
#         serving/
#             api.py
#
# Therefore:
#
# parents[0] = serving
# parents[1] = src
# parents[2] = project root

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


MODEL_STORE = (
    PROJECT_ROOT / "model_store"
)


MODEL_PATH = (
    MODEL_STORE / "eta_model.pkl"
)


ENCODER_PATH = (
    MODEL_STORE / "eta_encoder.pkl"
)


FEATURE_COLUMNS_PATH = (
    MODEL_STORE / "feature_columns.json"
)


METADATA_PATH = (
    MODEL_STORE / "model_metadata.json"
)


# ============================================================
# 3. MONITORING PATH
# ============================================================

MONITORING_DIR = (
    PROJECT_ROOT / "monitoring"
)


PREDICTION_LOG_PATH = (
    MONITORING_DIR / "prediction_logs.csv"
)


MONITORING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


logger = logging.getLogger(
    __name__
)


# ============================================================
# 5. FASTAPI APPLICATION
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
# 6. MODEL ARTIFACT VALIDATION
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
# 7. LOAD MODEL ARTIFACTS
# ============================================================

logger.info(
    "Loading model artifacts..."
)


# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------

model = joblib.load(
    MODEL_PATH
)


# ------------------------------------------------------------
# Load encoder
# ------------------------------------------------------------

encoder = joblib.load(
    ENCODER_PATH
)


# ------------------------------------------------------------
# Load feature columns
# ------------------------------------------------------------

with open(
    FEATURE_COLUMNS_PATH,
    "r",
    encoding="utf-8"
) as file:

    FEATURE_COLUMNS = json.load(
        file
    )


# ------------------------------------------------------------
# Load model metadata
# ------------------------------------------------------------

MODEL_METADATA = {}


if METADATA_PATH.exists():

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        MODEL_METADATA = json.load(
            file
        )


# ============================================================
# 8. MODEL VERSION
# ============================================================

MODEL_VERSION = MODEL_METADATA.get(
    "model",
    "GradientBoostingRegressor"
)


logger.info(
    "Model version: %s",
    MODEL_VERSION
)


logger.info(
    "Model artifacts loaded successfully."
)


logger.info(
    "Feature count: %s",
    len(FEATURE_COLUMNS)
)


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

    @field_validator(
        "pickup_location"
    )
    @classmethod
    def validate_pickup_location(
        cls,
        value
    ):

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

    @field_validator(
        "drop_location"
    )
    @classmethod
    def validate_drop_location(
        cls,
        value
    ):

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

    @field_validator(
        "pickup_date"
    )
    @classmethod
    def validate_date(
        cls,
        value
    ):

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

    @field_validator(
        "pickup_time"
    )
    @classmethod
    def validate_time(
        cls,
        value
    ):

        try:

            parsed_time = datetime.strptime(
                value,
                "%H:%M"
            ).time()

        except ValueError:

            raise ValueError(
                "pickup_time must be in HH:MM format."
            )

        return parsed_time.strftime(
            "%H:%M"
        )


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
        LOCATION_COORDINATES[
            pickup_location
        ]
    )


    drop_lat, drop_lon = (
        LOCATION_COORDINATES[
            drop_location
        ]
    )


    # --------------------------------------------------------
    # Convert degrees to radians
    # --------------------------------------------------------

    lat1 = np.radians(
        pickup_lat
    )

    lon1 = np.radians(
        pickup_lon
    )

    lat2 = np.radians(
        drop_lat
    )

    lon2 = np.radians(
        drop_lon
    )


    # --------------------------------------------------------
    # Differences
    # --------------------------------------------------------

    delta_lat = (
        lat2 - lat1
    )

    delta_lon = (
        lon2 - lon1
    )


    # --------------------------------------------------------
    # Haversine formula
    # --------------------------------------------------------

    a = (

        np.sin(
            delta_lat / 2
        ) ** 2

        +

        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(
            delta_lon / 2
        ) ** 2

    )


    c = 2 * np.arctan2(

        np.sqrt(a),

        np.sqrt(
            1 - a
        )

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

        "pickup_hour":
            pickup_hour,

        "pickup_minute":
            pickup_minute,

        "month":
            month,

        "day":
            day,

        "day_of_year":
            day_of_year,

        "weekday":
            weekday,

        "is_weekend":
            is_weekend,

        "season":
            season

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

    It is a deterministic backend assumption.
    """

    # --------------------------------------------------------
    # Weekend
    # --------------------------------------------------------

    if is_weekend:

        if 11 <= pickup_hour <= 18:

            return "Medium"

        return "Low"


    # --------------------------------------------------------
    # Weekday morning rush
    # --------------------------------------------------------

    if 7 <= pickup_hour <= 10:

        return "High"


    # --------------------------------------------------------
    # Weekday evening rush
    # --------------------------------------------------------

    if 16 <= pickup_hour <= 19:

        return "High"


    # --------------------------------------------------------
    # Midday
    # --------------------------------------------------------

    if 11 <= pickup_hour <= 15:

        return "Medium"


    # --------------------------------------------------------
    # Night
    # --------------------------------------------------------

    return "Low"


# ============================================================
# 13. CREATE BACKEND FEATURES
# ============================================================

def create_backend_features(
    request: ETAPredictionRequest
) -> dict:

    """
    Create all features that are not directly entered
    by the user.
    """

    # --------------------------------------------------------
    # Date/time features
    # --------------------------------------------------------

    datetime_features = (
        create_datetime_features(

            request.pickup_date,

            request.pickup_time

        )
    )


    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    distance = calculate_distance_km(

        request.pickup_location,

        request.drop_location

    )


    # --------------------------------------------------------
    # Traffic
    # --------------------------------------------------------

    traffic_level = (
        estimate_traffic_level(

            datetime_features[
                "pickup_hour"
            ],

            datetime_features[
                "is_weekend"
            ]

        )
    )


    # --------------------------------------------------------
    # Backend defaults
    # --------------------------------------------------------

    passenger_count = 1

    surge_multiplier = 1.0


    # --------------------------------------------------------
    # Final feature dictionary
    # --------------------------------------------------------

    features = {

        "pickup_location":
            request.pickup_location,

        "drop_location":
            request.drop_location,

        "pickup_hour":
            datetime_features[
                "pickup_hour"
            ],

        "pickup_minute":
            datetime_features[
                "pickup_minute"
            ],

        "weekday":
            datetime_features[
                "weekday"
            ],

        "is_weekend":
            datetime_features[
                "is_weekend"
            ],

        "season":
            datetime_features[
                "season"
            ],

        "month":
            datetime_features[
                "month"
            ],

        "day":
            datetime_features[
                "day"
            ],

        "day_of_year":
            datetime_features[
                "day_of_year"
            ],

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

        raw_df[
            "traffic_level"
        ]

        .map(
            traffic_map
        )

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
    # One-hot encoding
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


    # --------------------------------------------------------
    # Remove original categorical columns
    # --------------------------------------------------------

    raw_df = raw_df.drop(

        columns=categorical_columns

    )


    # --------------------------------------------------------
    # Combine numerical + encoded features
    # --------------------------------------------------------

    model_input = pd.concat(

        [

            raw_df.reset_index(
                drop=True
            ),

            encoded.reset_index(
                drop=True
            )

        ],

        axis=1

    )


    # --------------------------------------------------------
    # Ensure exact feature schema
    # --------------------------------------------------------

    for column in FEATURE_COLUMNS:

        if column not in model_input.columns:

            model_input[column] = 0


    # --------------------------------------------------------
    # Remove unexpected columns and order correctly
    # --------------------------------------------------------

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

    Steps:

        1. Create backend features
        2. Prepare model input
        3. Generate prediction
        4. Log production prediction
        5. Return prediction

    IMPORTANT:
        Monitoring failure does NOT cause the API
        prediction to fail.
    """

    # ========================================================
    # STEP 1 - CREATE BACKEND FEATURES
    # ========================================================

    features = create_backend_features(
        request
    )


    logger.info(

        "Prediction request: %s -> %s",

        request.pickup_location,

        request.drop_location

    )


    # ========================================================
    # STEP 2 - PREPARE MODEL INPUT
    # ========================================================

    model_input = prepare_model_input(
        features
    )


    # ========================================================
    # STEP 3 - GENERATE PREDICTION
    # ========================================================

    prediction = model.predict(
        model_input
    )


    eta_minutes = float(
        prediction[0]
    )


    # --------------------------------------------------------
    # Prevent negative ETA
    # --------------------------------------------------------

    eta_minutes = max(
        eta_minutes,
        0.0
    )


    # --------------------------------------------------------
    # Round prediction
    # --------------------------------------------------------

    eta_minutes = round(
        eta_minutes,
        2
    )


    # ========================================================
    # STEP 4 - LOG PRODUCTION PREDICTION
    # ========================================================
    #
    # IMPORTANT:
    #
    # prediction_logger.py expects:
    #
    #     features
    #     pickup_date
    #     pickup_time
    #     predicted_eta
    #     model_version
    #
    # We pass exactly those arguments.
    #
    # DO NOT pass pickup_location=, drop_location=,
    # pickup_hour=, etc. individually.
    #
    # All those values already exist inside "features".
    #
    # ========================================================

    try:

        log_prediction(

            features=features,

            pickup_date=request.pickup_date,

            pickup_time=request.pickup_time,

            predicted_eta=eta_minutes,

            model_version=MODEL_VERSION

        )


        logger.info(
            "Prediction successfully logged for monitoring."
        )


    except Exception:

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Monitoring failure must not cause a successful
        # prediction request to fail.
        # ----------------------------------------------------

        logger.exception(
            "Failed to log prediction for monitoring."
        )


    # ========================================================
    # STEP 5 - RETURN RESULT
    # ========================================================

    return {

        "eta_minutes":
            eta_minutes,

        "pickup_location":
            request.pickup_location,

        "drop_location":
            request.drop_location,

        "pickup_date":
            request.pickup_date,

        "pickup_time":
            request.pickup_time,

        "calculated_distance_km":
            features[
                "trip_distance_km"
            ],

        "estimated_traffic_level":
            features[
                "traffic_level"
            ]

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

        "status":
            "healthy",

        "service":
            "ETA Prediction API",

        "model_loaded":
            model is not None,

        "encoder_loaded":
            encoder is not None,

        "feature_count":
            len(FEATURE_COLUMNS),

        "model_version":
            MODEL_VERSION

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
            len(FEATURE_COLUMNS),

        "model_version":
            MODEL_VERSION

    }


# ============================================================
# 18. MONITORING INFORMATION
# ============================================================

@app.get(
    "/monitoring-info"
)
def monitoring_info():

    """
    Return basic information about the prediction log.
    """

    log_exists = PREDICTION_LOG_PATH.exists()

    prediction_count = 0


    if log_exists:

        try:

            df = pd.read_csv(
                PREDICTION_LOG_PATH
            )

            prediction_count = len(
                df
            )

        except Exception:

            logger.exception(
                "Unable to read prediction log."
            )


    return {

        "monitoring_enabled":
            True,

        "prediction_log":
            str(
                PREDICTION_LOG_PATH
            ),

        "log_exists":
            log_exists,

        "prediction_count":
            prediction_count

    }


# ============================================================
# 19. ETA PREDICTION ENDPOINT
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

            "success":
                True,

            "prediction":
                result

        }


    except Exception:

        logger.exception(
            "Prediction failed."
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "Unable to generate ETA prediction."
            )

        )