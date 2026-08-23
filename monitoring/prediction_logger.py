# ============================================================
# ETA PREDICTION - PREDICTION LOGGER
# ============================================================
#
# Purpose:
#   Log every successful API prediction.
#
# Output:
#   monitoring/prediction_logs.csv
#
# Used for:
#   - Production monitoring
#   - Data drift detection
#   - Prediction analysis
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import csv
from datetime import datetime
from pathlib import Path


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MONITORING_DIR = PROJECT_ROOT / "monitoring"

LOG_FILE = MONITORING_DIR / "prediction_logs.csv"


# ============================================================
# 3. CREATE MONITORING DIRECTORY
# ============================================================

MONITORING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. CSV COLUMNS
# ============================================================

FIELDNAMES = [

    "timestamp",

    "pickup_location",
    "drop_location",

    "pickup_date",
    "pickup_time",

    "pickup_hour",
    "pickup_minute",

    "weekday",
    "is_weekend",

    "season",

    "month",
    "day",
    "day_of_year",

    "trip_distance_km",

    "traffic_level",

    "passenger_count",
    "surge_multiplier",

    "predicted_eta",

    "model_version"
]


# ============================================================
# 5. LOG PREDICTION
# ============================================================

def log_prediction(
    features: dict,
    pickup_date: str,
    pickup_time: str,
    predicted_eta: float,
    model_version: str
):
    """
    Log one successful API prediction.

    Parameters
    ----------
    features:
        Backend-generated feature dictionary.

    pickup_date:
        User pickup date.

    pickup_time:
        User pickup time.

    predicted_eta:
        Model prediction in minutes.

    model_version:
        Model name/version.
    """

    # --------------------------------------------------------
    # Create timestamp
    # --------------------------------------------------------

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    # --------------------------------------------------------
    # Create CSV record
    # --------------------------------------------------------

    record = {

        "timestamp":
            timestamp,

        "pickup_location":
            features.get(
                "pickup_location"
            ),

        "drop_location":
            features.get(
                "drop_location"
            ),

        "pickup_date":
            pickup_date,

        "pickup_time":
            pickup_time,

        "pickup_hour":
            features.get(
                "pickup_hour"
            ),

        "pickup_minute":
            features.get(
                "pickup_minute"
            ),

        "weekday":
            features.get(
                "weekday"
            ),

        "is_weekend":
            features.get(
                "is_weekend"
            ),

        "season":
            features.get(
                "season"
            ),

        "month":
            features.get(
                "month"
            ),

        "day":
            features.get(
                "day"
            ),

        "day_of_year":
            features.get(
                "day_of_year"
            ),

        "trip_distance_km":
            features.get(
                "trip_distance_km"
            ),

        "traffic_level":
            features.get(
                "traffic_level"
            ),

        "passenger_count":
            features.get(
                "passenger_count"
            ),

        "surge_multiplier":
            features.get(
                "surge_multiplier"
            ),

        "predicted_eta":
            predicted_eta,

        "model_version":
            model_version
    }

    # --------------------------------------------------------
    # Determine whether file exists
    # --------------------------------------------------------

    file_exists = LOG_FILE.exists()

    # --------------------------------------------------------
    # Write record
    # --------------------------------------------------------

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        # Write header for a new file

        if not file_exists:

            writer.writeheader()

        writer.writerow(record)

    # --------------------------------------------------------
    # Console confirmation
    # --------------------------------------------------------

    print(
        "Prediction logged successfully: "
        f"{record['pickup_location']} -> "
        f"{record['drop_location']}, "
        f"ETA={predicted_eta} minutes"
    )


# ============================================================
# 6. TEST LOGGER
# ============================================================

if __name__ == "__main__":

    print(
        "Testing prediction logger..."
    )

    test_features = {

        "pickup_location":
            "Harlem",

        "drop_location":
            "East Harlem",

        "pickup_hour":
            16,

        "pickup_minute":
            0,

        "weekday":
            "Sunday",

        "is_weekend":
            1,

        "season":
            "Summer",

        "month":
            8,

        "day":
            23,

        "day_of_year":
            235,

        "trip_distance_km":
            2.5,

        "traffic_level":
            "Medium",

        "passenger_count":
            1,

        "surge_multiplier":
            1.0
    }

    log_prediction(

        features=test_features,

        pickup_date="2026-08-23",

        pickup_time="16:00",

        predicted_eta=6.68,

        model_version="GradientBoostingRegressor"
    )

    print(
        "\nLog file:"
    )

    print(
        LOG_FILE
    )