# ============================================================
# ETA PREDICTION - MODEL MONITORING
# ============================================================
#
# Purpose:
#   1. Load reference ETA dataset
#   2. Load current/production ETA data
#   3. Detect data drift
#   4. Monitor prediction behaviour
#   5. Calculate model performance when actual ETA is available
#   6. Log monitoring metrics to MLflow
#   7. Save monitoring results
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ETA_Model_data_1.csv"
)

MODEL_STORE = (
    PROJECT_ROOT
    / "model_store"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "metrics"
)

MONITORING_DIR = (
    PROJECT_ROOT
    / "monitoring"
)


# Create directories if required

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MONITORING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. MLFLOW CONFIGURATION
# ============================================================

MLFLOW_DB = (
    PROJECT_ROOT
    / "mlflow.db"
)

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_experiment(
    "ETA Prediction"
)

print(
    "\nMLflow tracking URI:"
)

print(
    mlflow.get_tracking_uri()
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING MODEL")
print("=" * 60)

model_path = (
    MODEL_STORE
    / "eta_model.pkl"
)

encoder_path = (
    MODEL_STORE
    / "eta_encoder.pkl"
)

feature_columns_path = (
    MODEL_STORE
    / "feature_columns.json"
)


if not model_path.exists():

    raise FileNotFoundError(
        f"Model not found: {model_path}"
    )


if not encoder_path.exists():

    raise FileNotFoundError(
        f"Encoder not found: {encoder_path}"
    )


if not feature_columns_path.exists():

    raise FileNotFoundError(
        f"Feature schema not found: "
        f"{feature_columns_path}"
    )


model = joblib.load(
    model_path
)

encoder = joblib.load(
    encoder_path
)


with open(
    feature_columns_path,
    "r"
) as file:

    feature_columns = json.load(
        file
    )


print(
    "Model loaded successfully."
)

print(
    "Number of model features:",
    len(feature_columns)
)


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)


df = pd.read_csv(
    DATA_PATH
)

print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# 6. BASIC VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("DATA VALIDATION")
print("=" * 60)


print(
    "\nMissing values:"
)

print(
    df.isnull().sum()
)


print(
    "\nDuplicate rows:",
    df.duplicated().sum()
)


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)


# ------------------------------------------------------------
# Remove identifier
# ------------------------------------------------------------

if "trip_id" in df.columns:

    df = df.drop(
        columns=["trip_id"]
    )


# ------------------------------------------------------------
# Pickup date
# ------------------------------------------------------------

df["pickup_date"] = pd.to_datetime(
    df["pickup_date"]
)

df["month"] = (
    df["pickup_date"].dt.month
)

df["day"] = (
    df["pickup_date"].dt.day
)

df["day_of_year"] = (
    df["pickup_date"].dt.dayofyear
)

df = df.drop(
    columns=["pickup_date"]
)


# ------------------------------------------------------------
# Pickup time
# ------------------------------------------------------------

df["pickup_time"] = pd.to_datetime(
    df["pickup_time"],
    format="%H:%M"
)

df["pickup_minute"] = (
    df["pickup_time"].dt.minute
)


if "pickup_hour" not in df.columns:

    df["pickup_hour"] = (
        df["pickup_time"].dt.hour
    )


df = df.drop(
    columns=["pickup_time"]
)


# ============================================================
# 8. DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)


# Remove unrealistic ETA values

if "actual_eta_minutes" in df.columns:

    df = df[
        df["actual_eta_minutes"] < 200
    ].copy()


# Remove features not used by model

columns_to_remove = [
    "average_speed",
    "time_of_day"
]

df = df.drop(
    columns=columns_to_remove,
    errors="ignore"
)


# ============================================================
# 9. HANDLE RARE LOCATIONS
# ============================================================

location_threshold = 20


# ------------------------------------------------------------
# Pickup locations
# ------------------------------------------------------------

if "pickup_location" in df.columns:

    pickup_counts = (
        df["pickup_location"]
        .value_counts()
    )

    rare_pickups = (
        pickup_counts[
            pickup_counts < location_threshold
        ]
        .index
    )

    df["pickup_location"] = (
        df["pickup_location"]
        .replace(
            rare_pickups,
            "Other"
        )
    )


# ------------------------------------------------------------
# Drop locations
# ------------------------------------------------------------

if "drop_location" in df.columns:

    drop_counts = (
        df["drop_location"]
        .value_counts()
    )

    rare_drops = (
        drop_counts[
            drop_counts < location_threshold
        ]
        .index
    )

    df["drop_location"] = (
        df["drop_location"]
        .replace(
            rare_drops,
            "Other"
        )
    )


# ============================================================
# 10. SEPARATE TARGET
# ============================================================

has_actual_eta = (
    "actual_eta_minutes" in df.columns
)


if has_actual_eta:

    y = df[
        "actual_eta_minutes"
    ].copy()

    X = df.drop(
        columns=["actual_eta_minutes"]
    )

else:

    X = df.copy()

    y = None


# ============================================================
# 11. ENCODE TRAFFIC LEVEL
# ============================================================

traffic_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}


if "traffic_level" in X.columns:

    X["traffic_level"] = (
        X["traffic_level"]
        .map(traffic_map)
    )


# ============================================================
# 12. ONE-HOT ENCODING
# ============================================================

categorical_columns = [
    "pickup_location",
    "drop_location",
    "weekday",
    "season"
]


existing_categorical_columns = [
    column
    for column in categorical_columns
    if column in X.columns
]


if existing_categorical_columns:

    encoded_data = pd.DataFrame(
        encoder.transform(
            X[
                existing_categorical_columns
            ]
        ),
        columns=encoder.get_feature_names_out(
            existing_categorical_columns
        ),
        index=X.index
    )

    X = X.drop(
        columns=existing_categorical_columns
    )

    X = pd.concat(
        [
            X,
            encoded_data
        ],
        axis=1
    )


# ============================================================
# 13. ALIGN FEATURES WITH TRAINING
# ============================================================

print("\n" + "=" * 60)
print("ALIGNING MODEL FEATURES")
print("=" * 60)


for column in feature_columns:

    if column not in X.columns:

        X[column] = 0


extra_columns = [
    column
    for column in X.columns
    if column not in feature_columns
]


if extra_columns:

    X = X.drop(
        columns=extra_columns
    )


X = X[
    feature_columns
]


print(
    "Monitoring feature shape:",
    X.shape
)


# ============================================================
# 14. GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)


predictions = model.predict(
    X
)


print(
    "Predictions generated:",
    len(predictions)
)


print(
    "\nPrediction statistics:"
)

print(
    pd.Series(predictions).describe()
)


# ============================================================
# 15. PREDICTION MONITORING
# ============================================================

prediction_mean = float(
    np.mean(predictions)
)

prediction_std = float(
    np.std(predictions)
)

prediction_min = float(
    np.min(predictions)
)

prediction_max = float(
    np.max(predictions)
)


# ============================================================
# 16. MODEL PERFORMANCE
# ============================================================

model_metrics = {}


if has_actual_eta:

    monitoring_r2 = r2_score(
        y,
        predictions
    )

    monitoring_mae = mean_absolute_error(
        y,
        predictions
    )

    monitoring_rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    model_metrics = {

        "monitoring_r2":
            float(monitoring_r2),

        "monitoring_mae":
            float(monitoring_mae),

        "monitoring_rmse":
            float(monitoring_rmse)
    }

    print(
        "\nMonitoring model performance:"
    )

    print(
        f"R²   : {monitoring_r2:.4f}"
    )

    print(
        f"MAE  : {monitoring_mae:.4f}"
    )

    print(
        f"RMSE : {monitoring_rmse:.4f}"
    )

else:

    print(
        "\nActual ETA is not available."
    )

    print(
        "Model performance cannot be calculated."
    )


# ============================================================
# 17. SIMPLE DATA DRIFT DETECTION
# ============================================================

print("\n" + "=" * 60)
print("DATA DRIFT DETECTION")
print("=" * 60)


# For this first monitoring implementation,
# we compare the current feature distribution
# against the training/reference dataset.

reference_df = pd.read_csv(
    DATA_PATH
)


# ------------------------------------------------------------
# Numeric drift
# ------------------------------------------------------------

numeric_columns = (
    reference_df
    .select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)


drift_results = {}


for column in numeric_columns:

    if column not in df.columns:

        continue

    reference_mean = float(
        reference_df[column]
        .mean()
    )

    current_mean = float(
        df[column]
        .mean()
    )

    reference_std = float(
        reference_df[column]
        .std()
    )

    if reference_std == 0:

        mean_difference = 0

    else:

        mean_difference = (
            abs(
                current_mean
                - reference_mean
            )
            / reference_std
        )


    drift_results[column] = {

        "reference_mean":
            reference_mean,

        "current_mean":
            current_mean,

        "mean_difference_std":
            float(mean_difference)
    }


# ============================================================
# 18. DETERMINE DRIFT STATUS
# ============================================================

drift_threshold = 0.5


drifted_features = []


for column, result in drift_results.items():

    if (
        result["mean_difference_std"]
        > drift_threshold
    ):

        drifted_features.append(
            column
        )


drift_count = len(
    drifted_features
)

total_checked_features = len(
    drift_results
)


if drift_count > 0:

    drift_status = "DRIFT DETECTED"

else:

    drift_status = "NO SIGNIFICANT DRIFT"


print(
    "\nDrift status:",
    drift_status
)

print(
    "Features checked:",
    total_checked_features
)

print(
    "Drifted features:",
    drift_count
)


if drifted_features:

    print(
        "\nDrifted features:"
    )

    for feature in drifted_features:

        print(
            "-",
            feature
        )


# ============================================================
# 19. CREATE MONITORING RESULTS
# ============================================================

monitoring_results = {

    "prediction_mean":
        prediction_mean,

    "prediction_std":
        prediction_std,

    "prediction_min":
        prediction_min,

    "prediction_max":
        prediction_max,

    "drift_status":
        drift_status,

    "drift_count":
        drift_count,

    "total_checked_features":
        total_checked_features,

    "drifted_features":
        drifted_features,

    "drift_threshold":
        drift_threshold,

    **model_metrics
}


# ============================================================
# 20. SAVE MONITORING JSON
# ============================================================

monitoring_json_path = (
    METRICS_DIR
    / "monitoring.json"
)


with open(
    monitoring_json_path,
    "w"
) as file:

    json.dump(
        monitoring_results,
        file,
        indent=4
    )


print(
    "\nMonitoring results saved to:"
)

print(
    monitoring_json_path
)


# ============================================================
# 21. MLFLOW MONITORING RUN
# ============================================================

print("\n" + "=" * 60)
print("LOGGING MONITORING RESULTS TO MLFLOW")
print("=" * 60)


with mlflow.start_run(
    run_name="ETA Model Monitoring"
):

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    mlflow.log_param(
        "monitoring_type",
        "data_drift_and_performance"
    )

    mlflow.log_param(
        "drift_threshold",
        drift_threshold
    )

    mlflow.log_param(
        "features_checked",
        total_checked_features
    )

    # --------------------------------------------------------
    # Prediction metrics
    # --------------------------------------------------------

    mlflow.log_metric(
        "prediction_mean",
        prediction_mean
    )

    mlflow.log_metric(
        "prediction_std",
        prediction_std
    )

    mlflow.log_metric(
        "prediction_min",
        prediction_min
    )

    mlflow.log_metric(
        "prediction_max",
        prediction_max
    )

    # --------------------------------------------------------
    # Drift metrics
    # --------------------------------------------------------

    mlflow.log_metric(
        "drift_count",
        drift_count
    )

    mlflow.log_metric(
        "drift_percentage",
        (
            drift_count
            / total_checked_features
            * 100
            if total_checked_features > 0
            else 0
        )
    )

    # --------------------------------------------------------
    # Model performance
    # --------------------------------------------------------

    if has_actual_eta:

        mlflow.log_metric(
            "monitoring_r2",
            float(
                model_metrics[
                    "monitoring_r2"
                ]
            )
        )

        mlflow.log_metric(
            "monitoring_mae",
            float(
                model_metrics[
                    "monitoring_mae"
                ]
            )
        )

        mlflow.log_metric(
            "monitoring_rmse",
            float(
                model_metrics[
                    "monitoring_rmse"
                ]
            )
        )

    # --------------------------------------------------------
    # Save monitoring report as artifact
    # --------------------------------------------------------

    mlflow.log_artifact(
        str(monitoring_json_path)
    )

    print(
        "\nMonitoring results logged to MLflow."
    )

    print(
        "MLflow Run ID:",
        mlflow.active_run().info.run_id
    )


# ============================================================
# 22. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("MONITORING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "\nDrift status:",
    drift_status
)

print(
    "Monitoring report:",
    monitoring_json_path
)

print(
    "\nMLflow experiment: ETA Prediction"
)