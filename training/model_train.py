# ============================================================
# ETA PREDICTION - MODEL TRAINING
# ============================================================
#
# Purpose:
#   1. Load ETA dataset
#   2. Validate data
#   3. Perform feature engineering
#   4. Clean and prepare data
#   5. Encode categorical features
#   6. Train multiple regression models
#   7. Select Gradient Boosting
#   8. Evaluate final model
#   9. Save model and feature information
#  10. Save metrics for DVC
#  11. Track parameters, metrics and model using MLflow
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import mlflow
import mlflow.sklearn

import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# model_train.py is inside:
#
# group-21-flavor-a/
#     training/
#         model_train.py
#
# Therefore parent of training/ = project root

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


# ------------------------------------------------------------
# Dataset path
# ------------------------------------------------------------

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ETA_Model_data_1.csv"
)


# ------------------------------------------------------------
# Model storage path
# ------------------------------------------------------------

MODEL_STORE = (
    PROJECT_ROOT
    / "model_store"
)


# Create model_store folder if it doesn't exist

MODEL_STORE.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. MLFLOW CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("MLFLOW CONFIGURATION")
print("=" * 60)


# ------------------------------------------------------------
# MLflow database
# ------------------------------------------------------------

MLFLOW_DB = (
    PROJECT_ROOT
    / "mlflow.db"
)


# ------------------------------------------------------------
# Use SQLite backend
# ------------------------------------------------------------

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)


# ------------------------------------------------------------
# Set MLflow experiment
# ------------------------------------------------------------

mlflow.set_experiment(
    "ETA Prediction"
)


print(
    "MLflow tracking URI:",
    mlflow.get_tracking_uri()
)

print(
    "MLflow experiment: ETA Prediction"
)


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)


df = pd.read_csv(
    DATA_PATH
)


print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# 5. BASIC DATA VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("DATA VALIDATION")
print("=" * 60)


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

missing_values = (
    df.isnull().sum()
)


print(
    "\nMissing values:"
)

print(
    missing_values
)


# ------------------------------------------------------------
# Duplicate rows
# ------------------------------------------------------------

duplicate_count = (
    df.duplicated().sum()
)


print(
    "\nDuplicate rows:",
    duplicate_count
)


if duplicate_count > 0:

    print(
        "Warning: duplicate rows found."
    )

else:

    print(
        "No duplicate rows found."
    )


# ============================================================
# 6. REMOVE IDENTIFIER
# ============================================================

# trip_id is only an identifier.
# It should not be used as a predictive feature.

if "trip_id" in df.columns:

    df.drop(
        columns=["trip_id"],
        inplace=True
    )


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)


# ------------------------------------------------------------
# Pickup date
# ------------------------------------------------------------

df["pickup_date"] = pd.to_datetime(
    df["pickup_date"]
)


# Extract month

df["month"] = (
    df["pickup_date"].dt.month
)


# Extract day

df["day"] = (
    df["pickup_date"].dt.day
)


# Extract day of year

df["day_of_year"] = (
    df["pickup_date"].dt.dayofyear
)


# Original date is no longer required

df.drop(
    columns=["pickup_date"],
    inplace=True
)


# ------------------------------------------------------------
# Pickup time
# ------------------------------------------------------------

df["pickup_time"] = pd.to_datetime(
    df["pickup_time"],
    format="%H:%M"
)


# Extract minute

df["pickup_minute"] = (
    df["pickup_time"].dt.minute
)


# Extract hour if it doesn't already exist

if "pickup_hour" not in df.columns:

    df["pickup_hour"] = (
        df["pickup_time"].dt.hour
    )


# Original time is no longer required

df.drop(
    columns=["pickup_time"],
    inplace=True
)


# ============================================================
# 8. DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)


# ------------------------------------------------------------
# Remove unrealistic ETA values
# ------------------------------------------------------------

initial_rows = len(df)


df = df[
    df["actual_eta_minutes"] < 200
].copy()


removed_rows = (
    initial_rows - len(df)
)


print(
    "Rows removed due to unrealistic ETA:",
    removed_rows
)


# ------------------------------------------------------------
# Remove features that should not be used
# ------------------------------------------------------------

columns_to_remove = [
    "average_speed",
    "time_of_day"
]


df.drop(
    columns=columns_to_remove,
    inplace=True,
    errors="ignore"
)


# ============================================================
# 9. HANDLE RARE LOCATIONS
# ============================================================

print("\n" + "=" * 60)
print("HANDLING RARE LOCATIONS")
print("=" * 60)


location_threshold = 20


# ------------------------------------------------------------
# Pickup locations
# ------------------------------------------------------------

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


print(
    "Rare pickup locations grouped:",
    len(rare_pickups)
)


print(
    "Rare drop locations grouped:",
    len(rare_drops)
)


# ============================================================
# 10. SEPARATE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 60)
print("PREPARING FEATURES AND TARGET")
print("=" * 60)


X = df.drop(
    columns=["actual_eta_minutes"]
)


y = df[
    "actual_eta_minutes"
]


# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
)


print(
    "Training rows:",
    len(X_train)
)


print(
    "Testing rows:",
    len(X_test)
)


# ============================================================
# 12. ENCODE TRAFFIC LEVEL
# ============================================================

traffic_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}


X_train = X_train.copy()

X_test = X_test.copy()


X_train["traffic_level"] = (
    X_train["traffic_level"]
    .map(traffic_map)
)


X_test["traffic_level"] = (
    X_test["traffic_level"]
    .map(traffic_map)
)


# ============================================================
# 13. ONE-HOT ENCODING
# ============================================================

categorical_columns = [
    "pickup_location",
    "drop_location",
    "weekday",
    "season"
]


encoder = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)


# ------------------------------------------------------------
# Fit encoder ONLY on training data
# ------------------------------------------------------------

encoder.fit(
    X_train[categorical_columns]
)


# ------------------------------------------------------------
# Transform training data
# ------------------------------------------------------------

encoded_train = pd.DataFrame(

    encoder.transform(
        X_train[categorical_columns]
    ),

    columns=encoder.get_feature_names_out(
        categorical_columns
    ),

    index=X_train.index
)


# ------------------------------------------------------------
# Transform test data
# ------------------------------------------------------------

encoded_test = pd.DataFrame(

    encoder.transform(
        X_test[categorical_columns]
    ),

    columns=encoder.get_feature_names_out(
        categorical_columns
    ),

    index=X_test.index
)


# ------------------------------------------------------------
# Remove original categorical columns
# ------------------------------------------------------------

X_train = X_train.drop(
    columns=categorical_columns
)


X_test = X_test.drop(
    columns=categorical_columns
)


# ------------------------------------------------------------
# Add encoded columns
# ------------------------------------------------------------

X_train = pd.concat(
    [
        X_train,
        encoded_train
    ],
    axis=1
)


X_test = pd.concat(
    [
        X_test,
        encoded_test
    ],
    axis=1
)


# ============================================================
# 14. SAVE FEATURE SCHEMA
# ============================================================

feature_columns = list(
    X_train.columns
)


with open(
    MODEL_STORE / "feature_columns.json",
    "w"
) as file:

    json.dump(
        feature_columns,
        file,
        indent=4
    )


print(
    "\nFeature schema saved."
)


# ============================================================
# 15. CREATE MODELS
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)


models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            random_state=42,
            n_estimators=100
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        )
}


# ============================================================
# 16. TRAIN AND EVALUATE MODELS
# ============================================================

results = []


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = (
        model.predict(
            X_test
        )
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    results.append({

        "Model": name,

        "MAE": round(
            mae,
            3
        ),

        "RMSE": round(
            rmse,
            3
        ),

        "R2 Score": round(
            r2,
            3
        )
    })


# ============================================================
# 17. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 18. SELECT GRADIENT BOOSTING + MLFLOW
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL + MLFLOW TRACKING")
print("=" * 60)


# ------------------------------------------------------------
# Create final model
# ------------------------------------------------------------

final_model = GradientBoostingRegressor(
    random_state=42
)


# ------------------------------------------------------------
# Start MLflow run
# ------------------------------------------------------------

with mlflow.start_run(
    run_name="Gradient Boosting ETA Model"
):

    print(
        "\nStarting MLflow run..."
    )


    # --------------------------------------------------------
    # Get active MLflow run
    # --------------------------------------------------------

    active_run = mlflow.active_run()


    print(
        "MLflow Run ID:",
        active_run.info.run_id
    )


    print(
        "MLflow Experiment ID:",
        active_run.info.experiment_id
    )


    # ========================================================
    # LOG PARAMETERS
    # ========================================================

    print(
        "\nLogging parameters to MLflow..."
    )


    mlflow.log_params({

        "model":
            "GradientBoostingRegressor",

        "test_size":
            0.20,

        "random_state":
            42,

        "location_threshold":
            location_threshold,

        "n_estimators":
            final_model.n_estimators,

        "learning_rate":
            final_model.learning_rate,

        "max_depth":
            final_model.max_depth,

        "min_samples_split":
            final_model.min_samples_split,

        "min_samples_leaf":
            final_model.min_samples_leaf,

        "subsample":
            final_model.subsample,

        "train_rows":
            len(X_train),

        "test_rows":
            len(X_test),

        "feature_count":
            len(feature_columns),

        "categorical_feature_count":
            len(categorical_columns)

    })


    print(
        "MLflow parameters logged successfully."
    )


    # ========================================================
    # TRAIN FINAL MODEL
    # ========================================================

    print(
        "\nTraining final Gradient Boosting model..."
    )


    final_model.fit(
        X_train,
        y_train
    )


    print(
        "Final model training completed."
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    train_pred = (
        final_model.predict(
            X_train
        )
    )


    test_pred = (
        final_model.predict(
            X_test
        )
    )


    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    train_r2 = r2_score(
        y_train,
        train_pred
    )


    test_r2 = r2_score(
        y_test,
        test_pred
    )


    train_mae = mean_absolute_error(
        y_train,
        train_pred
    )


    test_mae = mean_absolute_error(
        y_test,
        test_pred
    )


    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_pred
        )
    )


    # ========================================================
    # PRINT METRICS
    # ========================================================

    print(
        "\nModel Metrics:"
    )


    print(
        f"Train R²  : {train_r2:.4f}"
    )


    print(
        f"Test R²   : {test_r2:.4f}"
    )


    print(
        f"Train MAE : {train_mae:.4f} minutes"
    )


    print(
        f"Test MAE  : {test_mae:.4f} minutes"
    )


    print(
        f"Test RMSE : {test_rmse:.4f} minutes"
    )


    # ========================================================
    # LOG METRICS TO MLFLOW
    # ========================================================

    print(
        "\nLogging metrics to MLflow..."
    )


    mlflow.log_metrics({

        "train_r2":
            float(train_r2),

        "test_r2":
            float(test_r2),

        "train_mae":
            float(train_mae),

        "test_mae":
            float(test_mae),

        "test_rmse":
            float(test_rmse)

    })


    print(
        "Metrics successfully logged to MLflow."
    )


    # ========================================================
    # SAVE METRICS JSON FOR DVC
    # ========================================================

    metrics_dir = (
        PROJECT_ROOT
        / "metrics"
    )


    metrics_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    metrics = {

        "train_r2":
            float(train_r2),

        "test_r2":
            float(test_r2),

        "train_mae":
            float(train_mae),

        "test_mae":
            float(test_mae),

        "test_rmse":
            float(test_rmse)

    }


    metrics_path = (
        metrics_dir
        / "metrics.json"
    )


    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


    print(
        "\nMetrics saved to:",
        metrics_path
    )


    # ========================================================
    # LOG MODEL TO MLFLOW
    # ========================================================

    print(
        "\nLogging final model to MLflow..."
    )


    mlflow.sklearn.log_model(
        final_model,
        name="eta_model"
    )


    print(
        "Final model logged to MLflow."
    )


    # ========================================================
    # VERIFY MLFLOW RUN
    # ========================================================

    print(
        "\nVerifying MLflow run..."
    )


    current_run = mlflow.get_run(
        active_run.info.run_id
    )


    print(
        "\nMLflow recorded parameters:"
    )


    print(
        current_run.data.params
    )


    print(
        "\nMLflow recorded metrics:"
    )


    print(
        current_run.data.metrics
    )


# ============================================================
# 19. SAVE FINAL MODEL LOCALLY
# ============================================================

model_path = (
    MODEL_STORE
    / "eta_model.pkl"
)


joblib.dump(
    final_model,
    model_path
)


# ============================================================
# 20. SAVE ENCODER
# ============================================================

encoder_path = (
    MODEL_STORE
    / "eta_encoder.pkl"
)


joblib.dump(
    encoder,
    encoder_path
)


# ============================================================
# 21. SAVE METADATA
# ============================================================

metadata = {

    "model":
        "GradientBoostingRegressor",

    "random_state":
        42,

    "test_size":
        0.20,

    "train_r2":
        float(train_r2),

    "test_r2":
        float(test_r2),

    "train_mae":
        float(train_mae),

    "test_mae":
        float(test_mae),

    "test_rmse":
        float(test_rmse),

    "traffic_mapping":
        traffic_map,

    "categorical_columns":
        categorical_columns,

    "location_threshold":
        location_threshold

}


with open(
    MODEL_STORE / "model_metadata.json",
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# 22. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)


print(
    "\nSaved files:"
)


print(
    "1.",
    model_path
)


print(
    "2.",
    encoder_path
)


print(
    "3.",
    MODEL_STORE / "feature_columns.json"
)


print(
    "4.",
    MODEL_STORE / "model_metadata.json"
)


print(
    "5.",
    PROJECT_ROOT / "metrics" / "metrics.json"
)


print(
    "\nMLflow experiment: ETA Prediction"
)


print(
    "MLflow tracking completed successfully."
)