import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_DATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ETA_Model_data_1.csv"
)

MONITORING_DIR = (
    PROJECT_ROOT
    / "monitoring"
)

REFERENCE_DATA = (
    MONITORING_DIR
    / "reference_data.csv"
)


# ============================================================
# CREATE MONITORING DIRECTORY
# ============================================================

MONITORING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("Loading training dataset...")

df = pd.read_csv(
    TRAINING_DATA
)

print(
    "Training dataset shape:",
    df.shape
)


# ============================================================
# SELECT MONITORING FEATURES
# ============================================================

monitoring_columns = [
    "pickup_location",
    "drop_location",
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
    "surge_multiplier"
]


# Only keep columns that actually exist
available_columns = [
    column
    for column in monitoring_columns
    if column in df.columns
]


reference_df = df[
    available_columns
].copy()


# ============================================================
# SAVE REFERENCE DATA
# ============================================================

reference_df.to_csv(
    REFERENCE_DATA,
    index=False
)


print()
print("=" * 60)
print("REFERENCE DATA CREATED")
print("=" * 60)

print(
    "Rows:",
    len(reference_df)
)

print(
    "Columns:",
    list(reference_df.columns)
)

print(
    "\nSaved to:",
    REFERENCE_DATA
)