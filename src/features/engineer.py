from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.config import config
from src.serving.locations import LOCATION_COORDINATES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def calculate_haversine_distance(
    pickup_locations: pd.Series, drop_locations: pd.Series
) -> pd.Series:
    """Calculate Haversine distance in kilometers for series of location names."""
    distances = []
    earth_radius_km = 6371.0

    for p_loc, d_loc in zip(pickup_locations, drop_locations):
        if p_loc in LOCATION_COORDINATES and d_loc in LOCATION_COORDINATES:
            lat1, lon1 = np.radians(LOCATION_COORDINATES[p_loc])
            lat2, lon2 = np.radians(LOCATION_COORDINATES[d_loc])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distances.append(round(float(earth_radius_km * c), 3))
        else:
            distances.append(1.0)  # Default fallback distance

    return pd.Series(distances, index=pickup_locations.index)


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, minute, month, day, day_of_year, weekday, is_weekend, season."""
    df = df.copy()

    if "pickup_date" in df.columns:
        dt_date = pd.to_datetime(df["pickup_date"], format="%d-%m-%Y", errors="coerce")
        dt_date = dt_date.fillna(pd.to_datetime(df["pickup_date"], errors="coerce"))

        df["month"] = dt_date.dt.month
        df["day"] = dt_date.dt.day
        df["day_of_year"] = dt_date.dt.dayofyear

        if "weekday" not in df.columns or df["weekday"].isnull().all():
            df["weekday"] = dt_date.dt.strftime("%A")
        if "is_weekend" not in df.columns or df["is_weekend"].isnull().all():
            df["is_weekend"] = (dt_date.dt.weekday >= 5).astype(int)
        if "season" not in df.columns or df["season"].isnull().all():
            month_to_season = {
                12: "Winter", 1: "Winter", 2: "Winter",
                3: "Spring", 4: "Spring", 5: "Spring",
                6: "Summer", 7: "Summer", 8: "Summer",
                9: "Fall", 10: "Fall", 11: "Fall"
            }
            df["season"] = df["month"].map(month_to_season)

    if "pickup_time" in df.columns:
        dt_time = pd.to_datetime(df["pickup_time"].astype(str), format="%H:%M", errors="coerce")
        df["pickup_hour"] = dt_time.dt.hour
        df["pickup_minute"] = dt_time.dt.minute

    return df


def handle_rare_locations(
    df: pd.DataFrame, threshold: int = 20
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Group infrequent locations into 'Other' category."""
    df = df.copy()

    p_counts = df["pickup_location"].value_counts()
    rare_pickups = list(p_counts[p_counts < threshold].index)
    df["pickup_location"] = df["pickup_location"].replace(rare_pickups, "Other")

    d_counts = df["drop_location"].value_counts()
    rare_drops = list(d_counts[d_counts < threshold].index)
    df["drop_location"] = df["drop_location"].replace(rare_drops, "Other")

    return df, rare_pickups, rare_drops


def engineer_features_pipeline(
    input_path: Path | str | None = None,
    save_artifacts: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full feature engineering and train/test preparation pipeline."""
    in_path = Path(input_path) if input_path else config.ingested_data_path
    if not in_path.exists():
        raise FileNotFoundError(f"Ingested data not found at: {in_path}")

    logger.info(f"Running feature engineering on: {in_path}")
    df = pd.read_csv(in_path)

    # 1. Temporal feature extraction
    df = extract_temporal_features(df)

    # 2. Compute distance if not present
    if "trip_distance_km" not in df.columns:
        df["trip_distance_km"] = calculate_haversine_distance(
            df["pickup_location"], df["drop_location"]
        )

    # 3. Default features if missing
    if "passenger_count" not in df.columns:
        df["passenger_count"] = config.default_passenger_count
    if "surge_multiplier" not in df.columns:
        df["surge_multiplier"] = config.default_surge_multiplier

    # 4. Handle rare categories
    df, _, _ = handle_rare_locations(df, threshold=config.location_threshold)

    # 5. Clean up temporary columns
    drop_candidates = ["pickup_date", "pickup_time", "trip_id", "average_speed", "time_of_day"]
    df = df.drop(columns=[c for c in drop_candidates if c in df.columns])

    # 6. Separate features and target
    target_col = config.target_column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 7. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    # 8. Encode traffic level ordinal
    X_train["traffic_level"] = X_train["traffic_level"].map(config.traffic_mapping).fillna(1).astype(int)
    X_test["traffic_level"] = X_test["traffic_level"].map(config.traffic_mapping).fillna(1).astype(int)

    # 9. One-Hot Encoding for categorical features
    cat_cols = config.categorical_columns
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_train[cat_cols])

    encoded_train = pd.DataFrame(
        encoder.transform(X_train[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_train.index,
    )
    encoded_test = pd.DataFrame(
        encoder.transform(X_test[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_test.index,
    )

    X_train_final = pd.concat([X_train.drop(columns=cat_cols), encoded_train], axis=1)
    X_test_final = pd.concat([X_test.drop(columns=cat_cols), encoded_test], axis=1)

    feature_column_names = list(X_train_final.columns)

    # 10. Combine with target for persistent datasets
    train_df = pd.concat([X_train_final, y_train.rename(target_col)], axis=1)
    test_df = pd.concat([X_test_final, y_test.rename(target_col)], axis=1)

    if save_artifacts:
        config.processed_dir.mkdir(parents=True, exist_ok=True)
        config.store_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(config.train_data_path, index=False)
        test_df.to_csv(config.test_data_path, index=False)

        joblib.dump(encoder, config.encoder_path)
        with open(config.feature_columns_path, "w", encoding="utf-8") as f:
            json.dump(feature_column_names, f, indent=2)

        logger.info(f"Saved train features to: {config.train_data_path} ({len(train_df)} rows)")
        logger.info(f"Saved test features to: {config.test_data_path} ({len(test_df)} rows)")
        logger.info(f"Saved encoder to: {config.encoder_path}")
        logger.info(f"Saved {len(feature_column_names)} feature column definitions to: {config.feature_columns_path}")

    return train_df, test_df


if __name__ == "__main__":
    engineer_features_pipeline()
