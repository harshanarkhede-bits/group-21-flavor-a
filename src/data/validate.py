from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import config
from src.serving.locations import ALLOWED_LOCATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data fails validation assertions."""
    pass


def validate_schema_and_quality(
    df: pd.DataFrame,
    strict_locations: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate data schema, types, range bounds, and categorical values.
    Returns (is_valid, list_of_errors).
    """
    errors: List[str] = []

    # Required columns
    expected_cols = [
        "pickup_location",
        "drop_location",
        "pickup_date",
        "pickup_time",
        "traffic_level",
        config.target_column,
    ]
    for col in expected_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    if errors:
        return False, errors

    # Check for empty dataframe
    if df.empty:
        errors.append("Dataset is empty.")
        return False, errors

    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        errors.append(f"Found missing values in columns: {cols_with_nulls}")

    # Check target column range
    if config.target_column in df.columns:
        invalid_target = df[
            (df[config.target_column] < config.min_eta_minutes)
            | (df[config.target_column] > config.max_eta_minutes)
        ]
        if not invalid_target.empty:
            errors.append(
                f"Found {len(invalid_target)} rows with '{config.target_column}' outside [{config.min_eta_minutes}, {config.max_eta_minutes}]"
            )

    # Check traffic_level categories
    if "traffic_level" in df.columns:
        valid_traffic = set(config.traffic_mapping.keys())
        actual_traffic = set(df["traffic_level"].unique())
        invalid_traffic = actual_traffic - valid_traffic
        if invalid_traffic:
            errors.append(f"Found unexpected traffic levels: {invalid_traffic}. Expected subset of {valid_traffic}")

    # Check location constraints if strict
    if strict_locations and "pickup_location" in df.columns:
        invalid_pickups = set(df["pickup_location"].unique()) - set(ALLOWED_LOCATIONS)
        if invalid_pickups:
            errors.append(f"Found unknown pickup locations: {invalid_pickups}")

    is_valid = len(errors) == 0
    return is_valid, errors


def run_validation(data_path: Path | str | None = None) -> bool:
    """Run validation against the ingested dataset and report results."""
    path = Path(data_path) if data_path else config.ingested_data_path
    if not path.exists():
        raise FileNotFoundError(f"Data to validate not found at: {path}")

    logger.info(f"Validating dataset from: {path}")
    df = pd.read_csv(path)
    is_valid, errors = validate_schema_and_quality(df)

    if not is_valid:
        logger.error(f"Data validation FAILED with {len(errors)} issues:")
        for err in errors:
            logger.error(f"  - {err}")
        raise DataValidationError("\n".join(errors))

    logger.info(f"Data validation PASSED! Verified {len(df)} rows and {df.shape[1]} columns successfully.")
    return True


if __name__ == "__main__":
    run_validation()
