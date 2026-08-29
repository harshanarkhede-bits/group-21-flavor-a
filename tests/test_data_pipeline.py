import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import config
from src.data.ingest import ingest_data
from src.data.validate import validate_schema_and_quality, DataValidationError


def test_validate_schema_and_quality_valid_data():
    df = pd.DataFrame({
        "pickup_location": ["Upper West Side", "Harlem"],
        "drop_location": ["Harlem", "Midtown"],
        "pickup_date": ["2026-08-27", "2026-08-27"],
        "pickup_time": ["12:00", "15:30"],
        "traffic_level": ["Low", "High"],
        "actual_eta_minutes": [12.5, 25.0],
        "passenger_count": [1, 2],
        "trip_distance_km": [2.5, 5.0],
    })
    is_valid, errors = validate_schema_and_quality(df)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_schema_and_quality_detects_missing_cols():
    df = pd.DataFrame({"pickup_location": ["Harlem"]})
    is_valid, errors = validate_schema_and_quality(df)
    assert is_valid is False
    assert any("Missing required column" in e for e in errors)


def test_validate_schema_and_quality_detects_out_of_bounds_target():
    df = pd.DataFrame({
        "pickup_location": ["Upper West Side"],
        "drop_location": ["Harlem"],
        "pickup_date": ["2026-08-27"],
        "pickup_time": ["12:00"],
        "traffic_level": ["Low"],
        "actual_eta_minutes": [999.0],  # Out of bounds (>200)
    })
    is_valid, errors = validate_schema_and_quality(df)
    assert is_valid is False
    assert any("outside" in e for e in errors)


def test_ingest_data_cleans_and_creates_file(tmp_path):
    raw_csv = tmp_path / "sample_raw.csv"
    out_csv = tmp_path / "sample_ingested.csv"

    df_raw = pd.DataFrame({
        "trip_id": ["id1", "id2", "id3"],
        "pickup_location": ["Upper West Side", "Harlem", "Harlem"],
        "drop_location": ["Harlem", "Midtown", "Midtown"],
        "pickup_date": ["14-03-2016", "12-06-2016", "12-06-2016"],
        "pickup_time": ["17:24", "00:43", "00:43"],
        "time_of_day": ["2. Afternoon", "4. Night", "4. Night"],
        "season": ["2. Spring", "3. Summer", "3. Summer"],
        "passenger_count": [1, 2, 0],  # 0 passengers should be dropped
        "trip_distance_km": [1.96, 2.27, 3.0],
        "traffic_level": ["High", "Low", "Low"],
        "actual_eta_minutes": [7.6, 11.0, 15.0],
    })
    df_raw.to_csv(raw_csv, index=False)

    df_ingested = ingest_data(raw_data_path=raw_csv, output_path=out_csv)
    assert len(df_ingested) == 2  # 0-passenger row dropped
    assert "trip_id" not in df_ingested.columns
    assert out_csv.exists()
