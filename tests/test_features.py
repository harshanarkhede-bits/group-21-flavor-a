import pytest
import pandas as pd
import numpy as np
from src.features.engineer import (
    calculate_haversine_distance,
    extract_temporal_features,
    handle_rare_locations,
)
from src.serving.locations import calculate_haversine_distance_km


def test_calculate_haversine_distance_km():
    dist = calculate_haversine_distance_km("Upper West Side", "Harlem")
    assert isinstance(dist, float)
    assert dist > 1.0  # Approx 3-5 km in Manhattan


def test_calculate_haversine_distance_series():
    pickups = pd.Series(["Upper West Side", "Harlem"])
    drops = pd.Series(["Harlem", "Midtown"])
    distances = calculate_haversine_distance(pickups, drops)
    assert len(distances) == 2
    assert (distances > 0).all()


def test_extract_temporal_features():
    df = pd.DataFrame({
        "pickup_date": ["14-03-2016"],
        "pickup_time": ["17:24"],
    })
    df_feat = extract_temporal_features(df)
    assert df_feat["month"].iloc[0] == 3
    assert df_feat["day"].iloc[0] == 14
    assert df_feat["pickup_hour"].iloc[0] == 17
    assert df_feat["pickup_minute"].iloc[0] == 24
    assert df_feat["weekday"].iloc[0] == "Monday"
    assert df_feat["is_weekend"].iloc[0] == 0
    assert df_feat["season"].iloc[0] == "Spring"


def test_handle_rare_locations():
    df = pd.DataFrame({
        "pickup_location": ["Harlem"] * 25 + ["RareLoc1"] * 2,
        "drop_location": ["Midtown"] * 25 + ["RareLoc2"] * 2,
    })
    df_out, rare_p, rare_d = handle_rare_locations(df, threshold=20)
    assert "Other" in df_out["pickup_location"].values
    assert "RareLoc1" in rare_p
    assert "RareLoc2" in rare_d
