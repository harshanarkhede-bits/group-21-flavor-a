from __future__ import annotations

import numpy as np

ALLOWED_LOCATIONS = [
    "Upper West Side",
    "Harlem",
    "East Harlem",
    "Bedford-Stuyvesant",
    "South Slope",
    "Hell's Kitchen",
    "West Village",
    "Kensington",
    "Fort Greene",
    "Clinton Hill",
    "Williamsburg",
    "Murray Hill",
    "Midtown",
    "Park Slope",
    "Crown Heights",
    "Inwood",
    "Chinatown",
    "Tottenville",
    "Sheepshead Bay",
    "Midwood",
    "Chelsea",
    "Great Kills",
    "Bensonhurst",
    "Arverne",
    "Arden Heights",
]

LOCATION_COORDINATES = {
    "Upper West Side": (40.78705, -73.97542),
    "Harlem": (40.81160, -73.94650),
    "East Harlem": (40.79793, -73.93999),
    "Bedford-Stuyvesant": (40.68902, -73.93925),
    "South Slope": (40.66300, -73.99000),
    "Hell's Kitchen": (40.76281, -73.99317),
    "West Village": (40.73368, -74.00931),
    "Kensington": (40.64200, -73.97200),
    "Fort Greene": (40.69011, -73.97477),
    "Clinton Hill": (40.68972, -73.96528),
    "Williamsburg": (40.70810, -73.95710),
    "Murray Hill": (40.74763, -73.97677),
    "Midtown": (40.75490, -73.98400),
    "Park Slope": (40.67010, -73.98597),
    "Crown Heights": (40.66747, -73.94357),
    "Inwood": (40.86770, -73.92120),
    "Chinatown": (40.71500, -73.99700),
    "Tottenville": (40.51280, -74.25120),
    "Sheepshead Bay": (40.59120, -73.94460),
    "Midwood": (40.62500, -73.95700),
    "Chelsea": (40.74650, -74.00140),
    "Great Kills": (40.55000, -74.15000),
    "Bensonhurst": (40.60140, -73.99430),
    "Arverne": (40.59000, -73.79000),
    "Arden Heights": (40.55500, -74.18500),
}


def calculate_haversine_distance_km(pickup: str, drop: str) -> float:
    """Calculate approximate Haversine distance in km between two NYC neighborhoods."""
    if pickup not in LOCATION_COORDINATES or drop not in LOCATION_COORDINATES:
        return 1.0

    lat1, lon1 = np.radians(LOCATION_COORDINATES[pickup])
    lat2, lon2 = np.radians(LOCATION_COORDINATES[drop])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    earth_radius_km = 6371.0

    return round(float(earth_radius_km * c), 3)