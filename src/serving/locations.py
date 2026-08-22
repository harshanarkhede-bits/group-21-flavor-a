# ============================================================
# locations.py
# ETA PROJECT - LOCATION CONFIGURATION
# ============================================================
#
# Contains the 25 locations exposed by the application.
#
# Coordinates are representative coordinates for each
# neighborhood and are used for approximate distance
# calculation.
#
# No external API is required.
# ============================================================


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


# ------------------------------------------------------------
# Location -> (latitude, longitude)
# ------------------------------------------------------------

LOCATION_COORDINATES = {

    "Upper West Side": (
        40.78705,
        -73.97542
    ),

    "Harlem": (
        40.8116,
        -73.9465
    ),

    "East Harlem": (
        40.79793,
        -73.93999
    ),

    "Bedford-Stuyvesant": (
        40.68902,
        -73.93925
    ),

    "South Slope": (
        40.66300,
        -73.99000
    ),

    "Hell's Kitchen": (
        40.76281,
        -73.99317
    ),

    "West Village": (
        40.73368,
        -74.00931
    ),

    "Kensington": (
        40.64200,
        -73.97200
    ),

    "Fort Greene": (
        40.69011,
        -73.97477
    ),

    "Clinton Hill": (
        40.68972,
        -73.96528
    ),

    "Williamsburg": (
        40.70810,
        -73.95710
    ),

    "Murray Hill": (
        40.74763,
        -73.97677
    ),

    "Midtown": (
        40.75490,
        -73.98400
    ),

    "Park Slope": (
        40.67010,
        -73.98597
    ),

    "Crown Heights": (
        40.66747,
        -73.94357
    ),

    "Inwood": (
        40.86770,
        -73.92120
    ),

    "Chinatown": (
        40.71500,
        -73.99700
    ),

    "Tottenville": (
        40.51280,
        -74.25120
    ),

    "Sheepshead Bay": (
        40.59120,
        -73.94460
    ),

    "Midwood": (
        40.62500,
        -73.95700
    ),

    "Chelsea": (
        40.74650,
        -74.00140
    ),

    "Great Kills": (
        40.55000,
        -74.15000
    ),

    "Bensonhurst": (
        40.60140,
        -73.99430
    ),

    "Arverne": (
        40.59000,
        -73.79000
    ),

    "Arden Heights": (
        40.55500,
        -74.18500
    ),
}


# ============================================================
# VALIDATION
# ============================================================

# Make sure every allowed location has coordinates.

missing_coordinates = [
    location
    for location in ALLOWED_LOCATIONS
    if location not in LOCATION_COORDINATES
]


if missing_coordinates:

    raise ValueError(
        "Missing coordinates for: "
        + ", ".join(missing_coordinates)
    )


# Make sure we don't have coordinates for a location
# that isn't allowed.

extra_coordinates = [
    location
    for location in LOCATION_COORDINATES
    if location not in ALLOWED_LOCATIONS
]


if extra_coordinates:

    raise ValueError(
        "Coordinates exist for locations that are not "
        "in ALLOWED_LOCATIONS: "
        + ", ".join(extra_coordinates)
    )


print(
    f"Location configuration loaded successfully: "
    f"{len(ALLOWED_LOCATIONS)} locations"
)