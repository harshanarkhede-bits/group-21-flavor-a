# ============================================================
# ETA PREDICTION - DATA DRIFT MONITORING
# ============================================================
#
# Purpose:
#   Compare training/reference data against real API
#   prediction traffic.
#
# Reference:
#   monitoring/reference_data.csv
#
# Production:
#   monitoring/prediction_logs.csv
#
# Output:
#   monitoring/drift_report.json
#
# The script monitors:
#
#   Numerical features:
#       - pickup_hour
#       - trip_distance_km
#       - passenger_count
#       - surge_multiplier
#
#   Categorical features:
#       - pickup_location
#       - drop_location
#       - traffic_level
#       - weekday
#       - season
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# drift_monitor.py is inside:
#
# group-21-flavor-a/
#     monitoring/
#         drift_monitor.py
#
# Therefore:
# parents[1] = project root

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


MONITORING_DIR = (
    PROJECT_ROOT / "monitoring"
)


REFERENCE_DATA_PATH = (
    MONITORING_DIR
    / "reference_data.csv"
)


PRODUCTION_DATA_PATH = (
    MONITORING_DIR
    / "prediction_logs.csv"
)


DRIFT_REPORT_PATH = (
    MONITORING_DIR
    / "drift_report.json"
)


# ============================================================
# 3. MONITORING CONFIGURATION
# ============================================================

# Minimum number of production predictions required
# before we attempt meaningful drift monitoring.

MIN_PRODUCTION_ROWS = 10


# ------------------------------------------------------------
# Numerical drift threshold
# ------------------------------------------------------------
#
# We compare the mean percentage change.
#
# Example:
#
# Reference mean = 4.0
# Production mean = 6.0
#
# Change = 50%
#
# This would be considered drift.
#
# ------------------------------------------------------------

NUMERICAL_DRIFT_THRESHOLD = 0.20


# ------------------------------------------------------------
# Categorical drift threshold
# ------------------------------------------------------------
#
# We calculate the maximum difference between category
# proportions.
#
# Example:
#
# Reference:
# High = 25%
#
# Production:
# High = 60%
#
# Difference = 35 percentage points
#
# This exceeds 20 percentage points and is flagged.
# ------------------------------------------------------------

CATEGORICAL_DRIFT_THRESHOLD = 0.20


# ============================================================
# 4. FEATURES TO MONITOR
# ============================================================

NUMERICAL_FEATURES = [

    "pickup_hour",

    "trip_distance_km",

    "passenger_count",

    "surge_multiplier"
]


CATEGORICAL_FEATURES = [

    "pickup_location",

    "drop_location",

    "traffic_level",

    "weekday",

    "season"
]


# ============================================================
# 5. LOAD DATA
# ============================================================

def load_data():
    """
    Load reference and production datasets.
    """

    print()
    print("=" * 70)
    print("LOADING MONITORING DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Check reference data
    # --------------------------------------------------------

    if not REFERENCE_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Reference dataset not found:\n"
            f"{REFERENCE_DATA_PATH}\n\n"
            f"Run:\n"
            f"python monitoring/create_reference_data.py"
        )


    # --------------------------------------------------------
    # Check production data
    # --------------------------------------------------------

    if not PRODUCTION_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Production prediction log not found:\n"
            f"{PRODUCTION_DATA_PATH}\n\n"
            f"Generate predictions through the API first."
        )


    # --------------------------------------------------------
    # Read CSV files
    # --------------------------------------------------------

    reference_df = pd.read_csv(
        REFERENCE_DATA_PATH
    )


    production_df = pd.read_csv(
        PRODUCTION_DATA_PATH
    )


    print(
        "Reference rows:",
        len(reference_df)
    )


    print(
        "Production rows:",
        len(production_df)
    )


    return (
        reference_df,
        production_df
    )


# ============================================================
# 6. VALIDATE PRODUCTION DATA SIZE
# ============================================================

def validate_production_data(
    production_df
):
    """
    Check whether enough production predictions exist
    for monitoring.
    """

    print()
    print("=" * 70)
    print("CHECKING PRODUCTION DATA")
    print("=" * 70)


    if len(production_df) < MIN_PRODUCTION_ROWS:

        print(
            f"WARNING: Only {len(production_df)} "
            f"production predictions found."
        )

        print(
            f"At least {MIN_PRODUCTION_ROWS} "
            f"predictions are recommended."
        )

        print(
            "Generate more predictions before "
            "drawing conclusions from the drift results."
        )

        return False


    print(
        "Enough production predictions available."
    )

    return True


# ============================================================
# 7. NUMERICAL DRIFT
# ============================================================

def calculate_numerical_drift(
    reference_df,
    production_df
):
    """
    Compare numerical feature distributions.

    For each feature we calculate:

        reference mean
        production mean
        reference std
        production std
        absolute mean difference
        relative mean change
        drift status
    """

    print()
    print("=" * 70)
    print("NUMERICAL DATA DRIFT")
    print("=" * 70)


    results = {}


    for feature in NUMERICAL_FEATURES:

        # ----------------------------------------------------
        # Check columns
        # ----------------------------------------------------

        if feature not in reference_df.columns:

            print(
                f"Skipping {feature}: "
                f"not found in reference data."
            )

            continue


        if feature not in production_df.columns:

            print(
                f"Skipping {feature}: "
                f"not found in production data."
            )

            continue


        # ----------------------------------------------------
        # Convert to numeric
        # ----------------------------------------------------

        reference_values = pd.to_numeric(
            reference_df[feature],
            errors="coerce"
        ).dropna()


        production_values = pd.to_numeric(
            production_df[feature],
            errors="coerce"
        ).dropna()


        # ----------------------------------------------------
        # Check data
        # ----------------------------------------------------

        if len(reference_values) == 0:

            print(
                f"Skipping {feature}: "
                f"no valid reference values."
            )

            continue


        if len(production_values) == 0:

            print(
                f"Skipping {feature}: "
                f"no valid production values."
            )

            continue


        # ----------------------------------------------------
        # Calculate statistics
        # ----------------------------------------------------

        reference_mean = float(
            reference_values.mean()
        )


        production_mean = float(
            production_values.mean()
        )


        reference_std = float(
            reference_values.std()
        )


        production_std = float(
            production_values.std()
        )


        absolute_difference = abs(
            production_mean
            -
            reference_mean
        )


        # ----------------------------------------------------
        # Relative change
        # ----------------------------------------------------
        #
        # Avoid division by zero.
        #
        # ----------------------------------------------------

        if abs(reference_mean) > 1e-9:

            relative_change = (
                absolute_difference
                /
                abs(reference_mean)
            )

        else:

            relative_change = (
                0.0
                if absolute_difference == 0
                else 1.0
            )


        # ----------------------------------------------------
        # Determine drift
        # ----------------------------------------------------

        drift_detected = (
            relative_change
            >
            NUMERICAL_DRIFT_THRESHOLD
        )


        status = (
            "DRIFT DETECTED"
            if drift_detected
            else
            "NO DRIFT"
        )


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results[feature] = {

            "reference_mean":
                round(
                    reference_mean,
                    4
                ),

            "production_mean":
                round(
                    production_mean,
                    4
                ),

            "reference_std":
                round(
                    reference_std,
                    4
                ),

            "production_std":
                round(
                    production_std,
                    4
                ),

            "absolute_mean_difference":
                round(
                    absolute_difference,
                    4
                ),

            "relative_mean_change":
                round(
                    relative_change,
                    4
                ),

            "drift_threshold":
                NUMERICAL_DRIFT_THRESHOLD,

            "drift_detected":
                drift_detected,

            "status":
                status
        }


        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            f"\nFeature: {feature}"
        )


        print(
            f"  Reference mean : "
            f"{reference_mean:.4f}"
        )


        print(
            f"  Production mean: "
            f"{production_mean:.4f}"
        )


        print(
            f"  Mean change    : "
            f"{relative_change * 100:.2f}%"
        )


        print(
            f"  Status         : "
            f"{status}"
        )


    return results


# ============================================================
# 8. CATEGORICAL DRIFT
# ============================================================

def calculate_categorical_drift(
    reference_df,
    production_df
):
    """
    Compare categorical feature distributions.

    Example:

    Reference:

        Low       40%
        Medium    35%
        High      25%

    Production:

        Low       10%
        Medium    30%
        High      60%

    We calculate the maximum difference in proportions.
    """

    print()
    print("=" * 70)
    print("CATEGORICAL DATA DRIFT")
    print("=" * 70)


    results = {}


    for feature in CATEGORICAL_FEATURES:

        # ----------------------------------------------------
        # Check columns
        # ----------------------------------------------------

        if feature not in reference_df.columns:

            print(
                f"Skipping {feature}: "
                f"not found in reference data."
            )

            continue


        if feature not in production_df.columns:

            print(
                f"Skipping {feature}: "
                f"not found in production data."
            )

            continue


        # ----------------------------------------------------
        # Convert to string
        # ----------------------------------------------------

        reference_values = (
            reference_df[feature]
            .dropna()
            .astype(str)
        )


        production_values = (
            production_df[feature]
            .dropna()
            .astype(str)
        )


        if len(reference_values) == 0:

            print(
                f"Skipping {feature}: "
                f"no reference values."
            )

            continue


        if len(production_values) == 0:

            print(
                f"Skipping {feature}: "
                f"no production values."
            )

            continue


        # ----------------------------------------------------
        # Calculate distributions
        # ----------------------------------------------------

        reference_distribution = (
            reference_values
            .value_counts(
                normalize=True
            )
        )


        production_distribution = (
            production_values
            .value_counts(
                normalize=True
            )
        )


        # ----------------------------------------------------
        # Combine categories
        # ----------------------------------------------------

        categories = sorted(
            set(
                reference_distribution.index
            )
            |
            set(
                production_distribution.index
            )
        )


        # ----------------------------------------------------
        # Calculate differences
        # ----------------------------------------------------

        category_differences = {}


        for category in categories:

            reference_percentage = (
                float(
                    reference_distribution.get(
                        category,
                        0.0
                    )
                )
            )


            production_percentage = (
                float(
                    production_distribution.get(
                        category,
                        0.0
                    )
                )
            )


            difference = abs(
                production_percentage
                -
                reference_percentage
            )


            category_differences[
                category
            ] = {

                "reference_percentage":
                    round(
                        reference_percentage,
                        4
                    ),

                "production_percentage":
                    round(
                        production_percentage,
                        4
                    ),

                "absolute_difference":
                    round(
                        difference,
                        4
                    )
            }


        # ----------------------------------------------------
        # Maximum distribution difference
        # ----------------------------------------------------

        max_difference = max(

            item[
                "absolute_difference"
            ]

            for item
            in category_differences.values()
        )


        # ----------------------------------------------------
        # Determine drift
        # ----------------------------------------------------

        drift_detected = (
            max_difference
            >
            CATEGORICAL_DRIFT_THRESHOLD
        )


        status = (

            "DRIFT DETECTED"

            if drift_detected

            else

            "NO DRIFT"
        )


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results[feature] = {

            "category_distribution":
                category_differences,

            "maximum_distribution_difference":
                round(
                    max_difference,
                    4
                ),

            "drift_threshold":
                CATEGORICAL_DRIFT_THRESHOLD,

            "drift_detected":
                drift_detected,

            "status":
                status
        }


        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            f"\nFeature: {feature}"
        )


        print(
            "  Category distribution:"
        )


        for category, values in (
            category_differences.items()
        ):

            print(
                f"    {category}: "
                f"reference="
                f"{values['reference_percentage'] * 100:.2f}% "
                f"| production="
                f"{values['production_percentage'] * 100:.2f}%"
            )


        print(
            f"  Maximum difference: "
            f"{max_difference * 100:.2f}%"
        )


        print(
            f"  Status: {status}"
        )


    return results


# ============================================================
# 9. OVERALL DRIFT STATUS
# ============================================================

def calculate_overall_status(
    numerical_results,
    categorical_results
):
    """
    Determine the overall monitoring status.
    """

    numerical_drift_count = sum(

        1

        for result
        in numerical_results.values()

        if result[
            "drift_detected"
        ]
    )


    categorical_drift_count = sum(

        1

        for result
        in categorical_results.values()

        if result[
            "drift_detected"
        ]
    )


    total_drift_count = (
        numerical_drift_count
        +
        categorical_drift_count
    )


    if total_drift_count > 0:

        overall_status = (
            "DRIFT DETECTED"
        )

    else:

        overall_status = (
            "NO DRIFT DETECTED"
        )


    return {

        "overall_status":
            overall_status,

        "numerical_features_with_drift":
            numerical_drift_count,

        "categorical_features_with_drift":
            categorical_drift_count,

        "total_features_with_drift":
            total_drift_count
    }


# ============================================================
# 10. GENERATE DRIFT REPORT
# ============================================================

def generate_report(
    reference_df,
    production_df,
    numerical_results,
    categorical_results,
    overall_status
):
    """
    Create JSON monitoring report.
    """

    report = {

        "monitoring_timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "reference_dataset":
            str(
                REFERENCE_DATA_PATH
            ),

        "production_dataset":
            str(
                PRODUCTION_DATA_PATH
            ),

        "reference_rows":
            int(
                len(reference_df)
            ),

        "production_rows":
            int(
                len(production_df)
            ),

        "configuration": {

            "minimum_production_rows":
                MIN_PRODUCTION_ROWS,

            "numerical_drift_threshold":
                NUMERICAL_DRIFT_THRESHOLD,

            "categorical_drift_threshold":
                CATEGORICAL_DRIFT_THRESHOLD
        },

        "overall":
            overall_status,

        "numerical_features":
            numerical_results,

        "categorical_features":
            categorical_results
    }


    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        DRIFT_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )


    return report


# ============================================================
# 11. PRINT FINAL SUMMARY
# ============================================================

def print_summary(
    overall_status
):
    """
    Print final monitoring summary.
    """

    print()
    print()
    print("=" * 70)
    print("DATA DRIFT MONITORING SUMMARY")
    print("=" * 70)


    print(
        "\nOverall status:",
        overall_status[
            "overall_status"
        ]
    )


    print(
        "\nNumerical features with drift:",
        overall_status[
            "numerical_features_with_drift"
        ]
    )


    print(
        "Categorical features with drift:",
        overall_status[
            "categorical_features_with_drift"
        ]
    )


    print(
        "Total features with drift:",
        overall_status[
            "total_features_with_drift"
        ]
    )


    print(
        "\nDrift report saved to:"
    )


    print(
        DRIFT_REPORT_PATH
    )


    print()
    print("=" * 70)


# ============================================================
# 12. MAIN MONITORING WORKFLOW
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ETA MODEL - DATA DRIFT MONITOR")
    print("=" * 70)


    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    (
        reference_df,
        production_df
    ) = load_data()


    # --------------------------------------------------------
    # Validate production data
    # --------------------------------------------------------

    enough_data = (
        validate_production_data(
            production_df
        )
    )


    # --------------------------------------------------------
    # Numerical drift
    # --------------------------------------------------------

    numerical_results = (
        calculate_numerical_drift(
            reference_df,
            production_df
        )
    )


    # --------------------------------------------------------
    # Categorical drift
    # --------------------------------------------------------

    categorical_results = (
        calculate_categorical_drift(
            reference_df,
            production_df
        )
    )


    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    overall_status = (
        calculate_overall_status(
            numerical_results,
            categorical_results
        )
    )


    # Add warning if production dataset is small

    if not enough_data:

        overall_status[
            "warning"
        ] = (
            "Production dataset contains fewer "
            "than the recommended number of predictions."
        )


    # --------------------------------------------------------
    # Generate report
    # --------------------------------------------------------

    generate_report(

        reference_df,

        production_df,

        numerical_results,

        categorical_results,

        overall_status
    )


    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(
        overall_status
    )


# ============================================================
# 13. RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    main()