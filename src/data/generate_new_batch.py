"""
src/data/generate_new_batch.py
-------------------------------
Simulates real-world new data arriving in production by generating
synthetic NYC trip records with configurable drift scenarios.

Scenarios:
  - normal      : Same distribution as training data (no drift)
  - rush_hour   : Evening surge — higher ETAs, more High traffic
  - weather     : Storm/rain scenario — all ETAs inflated
  - weekend     : Weekend leisure pattern — different location distribution
  - gradual     : Mild 20% ETA creep (subtle model decay)
  - extreme     : Combined rush + weather (triggers retrain)

Usage:
  python src/data/generate_new_batch.py --scenario rush_hour --n 500
  python src/data/generate_new_batch.py --scenario extreme --n 1000 --append
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import config
from src.serving.locations import ALLOWED_LOCATIONS, LOCATION_COORDINATES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Scenario definitions — how they shift ETA distributions
# ─────────────────────────────────────────────────────────
SCENARIOS = {
    "normal": {
        "description": "Normal operations — same distribution as training",
        "eta_multiplier": 1.00,
        "traffic_high_prob": 0.25,
        "surge_range": (1.0, 1.2),
        "peak_hours": list(range(0, 24)),
    },
    "rush_hour": {
        "description": "Evening rush hour surge (16:00–20:00) — ETAs +45%",
        "eta_multiplier": 1.45,
        "traffic_high_prob": 0.70,
        "surge_range": (1.3, 2.5),
        "peak_hours": list(range(16, 21)),
    },
    "weather": {
        "description": "Severe weather / rainstorm — ETAs +60% across all trips",
        "eta_multiplier": 1.60,
        "traffic_high_prob": 0.80,
        "surge_range": (1.5, 3.0),
        "peak_hours": list(range(7, 22)),
    },
    "weekend": {
        "description": "Weekend leisure pattern — slower trips, different routes",
        "eta_multiplier": 1.20,
        "traffic_high_prob": 0.30,
        "surge_range": (1.0, 1.5),
        "peak_hours": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    },
    "gradual": {
        "description": "Gradual model decay — subtle +20% ETA creep",
        "eta_multiplier": 1.20,
        "traffic_high_prob": 0.35,
        "surge_range": (1.0, 1.3),
        "peak_hours": list(range(0, 24)),
    },
    "extreme": {
        "description": "Extreme scenario (rush + weather combined) — triggers retrain",
        "eta_multiplier": 1.90,
        "traffic_high_prob": 0.90,
        "surge_range": (2.0, 4.5),
        "peak_hours": list(range(15, 22)),
    },
}

TRAFFIC_LEVELS = ["Low", "Medium", "High"]
NYC_LOCATIONS: List[str] = list(ALLOWED_LOCATIONS)
SEASONS = {
    1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall",
    11: "Fall", 12: "Winter",
}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _sample_date(peak_hours: List[int]) -> tuple[str, str, int, int, str, int, int, str]:
    """Sample a random date/time weighted toward peak hours for the scenario."""
    # Random date in last 90 days
    base_date = date.today() - timedelta(days=random.randint(0, 90))
    hour = random.choice(peak_hours)
    minute = random.randint(0, 59)
    weekday_idx = base_date.weekday()
    weekday_name = WEEKDAYS[weekday_idx]
    is_weekend = int(weekday_idx >= 5)
    month = base_date.month
    season = SEASONS[month]
    return (
        base_date.strftime("%Y-%m-%d"),
        f"{hour:02d}:{minute:02d}",
        hour, minute,
        weekday_name, is_weekend,
        month, season,
    )


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Vectorised Haversine distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def generate_new_batch(
    n: int = 500,
    scenario: str = "normal",
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate n synthetic trip records under the specified drift scenario.

    Returns a DataFrame matching the raw ETA_Model_data.csv schema so it
    can be directly appended and re-ingested through the pipeline.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {list(SCENARIOS.keys())}")

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    cfg = SCENARIOS[scenario]
    logger.info(f"Generating {n} records — Scenario: '{scenario}' — {cfg['description']}")

    records = []
    location_names = NYC_LOCATIONS

    for _ in range(n):
        pickup_loc = random.choice(location_names)
        drop_loc = random.choice([l for l in location_names if l != pickup_loc])

        # Get coordinates for distance calculation
        pickup_coords = LOCATION_COORDINATES.get(pickup_loc, (40.7580, -73.9855))
        drop_coords = LOCATION_COORDINATES.get(drop_loc, (40.6501, -73.9496))

        distance_km = round(
            _haversine(
                pickup_coords[0], pickup_coords[1],
                drop_coords[0], drop_coords[1],
            ),
            3,
        )
        distance_km = max(distance_km, 0.5)

        # Sample date/time
        pickup_date, pickup_time, hour, minute, weekday, is_weekend, month, season = _sample_date(
            cfg["peak_hours"]
        )

        # Traffic level — biased by scenario
        traffic_roll = random.random()
        if traffic_roll < cfg["traffic_high_prob"]:
            traffic_level = "High"
        elif traffic_roll < cfg["traffic_high_prob"] + 0.30:
            traffic_level = "Medium"
        else:
            traffic_level = "Low"

        traffic_numeric = {"Low": 0, "Medium": 1, "High": 2}[traffic_level]

        # Surge multiplier
        surge = round(random.uniform(*cfg["surge_range"]), 2)

        # Passenger count
        passenger_count = random.choices([1, 2, 3, 4], weights=[0.5, 0.25, 0.15, 0.10])[0]

        # Base ETA from distance + traffic heuristic
        speed_kmh = {"Low": 45, "Medium": 28, "High": 15}[traffic_level]
        base_eta = (distance_km / speed_kmh) * 60  # minutes
        noise = np.random.normal(0, base_eta * 0.10)
        actual_eta = max(0.5, (base_eta + noise) * cfg["eta_multiplier"])
        actual_eta = round(actual_eta, 2)

        records.append({
            "pickup_location": pickup_loc,
            "drop_location": drop_loc,
            "pickup_date": pickup_date,
            "pickup_time": pickup_time,
            "pickup_hour": hour,
            "pickup_minute": minute,
            "weekday": weekday,
            "is_weekend": is_weekend,
            "month": month,
            "season": season,
            "trip_distance_km": distance_km,
            "traffic_level": traffic_level,
            "passenger_count": passenger_count,
            "surge_multiplier": surge,
            "actual_eta_minutes": actual_eta,
        })

    df = pd.DataFrame(records)
    logger.info(
        f"Generated {len(df)} trips | Mean ETA: {df['actual_eta_minutes'].mean():.1f} min "
        f"| Traffic High %: {(df['traffic_level']=='High').mean()*100:.1f}%"
    )
    return df


def save_new_batch(
    df: pd.DataFrame,
    append_to_raw: bool = False,
    batch_name: str | None = None,
) -> Path:
    """
    Save the new batch.
    - If append_to_raw=False: saves to data/incoming/<batch_name>.csv (staging area)
    - If append_to_raw=True:  appends to data/raw/ETA_Model_data.csv (triggers full repro)
    """
    incoming_dir = PROJECT_ROOT / "data" / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)

    if batch_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"batch_{ts}"

    if append_to_raw:
        raw_path = config.raw_data_path
        existing = pd.read_csv(raw_path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_csv(raw_path, index=False)
        logger.info(
            f"Appended {len(df)} records to raw dataset. "
            f"New total: {len(combined)} rows at {raw_path}"
        )
        return raw_path
    else:
        out_path = incoming_dir / f"{batch_name}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved new batch ({len(df)} rows) to staging: {out_path}")
        return out_path


def run_full_pipeline_on_new_data() -> None:
    """Re-run the full DVC pipeline after new data is staged."""
    import subprocess
    logger.info("Re-running full ML pipeline (ingest → validate → features → train → evaluate → drift)...")
    steps = [
        ["python", "src/data/ingest.py"],
        ["python", "src/data/validate.py"],
        ["python", "src/features/engineer.py"],
        ["python", "src/models/train.py"],
        ["python", "src/models/evaluate.py"],
        ["python", "src/monitoring/drift_detector.py"],
        ["python", "src/monitoring/retrain_trigger.py"],
    ]
    for cmd in steps:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False)
        if result.returncode != 0:
            logger.error(f"Pipeline step failed: {' '.join(cmd)}")
            break
    else:
        logger.info("Full pipeline completed. Check monitoring/retrain_decision.json for action.")


# ─────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate new production data batch for MLOps pipeline")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="normal",
        help="Drift scenario to simulate (default: normal)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Number of synthetic trip records to generate (default: 500)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append directly to data/raw/ETA_Model_data.csv (triggers full repro)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        default=False,
        help="Automatically re-run the full ML pipeline after data generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Print available scenarios
    print("\n" + "=" * 60)
    print("  AVAILABLE SCENARIOS")
    print("=" * 60)
    for name, info in SCENARIOS.items():
        marker = "→ " if name == args.scenario else "  "
        print(f"{marker}{name:12s}: {info['description']}")
    print("=" * 60 + "\n")

    df = generate_new_batch(n=args.n, scenario=args.scenario, seed=args.seed)

    out_path = save_new_batch(df, append_to_raw=args.append)

    print(f"\n📊  Batch Statistics:")
    print(f"    Rows generated  : {len(df)}")
    print(f"    Scenario        : {args.scenario}")
    print(f"    Mean ETA (min)  : {df['actual_eta_minutes'].mean():.2f}")
    print(f"    Std ETA (min)   : {df['actual_eta_minutes'].std():.2f}")
    print(f"    Traffic High %  : {(df['traffic_level']=='High').mean()*100:.1f}%")
    print(f"    Mean Surge      : {df['surge_multiplier'].mean():.2f}x")
    print(f"    Saved to        : {out_path}")

    if args.retrain:
        run_full_pipeline_on_new_data()
