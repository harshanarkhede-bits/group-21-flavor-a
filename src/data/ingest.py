from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def ingest_data(raw_data_path: Path | str | None = None, output_path: Path | str | None = None) -> pd.DataFrame:
    """Ingest raw ETA data, remove malformed records, and persist to processed directory."""
    raw_path = Path(raw_data_path) if raw_data_path else config.raw_data_path
    out_path = Path(output_path) if output_path else config.ingested_data_path

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    logger.info(f"Ingesting raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    logger.info(f"Loaded {initial_rows} records with {df.shape[1]} columns.")

    # 1. Clean categorical prefixes if present (e.g. '2. Afternoon' -> 'Afternoon')
    if "time_of_day" in df.columns:
        df["time_of_day"] = df["time_of_day"].astype(str).str.replace(r"^\d+\.\s*", "", regex=True)
    if "season" in df.columns:
        df["season"] = df["season"].astype(str).str.replace(r"^\d+\.\s*", "", regex=True)

    # 2. Drop unique trip identifier if present
    if "trip_id" in df.columns:
        df = df.drop(columns=["trip_id"])

    # 3. Drop exact duplicates & null rows
    df = df.drop_duplicates().dropna()

    # 4. Range validation on target and numeric bounds
    if config.target_column in df.columns:
        df = df[
            (df[config.target_column] >= config.min_eta_minutes)
            & (df[config.target_column] <= config.max_eta_minutes)
        ]

    if "passenger_count" in df.columns:
        df = df[df["passenger_count"] > 0]

    if "trip_distance_km" in df.columns:
        df = df[df["trip_distance_km"] > 0]

    dropped_rows = initial_rows - len(df)
    logger.info(f"Data ingestion complete. Kept {len(df)} records (dropped {dropped_rows} invalid/outlier rows).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Ingested dataset saved to: {out_path}")

    return df


if __name__ == "__main__":
    ingest_data()
