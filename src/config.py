from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_FILE = PROJECT_ROOT / "params.yaml"


def load_params(params_path: Path | str | None = None) -> Dict[str, Any]:
    """Load configuration from params.yaml with fallback to defaults."""
    if params_path is None:
        params_path = PARAMS_FILE
    else:
        params_path = Path(params_path)

    if not params_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


class Config:
    """Helper class to access project configuration parameters easily."""

    def __init__(self, params_path: Path | str | None = None):
        self.raw_params = load_params(params_path)

        # Project
        self.project_name: str = self.raw_params.get("project", {}).get("name", "ride-eta-prediction")
        self.project_version: str = self.raw_params.get("project", {}).get("version", "1.0.0")

        # Data
        data_cfg = self.raw_params.get("data", {})
        self.raw_data_path: Path = PROJECT_ROOT / data_cfg.get("raw_data_path", "data/raw/ETA_Model_data.csv")
        self.processed_dir: Path = PROJECT_ROOT / data_cfg.get("processed_dir", "data/processed")
        self.ingested_data_path: Path = PROJECT_ROOT / data_cfg.get("ingested_data_path", "data/processed/ingested_eta.csv")
        self.train_data_path: Path = PROJECT_ROOT / data_cfg.get("train_data_path", "data/processed/train_features.csv")
        self.test_data_path: Path = PROJECT_ROOT / data_cfg.get("test_data_path", "data/processed/test_features.csv")
        self.target_column: str = data_cfg.get("target_column", "actual_eta_minutes")
        self.test_size: float = float(data_cfg.get("test_size", 0.20))
        self.random_state: int = int(data_cfg.get("random_state", 42))
        self.location_threshold: int = int(data_cfg.get("location_threshold", 20))
        self.max_eta_minutes: float = float(data_cfg.get("max_eta_minutes", 200.0))
        self.min_eta_minutes: float = float(data_cfg.get("min_eta_minutes", 0.5))

        # Features
        feat_cfg = self.raw_params.get("features", {})
        self.categorical_columns: List[str] = feat_cfg.get(
            "categorical_columns", ["pickup_location", "drop_location", "weekday", "season"]
        )
        self.numerical_columns: List[str] = feat_cfg.get(
            "numerical_columns",
            [
                "pickup_hour",
                "pickup_minute",
                "month",
                "day",
                "day_of_year",
                "trip_distance_km",
                "passenger_count",
                "surge_multiplier",
                "is_weekend",
                "traffic_level",
            ],
        )
        self.traffic_mapping: Dict[str, int] = feat_cfg.get(
            "traffic_mapping", {"Low": 0, "Medium": 1, "High": 2}
        )

        # Models
        model_cfg = self.raw_params.get("models", {})
        self.store_dir: Path = PROJECT_ROOT / model_cfg.get("store_dir", "model_store")
        self.model_path: Path = PROJECT_ROOT / model_cfg.get("model_path", "model_store/eta_model.pkl")
        self.encoder_path: Path = PROJECT_ROOT / model_cfg.get("encoder_path", "model_store/eta_encoder.pkl")
        self.feature_columns_path: Path = PROJECT_ROOT / model_cfg.get(
            "feature_columns_path", "model_store/feature_columns.json"
        )
        self.metadata_path: Path = PROJECT_ROOT / model_cfg.get("metadata_path", "model_store/model_metadata.json")
        self.metrics_path: Path = PROJECT_ROOT / model_cfg.get("metrics_path", "model_store/evaluation_metrics.json")
        self.experiment_name: str = model_cfg.get("experiment_name", "Ride_ETA_Prediction_Experiment")
        self.registered_model_name: str = model_cfg.get("registered_model_name", "RideETAPredictor")
        self.hyperparameters: Dict[str, Any] = model_cfg.get("hyperparameters", {})

        # Serving
        srv_cfg = self.raw_params.get("serving", {})
        self.serving_host: str = srv_cfg.get("host", "0.0.0.0")
        self.serving_port: int = int(srv_cfg.get("port", 8000))
        self.default_passenger_count: int = int(srv_cfg.get("default_passenger_count", 1))
        self.default_surge_multiplier: float = float(srv_cfg.get("default_surge_multiplier", 1.0))

        # Monitoring
        mon_cfg = self.raw_params.get("monitoring", {})
        self.monitoring_dir: Path = PROJECT_ROOT / mon_cfg.get("dir", "monitoring")
        self.prediction_log_path: Path = PROJECT_ROOT / mon_cfg.get(
            "prediction_log_path", "monitoring/prediction_log.csv"
        )
        self.drift_report_html: Path = PROJECT_ROOT / mon_cfg.get(
            "drift_report_html", "monitoring/drift_report.html"
        )
        self.drift_report_json: Path = PROJECT_ROOT / mon_cfg.get(
            "drift_report_json", "monitoring/drift_report.json"
        )
        self.retrain_decision_path: Path = PROJECT_ROOT / mon_cfg.get(
            "retrain_decision_path", "monitoring/retrain_decision.json"
        )
        self.rmse_threshold_pct: float = float(mon_cfg.get("rmse_threshold_pct", 0.15))
        self.drift_share_threshold: float = float(mon_cfg.get("drift_share_threshold", 0.30))
        self.stat_test_threshold: float = float(mon_cfg.get("stat_test_threshold", 0.05))


config = Config()
