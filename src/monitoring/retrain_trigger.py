from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_baseline_metrics() -> Dict[str, Any]:
    """Load baseline training metrics from model metadata."""
    if not config.metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found at: {config.metadata_path}")

    with open(config.metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "model_name": meta.get("model", "GradientBoosting"),
        "baseline_rmse": float(meta.get("test_rmse", 5.6)),
        "baseline_mae": float(meta.get("test_mae", 3.8)),
        "baseline_r2": float(meta.get("test_r2", 0.72)),
    }


def evaluate_retraining_policy(
    current_rmse: float | None = None,
    threshold_pct: float | None = None,
    auto_trigger: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate whether production metrics or data drift warrant automated model retraining.
    Decision Rules:
    1. Performance Degradation: Current RMSE > Baseline RMSE * (1 + threshold_pct) [Default 15%]
    2. Data Drift Alert: If dataset drift is detected in drift_report.json
    """
    threshold = threshold_pct if threshold_pct is not None else config.rmse_threshold_pct
    baseline_info = load_baseline_metrics()
    baseline_rmse = baseline_info["baseline_rmse"]

    # Check drift report if available
    drift_detected = False
    drift_share = 0.0
    if config.drift_report_json.exists():
        try:
            with open(config.drift_report_json, "r", encoding="utf-8") as f:
                drift_info = json.load(f)
            drift_summary = drift_info.get("drift_summary", {})
            drift_detected = drift_summary.get("dataset_drift_detected", False)
            drift_share = drift_summary.get("drift_share", 0.0)
        except Exception as e:
            logger.warning(f"Could not parse drift report: {e}")

    # Fallback simulation RMSE if none provided
    eval_rmse = current_rmse if current_rmse is not None else baseline_rmse * 1.20
    rmse_increase_pct = (eval_rmse - baseline_rmse) / baseline_rmse if baseline_rmse > 0 else 0.0
    rmse_breach = rmse_increase_pct > threshold

    should_retrain = bool(rmse_breach or drift_detected)

    reasons = []
    if rmse_breach:
        reasons.append(
            f"RMSE degradation of {rmse_increase_pct * 100:.1f}% exceeded allowed threshold of {threshold * 100:.1f}%."
        )
    if drift_detected:
        reasons.append(
            f"Significant feature drift detected with drift share of {drift_share * 100:.1f}%."
        )
    if not should_retrain:
        reasons.append("Model performance and data distributions are within healthy operational thresholds.")

    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_rmse": baseline_rmse,
        "current_rmse": float(round(eval_rmse, 4)),
        "rmse_increase_pct": float(round(rmse_increase_pct * 100, 2)),
        "threshold_pct": float(round(threshold * 100, 2)),
        "rmse_breach_detected": rmse_breach,
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "retraining_triggered": should_retrain,
        "status": "ACTION_REQUIRED_RETRAIN" if should_retrain else "HEALTHY",
        "reasons": reasons,
        "recommended_action": "Execute pipeline retraining (src.models.train)" if should_retrain else "Continue active monitoring",
    }

    config.monitoring_dir.mkdir(parents=True, exist_ok=True)
    with open(config.retrain_decision_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)
    logger.info(f"Retraining decision saved to: {config.retrain_decision_path}")

    if should_retrain and auto_trigger:
        logger.info("⚡ Automatically triggering model retraining pipeline...")
        from src.models.train import train_and_evaluate_models
        train_and_evaluate_models()
        decision["auto_retrain_executed"] = True

    return decision


if __name__ == "__main__":
    result = evaluate_retraining_policy(current_rmse=6.85, auto_trigger=False)
    print(json.dumps(result, indent=2))
