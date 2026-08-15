from __future__ import annotations

import json
from pathlib import Path

from src.monitoring import DRIFT_REPORT_PATH, load_baseline_rmse, should_retrain

ROOT_DIR = Path(__file__).resolve().parent.parent


def evaluate_retraining_need(current_rmse: float, threshold_pct: float = 0.15) -> dict:
    """Check whether a retraining trigger should fire based on the current monitoring RMSE."""
    baseline_rmse = load_baseline_rmse()
    triggered = should_retrain(current_rmse, baseline_rmse, threshold_pct)

    result = {
        "baseline_rmse": baseline_rmse,
        "current_rmse": current_rmse,
        "threshold_pct": threshold_pct,
        "triggered_retraining": triggered,
        "recommended_action": "Retrain model" if triggered else "Continue monitoring",
    }

    with open(DRIFT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    demo_current_rmse = 520.0
    print(json.dumps(evaluate_retraining_need(demo_current_rmse), indent=2))
