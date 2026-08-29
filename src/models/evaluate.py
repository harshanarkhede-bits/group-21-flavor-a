from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model_on_test_data(
    model_path: Path | str | None = None,
    test_data_path: Path | str | None = None,
    output_report_path: Path | str | None = None,
) -> Dict[str, float]:
    """Evaluate the serialized model on the test dataset and generate markdown & JSON reports."""
    m_path = Path(model_path) if model_path else config.model_path
    t_path = Path(test_data_path) if test_data_path else config.test_data_path
    r_path = Path(output_report_path) if output_report_path else config.store_dir / "evaluation_report.md"

    if not m_path.exists():
        raise FileNotFoundError(f"Model file not found: {m_path}")
    if not t_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {t_path}")

    logger.info(f"Loading model from: {m_path}")
    model = joblib.load(m_path)

    logger.info(f"Loading test features from: {t_path}")
    test_df = pd.read_csv(t_path)

    target_col = config.target_column
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    preds = model.predict(X_test)
    preds = np.maximum(preds, 0.0)  # non-negative

    mae = float(mean_absolute_error(y_test, preds))
    mse = float(mean_squared_error(y_test, preds))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, preds))
    mape = float(mean_absolute_percentage_error(y_test, preds)) * 100

    metrics = {
        "test_r2": round(r2, 4),
        "test_rmse": round(rmse, 4),
        "test_mae": round(mae, 4),
        "test_mape_percent": round(mape, 2),
        "num_test_samples": int(len(test_df)),
    }

    logger.info(
        f"Evaluation Results -> R2: {metrics['test_r2']} | RMSE: {metrics['test_rmse']} | MAE: {metrics['test_mae']} min | MAPE: {metrics['test_mape_percent']}%"
    )

    # Generate Markdown Summary for CML / GitHub PRs
    markdown_content = f"""# 📊 Model Evaluation Report

**Model Artifact**: `{m_path.name}`  
**Test Samples**: {metrics['num_test_samples']}  

## Performance Metrics

| Metric | Value | Description |
| :--- | :--- | :--- |
| **$R^2$ Score** | `{metrics['test_r2']}` | Proportion of variance explained by model |
| **RMSE** | `{metrics['test_rmse']} min` | Root Mean Squared Error |
| **MAE** | `{metrics['test_mae']} min` | Mean Absolute Error |
| **MAPE** | `{metrics['test_mape_percent']}%` | Mean Absolute Percentage Error |

---
*Report generated automatically by MLOps Evaluation Pipeline.*
"""

    r_path.parent.mkdir(parents=True, exist_ok=True)
    with open(r_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    logger.info(f"Saved evaluation markdown report to: {r_path}")

    # Also save JSON format
    json_path = r_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    evaluate_model_on_test_data()
