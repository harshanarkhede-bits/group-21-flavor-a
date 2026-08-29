from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False

from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def calculate_statistical_drift(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05
) -> Dict[str, Any]:
    """Calculate Kolmogorov-Smirnov statistical test across numerical features."""
    drift_by_feature = {}
    drifted_features = []

    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.intersection(
        current_df.select_dtypes(include=[np.number]).columns
    )

    for col in numeric_cols:
        ref_vals = reference_df[col].dropna()
        curr_vals = current_df[col].dropna()

        if len(ref_vals) < 5 or len(curr_vals) < 5:
            continue

        ks_stat, p_value = ks_2samp(ref_vals, curr_vals)
        has_drifted = bool(p_value < threshold)

        if has_drifted:
            drifted_features.append(col)

        drift_by_feature[col] = {
            "drift_detected": has_drifted,
            "p_value": float(round(p_value, 5)),
            "ks_statistic": float(round(ks_stat, 5)),
            "ref_mean": float(round(ref_vals.mean(), 3)),
            "curr_mean": float(round(curr_vals.mean(), 3)),
        }

    total_features = len(drift_by_feature)
    drift_share = len(drifted_features) / total_features if total_features > 0 else 0.0
    dataset_drift = drift_share >= config.drift_share_threshold

    return {
        "dataset_drift_detected": dataset_drift,
        "drift_share": float(round(drift_share, 4)),
        "drifted_features_count": len(drifted_features),
        "drifted_features": drifted_features,
        "total_features_tested": total_features,
        "feature_metrics": drift_by_feature,
    }


def generate_drift_report(
    reference_data_path: Path | str | None = None,
    current_data_path: Path | str | None = None,
    save_reports: bool = True,
) -> Dict[str, Any]:
    """Generate comprehensive drift report using Evidently AI and statistical testing."""
    ref_path = Path(reference_data_path) if reference_data_path else config.train_data_path
    curr_path = Path(current_data_path) if current_data_path else config.test_data_path

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference data not found at: {ref_path}")
    if not curr_path.exists():
        raise FileNotFoundError(f"Current data not found at: {curr_path}")

    logger.info(f"Loading reference dataset: {ref_path}")
    ref_df = pd.read_csv(ref_path)

    logger.info(f"Loading current/production dataset: {curr_path}")
    curr_df = pd.read_csv(curr_path)

    # 1. Run statistical drift calculations
    stat_drift = calculate_statistical_drift(
        ref_df, curr_df, threshold=config.stat_test_threshold
    )

    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reference_dataset": str(ref_path),
        "current_dataset": str(curr_path),
        "reference_rows": len(ref_df),
        "current_rows": len(curr_df),
        "drift_summary": stat_drift,
    }

    # 2. Evidently AI HTML & JSON generation if installed
    if HAS_EVIDENTLY:
        try:
            logger.info("Generating Evidently AI drift report...")
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_df, current_data=curr_df)

            if save_reports:
                config.monitoring_dir.mkdir(parents=True, exist_ok=True)
                report.save_html(str(config.drift_report_html))
                logger.info(f"Saved interactive Evidently HTML report to: {config.drift_report_html}")
        except Exception as e:
            logger.warning(f"Evidently report generation failed: {e}")
    else:
        # Fallback HTML report
        if save_reports:
            config.monitoring_dir.mkdir(parents=True, exist_ok=True)
            html_content = f"""<!DOCTYPE html>
<html>
<head><title>MLOps Drift Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #0f172a; color: #f8fafc; }}
.card {{ background: #1e293b; padding: 24px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #334155; }}
h1 {{ color: #38bdf8; }}
.badge-alert {{ background: #ef4444; color: white; padding: 6px 12px; border-radius: 6px; font-weight: bold; }}
.badge-ok {{ background: #10b981; color: white; padding: 6px 12px; border-radius: 6px; font-weight: bold; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #334155; }}
th {{ background: #0f172a; color: #94a3b8; }}
</style>
</head>
<body>
<h1>🚗 Ride ETA Prediction - Data Drift Report</h1>
<div class="card">
<h3>Status: {'<span class="badge-alert">DATA DRIFT DETECTED</span>' if stat_drift['dataset_drift_detected'] else '<span class="badge-ok">NO DRIFT DETECTED</span>'}</h3>
<p>Drift Share: <strong>{stat_drift['drift_share'] * 100:.1f}%</strong> | Drifted Features: <strong>{stat_drift['drifted_features_count']} / {stat_drift['total_features_tested']}</strong></p>
<p>Timestamp: {report_data['timestamp']}</p>
</div>
<div class="card">
<h3>Feature-Level Drift Breakdown</h3>
<table>
<tr><th>Feature</th><th>Drift Detected</th><th>KS Statistic</th><th>p-value</th><th>Reference Mean</th><th>Current Mean</th></tr>
{"".join(f"<tr><td>{col}</td><td>{'⚠️ Yes' if d['drift_detected'] else '✅ No'}</td><td>{d['ks_statistic']}</td><td>{d['p_value']}</td><td>{d['ref_mean']}</td><td>{d['curr_mean']}</td></tr>" for col, d in stat_drift['feature_metrics'].items())}
</table>
</div>
</body>
</html>"""
            with open(config.drift_report_html, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Saved custom Drift HTML report to: {config.drift_report_html}")

    if save_reports:
        config.monitoring_dir.mkdir(parents=True, exist_ok=True)
        with open(config.drift_report_json, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Saved Drift JSON metrics to: {config.drift_report_json}")

    return report_data


if __name__ == "__main__":
    generate_drift_report()
