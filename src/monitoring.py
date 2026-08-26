"""Legacy entrypoint for monitoring."""
from src.monitoring.drift_detector import generate_drift_report, calculate_statistical_drift
from src.monitoring.retrain_trigger import evaluate_retraining_policy, load_baseline_metrics

DRIFT_REPORT_PATH = generate_drift_report

if __name__ == "__main__":
    generate_drift_report()
