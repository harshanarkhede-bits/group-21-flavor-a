"""Legacy entrypoint for retrain trigger."""
from src.monitoring.retrain_trigger import evaluate_retraining_policy, load_baseline_metrics

if __name__ == "__main__":
    evaluate_retraining_policy(current_rmse=6.85)
