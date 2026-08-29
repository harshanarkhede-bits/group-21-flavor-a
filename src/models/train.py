from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> Dict[str, float]:
    """Compute regression metrics: R2, MAE, RMSE, MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    try:
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
    except Exception:
        mape = 0.0

    return {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
    }


def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray, model_name: str, output_path: Path) -> None:
    """Generate and save residual distribution plot."""
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.3, color="#2b5c8f")
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes[0].set_xlabel("Actual ETA (minutes)")
    axes[0].set_ylabel("Predicted ETA (minutes)")
    axes[0].set_title(f"{model_name}: Actual vs. Predicted")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Residuals distribution
    axes[1].hist(residuals, bins=40, color="#2b5c8f", edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--", lw=2)
    axes[1].set_xlabel("Residual (Actual - Predicted)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{model_name}: Residual Distribution")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_feature_importance(model: Any, feature_names: list, model_name: str, output_path: Path) -> None:
    """Generate and save top 20 feature importances."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices][::-1], align="center", color="#3b82f6")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=8)
    ax.set_xlabel("Relative Importance")
    ax.set_title(f"{model_name}: Top 20 Feature Importances")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def train_and_evaluate_models() -> Tuple[Any, Dict[str, Any]]:
    """Train candidate regression models, track with MLflow, and persist top model."""
    if not config.train_data_path.exists() or not config.test_data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found. Run feature engineering first: python src/features/engineer.py"
        )

    logger.info("Loading preprocessed training and testing data...")
    train_df = pd.read_csv(config.train_data_path)
    test_df = pd.read_csv(config.test_data_path)

    target_col = config.target_column
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    feature_names = list(X_train.columns)

    # Define model candidates
    hp = config.hyperparameters
    models_dict = {
        "RidgeRegression": Ridge(**hp.get("ridge", {"alpha": 1.0})),
        "RandomForest": RandomForestRegressor(
            **hp.get("random_forest", {"n_estimators": 100, "max_depth": 12, "random_state": 42, "n_jobs": -1})
        ),
        "GradientBoosting": GradientBoostingRegressor(
            **hp.get("gradient_boosting", {"n_estimators": 120, "learning_rate": 0.1, "max_depth": 5, "random_state": 42})
        ),
    }

    if HAS_XGBOOST:
        models_dict["XGBoost"] = xgb.XGBRegressor(
            **hp.get("xgboost", {"n_estimators": 120, "learning_rate": 0.1, "max_depth": 5, "random_state": 42})
        )

    if HAS_MLFLOW:
        try:
            mlflow.set_experiment(config.experiment_name)
        except Exception as e:
            logger.warning(f"Failed to set MLflow experiment: {e}")

    best_model_name = ""
    best_model = None
    best_rmse = float("inf")
    results_summary = {}

    plots_dir = config.store_dir / "plots"

    for name, model in models_dict.items():
        logger.info(f"--- Training candidate model: {name} ---")

        if HAS_MLFLOW:
            run_context = mlflow.start_run(run_name=f"Run_{name}")
        else:
            run_context = None

        try:
            model.fit(X_train, y_train)

            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)

            train_metrics = compute_metrics(y_train, train_preds)
            test_metrics = compute_metrics(y_test, test_preds)

            logger.info(f"{name} -> Test R2: {test_metrics['r2']:.4f} | Test RMSE: {test_metrics['rmse']:.4f} | Test MAE: {test_metrics['mae']:.4f}")

            # Plot residuals & feature importances
            residual_plot_path = plots_dir / f"{name}_residuals.png"
            plot_residuals(y_test.values, test_preds, name, residual_plot_path)

            importance_plot_path = plots_dir / f"{name}_feature_importance.png"
            plot_feature_importance(model, feature_names, name, importance_plot_path)

            results_summary[name] = {
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "hyperparameters": model.get_params() if hasattr(model, "get_params") else {},
            }

            if HAS_MLFLOW:
                # Log hyperparams
                if hasattr(model, "get_params"):
                    mlflow.log_params({k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool))})

                # Log metrics
                for metric_name, val in train_metrics.items():
                    mlflow.log_metric(f"train_{metric_name}", val)
                for metric_name, val in test_metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", val)

                # Log artifacts
                if residual_plot_path.exists():
                    mlflow.log_artifact(str(residual_plot_path))
                if importance_plot_path.exists():
                    mlflow.log_artifact(str(importance_plot_path))

                # Log model
                try:
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name=config.registered_model_name if name == "GradientBoosting" else None
                    )
                except Exception as log_err:
                    logger.warning(f"Could not register model in MLflow: {log_err}")

            if test_metrics["rmse"] < best_rmse:
                best_rmse = test_metrics["rmse"]
                best_model = model
                best_model_name = name

        finally:
            if HAS_MLFLOW and run_context:
                mlflow.end_run()

    logger.info(f"🏆 Best Performing Model: {best_model_name} with Test RMSE: {best_rmse:.4f}")

    # Persist the best model to model_store/
    config.store_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, config.model_path)
    logger.info(f"Persisted best model artifact to: {config.model_path}")

    # Write model metadata
    metadata = {
        "model": best_model_name,
        "best_model_class": type(best_model).__name__,
        "random_state": config.random_state,
        "test_size": config.test_size,
        "train_r2": results_summary[best_model_name]["train_metrics"]["r2"],
        "test_r2": results_summary[best_model_name]["test_metrics"]["r2"],
        "train_mae": results_summary[best_model_name]["train_metrics"]["mae"],
        "test_mae": results_summary[best_model_name]["test_metrics"]["mae"],
        "test_rmse": results_summary[best_model_name]["test_metrics"]["rmse"],
        "test_mape": results_summary[best_model_name]["test_metrics"]["mape"],
        "traffic_mapping": config.traffic_mapping,
        "categorical_columns": config.categorical_columns,
        "location_threshold": config.location_threshold,
        "all_model_comparisons": {
            k: {
                "test_r2": v["test_metrics"]["r2"],
                "test_rmse": v["test_metrics"]["rmse"],
                "test_mae": v["test_metrics"]["mae"]
            }
            for k, v in results_summary.items()
        }
    }

    with open(config.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved model metadata to: {config.metadata_path}")

    # Write evaluation metrics
    with open(config.metrics_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4)
    logger.info(f"Saved comprehensive evaluation metrics to: {config.metrics_path}")

    return best_model, metadata


if __name__ == "__main__":
    train_and_evaluate_models()
