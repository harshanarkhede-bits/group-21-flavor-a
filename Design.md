# System Design Document: Ride ETA Prediction Pipeline (Flavor A)

## 1. Project Overview
**Problem Statement:** Predict delivery time (Ride ETA) based on trip distance, time of day, and location.
**Objective:** Build an end-to-end Machine Learning pipeline encompassing data engineering, experiment tracking, model deployment, and drift monitoring.
**Dataset:** [Synthetic data based on NYC Taxi Trip Duration (Kaggle)](https://www.kaggle.com/c/nyc-taxi-trip-duration/data) 

## 2. System Architecture
The system follows a modular MLOps architecture using entirely open-source tools:
*   **Data Versioning:** DVC (Data Version Control)
*   **Experiment Tracking:** MLflow
*   **Serving:** FastAPI & Uvicorn
*   **Containerization:** Docker

```text
[Raw Data] -> (Ingestion & Validation) -> (Feature Engineering) -> [Processed Data]
                                                                          |
                                                                          v
[MLflow UI] <--- (Logs Metrics/Models) <--- (Model Training: Baseline vs XGBoost)
                                                                          |
                                                                          v
[Client/User] ---> (REST API POST /predict) ---> [FastAPI + Docker Container]
                                                                          |
                                                                          v
[Monitoring System] <--- (Logs Predictions) <--- (Drift Simulation & Retraining)
```
## 3. Pipeline Phases & Design Justifications
Phase 1: Data Engineering & Versioning (Module 2)
Ingestion: Raw CSV data is sampled (100k rows for local development) to ensure rapid iteration. Invalid rows (e.g., zero passengers, missing coordinates, negative durations) are dropped.

Feature Engineering:

Distance: GPS coordinates (Latitude/Longitude) are converted to linear distance (Kilometers) using the Haversine formula.

Temporal: pickup_datetime is decomposed into hour_of_day, day_of_week, and an is_weekend flag to capture traffic seasonality.

Versioning Justification: DVC is used instead of Git for data files because Git is inefficient with large binary or CSV files. DVC creates lightweight pointer files (.dvc) that are tracked by Git, ensuring code and data states are tightly coupled.

Phase 2: Experimentation & Reproducibility (Module 3)
Model Selection: We will train and compare two distinct architectures:

Baseline: Linear Regression (interpretable, fast).

Advanced: XGBoost / Gradient Boosting Regressor (handles non-linear relationships like traffic spikes during specific hours).

Tracking Justification: MLflow is integrated into the training script. It automatically logs hyperparameters (e.g., learning_rate, max_depth), evaluation metrics (RMSE, MAE), and the serialized model artifacts (.pkl). This ensures full reproducibility of any given experiment run.

Phase 3: Model Packaging & Deployment (Module 4)
API Framework: FastAPI is chosen for its speed, built-in async support, and automatic OpenAPI (Swagger) documentation generation.

Data Validation: Pydantic models will strictly define the expected incoming JSON payload structure, catching malformed requests (e.g., missing coordinates or text where numbers are expected) before they hit the ML model.

Packaging Justification: The entire API and its dependencies will be containerized using Docker. This guarantees the "it works on my machine" principle extends to any production environment.

Phase 4: Monitoring, Drift & Retraining (Module 5)
Drift Simulation Strategy: Once deployed, we will simulate Data Drift by feeding the model requests heavily skewed towards rush hour (e.g., 5:00 PM - 7:00 PM) or simulating extreme weather conditions that increase travel times significantly.

Retraining Trigger: A script will monitor the Root Mean Squared Error (RMSE) of incoming synthetic "actuals" vs. predictions.

Trigger Justification: If the rolling RMSE exceeds our baseline validation RMSE by more than 15%, an automated alert (log) is generated, signaling the need to pull recent data and trigger a new MLflow training run.