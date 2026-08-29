# System Design Document: Ride & Delivery ETA Prediction Pipeline (Flavor A)

## 1. Executive Summary & Problem Statement
* **Business Objective:** Accurately predict delivery/ride Estimated Time of Arrival (ETA) in New York City using trip origin, destination, time of day, and traffic conditions.
* **ML Problem Type:** Supervised Regression ($y = \text{actual\_eta\_minutes}$).
* **Target Metric:** Root Mean Squared Error (RMSE) $\le 6.0$ minutes and $R^2 \ge 0.70$.
* **System Type:** Low-latency online inference API (FastAPI) coupled with real-time monitoring and automated MLOps lifecycle pipelines.

---

## 2. Architecture & Tech Stack

```text
[Raw Dataset (10k Trips)] 
        │
        ▼ (src.data.ingest & src.data.validate)
[Validated Data + Schema Assertions]
        │
        ▼ (src.features.engineer)
[Engineered Splits: Train & Test (DVC Versioned)]
        │
        ▼ (src.models.train)
[MLflow Experiment Tracking & Multi-Model Benchmark] ──► [MLflow Model Registry]
        │
        ▼
[Production Model Artifacts (model_store/)]
        │
        ▼ (src.serving.api)
[FastAPI REST API (:8000)] ──► [Prometheus Metrics (:9090)] ──► [Grafana (:3000)]
        ▲
        │
[Streamlit Ops UI (:8501)]
        │
        ▼ (src.monitoring.drift_detector)
[Evidently AI & Statistical Drift Monitor]
        │
        ▼ (src.monitoring.retrain_trigger)
[Automated Retraining Decision Engine]
```

### Technology Matrix:
| Function | Technology | Rationale |
| :--- | :--- | :--- |
| **Data Versioning** | DVC (Data Version Control) | Lightweight pointer files (`dvc.yaml`), tight coupling with Git, non-polluting data pipelines. |
| **Experiment Tracking** | MLflow | Automated parameter logging, metrics comparison ($R^2, RMSE, MAE$), artifact storage, model registry. |
| **Serving Layer** | FastAPI & Uvicorn | Async performance, strict Pydantic validation, OpenAPI documentation, Prometheus metrics. |
| **Monitoring** | Evidently AI & Prometheus | Statistical feature drift (KS-test, Chi-sq) and real-time operational metric collection. |
| **Operations UI** | Streamlit | Rapid interactive prototyping for single & batch inference, model performance, and drift control. |
| **Containerization** | Docker & Docker Compose | Multi-stage slim container builds and multi-service local/cloud orchestration. |
| **CI/CD & CML** | GitHub Actions | Automated linting, test suites with code coverage, and CML automated model PR reporting. |

---

## 3. Pipeline Phases & Detailed Design

### Phase 1: Data Engineering & Validation (Module 2)
* **Ingestion (`src/data/ingest.py`)**: Sanitizes categorical prefixes (`2. Spring` $\rightarrow$ `Spring`), drops trip IDs, eliminates duplicates, and removes impossible durations ($< 0.5$ min or $> 200$ min).
* **Validation (`src/data/validate.py`)**: Strict schema enforcement checking required columns, coordinate validity, traffic category sets, and numeric domain bounds.
* **Feature Engineering (`src/features/engineer.py`)**:
  * **Distance**: Haversine formula based on NYC neighborhood coordinates.
  * **Temporal Decomposition**: `pickup_hour`, `pickup_minute`, `month`, `day`, `day_of_year`, `weekday`, `is_weekend`, `season`.
  * **Categorical Handling**: Grouping infrequent locations ($< 20$ occurrences) into `"Other"` and fitting a `OneHotEncoder(drop='first', handle_unknown='ignore')`.
  * **Traffic Mapping**: Ordinal mapping (`Low: 0, Medium: 1, High: 2`).

### Phase 2: Experimentation, Modeling & Registry (Module 3)
* **Model Benchmark**:
  1. *Baseline*: Ridge Regression.
  2. *Ensemble*: Random Forest Regressor ($n=100, \text{depth}=12$).
  3. *Boosting*: Gradient Boosting Regressor ($n=120, \text{lr}=0.1, \text{depth}=5$).
  4. *Optimized*: XGBoost Regressor.
* **Tracking & Registration**:
  * Metrics logged: Train/Test $R^2$, $RMSE$, $MAE$, $MAPE$.
  * Artifacts: Residual plots, actual vs predicted scatter plots, top 20 feature importance bar charts.
  * The best-performing model is automatically persisted to `model_store/eta_model.pkl` and registered in MLflow Model Registry.

### Phase 3: Packaging, Serving & Observability (Module 4)
* **REST API**:
  * `POST /predict`: Single trip ETA prediction with Haversine distance calculation and traffic estimation.
  * `POST /predict/batch`: High-throughput batch inference endpoint.
  * `GET /health` & `GET /ready`: Health and artifact readiness probes.
  * `GET /model-info`: Active model metadata, test metrics, and lineage.
  * `GET /metrics`: Prometheus scraper endpoint tracking latency histograms and request counters.
* **Containerization**: Multi-stage `Dockerfile` with non-root security and a 5-service `docker-compose.yml`.

### Phase 4: Monitoring, Drift Detection & Retraining (Module 5)
* **Drift Detection Engine (`src/monitoring/drift_detector.py`)**:
  * Continuous statistical comparison of production request logs against baseline distributions using Kolmogorov-Smirnov (for continuous features) and Evidently AI presets.
  * Interactive HTML reports (`monitoring/drift_report.html`) and structured JSON summaries (`monitoring/drift_report.json`).
* **Retraining Trigger Policy (`src/monitoring/retrain_trigger.py`)**:
  * $\text{RMSE}_{\text{monitored}} > 1.15 \times \text{RMSE}_{\text{baseline}}$ OR $\text{Drift Share} \ge 30\% \rightarrow$ Fire Retraining Trigger.
  * Can automatically trigger retraining pipeline execution upon trigger activation.