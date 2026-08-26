.PHONY: help setup data features train evaluate pipeline serve ui monitor test docker-up docker-down clean

PYTHON ?= python

help:
	@echo "=== Ride ETA Prediction MLOps CLI ==="
	@echo "Available commands:"
	@echo "  make setup        : Install dependencies"
	@echo "  make data         : Ingest and validate data"
	@echo "  make features     : Run feature engineering"
	@echo "  make train        : Train candidate models & log to MLflow"
	@echo "  make evaluate     : Evaluate model on test set"
	@echo "  make pipeline     : Execute end-to-end pipeline (DVC)"
	@echo "  make serve        : Start FastAPI serving server"
	@echo "  make ui           : Start Streamlit interactive UI"
	@echo "  make monitor      : Generate data drift report"
	@echo "  make test         : Run pytest test suite"
	@echo "  make docker-up    : Launch full Docker Compose stack"
	@echo "  make docker-down  : Stop Docker Compose stack"
	@echo "  make clean        : Clean cache and temporary files"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) src/data/ingest.py
	$(PYTHON) src/data/validate.py

features:
	$(PYTHON) src/features/engineer.py

train:
	$(PYTHON) src/models/train.py

evaluate:
	$(PYTHON) src/models/evaluate.py

pipeline: data features train evaluate monitor
	@echo "✅ End-to-End Pipeline Completed Successfully!"

serve:
	$(PYTHON) -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

ui:
	$(PYTHON) -m streamlit run ui/app.py --server.port 8501

monitor:
	$(PYTHON) src/monitoring/drift_detector.py

test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .coverage htmlcov
