# Heart Disease MLOps Report

## Setup & Reproducibility
- Environment: `conda env create -f environment.yml` → `conda activate heart-disease-mlops`.
- Data: `python download_data.py` downloads raw CSV; cleaned data lives in `data/processed/heart_disease_clean_binary.csv`.
- Training: `python -c "from src.training import train_pipeline; train_pipeline('data/raw/heart_disease_raw.csv')"` saves `models/best_model.pkl` and `models/preprocessor.pkl`.
- Tests: `pytest tests/ -v` for preprocessing sanity.
- Serving: `uvicorn app:app --reload` (or Docker/K8s options below). API docs at `http://localhost:8000/docs`.

## Data & EDA
- Dataset: UCI Heart Disease (14+ clinical features, binary target). Raw download handled in `download_data.py`.
- Notebook `EDA_UCI.ipynb` covers distributions, correlations, and class balance; target converted to binary `target_binary`.
- Cleaning: NaN handling and column normalization performed by `HeartDiseasePreprocessor` in `src/preprocessing.py`.

## Feature Engineering & Models
- Preprocessing: numeric median fill + `StandardScaler`; categorical label encoding; column order enforced to avoid training/serving drift.
- Models: Logistic Regression and Random Forest (grid search). Metrics include accuracy, precision, recall, F1, ROC-AUC; feature importance plot saved to `logs/feature_importance.png`.
- Best model selected by test AUC and persisted to `models/best_model.pkl`; preprocessor saved to `models/preprocessor.pkl`.

## Experiment Tracking (MLflow)
- Experiment name: `heart-disease-mlops`.
- Parameters, metrics, and artifacts (including feature importance plot) logged per run.
- Launch UI: `mlflow ui --host 0.0.0.0 --port 5000` → `http://localhost:5000`.

## Packaging, Containerization, Deployment
- Docker: `Dockerfile` builds FastAPI service with model artifacts mounted at `/app/models`; run via `docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro heart-disease-mlops:latest`.
- Kubernetes: manifests in `k8s/deployment.yaml` (deployment + service + HPA). Probes hit `/health`; service exposed as LoadBalancer. Prometheus annotations included for `/metrics`.
- Health & inference checks: `curl http://localhost:8000/health` and POST `/predict` with sample JSON from `README.md`.

## CI/CD & Testing Notes
- Unit tests reside in `tests/`; extend coverage to training and API schema as needed.
- Suggested pipeline (GitHub Actions/Jenkins): steps for linting, `pytest`, model training, image build/push, and deploy to cluster. Add run artifacts/screenshots to `screenshots/` for submission.

## Monitoring & Logging
- Structured logs written to `logs/api.log`; set verbosity via `LOG_LEVEL`.
- Request/response metrics exported at `/metrics` (Prometheus). Sample config: `monitoring/prometheus.yml`.
- Quick start Prometheus: `docker run -p 9090:9090 -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus`; optional Grafana: `docker run -d -p 3000:3000 grafana/grafana` (add Prom data source at `http://host.docker.internal:9090`).
- K8s manifests include `prometheus.io/*` annotations for automatic scraping.

## Architecture (high level)
```mermaid
flowchart LR
    subgraph Data
      A[Raw CSV] --> B[download_data.py]
      B --> C[EDA_UCI.ipynb]
    end
    C --> D[HeartDiseasePreprocessor\nsrc/preprocessing.py]
    D --> E[ModelTrainer\nsrc/training.py]
    E -->|best model + preprocessor| F[(models/)]
    F --> G[FastAPI Service\napp.py]
    G --> H[/predict]
    G --> I[/metrics]
    G --> J[/health]
    H --> K[Prometheus/Grafana]
    I --> K
```

## API Quick Reference
- `GET /health` – readiness check; reports model load status.
- `POST /predict` – returns `prediction`, `confidence`, `risk_level`.
- `GET /metrics` – Prometheus scrape endpoint for request counts/latency.

## Submission Links & Artifacts
- Repository: https://github.com/<your-username>/heart-disease-mlops
- Screenshots: place CI/CD, deployment, and monitoring captures in `screenshots/`.
- Short video: record end-to-end run (data → train → serve → predict → monitor).

_(Convert this Markdown to PDF if your submission portal requires a PDF.)_

