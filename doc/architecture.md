# Architecture

## High-level flow

```mermaid
flowchart LR
  A[UCI Dataset CSV] --> B[download.py]
  B --> C[data/raw/heart_disease_raw.csv]
  C --> D[Preprocessing\nHeartDiseasePreprocessor]
  D --> E[Training\nLogistic Regression + Random Forest]
  E --> F[MLflow Tracking\nparams/metrics/artifacts/models]
  E --> G[Saved Artifacts\nmodels/best_model.pkl\nmodels/preprocessor.pkl]
  G --> H[FastAPI Service\napp.py]
  H --> I[/predict]
  H --> J[/health]
  H --> K[/metrics]
  K --> L[Prometheus/Grafana]
```

## Components

- Data acquisition: `download.py`
- Preprocessing: `src/preprocessing.py` (fit/transform, NaN safety, consistent column ordering)
- Training: `src/training.py`
  - Grid search + cross-validation
  - Logs to MLflow
  - Saves final artifacts to `models/`
- Serving: `app.py` (FastAPI)
- Monitoring: Prometheus scrape config in `monitoring/prometheus.yml`
- Deployment: Kubernetes manifest in `k8s/deployment.yaml`
- Automation: `noxfile.py` + GitHub Actions workflows
