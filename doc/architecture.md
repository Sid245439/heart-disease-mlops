# Architecture

## High-level flow

```mermaid
flowchart TD
  A[UCI Dataset CSV] --> B[download.py]
  B --> C[data/raw/heart_disease_raw.csv]
  C --> D[Preprocessing<br/>HeartDiseasePreprocessor]
  D --> E[Training<br/>Logistic Regression + Random Forest]
  E --> F[MLflow Tracking<br/>params/metrics/artifacts/models]
  E --> G[Saved Artifacts<br/>models/best_model.pkl<br/>models/preprocessor.pkl]
  G --> H[FastAPI Service<br/>app.py]
  H --> I["/predict"]
  H --> J["/health"]
  H --> K["/metrics"]
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
