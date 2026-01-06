# Heart Disease MLOps (End-to-End)

An end-to-end MLOps project using the **UCI Heart Disease** dataset:

- EDA + preprocessing
- Train two models (Logistic Regression + Random Forest)
- MLflow experiment tracking
- FastAPI inference service (`/predict`) with Prometheus metrics (`/metrics`)
- CI (ruff/format/mypy/pytest) using `nox` + `uv`
- Docker image + Kubernetes deployment manifest

Full project documentation is in the MkDocs site under `doc/`.

## Quickstart (recommended): uv + nox

1. Install `uv` (once)

https://docs.astral.sh/uv/

```bash
pip install uv
```

2. Install nox

```bash
uv pip install --system nox nox-uv
```

3. Run the full CI suite locally

```bash
nox
```

4. Download data + train models (also logs to MLflow)

```bash
nox -s train
```

This creates:

- `data/raw/heart_disease_raw.csv`
- `models/best_model.pkl`
- `models/preprocessor.pkl`
- MLflow runs under `mlruns/`

## Run the API (local)

```bash
uvicorn app:app --reload
```

- Swagger UI: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

## Run MLflow UI

```bash
mlflow ui --backend-store-uri file:./mlruns --host localhost --port 5000
```

- UI: http://localhost:5000
- Alternative: `nox -s mlflow_ui`

## Docker

Build:

```bash
docker build -t heart-disease-mlops:latest .
```

Run (models baked into the image if you trained before building):

```bash
docker run -p 8000:8000 heart-disease-mlops:latest
```

Optional (development): mount local `models/` into the container.

PowerShell:

```powershell
docker run -p 8000:8000 -v ${PWD}/models:/app/models:ro heart-disease-mlops:latest
```

## Sample prediction

PowerShell:

```powershell
$body = @{
  age = 63
  sex = 1
  cp = 3
  trestbps = 145
  chol = 233
  fbs = 1
  restecg = 0
  thalach = 150
  exang = 0
  oldpeak = 2.3
  slope = 0
  ca = 0
  thal = 1
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType application/json -Body $body
```

## Kubernetes (local cluster)

See `k8s/deployment.yaml` and the docs page `doc/deployment.md`.
