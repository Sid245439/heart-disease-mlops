# Heart Disease MLOps — Assignment Documentation

This repository implements an end-to-end MLOps workflow for a **Heart Disease risk classifier** using the **UCI Heart Disease dataset**.

It is structured to satisfy the assignment requirements:

- Data acquisition + EDA
- Feature engineering + training of at least two models
- Experiment tracking (MLflow)
- Reproducible packaging (saved preprocessor + model, requirements)
- CI with lint/format/typing/tests + artifacts
- Containerized model-serving API (FastAPI)
- Kubernetes deployment manifest
- Monitoring via Prometheus metrics + structured logging

## Quickstart

### Run CI sessions locally (same as GitHub Actions)

```bash
uv pip install --system nox nox-uv
nox
```

### Train models (also logs to MLflow)

```bash
nox -s train
```

### Run API locally

```bash
uvicorn app:app --reload
```

- Swagger UI: http://localhost:8000/docs
- Metrics: http://localhost:8000/metrics

## Assignment pages

- [Setup & Reproducibility](setup.md)
- [EDA](eda.md)
- [Modeling](modeling.md)
- [Experiment Tracking](experiment-tracking.md)
- [API Serving](api.md)
- [CI/CD](ci-cd.md)
- [Containerization](containerization.md)
- [Deployment (Kubernetes)](deployment.md)
- [Monitoring & Logging](monitoring.md)
- [Architecture](architecture.md)
- [Rubric Mapping](assignment-mapping.md)

## Reports & assets

- Automated workflow reports (JUnit HTML/XML, coverage HTML) are saved under `doc/reports/`.
- EDA figures are in `doc/images/`.
- MkDocs static styling is in `doc/_static/`.
