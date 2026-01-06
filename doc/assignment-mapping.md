# Rubric Mapping

This page maps the repository contents to the assignment requirements.

## 1. Data Acquisition & EDA

- Dataset: UCI Heart Disease CSV
- Download script: `download.py` → saves `data/raw/heart_disease_raw.csv`
- EDA notebook: `exploration/eda.ipynb`
- EDA writeup: `doc/eda.md`
- EDA figures (view in GitHub Pages):
  - [Target distribution](images/heart_disease_target_distribution.png)
  - [Correlation heatmap](images/heart_disease_correlation_matrix.png)
  - [Categorical distributions](images/heart_disease_categorical_distribution.png)
  - [Numerical distributions](images/heart_disease_numerical_distribution.png)

## 2. Feature Engineering & Model Development

- Preprocessing: `src/preprocessing.py`
  - Missing value handling
  - Encoding
  - Scaling
  - Reproducible transform at inference
- Two models trained: `src/training.py`
  - Logistic Regression
  - Random Forest
- Evaluation: accuracy/precision/recall/F1/ROC-AUC + cross-validated tuning (GridSearchCV)
- Modeling writeup: `doc/modeling.md`

## 3. Experiment Tracking

- Tool: MLflow
- Implementation: `src/training.py`
  - Logs params + metrics
  - Logs feature importance plot as artifact
  - Logs models via `mlflow.sklearn.log_model`
- Guide: `doc/experiment-tracking.md`

## 4. Model Packaging & Reproducibility

- Saved artifacts:
  - `models/best_model.pkl`
  - `models/preprocessor.pkl`
- Reproducible dependencies:
  - `requirements.txt` (generated via `nox -s requirements`)
  - `pyproject.toml` + `uv.lock`
- Reproducibility guide: `doc/setup.md`

## 5. CI/CD Pipeline & Automated Testing

- Tests: `tests/` (Pytest)
- CI tool: GitHub Actions + Nox
- Workflows:
  - `.github/workflows/ci.yml` (lint/format/typing/test in parallel)
  - `.github/workflows/ci-cd.yml` (reuses CI + builds Docker + trains)
- Reports/artifacts (view in GitHub Pages):
  - Ruff lint report: [reports/ruff/ruff-lint-report.html](reports/ruff/ruff-lint-report.html)
  - Format check report: [reports/format/ruff-format-report.html](reports/format/ruff-format-report.html)
  - Mypy report: [reports/typing/mypy-report.html](reports/typing/mypy-report.html)
  - Pytest report: [reports/pytest/pytest-report.html](reports/pytest/pytest-report.html)
  - Coverage report: [reports/coverage/htmlcov/index.html](reports/coverage/htmlcov/index.html)
- CI/CD guide: `doc/ci-cd.md`

## 6. Model Containerization

- Dockerfile: `Dockerfile`
- FastAPI API:
  - `/predict` accepts JSON
  - returns prediction + confidence
- Guide: `doc/containerization.md`

## 7. Production Deployment

- Kubernetes manifest: `k8s/deployment.yaml`
  - Deployment + Service + HPA
  - Prometheus scrape annotations
- Guide: `doc/deployment.md`

## 8. Monitoring & Logging

- Logging: Loguru → `logs/api.log`
- Metrics: Prometheus format at `/metrics`
- Prometheus config: `monitoring/prometheus.yml`
- Guide: `doc/monitoring.md`

Monitoring evidence (view in GitHub Pages):

- [Prometheus targets](images/screenshots/monitoring-targets.png)
- [Requests total](images/screenshots/monitoring-requests-total.png)
- [Request rate v1](images/screenshots/monitoring-requests-rate-v1.png)
- [Request rate v2](images/screenshots/monitoring-requests-rate-v2.png)
- [/metrics output](images/screenshots/monitoring-metrics-terminal-view.png)

## 9. Documentation & Reporting

- Repo README: `README.md`
- MkDocs site source: `doc/`
- Report text: `REPORT.md` (summary report content)
