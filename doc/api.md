# API Serving (FastAPI)

The inference service is implemented in `app.py`.

## Endpoints

- `GET /health`
  - Returns service status and whether the model is loaded.
- `POST /predict`
  - Accepts a JSON payload matching the `PatientData` schema.
  - Returns `prediction`, `confidence`, and `risk_level`.
- `GET /metrics`
  - Prometheus metrics endpoint (request counts and latency histogram).

## Run locally (dev)

```bash
uvicorn app:app --reload
```

Open:

- Swagger UI: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

## Sample request

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

## Logging

- The service uses Loguru.
- Logs are written to `logs/api.log` and also printed to stdout.
- Control verbosity with `LOG_LEVEL` (e.g., `INFO`, `DEBUG`).

## Model artifacts

On startup, the app loads:

- `models/best_model.pkl`
- `models/preprocessor.pkl`

Make sure you have trained at least once (`nox -s train`) before starting the service, or build a Docker image after training (so the `models/` folder is present in the image).
