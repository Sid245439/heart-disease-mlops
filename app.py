"""FastAPI Prediction Service with logging + Prometheus metrics"""

import logging
import os
import pickle
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

# Structured logging to console + file
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("heart_disease_api")

file_handler = logging.FileHandler(LOG_DIR / "api.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
logger.addHandler(file_handler)


# Prometheus metrics
REQUEST_COUNT = Counter(
    "heart_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "heart_api_request_latency_seconds",
    "Request latency (s)",
    ["endpoint"],
)

MODEL = None
PREPROCESSOR = None


class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


def load_models():
    """Load trained model and preprocessor from disk"""
    global MODEL, PREPROCESSOR
    try:
        with Path("models/best_model.pkl").open("rb") as f:
            MODEL = pickle.load(f)
        with Path("models/preprocessor.pkl").open("rb") as f:
            PREPROCESSOR = pickle.load(f)
        logger.info("✓ Models loaded")
    except Exception as e:
        logger.error("✗ Failed to load models: %s", e)


app = FastAPI(title="Heart Disease Prediction API")


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    """Log every request and track Prometheus metrics"""
    start_time = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration = time.perf_counter() - start_time
        endpoint = request.url.path
        REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, http_status=status_code).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        logger.info(
            f"{request.method} {endpoint} status={status_code} duration_ms={duration * 1000:.2f}",
        )


@app.on_event("startup")
async def startup_event():
    load_models()


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(patient: PatientData):
    try:
        if MODEL is None or PREPROCESSOR is None:
            raise HTTPException(500, "Model not loaded")

        input_df = pd.DataFrame([patient.model_dump()])
        input_processed = PREPROCESSOR.transform(input_df)
        prediction = MODEL.predict(input_processed)[0]
        confidence = max(MODEL.predict_proba(input_processed)[0])
        risk_level = "low" if confidence < 0.6 else ("medium" if confidence < 0.8 else "high")

        logger.info(
            f"prediction={int(prediction)} risk={risk_level} confidence={confidence:.3f} age={patient.age}",
        )

        return {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "risk_level": risk_level,
        }
    except HTTPException:
        # Re-raise HTTP errors untouched
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(400, str(e))


@app.get("/")
async def root():
    return {
        "service": "Heart Disease Prediction API",
        "endpoints": {"/health": "GET", "/predict": "POST", "/metrics": "GET"},
    }
