"""FastAPI Prediction Service"""
import logging
from pathlib import Path
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    global MODEL, PREPROCESSOR
    try:
        with open('models/best_model.pkl', 'rb') as f:
            MODEL = pickle.load(f)
        with open('models/preprocessor.pkl', 'rb') as f:
            PREPROCESSOR = pickle.load(f)
        logger.info("✓ Models loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load models: {e}")

app = FastAPI(title="Heart Disease Prediction API")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": MODEL is not None}

@app.post("/predict")
async def predict(patient: PatientData):
    try:
        if MODEL is None or PREPROCESSOR is None:
            raise HTTPException(500, "Model not loaded")
        
        input_df = pd.DataFrame([patient.dict()])
        input_processed = PREPROCESSOR.transform(input_df)
        prediction = MODEL.predict(input_processed)[0]
        confidence = max(MODEL.predict_proba(input_processed)[0])
        
        return {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "risk_level": "low" if confidence < 0.6 else ("medium" if confidence < 0.8 else "high")
        }
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/")
async def root():
    return {"service": "Heart Disease Prediction API", "endpoints": {"/health": "GET", "/predict": "POST"}}
