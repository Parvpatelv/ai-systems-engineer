# churn_api/app/main.py
import os
import glob
import json
import datetime
import pathlib

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Logging setup (prediction logs) ---
LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(exist_ok=True)
PRED_LOG = LOG_DIR / "prediction_log.jsonl"

def log_prediction(payload: dict, prediction: dict, model_version: str):
    """Append prediction event to JSONL log."""
    event = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "model_version": model_version,
        "request": payload,
        "prediction": prediction,
    }
    with open(PRED_LOG, "a") as f:
        f.write(json.dumps(event) + "\n")

# ---- Config ----
MODEL_DIR = "models"
MODEL_VERSION = os.getenv("MODEL_VERSION")  # optional; if unset, pick latest
app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# CORS (allow all by default; tighten later if you have a frontend domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request Schema ----
class CustomerFeatures(BaseModel):
    gender: str
    senior_citizen: int
    partner: str
    dependents: str
    tenure: float
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: float

# ---- Model Loading ----
_model = None
_loaded_version = None

def _get_model_path() -> str:
    """Return model path: either specific version or latest."""
    if MODEL_VERSION:
        path = f"{MODEL_DIR}/churn_model_{MODEL_VERSION}.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model version {MODEL_VERSION} not found at {path}")
        return path
    # Fallback: pick latest by filename sort
    files = sorted(glob.glob(f"{MODEL_DIR}/churn_model_*.pkl"))
    if not files:
        raise FileNotFoundError("No trained models found in models/. Train and commit a model first.")
    return files[-1]

def _load_model():
    """Lazy-load model once."""
    global _model, _loaded_version
    if _model is None:
        path = _get_model_path()
        _model = joblib.load(path)
        fname = os.path.basename(path)
        _loaded_version = fname.replace("churn_model_", "").replace(".pkl", "")
    return _model

# ---- Lifecycle ----
@app.on_event("startup")
def startup_event():
    # Try to load the model on startup; still allow app to boot and show a clear error later if it fails.
    try:
        _load_model()
        print(f"✅ Loaded model version: {_loaded_version}")
    except Exception as e:
        # Don't crash the process; health endpoint will expose the error
        print(f"⚠️  Model not loaded at startup: {e}")

# ---- Routes ----
@app.get("/")
def root():
    """Landing endpoint with helpful links."""
    body = {
        "status": "ok",
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "version": "/version",
        "predict": "/predict",
    }
    return JSONResponse(content=body)

@app.get("/health")
def health():
    try:
        _load_model()
        return JSONResponse(content={"status": "ok", "model_version": _loaded_version})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/version")
def version():
    try:
        _load_model()
        return JSONResponse(content={"model_version": _loaded_version})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(features: CustomerFeatures):
    try:
        model = _load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not available: {e}")

    X = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(X)[:, 1][0])
    label = int(proba >= 0.5)

    result = {
        "churn_probability": proba,
        "churn_label": label,
        "model_version": _loaded_version,
    }
    # Log the prediction event
    log_prediction(features.model_dump(), result, _loaded_version)
    return JSONResponse(content=result)
