# churn_api/app/main.py
import os
import glob
import json
import datetime
import pathlib
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# --- Logging setup (prediction logs) ---
LOG_DIR = pathlib.Path(os.getenv("LOG_DIR", "/tmp/logs"))  # cloud-safe default
LOG_DIR.mkdir(parents=True, exist_ok=True)
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

# ---- Prometheus metrics ----
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total number of predictions served",
    labelnames=("model_version",),
)

PREDICTION_ERRORS_TOTAL = Counter(
    "prediction_errors_total",
    "Total number of failed prediction requests",
    labelnames=("model_version",),
)

PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Time taken (seconds) to run a prediction",
    labelnames=("model_version",),
)

REQUEST_COUNT = Counter(
    "churn_api_requests_total",
    "Total number of requests received",
    ["method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "churn_api_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"]
)

# Optional: middleware to capture generic request metrics
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    try:
        REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)
    except Exception:
        # never let metrics break the response path
        pass
    return response

# ---- Lifecycle ----
@app.on_event("startup")
def startup_event():
    # Try to load the model on startup; still allow app to boot and show a clear error later if it fails.
    try:
        _load_model()
        print(f"✅ Loaded model version: {_loaded_version}")
    except Exception as e:
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
        "metrics": "/metrics",
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
    # Time just the prediction (including preprocessing) for latency metric
    start = time.time()
    try:
        model = _load_model()

        # Your pipeline expects a DataFrame with named columns (kept from your earlier implementation)
        X = pd.DataFrame([features.model_dump()])

        proba = float(model.predict_proba(X)[:, 1][0])
        label = int(proba >= 0.5)

        duration = time.time() - start

        result = {
            "churn_probability": proba,
            "churn_label": label,
            "model_version": _loaded_version,
            "latency_sec": round(duration, 3),
        }

        # Metrics
        PREDICTIONS_TOTAL.labels(_loaded_version).inc()
        PREDICTION_LATENCY.labels(_loaded_version).observe(duration)

        # Log the prediction event
        log_prediction(features.model_dump(), result, _loaded_version)

        return JSONResponse(content=result)

    except Exception as e:
        # Count the error against the current resolved model version if available
        version_for_error = _loaded_version or (MODEL_VERSION or "unknown")
        try:
            PREDICTION_ERRORS_TOTAL.labels(version_for_error).inc()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    # Standard Prometheus text format
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
