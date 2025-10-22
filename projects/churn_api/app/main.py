# app/main.py
from pathlib import Path
import os
import json
import traceback

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---- Model/artifact paths ----
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
COLUMNS_PATH = MODEL_PATH.parent / "columns.json"

app = FastAPI(title="Churn API", version="1.0.0")

# ---- Load model at startup (once) ----
try:
    pipe = joblib.load(MODEL_PATH)
except Exception as e:
    # Fail fast with a clear message if model is missing or incompatible
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}") from e

# ---- Load feature metadata to keep inference aligned with training ----
try:
    with open(COLUMNS_PATH, "r") as f:
        cols_meta = json.load(f)
    CATEGORICAL = cols_meta["categorical"]
    NUMERIC = cols_meta["numeric"]
    FEATURES = CATEGORICAL + NUMERIC
except Exception as e:
    raise RuntimeError(f"Failed to load columns from {COLUMNS_PATH}: {e}") from e

# ---- Request schema (align types with training) ----
class PredictRequest(BaseModel):
    gender: str
    senior_citizen: int = Field(ge=0, le=1)  # 0 or 1
    partner: str
    dependents: str
    tenure: int = Field(ge=0)
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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        record = payload.model_dump()  # pydantic v2
        df = pd.DataFrame([record])

        # Ensure all required columns exist
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required fields: {missing}")

        # Enforce exact feature order
        df = df[FEATURES].copy()

        # Defensive numeric coercion
        for col in NUMERIC:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[NUMERIC].isna().any().any():
            bad = df[NUMERIC].columns[df[NUMERIC].isna().any()].tolist()
            raise HTTPException(status_code=400, detail=f"Invalid numeric values for: {bad}")

        # Predict
        proba = float(pipe.predict_proba(df)[:, 1][0])
        pred = int(proba >= 0.5)
        return {"churn_probability": proba, "churn": pred}

    except HTTPException:
        # Bubble up 4xx with clear message
        raise
    except Exception as e:
        # Log full traceback to container logs, return readable error to client
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {type(e).__name__}: {e}")
