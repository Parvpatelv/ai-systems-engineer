# churn_api/src/predict.py
import joblib
import numpy as np
from typing import Dict

def load_model(model_path="models/model.pkl"):
    return joblib.load(model_path)

def predict_one(model, payload: Dict) -> Dict:
    """
    payload: dict with keys matching training features (categorical + numeric).
    Returns: dict with probability and label.
    """
    import pandas as pd
    X = pd.DataFrame([payload])  # single row
    proba = float(model.predict_proba(X)[:, 1][0])
    return {"churn_probability": proba, "churn_label": int(proba >= 0.5)}
