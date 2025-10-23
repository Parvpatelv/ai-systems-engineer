# churn_api/src/train.py
import os
import json
import pathlib
import subprocess
import datetime as dt

import joblib
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    accuracy_score,
    log_loss,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---- Your data prep helpers ----
from data_prep import load_data, train_val_split, get_X_y, CATEGORICAL, NUMERIC

# ---- Paths & Logging setup ----
MODELS_DIR = pathlib.Path("models")
LOGS_DIR = pathlib.Path("logs")
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
logger.add(LOGS_DIR / "train.log", rotation="1 MB", retention=5, enqueue=True, level="INFO")

# ---- Model version (read from environment variable) ----
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0")


def get_git_sha() -> str:
    """Return short git SHA for traceability; 'unknown' if not a git repo."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def build_pipeline() -> Pipeline:
    """Build a simple ML pipeline with preprocessing and logistic regression."""
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    clf = LogisticRegression(max_iter=200)
    return Pipeline([("pre", pre), ("clf", clf)])


def main(data_path: str):
    logger.info(f"Loading data from {data_path}")
    df = load_data(data_path)
    logger.info(f"Rows: {len(df)} | churn counts: {df['churn'].value_counts(dropna=False).to_dict()}")

    # Split data
    train_df, val_df = train_val_split(df)
    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)

    # Extra visibility (especially useful on tiny datasets)
    logger.info(f"Train size: {len(train_df)} | Val size: {len(val_df)}")
    try:
        logger.info("Train class counts:\n" + y_train.value_counts().to_string())
        logger.info("Val class counts:\n" + y_val.value_counts().to_string())
    except Exception:
        # y could be a numpy array; this keeps logging resilient
        pass

    # Build and train model
    pipe = build_pipeline()
    logger.info(f"Training churn model | version={MODEL_VERSION}")
    pipe.fit(X_train, y_train)

    # ---------- Robust evaluation block ----------
    proba = pipe.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Base metrics always available
    metrics = {
        "accuracy": float(accuracy_score(y_val, pred)),
        "log_loss": float(log_loss(y_val, proba, labels=[0, 1])),
    }

    # ROC-AUC only if both classes present
    if len(set(y_val)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_val, proba))
    else:
        metrics["roc_auc"] = None
        logger.warning("Validation set has a single class; ROC AUC is undefined.")

    logger.info(f"Validation metrics: {metrics}")

    # Classification report (safe for zero-division)
    cls_report = classification_report(y_val, pred, zero_division=0)
    logger.info("Classification report:\n" + cls_report)
    # ---------- End evaluation block ----------

    # ---- Save model & metadata with version ----
    model_path = MODELS_DIR / f"churn_model_{MODEL_VERSION}.pkl"
    meta_path = MODELS_DIR / f"metadata_{MODEL_VERSION}.json"

    joblib.dump(pipe, model_path)

    metadata = {
        "version": MODEL_VERSION,
        "saved_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "git_sha": get_git_sha(),
        "metrics": metrics,
        "data": {
            "path": data_path,
            "n_rows": int(len(df)),
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
        },
        "features": {"categorical": CATEGORICAL, "numeric": NUMERIC},
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved model -> {model_path}")
    logger.info(f"Saved metadata -> {meta_path}")
    logger.success("Training completed successfully ✅")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn_sample.csv")
    args = parser.parse_args()

    main(args.data)
