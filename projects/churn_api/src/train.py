# churn_api/src/train.py
import json
import argparse
import joblib
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, log_loss

from data_prep import load_data, train_val_split, get_X_y, CATEGORICAL, NUMERIC


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    clf = LogisticRegression(max_iter=200)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe


def safe_metrics(y_true, proba):
    """
    Compute metrics safely:
    - accuracy always
    - log_loss and roc_auc only if both classes present in y_true
    Accepts proba as shape (n,) or (n,2); uses positive-class probs.
    """
    y_true = np.asarray(y_true)

    if proba.ndim == 2:
        pos_proba = proba[:, 1]
    else:
        pos_proba = proba

    y_pred = (pos_proba >= 0.5).astype(int)

    m = {"accuracy": float(accuracy_score(y_true, y_pred))}

    unique = np.unique(y_true)
    if unique.size >= 2:
        eps = 1e-15
        p = np.clip(pos_proba, eps, 1 - eps)
        try:
            m["log_loss"] = float(log_loss(y_true, p))
        except Exception:
            pass
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, pos_proba))
        except Exception:
            pass

    return m


def main(args):
    logger.info(f"Loading data from {args.data}")
    df = load_data(args.data)

    # Split (robustness handled inside train_val_split)
    train_df, val_df = train_val_split(df, test_size=args.test_size, random_state=42)

    # Show class balance to help diagnose tiny/imbalanced folds
    logger.info(f"Train size: {len(train_df)} | Val size: {len(val_df)}")
    logger.info(f"Train class counts:\n{train_df['churn'].value_counts(dropna=False)}")
    logger.info(f"Val class counts:\n{val_df['churn'].value_counts(dropna=False)}")

    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)

    pipe = build_pipeline()
    logger.info("Fitting model...")
    pipe.fit(X_train, y_train)

    logger.info("Evaluating...")
    proba = pipe.predict_proba(X_val)
    metrics = safe_metrics(y_val, proba)
    logger.info(f"Validation metrics: {metrics}")

    # Optional pretty classification report (only if both classes exist)
    if len(np.unique(y_val)) >= 2:
        y_pred = (proba[:, 1] >= 0.5).astype(int)
        logger.info("\n" + classification_report(y_val, y_pred))

    # Ensure output directory exists, then save
    model_path = Path(args.model_out)
    cols_path = Path(args.columns_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    cols_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, model_path)
    with open(cols_path, "w") as f:
        json.dump({"categorical": CATEGORICAL, "numeric": NUMERIC}, f)

    logger.info(f"Saved model to {model_path} and columns to {cols_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn_sample.csv")
    parser.add_argument("--model_out", default="models/model.pkl")
    parser.add_argument("--columns_out", default="models/columns.json")
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,  # slightly larger to reduce single-class val splits on small data
        help="Validation size as a fraction (0-1) or integer rows if your train_val_split supports it.",
    )
    args = parser.parse_args()
    main(args)
