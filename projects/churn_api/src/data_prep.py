# churn_api/src/data_prep.py
from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "churn"  # 'Yes'/'No'

CATEGORICAL = [
    "gender","senior_citizen","partner","dependents","phone_service","multiple_lines",
    "internet_service","online_security","online_backup","device_protection",
    "tech_support","streaming_tv","streaming_movies","contract","paperless_billing",
    "payment_method",
]
NUMERIC = ["tenure", "monthly_charges", "total_charges"]

FEATURES = CATEGORICAL + NUMERIC

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # basic cleanup
    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df = df.dropna(subset=["total_charges"])
    return df

def train_val_split(df: pd.DataFrame, test_size=0.2, random_state=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Robust split that:
    - Accepts float (fraction) or int (absolute rows) for test_size
    - Uses stratify when feasible
    - Falls back to non-stratified split if the test or train fold would be too small
    """
    y = df[TARGET]
    n_samples = len(df)
    n_classes = y.nunique()

    # Compute absolute test size from float or int
    if isinstance(test_size, float):
        ts_abs = max(1, int(round(test_size * n_samples)))
    else:
        ts_abs = int(test_size)

    # Keep within valid bounds (at least 1 row in each split)
    ts_abs = max(1, min(ts_abs, n_samples - 1))

    # We can only stratify if BOTH folds can contain at least one sample per class
    can_stratify = (ts_abs >= n_classes) and ((n_samples - ts_abs) >= n_classes)
    strat = y if can_stratify else None

    return train_test_split(df, test_size=ts_abs, random_state=random_state, stratify=strat)

def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURES].copy()
    y = (df[TARGET].astype(str).str.lower() == "yes").astype(int)
    return X, y
