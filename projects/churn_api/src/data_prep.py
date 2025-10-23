# churn_api/src/data_prep.py
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "churn"

# Define which columns are categorical and numeric
CATEGORICAL = [
    "gender", "partner", "dependents", "phone_service", "multiple_lines",
    "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "contract",
    "paperless_billing", "payment_method"
]

NUMERIC = ["tenure", "monthly_charges", "total_charges"]


def load_data(path: str) -> pd.DataFrame:
    """Load churn dataset from CSV."""
    df = pd.read_csv(path)
    # Normalize churn column to binary
    df[TARGET] = (df[TARGET].astype(str).str.lower() == "yes").astype(int)
    return df


def get_X_y(df: pd.DataFrame):
    """Split dataframe into X and y."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def train_val_split(df: pd.DataFrame, test_size=0.3, random_state=42):
    """Split data ensuring at least 2 validation samples and valid stratification."""
    y = df[TARGET]
    vc = y.value_counts()
    n = len(df)

    # Ensure at least 2 samples in validation
    ts = max(test_size, 2 / n)

    # Stratify only if both classes exist and each has ≥2 samples
    use_stratify = len(vc) == 2 and all(vc.get(c, 0) >= 2 for c in vc.index)

    if use_stratify:
        return train_test_split(df, test_size=ts, random_state=random_state, stratify=y)
    else:
        return train_test_split(df, test_size=ts, random_state=random_state)
