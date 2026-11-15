import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from blastvibe.utils.logger import setup_logger

logger = setup_logger("blastvibe.data.loader")


@dataclass
class ProcessedFoldData:
    X_train_scaled: np.ndarray
    X_val_scaled: np.ndarray
    y_train_scaled: np.ndarray
    y_val_scaled: np.ndarray
    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    scaler_x: StandardScaler
    scaler_y: StandardScaler


def load_raw_dataset(
    file_path: str,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "SUM",
) -> pd.DataFrame:
    """
    Loads blast monitoring dataset, performs data validation, and engineers
    physics-informed Scaled Distance (SD = DIS / sqrt(CHARGE)).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format for {file_path}. Use .csv or .xlsx")

    required = ["DIS", "CHARGE", target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Present: {list(df.columns)}")

    # Data hygiene & domain validation
    if (df["CHARGE"] <= 0).any():
        raise ValueError("Detected non-positive CHARGE values; cannot compute square root.")
    if (df["DIS"] < 0).any():
        raise ValueError("Detected negative DIS (distance) values.")

    # Physics-based feature engineering (USBM Scaled Distance)
    df["SD"] = df["DIS"] / np.sqrt(df["CHARGE"])

    if feature_cols is None:
        feature_cols = ["DIS", "CHARGE", "SD"]

    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not present after feature engineering.")

    clean_df = df[feature_cols + [target_col]].dropna().copy()
    logger.info(
        f"Loaded and validated {len(clean_df)} records from {file_path}. "
        f"Features: {feature_cols}, Target: {target_col}"
    )
    return clean_df


def prepare_fold_data(
    X_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    y_train_raw: np.ndarray,
    y_val_raw: np.ndarray,
    train_indices: Optional[np.ndarray] = None,
    val_indices: Optional[np.ndarray] = None,
) -> ProcessedFoldData:
    """
    Scales features and targets strictly using training statistics to prevent data leakage.
    """
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train_raw)
    X_val_scaled = scaler_x.transform(X_val_raw)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)
    y_val_scaled = scaler_y.transform(y_val_raw.reshape(-1, 1)).reshape(-1)

    if train_indices is None:
        train_indices = np.arange(len(X_train_raw), dtype=int)
    if val_indices is None:
        val_indices = np.arange(len(X_val_raw), dtype=int)

    return ProcessedFoldData(
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        y_train_scaled=y_train_scaled,
        y_val_scaled=y_val_scaled,
        y_train_raw=y_train_raw,
        y_val_raw=y_val_raw,
        train_indices=train_indices,
        val_indices=val_indices,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
    )
