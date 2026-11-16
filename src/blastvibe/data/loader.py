from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from blastvibe.utils.logger import setup_logger

logger = setup_logger("blastvibe.data.loader")


def compute_scaled_distance(
    dis: float | np.ndarray | pd.Series,
    charge: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Centralized physics calculation of USBM Scaled Distance:
        SD = DIS / sqrt(CHARGE)

    Args:
        dis: Distance from blast face in meters (must be >= 0).
        charge: Maximum explosive charge per delay in kilograms (must be > 0).

    Returns:
        Scaled Distance in m / kg^0.5.
    """
    if isinstance(dis, (int, float)) and isinstance(charge, (int, float)):
        if charge <= 0:
            raise ValueError(f"Detected non-positive CHARGE ({charge}); explosive charge must be strictly positive.")
        if dis < 0:
            raise ValueError(f"Detected negative DIS ({dis}); distance must be non-negative.")
        return float(dis / np.sqrt(charge))

    dis_arr = np.asarray(dis)
    charge_arr = np.asarray(charge)

    if np.any(charge_arr <= 0):
        raise ValueError("Detected non-positive CHARGE values; explosive charge must be strictly positive.")
    if np.any(dis_arr < 0):
        raise ValueError("Detected negative DIS values; distance must be non-negative.")

    return dis / np.sqrt(charge)


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
    file_path: str | Path,
    feature_cols: list[str] | None = None,
    target_col: str = "SUM",
) -> pd.DataFrame:
    """
    Loads blast monitoring dataset, performs data validation, and engineers
    physics-informed Scaled Distance (SD = DIS / sqrt(CHARGE)).
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format for {path}. Use .csv or .xlsx")

    required = ["DIS", "CHARGE", target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Present: {list(df.columns)}")

    # 1. Coerce required and target columns to numeric, replacing unparseable dirty strings with NaN
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Filter out null/NaN records prior to physics computation
    df = df.dropna(subset=required).copy()

    # 3. Execute domain validations (CHARGE > 0, DIS >= 0)
    if (df["CHARGE"] <= 0).any():
        raise ValueError("Detected non-positive CHARGE values; explosive charge must be strictly positive.")
    if (df["DIS"] < 0).any():
        raise ValueError("Detected negative DIS values; distance must be non-negative.")

    # 4. Physics-based feature engineering (USBM Scaled Distance: SD = DIS / sqrt(CHARGE))
    df["SD"] = compute_scaled_distance(df["DIS"], df["CHARGE"])

    if feature_cols is None:
        feature_cols = ["DIS", "CHARGE", "SD"]

    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not present after feature engineering.")

    clean_df = df[feature_cols + [target_col]].dropna().copy()
    logger.info(
        f"Loaded and validated {len(clean_df)} records from {path}. " f"Features: {feature_cols}, Target: {target_col}"
    )
    return clean_df


def prepare_fold_data(
    X_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    y_train_raw: np.ndarray,
    y_val_raw: np.ndarray,
    train_indices: np.ndarray | None = None,
    val_indices: np.ndarray | None = None,
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
