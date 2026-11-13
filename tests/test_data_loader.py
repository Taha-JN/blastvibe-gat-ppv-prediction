import os
import pytest
import numpy as np
import pandas as pd
from blastvibe.data.loader import load_raw_dataset, prepare_fold_data


def test_feature_engineering_scaled_distance(tmp_path):
    # Create temporary dummy CSV
    data = {
        "DIS": [100.0, 400.0, 900.0],
        "CHARGE": [25.0, 100.0, 225.0],
        "SUM": [50.0, 20.0, 10.0],
    }
    csv_file = tmp_path / "test_data.csv"
    pd.DataFrame(data).to_csv(csv_file, index=False)

    df = load_raw_dataset(str(csv_file))

    assert "SD" in df.columns
    # SD = DIS / sqrt(CHARGE)
    # 100 / sqrt(25) = 100 / 5 = 20.0
    # 400 / sqrt(100) = 400 / 10 = 40.0
    # 900 / sqrt(225) = 900 / 15 = 60.0
    expected_sd = [20.0, 40.0, 60.0]
    np.testing.assert_allclose(df["SD"].values, expected_sd, rtol=1e-5)


def test_prepare_fold_data_no_leakage():
    X_train = np.array([[10.0, 100.0, 1.0], [20.0, 200.0, 2.0]], dtype=float)
    X_val = np.array([[15.0, 150.0, 1.5]], dtype=float)
    y_train = np.array([5.0, 10.0], dtype=float)
    y_val = np.array([7.5], dtype=float)

    fold_data = prepare_fold_data(X_train, X_val, y_train, y_val)

    # Scaler mean must match X_train mean exactly, unaffected by X_val
    np.testing.assert_allclose(fold_data.scaler_x.mean_, [15.0, 150.0, 1.5])
    # Scaler y mean must match y_train mean (7.5)
    np.testing.assert_allclose(fold_data.scaler_y.mean_, [7.5])
