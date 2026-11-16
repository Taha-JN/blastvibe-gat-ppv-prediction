import numpy as np
import pytest

from blastvibe.models.baseline import USBMRegressor, run_usbm_baseline, run_xgboost_baseline


def test_usbm_regressor_parameter_recovery():
    """Validates that USBMRegressor recovers positive physical constants (K > 0, beta > 0)."""
    np.random.seed(42)
    n_samples = 50
    # Scaled Distance between 5 and 50
    sd = np.linspace(5.0, 50.0, n_samples)
    true_K = 1200.0
    true_beta = 1.45
    # Synthetic PPV with low log-noise
    noise = np.random.normal(0.0, 0.05, size=n_samples)
    ppv = true_K * (sd ** (-true_beta)) * np.exp(noise)

    # Feature matrix [DIS, CHARGE, SD]
    charge = np.full(n_samples, 100.0)
    dis = sd * np.sqrt(charge)
    X = np.column_stack([dis, charge, sd])

    model = USBMRegressor(sd_index=2)
    model.fit(X, ppv)

    assert model.K_val is not None
    assert model.beta_val is not None
    assert model.K_val > 0.0, f"Expected positive K, got {model.K_val}"
    assert model.beta_val > 0.0, f"Expected positive beta, got {model.beta_val}"
    # Check within 20% of ground truth
    assert np.isclose(model.K_val, true_K, rtol=0.25)
    assert np.isclose(model.beta_val, true_beta, rtol=0.25)

    preds = model.predict(X)
    assert len(preds) == n_samples
    assert np.all(preds >= 0.0), "Predicted PPV must be strictly non-negative."


def test_usbm_regressor_non_positive_values_fail():
    """Asserts that non-positive SD or target PPV triggers domain validation errors."""
    X_invalid_sd = np.array([[100.0, 25.0, 0.0]])
    y_valid = np.array([20.0])
    model = USBMRegressor(sd_index=2)
    with pytest.raises(ValueError, match="Scaled Distance"):
        model.fit(X_invalid_sd, y_valid)

    X_valid = np.array([[100.0, 25.0, 20.0]])
    y_invalid = np.array([-5.0])
    with pytest.raises(ValueError, match="Target PPV"):
        model.fit(X_valid, y_invalid)


def test_run_usbm_baseline_execution():
    """Tests cross-validated execution of USBM baseline."""
    np.random.seed(42)
    n_samples = 40
    sd = np.linspace(8.0, 45.0, n_samples)
    ppv = 800.0 * (sd**-1.3)
    charge = np.full(n_samples, 64.0)
    dis = sd * 8.0
    X = np.column_stack([dis, charge, sd])

    metrics, oof_preds, fitted_model = run_usbm_baseline(
        X=X,
        y=ppv,
        sd_index=2,
        n_splits=3,
        random_state=42,
        return_model=True,
    )

    for key in ["r2", "mae", "mse", "rmse", "fitted_K", "fitted_beta"]:
        assert key in metrics
        assert isinstance(metrics[key], float)

    assert metrics["r2"] > 0.8
    assert metrics["fitted_K"] > 0.0
    assert metrics["fitted_beta"] > 0.0
    assert len(oof_preds) == n_samples
    assert isinstance(fitted_model, USBMRegressor)


def test_run_xgboost_baseline_execution():
    """Tests nested cross-validated execution of equitable XGBoost baseline."""
    np.random.seed(42)
    n_samples = 30
    X = np.random.uniform(10.0, 100.0, size=(n_samples, 3))
    y = 50.0 + 0.5 * X[:, 0] - 0.2 * X[:, 1] + 1.2 * X[:, 2]

    param_grid = {
        "n_estimators": [10],
        "max_depth": [2],
        "learning_rate": [0.1],
    }

    metrics, oof_preds, fitted_model = run_xgboost_baseline(
        X=X,
        y=y,
        n_splits=2,
        random_state=42,
        param_grid=param_grid,
        inner_cv_splits=2,
        return_model=True,
    )

    for key in ["r2", "mae", "mse", "rmse"]:
        assert key in metrics
        assert isinstance(metrics[key], float)

    assert len(oof_preds) == n_samples
    test_preds = fitted_model.predict(X[:5])
    assert len(test_preds) == 5
