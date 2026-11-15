from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

from blastvibe.utils.logger import setup_logger

logger = setup_logger("blastvibe.models.baseline")


class USBMRegressor:
    """
    Classical USBM empirical attenuation regressor: PPV = K * (SD)^(-beta)
    Fitted via log-linear ordinary least squares.
    """
    def __init__(self, sd_index: int = 2) -> None:
        self.sd_index = sd_index
        self.reg = LinearRegression()
        self.K_val: Optional[float] = None
        self.beta_val: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "USBMRegressor":
        sd = X[:, self.sd_index]
        if (sd <= 0).any():
            raise ValueError("Scaled Distance (SD) must be strictly positive for USBM log transformation.")
        if (y <= 0).any():
            raise ValueError("Target PPV must be strictly positive for USBM log transformation.")
        log_sd = np.log(sd).reshape(-1, 1)
        log_y = np.log(y)
        self.reg.fit(log_sd, log_y)
        self.K_val = float(np.exp(self.reg.intercept_))
        self.beta_val = float(-self.reg.coef_[0])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        sd = X[:, self.sd_index]
        log_sd = np.log(sd).reshape(-1, 1)
        preds = np.exp(self.reg.predict(log_sd))
        return np.maximum(0.0, preds)


def run_usbm_baseline(
    X: np.ndarray,
    y: np.ndarray,
    sd_index: int = 2,
    n_splits: int = 5,
    random_state: int = 42,
    return_model: bool = False,
) -> Union[Tuple[Dict[str, float], np.ndarray], Tuple[Dict[str, float], np.ndarray, USBMRegressor]]:
    """
    Evaluates the classical USBM (United States Bureau of Mines) empirical blast
    attenuation formulation:
        PPV = K * (SD)^(-beta)
    fitted via log-linear ordinary least squares regression on the Scaled Distance (SD):
        ln(PPV) = ln(K) - beta * ln(SD)
        
    Args:
        X: Feature matrix [N, D] where column at sd_index is Scaled Distance.
        y: True Peak Particle Velocity (PPV) target values [N].
        sd_index: Column index of Scaled Distance (SD). Defaults to 2.
        n_splits: Number of cross-validation splits.
        random_state: Random state for deterministic K-Fold split.
        
    Returns:
        metrics: Dictionary of aggregated out-of-fold metrics (r2, mae, mse, rmse).
        oof_predictions: Array of out-of-fold predictions.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_predictions = np.zeros_like(y, dtype=float)

    r2_scores, mae_scores, mse_scores, rmse_scores = [], [], [], []
    k_params, beta_params = [], []

    sd_all = X[:, sd_index]
    if (sd_all <= 0).any():
        raise ValueError("Scaled Distance (SD) must be strictly positive for USBM log transformation.")
    if (y <= 0).any():
        raise ValueError("Target PPV must be strictly positive for USBM log transformation.")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        fold_model = USBMRegressor(sd_index=sd_index)
        fold_model.fit(X[train_idx], y[train_idx])

        k_params.append(fold_model.K_val)
        beta_params.append(fold_model.beta_val)

        y_val = y[val_idx]
        pred_val = fold_model.predict(X[val_idx])
        oof_predictions[val_idx] = pred_val

        r2_scores.append(r2_score(y_val, pred_val))
        mae_scores.append(mean_absolute_error(y_val, pred_val))
        mse = mean_squared_error(y_val, pred_val)
        mse_scores.append(mse)
        rmse_scores.append(np.sqrt(mse))

    metrics = {
        "r2": float(np.mean(r2_scores)),
        "mae": float(np.mean(mae_scores)),
        "mse": float(np.mean(mse_scores)),
        "rmse": float(np.mean(rmse_scores)),
        "fitted_K": float(np.mean(k_params)),
        "fitted_beta": float(np.mean(beta_params)),
    }

    logger.info(
        f"Classical USBM Physics Baseline ({n_splits}-Fold CV) -> "
        f"R2: {metrics['r2']:.4f} | MAE: {metrics['mae']:.4f} | "
        f"RMSE: {metrics['rmse']:.4f} | [Fitted K={metrics['fitted_K']:.2f}, beta={metrics['fitted_beta']:.2f}]"
    )

    if return_model:
        final_model = USBMRegressor(sd_index=sd_index).fit(X, y)
        return metrics, oof_predictions, final_model

    return metrics, oof_predictions


def run_xgboost_baseline(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    param_grid: Optional[Dict[str, Any]] = None,
    inner_cv_splits: int = 3,
    return_model: bool = False,
) -> Union[Tuple[Dict[str, float], np.ndarray], Tuple[Dict[str, float], np.ndarray, xgb.XGBRegressor]]:
    """
    Evaluates an equitable industrial baseline XGBoost Regressor using nested
    cross-validation (GridSearchCV within outer K-Fold).
    
    Guarantees equivalent hyperparameter tuning rigor to the GAT grid search.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [2, 3, 4, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_predictions = np.zeros_like(y, dtype=float)

    r2_scores = []
    mae_scores = []
    mse_scores = []
    rmse_scores = []
    best_params_list = []

    logger.info(
        f"Running Equitable Tuned XGBoost Baseline across {n_splits} outer folds "
        f"with {inner_cv_splits}-fold inner GridSearchCV..."
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        base_estimator = xgb.XGBRegressor(
            random_state=random_state,
            verbosity=0,
            n_jobs=1,
        )

        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=inner_cv_splits,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        best_estimator = grid_search.best_estimator_
        best_params_list.append(grid_search.best_params_)

        preds = best_estimator.predict(X_val)
        oof_predictions[val_idx] = preds

        r2 = r2_score(y_val, preds)
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)

        r2_scores.append(r2)
        mae_scores.append(mae)
        mse_scores.append(mse)
        rmse_scores.append(rmse)

    metrics = {
        "r2": float(np.mean(r2_scores)),
        "mae": float(np.mean(mae_scores)),
        "mse": float(np.mean(mse_scores)),
        "rmse": float(np.mean(rmse_scores)),
    }

    logger.info(
        f"Equitable Tuned XGBoost ({n_splits}-Fold CV) -> "
        f"R2: {metrics['r2']:.4f} | MAE: {metrics['mae']:.4f} | "
        f"MSE: {metrics['mse']:.4f} | RMSE: {metrics['rmse']:.4f}"
    )

    if return_model:
        logger.info("Fitting production Tuned XGBoost on 100% of training data...")
        prod_grid_search = GridSearchCV(
            estimator=xgb.XGBRegressor(
                random_state=random_state,
                verbosity=0,
                n_jobs=1,
            ),
            param_grid=param_grid,
            cv=inner_cv_splits,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        prod_grid_search.fit(X, y)
        return metrics, oof_predictions, prod_grid_search.best_estimator_

    return metrics, oof_predictions
