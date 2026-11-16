import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from blastvibe.configs.schema import PipelineConfig
from blastvibe.data.loader import load_raw_dataset
from blastvibe.models.baseline import run_usbm_baseline, run_xgboost_baseline
from blastvibe.models.predictor import BlastVibePredictor
from blastvibe.training.grid import run_cross_validation_grid_search
from blastvibe.utils.logger import setup_logger

SCRIPT_DIR = Path(__file__).resolve().parent
logger = setup_logger("blastvibe.main")


def set_seed(seed: int = 42) -> None:
    """Enforces deterministic global seeding across Python, NumPy, and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BlastVibe-GNN: Production-Ready Graph Neural Network Pipeline for Ground Vibration Regression"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(SCRIPT_DIR / "configs" / "default_config.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override path to dataset file (CSV or Excel)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip running the XGBoost comparison baseline",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> PipelineConfig:
    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}
    # Strict fail-fast validation with Pydantic
    return PipelineConfig(**raw_config)


def main() -> None:
    args = parse_args()
    pipeline_config = load_config(args.config)
    set_seed(pipeline_config.cv.random_state)
    config = pipeline_config.to_dict()

    # Modernized path resolution using pathlib.Path
    raw_path = Path(args.data_path or pipeline_config.data.raw_path)
    if raw_path.is_absolute():
        data_path = raw_path
    else:
        candidate = (SCRIPT_DIR / raw_path).resolve()
        alt_candidate = (SCRIPT_DIR / "data" / raw_path.name).resolve()
        data_path = candidate if candidate.exists() else (alt_candidate if alt_candidate.exists() else candidate)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    feature_cols = pipeline_config.data.feature_cols
    target_col = pipeline_config.data.target_col

    logger.info("=" * 60)
    logger.info("BlastVibe-GNN: Production Machine Learning Pipeline")
    logger.info("=" * 60)

    # 1. Ingestion & Physics Feature Engineering
    df = load_raw_dataset(data_path, feature_cols=feature_cols, target_col=target_col)
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # 2. Strict 80/20 Train-Test Split (Isolated Hold-out Test Set)
    # The 20% test partition is strictly isolated BEFORE any CV, tuning, or fitting
    random_state = pipeline_config.cv.random_state
    shuffle = pipeline_config.cv.shuffle

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, shuffle=shuffle
    )
    logger.info(
        f"Data Partitioning -> 80% Training Set ({len(X_train)} samples) | "
        f"20% Unseen Hold-out Test Set ({len(X_test)} samples)"
    )

    # 3. Benchmark Baseline Evaluations on 80% Train Set (USBM Physics & Tuned XGBoost)
    usbm_model = None
    usbm_oof = None
    xgb_model = None
    if not args.skip_baseline:
        logger.info("-" * 40)
        logger.info("Running Empirical Physics Baseline: USBM Attenuation Law (on 80% Train)")
        logger.info("-" * 40)
        _usbm_metrics, usbm_oof, usbm_model = run_usbm_baseline(
            X=X_train,
            y=y_train,
            sd_index=feature_cols.index("SD") if "SD" in feature_cols else 2,
            n_splits=pipeline_config.cv.n_splits,
            random_state=random_state,
            return_model=True,
        )

        logger.info("-" * 40)
        logger.info("Running Equitable Industrial Baseline: Tuned XGBoost (on 80% Train)")
        logger.info("-" * 40)
        _baseline_metrics, _, xgb_model = run_xgboost_baseline(
            X=X_train,
            y=y_train,
            n_splits=pipeline_config.cv.n_splits,
            random_state=random_state,
            return_model=True,
        )

    # 4. GAT Model Cross-Validation & Grid Search strictly on 80% Train Set
    device_setting = pipeline_config.training.device
    if device_setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_setting)
    logger.info(f"Using Compute Device: {device}")

    logger.info("-" * 40)
    logger.info("Running Leak-Free Inductive GAT Cross-Validation & Tuning (on 80% Train)")
    logger.info("-" * 40)

    _results_df, best_bundle = run_cross_validation_grid_search(
        X=X_train,
        y=y_train,
        config=config,
        device=device,
    )

    logger.info("=" * 60)
    logger.info("Cross-Validation & Model Training Complete!")
    if best_bundle:
        best_cfg = best_bundle["config"]
        logger.info(f"Optimal Hyperparameters: {best_cfg}")
        logger.info(
            f"GAT 5-Fold CV Validation Performance (Train Set Only) -> "
            f"R2: {best_cfg['mean_r2']:.4f} | "
            f"MAE: {best_cfg['mean_mae']:.4f} | "
            f"RMSE: {best_cfg['mean_rmse']:.4f}"
        )

        # 5. Production Inference Engine & Final Evaluation on 20% Unseen Test Set
        model_file = Path(pipeline_config.artifacts.best_model_file)
        best_model_path = model_file if model_file.is_absolute() else (SCRIPT_DIR / model_file).resolve()

        logger.info("-" * 40)
        logger.info("SYMMETRICAL EVALUATION ON 20% UNSEEN HOLD-OUT TEST SET (40 SAMPLES)")
        logger.info("-" * 40)

        # Baseline 1: USBM Physics Law on Hold-out Test Set
        if usbm_model is not None:
            y_test_pred_usbm = usbm_model.predict(X_test)
            test_r2_usbm = float(r2_score(y_test, y_test_pred_usbm))
            test_mae_usbm = float(mean_absolute_error(y_test, y_test_pred_usbm))
            test_rmse_usbm = float(np.sqrt(mean_squared_error(y_test, y_test_pred_usbm)))
            logger.info(
                f"Hold-out Test Set (Classical USBM) -> "
                f"R2: {test_r2_usbm:.4f} | MAE: {test_mae_usbm:.4f} mm/s | RMSE: {test_rmse_usbm:.4f} mm/s"
            )

        # Baseline 2: Tuned XGBoost Regressor on Hold-out Test Set
        if xgb_model is not None:
            y_test_pred_xgb = xgb_model.predict(X_test)
            test_r2_xgb = float(r2_score(y_test, y_test_pred_xgb))
            test_mae_xgb = float(mean_absolute_error(y_test, y_test_pred_xgb))
            test_rmse_xgb = float(np.sqrt(mean_squared_error(y_test, y_test_pred_xgb)))
            logger.info(
                f"Hold-out Test Set (Tuned XGBoost) -> "
                f"R2: {test_r2_xgb:.4f} | MAE: {test_mae_xgb:.4f} mm/s | RMSE: {test_rmse_xgb:.4f} mm/s"
            )

        # Proposed: BlastVibe-GAT on Hold-out Test Set
        predictor = BlastVibePredictor(bundle_path=best_model_path, device="cpu")
        y_test_pred = predictor.predict(X_test)
        test_r2 = float(r2_score(y_test, y_test_pred))
        test_mae = float(mean_absolute_error(y_test, y_test_pred))
        test_mse = float(mean_squared_error(y_test, y_test_pred))
        test_rmse = float(np.sqrt(test_mse))

        logger.info(
            f"Hold-out Test Set (BlastVibe-GAT)  -> "
            f"R2: {test_r2:.4f} | MAE: {test_mae:.4f} mm/s | RMSE: {test_rmse:.4f} mm/s"
        )

        sample_dis = 400.0
        sample_charge = 1850.0
        sample_pred = predictor.predict_single(dis=sample_dis, charge=sample_charge)
        logger.info(
            f"Single Shot Inference Test -> [DIS={sample_dis}m, CHARGE={sample_charge}kg] | "
            f"Predicted PPV: {sample_pred:.2f} mm/s"
        )

        # 6. Statistical Rigor: Wilcoxon Signed-Rank Hypothesis Testing
        if usbm_model is not None:
            logger.info("-" * 40)
            logger.info("STATISTICAL RIGOR: Wilcoxon Signed-Rank Hypothesis Testing")
            logger.info("-" * 40)

            # Test A: Out-of-Fold Cross-Validation Errors
            if usbm_oof is not None and best_bundle.get("oof_predictions") is not None:
                oof_errors_gat = np.abs(y_train - best_bundle["oof_predictions"])
                oof_errors_usbm = np.abs(y_train - usbm_oof)
                stat_oof, p_val_oof = wilcoxon(oof_errors_gat, oof_errors_usbm)
                logger.info(
                    f"Out-of-Fold Absolute Error Comparison (N={len(y_train)}) -> "
                    f"Wilcoxon stat: {stat_oof:.2f}, p-value: {p_val_oof:.4e}"
                )
                if p_val_oof >= 0.05:
                    logger.warning(
                        f"Out-of-Fold performance difference between GAT and USBM is NOT statistically significant "
                        f"(p = {p_val_oof:.4f} >= 0.05)."
                    )
                else:
                    logger.info(
                        f"Out-of-Fold performance difference between GAT and USBM IS statistically significant "
                        f"(p = {p_val_oof:.4f} < 0.05)."
                    )

            # Test B: Hold-out Test Set Absolute Errors
            test_errors_gat = np.abs(y_test - y_test_pred)
            test_errors_usbm = np.abs(y_test - y_test_pred_usbm)
            stat_test, p_val_test = wilcoxon(test_errors_gat, test_errors_usbm)
            logger.info(
                f"Hold-out Test Absolute Error Comparison (N={len(y_test)}) -> "
                f"Wilcoxon stat: {stat_test:.2f}, p-value: {p_val_test:.4e}"
            )
            if p_val_test >= 0.05:
                logger.warning(
                    f"Hold-out test performance difference between GAT and USBM is NOT statistically significant "
                    f"(p = {p_val_test:.4f} >= 0.05)."
                )
            else:
                logger.info(
                    f"Hold-out test performance difference between GAT and USBM IS statistically significant "
                    f"(p = {p_val_test:.4f} < 0.05)."
                )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
