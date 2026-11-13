import argparse
import os
import sys
import yaml
import torch
import numpy as np

# Ensure src is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from blastvibe.data.loader import load_raw_dataset
from blastvibe.models.baseline import run_xgboost_baseline, run_usbm_baseline
from blastvibe.models.predictor import BlastVibePredictor
from blastvibe.training.trainer import run_cross_validation_grid_search
from blastvibe.utils.logger import setup_logger

logger = setup_logger("blastvibe.main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="BlastVibe-GNN: Production-Ready Graph Neural Network Pipeline for Ground Vibration Regression"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(SCRIPT_DIR, "configs", "default_config.yaml"),
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


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)

    # Resolve dataset path
    data_path = args.data_path or config.get("data", {}).get("raw_path", "data/data.csv")
    if not os.path.isabs(data_path):
        candidate = os.path.join(SCRIPT_DIR, data_path)
        if not os.path.exists(candidate):
            alt_candidate = os.path.join(SCRIPT_DIR, "data", os.path.basename(data_path))
            if os.path.exists(alt_candidate):
                data_path = alt_candidate
            else:
                data_path = candidate
        else:
            data_path = candidate

    feature_cols = config.get("data", {}).get("feature_cols", ["DIS", "CHARGE", "SD"])
    target_col = config.get("data", {}).get("target_col", "SUM")

    logger.info("=" * 60)
    logger.info("BlastVibe-GNN: Production Machine Learning Pipeline")
    logger.info("=" * 60)

    # 1. Ingestion & Physics Feature Engineering
    df = load_raw_dataset(data_path, feature_cols=feature_cols, target_col=target_col)
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # 2. Benchmark Baseline Evaluations (USBM Physics & Tuned XGBoost)
    if not args.skip_baseline:
        logger.info("-" * 40)
        logger.info("Running Empirical Physics Baseline: USBM Attenuation Law")
        logger.info("-" * 40)
        usbm_metrics, _ = run_usbm_baseline(
            X=X,
            y=y,
            sd_index=feature_cols.index("SD") if "SD" in feature_cols else 2,
            n_splits=config.get("cv", {}).get("n_splits", 5),
            random_state=config.get("cv", {}).get("random_state", 42),
        )

        logger.info("-" * 40)
        logger.info("Running Equitable Industrial Baseline: Tuned XGBoost (GridSearchCV)")
        logger.info("-" * 40)
        baseline_metrics, _ = run_xgboost_baseline(
            X=X,
            y=y,
            n_splits=config.get("cv", {}).get("n_splits", 5),
            random_state=config.get("cv", {}).get("random_state", 42),
        )

    # 3. GAT Model Cross-Validation & Grid Search
    device_setting = config.get("training", {}).get("device", "auto")
    if device_setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_setting)
    logger.info(f"Using Compute Device: {device}")

    logger.info("-" * 40)
    logger.info("Running Leak-Free Inductive GAT Cross-Validation & Tuning")
    logger.info("-" * 40)

    results_df, best_bundle = run_cross_validation_grid_search(
        X=X,
        y=y,
        config=config,
        device=device,
    )

    logger.info("=" * 60)
    logger.info("Pipeline Execution Complete!")
    if best_bundle:
        best_cfg = best_bundle["config"]
        logger.info(f"Optimal Hyperparameters: {best_cfg}")
        logger.info(
            f"Best GAT Validation Performance -> "
            f"R2: {best_cfg['mean_r2']:.4f} | "
            f"MAE: {best_cfg['mean_mae']:.4f} | "
            f"RMSE: {best_cfg['mean_rmse']:.4f}"
        )

        # 4. Production Inference Engine Verification
        best_model_path = config.get("artifacts", {}).get("best_model_file", "artifacts/models/best_gat_model.pt")
        if not os.path.isabs(best_model_path):
            best_model_path = os.path.join(SCRIPT_DIR, best_model_path)

        logger.info("-" * 40)
        logger.info("Verifying Standalone Production Inference Service (BlastVibePredictor)")
        logger.info("-" * 40)
        predictor = BlastVibePredictor(bundle_path=best_model_path, device="cpu")
        sample_dis = 400.0
        sample_charge = 1850.0
        sample_pred = predictor.predict_single(dis=sample_dis, charge=sample_charge)
        logger.info(
            f"Inference Test -> Input: [DIS={sample_dis}m, CHARGE={sample_charge}kg] | "
            f"Predicted PPV: {sample_pred:.2f} mm/s"
        )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
