import os
import itertools
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from blastvibe.data.loader import prepare_fold_data, ProcessedFoldData
from blastvibe.training.engine import TopologyCache, train_single_fold
from blastvibe.training.artifacts import fit_and_save_production_model, save_metrics_log
from blastvibe.utils.logger import setup_logger

logger = setup_logger("blastvibe.training.grid")


def build_cv_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: Optional[int] = 42,
) -> List[ProcessedFoldData]:
    """Prepares and scales all K-Fold cross-validation splits with zero leakage."""
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
    folds_data: List[ProcessedFoldData] = []
    for train_idx, val_idx in kf.split(X):
        fold_data = prepare_fold_data(
            X_train_raw=X[train_idx],
            X_val_raw=X[val_idx],
            y_train_raw=y[train_idx],
            y_val_raw=y[val_idx],
            train_indices=train_idx,
            val_indices=val_idx,
        )
        folds_data.append(fold_data)
    return folds_data


def evaluate_hyperparameter_grid(
    combos: List[Tuple],
    folds_data: List[ProcessedFoldData],
    topology_cache: TopologyCache,
    epochs: int,
    patience: int,
    eval_interval: int,
    device: torch.device,
    total_samples: int,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]], Optional[np.ndarray], List[int]]:
    """Evaluates all hyperparameter combinations across CV folds using cached graph topologies."""
    results_table: List[Dict[str, Any]] = []
    best_mae = float("inf")
    best_combo: Optional[Dict[str, Any]] = None
    best_oof_preds: Optional[np.ndarray] = None
    best_fold_epochs: List[int] = []

    for idx, (k_val, h_dim, heads, drop, lr, wd) in enumerate(combos, 1):
        fold_metrics_list = []
        oof_preds = np.zeros(total_samples, dtype=float)

        for fold_idx, fold_data in enumerate(folds_data):
            cached_topo = topology_cache.get(k=k_val, fold_idx=fold_idx)

            metrics, fold_preds, _ = train_single_fold(
                fold_idx=fold_idx,
                fold_data=fold_data,
                cached_topo=cached_topo,
                hidden_dim=h_dim,
                heads=heads,
                dropout=drop,
                lr=lr,
                weight_decay=wd,
                epochs=epochs,
                patience=patience,
                eval_interval=eval_interval,
                device=device,
            )
            oof_preds[fold_data.val_indices] = fold_preds
            fold_metrics_list.append(metrics)

        mean_r2 = float(np.mean([m["r2"] for m in fold_metrics_list]))
        mean_mae = float(np.mean([m["mae"] for m in fold_metrics_list]))
        mean_mse = float(np.mean([m["mse"] for m in fold_metrics_list]))
        mean_rmse = float(np.mean([m["rmse"] for m in fold_metrics_list]))
        fold_best_epochs = [int(m.get("best_epoch", epochs)) for m in fold_metrics_list]

        record = {
            "k_neighbors": k_val,
            "hidden_dim": h_dim,
            "heads": heads,
            "dropout": drop,
            "lr": lr,
            "weight_decay": wd,
            "mean_r2": mean_r2,
            "mean_mae": mean_mae,
            "mean_mse": mean_mse,
            "mean_rmse": mean_rmse,
            "avg_best_epoch": int(np.mean(fold_best_epochs)),
        }
        results_table.append(record)

        logger.info(
            f"Combo {idx}/{len(combos)} [k={k_val}, h={h_dim}, lr={lr}, wd={wd}] -> "
            f"Val R2: {mean_r2:.4f} | Val MAE: {mean_mae:.4f} | Val RMSE: {mean_rmse:.4f}"
        )

        if mean_mae < best_mae:
            best_mae = mean_mae
            best_combo = record
            best_oof_preds = oof_preds.copy()
            best_fold_epochs = fold_best_epochs

    results_df = pd.DataFrame(results_table)
    return results_df, best_combo, best_oof_preds, best_fold_epochs


def run_cross_validation_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    config: Union[Dict[str, Any], Any],
    device: torch.device,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Coordinates K-Fold Cross-Validation across a hyperparameter grid using topology caching,
    saves the results log, and serializes the optimal production model bundle to disk.
    """
    # Support both Pydantic PipelineConfig and standard dictionary
    cfg_dict = config.to_dict() if hasattr(config, "to_dict") else (config.model_dump() if hasattr(config, "model_dump") else config)

    grid_cfg = cfg_dict.get("grid_search", {})
    cv_cfg = cfg_dict.get("cv", {})
    train_cfg = cfg_dict.get("training", {})
    art_cfg = cfg_dict.get("artifacts", {})
    data_cfg = cfg_dict.get("data", {})

    k_neighbors_list = grid_cfg.get("k_neighbors", [8])
    hidden_dim_list = grid_cfg.get("hidden_dim", [64])
    lr_list = grid_cfg.get("lr", [0.001])
    weight_decay_list = grid_cfg.get("weight_decay", [0.0])
    heads_list = grid_cfg.get("heads", [4])
    dropout_list = grid_cfg.get("dropout", [0.2])

    n_splits = cv_cfg.get("n_splits", 5)
    shuffle = cv_cfg.get("shuffle", True)
    seed = cv_cfg.get("random_state", 42)

    epochs = train_cfg.get("epochs", 300)
    patience = train_cfg.get("patience", 30)
    eval_interval = train_cfg.get("eval_interval", 10)

    output_dir = art_cfg.get("output_dir", "artifacts")
    model_dir = art_cfg.get("model_dir", "artifacts/models")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    feature_names = data_cfg.get("feature_cols", ["DIS", "CHARGE", "SD"])
    target_name = data_cfg.get("target_col", "SUM")

    # 1. Prepare and scale all CV folds
    folds_data = build_cv_folds(X, y, n_splits=n_splits, shuffle=shuffle, seed=seed)

    # 2. Precompute and cache invariant graph topologies across (k, fold)
    topology_cache = TopologyCache(
        folds_data=folds_data,
        k_neighbors_list=k_neighbors_list,
        device=device,
    )

    # 3. Generate hyperparameter combinations
    combos = list(itertools.product(
        k_neighbors_list,
        hidden_dim_list,
        heads_list,
        dropout_list,
        lr_list,
        weight_decay_list,
    ))

    logger.info(f"Initiating Grid Search: {len(combos)} combinations across {n_splits} CV folds.")

    # 4. Evaluate grid with cached graph topologies
    results_df, best_combo, best_oof_preds, best_fold_epochs = evaluate_hyperparameter_grid(
        combos=combos,
        folds_data=folds_data,
        topology_cache=topology_cache,
        epochs=epochs,
        patience=patience,
        eval_interval=eval_interval,
        device=device,
        total_samples=len(X),
    )

    metrics_path = os.path.join(output_dir, "grid_search_results.csv")
    save_metrics_log(results_df, metrics_path)

    # 5. Fit production model on 100% of training data using CV-optimal mean epochs
    best_bundle = None
    if best_combo is not None:
        avg_best_epoch = max(1, int(np.mean(best_fold_epochs)))
        best_bundle = fit_and_save_production_model(
            X=X,
            y=y,
            best_combo=best_combo,
            avg_best_epoch=avg_best_epoch,
            feature_names=feature_names,
            target_name=target_name,
            output_dir=output_dir,
            model_dir=model_dir,
            device=device,
            best_oof_preds=best_oof_preds,
        )

    return results_df, best_bundle
