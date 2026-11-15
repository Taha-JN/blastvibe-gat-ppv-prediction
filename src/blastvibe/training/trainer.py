import copy
import os
import json
import itertools
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from blastvibe.data.loader import prepare_fold_data, ProcessedFoldData
from blastvibe.data.graph import build_train_knn_graph, build_inductive_graph
from blastvibe.models.gat import GATRegressor
from blastvibe.utils.logger import setup_logger

logger = setup_logger("blastvibe.training.trainer")


@dataclass
class CachedFoldTopology:
    """Stores precomputed tensors and edge indices for a specific (k, fold) pair."""
    train_edge_index: torch.Tensor
    x_eval: torch.Tensor
    eval_edge_index: torch.Tensor
    val_mask: torch.Tensor
    x_train_t: torch.Tensor
    y_train_t: torch.Tensor


class TopologyCache:
    """
    Precomputes and caches inductive KNN graph topologies across (k, fold) pairs.
    
    Eliminates redundant spatial distance calculations and graph constructions
    across hyperparameter grid iterations (such as hidden_dim, lr, heads).
    """
    def __init__(
        self,
        folds_data: List[ProcessedFoldData],
        k_neighbors_list: List[int],
        device: torch.device,
    ):
        self.device = device
        self.cache: Dict[Tuple[int, int], CachedFoldTopology] = {}
        logger.info(
            f"Precomputing TopologyCache for {len(k_neighbors_list)} k-values across "
            f"{len(folds_data)} CV folds on device: {device}..."
        )
        for fold_idx, fold_data in enumerate(folds_data):
            x_train_t = torch.tensor(fold_data.X_train_scaled, dtype=torch.float32, device=device)
            y_train_t = torch.tensor(fold_data.y_train_scaled, dtype=torch.float32, device=device)
            x_val_t = torch.tensor(fold_data.X_val_scaled, dtype=torch.float32, device=device)

            for k in k_neighbors_list:
                train_edge_index = build_train_knn_graph(x_train_t, k=k, loop=False)
                x_eval, eval_edge_index, val_mask = build_inductive_graph(
                    x_train=x_train_t,
                    x_query=x_val_t,
                    train_edge_index=train_edge_index,
                    k=k,
                )
                self.cache[(k, fold_idx)] = CachedFoldTopology(
                    train_edge_index=train_edge_index,
                    x_eval=x_eval,
                    eval_edge_index=eval_edge_index,
                    val_mask=val_mask,
                    x_train_t=x_train_t,
                    y_train_t=y_train_t,
                )
        logger.info(
            f"TopologyCache successfully precomputed and cached {len(self.cache)} unique graph topologies."
        )

    def get(self, k: int, fold_idx: int) -> CachedFoldTopology:
        return self.cache[(k, fold_idx)]


def fit_model_with_early_stopping(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    x_train: torch.Tensor,
    train_edge_index: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    eval_edge_index: torch.Tensor,
    val_mask: torch.Tensor,
    y_val_raw: np.ndarray,
    scaler_y: StandardScaler,
    epochs: int,
    patience: int,
    eval_interval: int,
) -> Tuple[Dict[str, float], np.ndarray, Optional[Dict[str, Any]]]:
    """
    Fits a GNN model with validation monitoring and early stopping.
    """
    best_val_mae = float("inf")
    best_state = None
    best_epoch = epochs
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out_train = model(x_train, train_edge_index)
        loss = criterion(out_train, y_train)
        loss.backward()
        optimizer.step()

        if epoch % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                out_eval = model(x_eval, eval_edge_index)
                pred_val_scaled = out_eval[val_mask].cpu().numpy().reshape(-1, 1)
                pred_val_raw = scaler_y.inverse_transform(pred_val_scaled).reshape(-1)

                val_mae = mean_absolute_error(y_val_raw, pred_val_raw)

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += eval_interval

                if patience > 0 and epochs_no_improve >= patience:
                    break

    # Restore best checkpoint based on validation performance
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on validation set
    model.eval()
    with torch.no_grad():
        out_eval = model(x_eval, eval_edge_index)
        pred_val_scaled = out_eval[val_mask].cpu().numpy().reshape(-1, 1)
        final_preds = scaler_y.inverse_transform(pred_val_scaled).reshape(-1)

    r2 = float(r2_score(y_val_raw, final_preds))
    mae = float(mean_absolute_error(y_val_raw, final_preds))
    mse = float(mean_squared_error(y_val_raw, final_preds))
    rmse = float(np.sqrt(mse))

    metrics = {"r2": r2, "mae": mae, "mse": mse, "rmse": rmse, "best_epoch": best_epoch}
    return metrics, final_preds, best_state


def build_cv_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: Optional[int] = 42,
) -> List[ProcessedFoldData]:
    """Prepares and scales all K-Fold cross-validation splits."""
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


def train_single_fold(
    fold_idx: int,
    fold_data: ProcessedFoldData,
    cached_topo: CachedFoldTopology,
    hidden_dim: int,
    heads: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    eval_interval: int,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, Optional[Dict[str, Any]]]:
    """
    Trains a GATRegressor on a single fold using precomputed cached graph topologies.
    """
    model = GATRegressor(
        in_dim=cached_topo.x_train_t.size(1),
        hidden_dim=hidden_dim,
        heads=heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    return fit_model_with_early_stopping(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        x_train=cached_topo.x_train_t,
        train_edge_index=cached_topo.train_edge_index,
        y_train=cached_topo.y_train_t,
        x_eval=cached_topo.x_eval,
        eval_edge_index=cached_topo.eval_edge_index,
        val_mask=cached_topo.val_mask,
        y_val_raw=fold_data.y_val_raw,
        scaler_y=fold_data.scaler_y,
        epochs=epochs,
        patience=patience,
        eval_interval=eval_interval,
    )


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


def fit_and_save_production_model(
    X: np.ndarray,
    y: np.ndarray,
    best_combo: Dict[str, Any],
    avg_best_epoch: int,
    feature_names: List[str],
    target_name: str,
    output_dir: str,
    model_dir: str,
    device: torch.device,
    best_oof_preds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fits the final production model deterministically on 100% of the training partition
    for avg_best_epoch epochs and persists secure weights, scalers, and schema metadata.
    """
    logger.info("-" * 40)
    logger.info(
        f"Fitting Production Model on 100% of Training Partition ({len(X)} records) "
        f"for {avg_best_epoch} epochs (mean optimal epoch across CV folds) "
        f"using optimal hyperparameters: {best_combo}"
    )
    logger.info("-" * 40)

    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(-1)

    x_train_ref = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    y_train_prod = torch.tensor(y_scaled, dtype=torch.float32, device=device)

    full_train_edge_index = build_train_knn_graph(
        x_train_ref,
        k=best_combo["k_neighbors"],
        loop=False,
    )

    final_model = GATRegressor(
        in_dim=x_train_ref.size(1),
        hidden_dim=best_combo["hidden_dim"],
        heads=best_combo["heads"],
        dropout=best_combo["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=best_combo["lr"],
        weight_decay=best_combo["weight_decay"],
    )
    criterion = nn.MSELoss()

    final_model.train()
    for epoch in range(1, avg_best_epoch + 1):
        optimizer.zero_grad()
        out = final_model(x_train_ref, full_train_edge_index)
        loss = criterion(out, y_train_prod)
        loss.backward()
        optimizer.step()

    final_model.eval()
    with torch.no_grad():
        out_final = final_model(x_train_ref, full_train_edge_index)
        pred_train_scaled = out_final.cpu().numpy().reshape(-1, 1)
        pred_train_raw = scaler_y.inverse_transform(pred_train_scaled).reshape(-1)
        final_train_mae = float(mean_absolute_error(y, pred_train_raw))

    logger.info(f"Fitted production model with training MAE: {final_train_mae:.4f}")

    # Decouple PyTorch weights from Scikit-Learn objects (CWE-502 remediation)
    scaler_params = {
        "scaler_x": {
            "mean": scaler_x.mean_.tolist(),
            "scale": scaler_x.scale_.tolist(),
            "var": scaler_x.var_.tolist(),
        },
        "scaler_y": {
            "mean": scaler_y.mean_.tolist(),
            "scale": scaler_y.scale_.tolist(),
            "var": scaler_y.var_.tolist(),
        },
    }
    scaler_params_path = os.path.join(model_dir, "scaler_params.json")
    with open(scaler_params_path, "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=2)
    logger.info(f"Saved decoupled scaler parameters to: {scaler_params_path}")

    best_bundle = {
        "config": best_combo,
        "model_state_dict": final_model.state_dict(),
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "feature_names": feature_names,
        "target_name": target_name,
        "x_train_ref": x_train_ref.detach().cpu(),
        "train_edge_index": full_train_edge_index.detach().cpu(),
        "oof_predictions": best_oof_preds,
    }

    best_model_path = os.path.join(model_dir, "best_gat_model.pt")
    # Save secure PyTorch model bundle with explicit schema metadata (CWE-502 compliant)
    torch.save(
        {
            "model_state_dict": best_bundle["model_state_dict"],
            "hyperparameters": best_bundle["config"],
            "feature_names": feature_names,
            "target_name": target_name,
            "x_train_ref": best_bundle["x_train_ref"],
            "train_edge_index": best_bundle["train_edge_index"],
        },
        best_model_path,
    )
    logger.info(
        f"Successfully serialized secure production GAT model bundle with reference topology "
        f"({x_train_ref.size(0)} nodes, {full_train_edge_index.size(1)} edges) to: {best_model_path}"
    )

    return best_bundle


def run_cross_validation_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Coordinates K-Fold Cross-Validation across a hyperparameter grid using topology caching,
    saves the results log, and serializes the optimal production model bundle to disk.
    """
    grid_cfg = config.get("grid_search", {})
    cv_cfg = config.get("cv", {})
    train_cfg = config.get("training", {})
    art_cfg = config.get("artifacts", {})
    data_cfg = config.get("data", {})

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
    results_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved complete grid search metric log to: {metrics_path}")

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
