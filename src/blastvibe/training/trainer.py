import copy
import os
import json
import itertools
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from blastvibe.data.loader import prepare_fold_data, ProcessedFoldData
from blastvibe.data.graph import build_train_knn_graph, build_inductive_graph
from blastvibe.models.gat import GATRegressor
from blastvibe.utils.logger import setup_logger

logger = setup_logger("blastvibe.training.trainer")


def train_single_fold(
    fold_idx: int,
    fold_data: ProcessedFoldData,
    k_neighbors: int,
    hidden_dim: int,
    heads: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    eval_interval: int,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Any]]:
    """
    Trains a GATRegressor on a single fold with strictly inductive, leak-free evaluation.
    
    Validates on the OUT-OF-FOLD validation set and saves the checkpoint with lowest
    VALIDATION error (eliminating the legacy bug where best_state was picked on train error).
    """
    x_train_t = torch.tensor(fold_data.X_train_scaled, dtype=torch.float, device=device)
    y_train_t = torch.tensor(fold_data.y_train_scaled, dtype=torch.float, device=device)
    x_val_t = torch.tensor(fold_data.X_val_scaled, dtype=torch.float, device=device)

    # 1. Build train graph strictly on X_train (No test nodes included)
    train_edge_index = build_train_knn_graph(x_train_t, k=k_neighbors, loop=False)

    # 2. Build leak-free inductive graph for out-of-sample validation
    x_eval, eval_edge_index, val_mask = build_inductive_graph(
        x_train=x_train_t,
        x_query=x_val_t,
        train_edge_index=train_edge_index,
        k=k_neighbors,
    )

    model = GATRegressor(
        in_dim=x_train_t.size(1),
        hidden_dim=hidden_dim,
        heads=heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out_train = model(x_train_t, train_edge_index)
        loss = criterion(out_train, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                out_eval = model(x_eval, eval_edge_index)
                pred_val_scaled = out_eval[val_mask].cpu().numpy().reshape(-1, 1)
                pred_val_raw = fold_data.scaler_y.inverse_transform(pred_val_scaled).reshape(-1)

                val_mae = mean_absolute_error(fold_data.y_val_raw, pred_val_raw)

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_state = copy.deepcopy(model.state_dict())
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
        final_preds = fold_data.scaler_y.inverse_transform(pred_val_scaled).reshape(-1)

    r2 = float(r2_score(fold_data.y_val_raw, final_preds))
    mae = float(mean_absolute_error(fold_data.y_val_raw, final_preds))
    mse = float(mean_squared_error(fold_data.y_val_raw, final_preds))
    rmse = float(np.sqrt(mse))

    metrics = {"r2": r2, "mae": mae, "mse": mse, "rmse": rmse}
    return metrics, final_preds, best_state


def run_cross_validation_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes K-Fold Cross-Validation across a hyperparameter grid.
    
    Saves results log and serializes the best model bundle to disk.
    """
    grid_cfg = config.get("grid_search", {})
    cv_cfg = config.get("cv", {})
    train_cfg = config.get("training", {})
    art_cfg = config.get("artifacts", {})

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

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    combos = list(itertools.product(
        k_neighbors_list,
        hidden_dim_list,
        heads_list,
        dropout_list,
        lr_list,
        weight_decay_list,
    ))

    logger.info(f"Initiating Grid Search: {len(combos)} combinations across {n_splits} CV folds.")

    results_table: List[Dict[str, Any]] = []
    best_mae = float("inf")
    best_combo: Optional[Dict[str, Any]] = None
    best_oof_preds: Optional[np.ndarray] = None
    best_bundle: Optional[Dict[str, Any]] = None

    for idx, (k_val, h_dim, heads, drop, lr, wd) in enumerate(combos, 1):
        fold_metrics_list = []
        oof_preds = np.zeros_like(y, dtype=float)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            fold_data = prepare_fold_data(
                X_train_raw=X[train_idx],
                X_val_raw=X[val_idx],
                y_train_raw=y[train_idx],
                y_val_raw=y[val_idx],
            )

            metrics, fold_preds, state = train_single_fold(
                fold_idx=fold,
                fold_data=fold_data,
                k_neighbors=k_val,
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
            oof_preds[val_idx] = fold_preds
            fold_metrics_list.append(metrics)

        mean_r2 = float(np.mean([m["r2"] for m in fold_metrics_list]))
        mean_mae = float(np.mean([m["mae"] for m in fold_metrics_list]))
        mean_mse = float(np.mean([m["mse"] for m in fold_metrics_list]))
        mean_rmse = float(np.mean([m["rmse"] for m in fold_metrics_list]))

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
        }
        results_table.append(record)

        logger.info(
            f"Combo {idx}/{len(combos)} [k={k_val}, h={h_dim}, lr={lr}, wd={wd}] -> "
            f"Val R2: {mean_r2:.4f} | Val MAE: {mean_mae:.4f} | Val RMSE: {mean_rmse:.4f}"
        )

        if mean_mae < best_mae:
            best_mae = mean_mae
            best_combo = record
            best_oof_preds = oof_preds

    results_df = pd.DataFrame(results_table)
    metrics_path = os.path.join(output_dir, "grid_search_results.csv")
    results_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved complete grid search metric log to: {metrics_path}")

    # Train Final Production Model with 90/10 Train/Holdout Early Stopping & Persist Serving Artifacts
    best_model_path = os.path.join(model_dir, "best_gat_model.pt")
    if best_combo is not None:
        logger.info("-" * 40)
        logger.info(
            f"Fitting Production Model with 90/10 Train/Holdout Split & Early Stopping ({len(X)} total records) "
            f"using optimal hyperparameters: {best_combo}"
        )
        logger.info("-" * 40)

        # Phase 3: 90/10 Train/Holdout split to prevent production overfitting
        X_prod_train, X_prod_val, y_prod_train, y_prod_val = train_test_split(
            X, y, test_size=0.10, random_state=seed, shuffle=shuffle
        )

        prod_fold_data = prepare_fold_data(
            X_train_raw=X_prod_train,
            X_val_raw=X_prod_val,
            y_train_raw=y_prod_train,
            y_val_raw=y_prod_val,
        )

        x_train_ref = torch.tensor(prod_fold_data.X_train_scaled, dtype=torch.float32, device=device)
        y_train_prod = torch.tensor(prod_fold_data.y_train_scaled, dtype=torch.float32, device=device)
        x_val_prod = torch.tensor(prod_fold_data.X_val_scaled, dtype=torch.float32, device=device)

        full_train_edge_index = build_train_knn_graph(
            x_train_ref,
            k=best_combo["k_neighbors"],
            loop=False,
        )

        # Build inductive graph for early-stopping validation on production holdout
        x_eval, eval_edge_index, val_mask = build_inductive_graph(
            x_train=x_train_ref,
            x_query=x_val_prod,
            train_edge_index=full_train_edge_index,
            k=best_combo["k_neighbors"],
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

        best_prod_mae = float("inf")
        best_prod_state = None
        epochs_no_improve = 0

        # Phase 3: Early stopping on the holdout validation split
        for epoch in range(1, epochs + 1):
            final_model.train()
            optimizer.zero_grad()
            out = final_model(x_train_ref, full_train_edge_index)
            loss = criterion(out, y_train_prod)
            loss.backward()
            optimizer.step()

            if epoch % eval_interval == 0:
                final_model.eval()
                with torch.no_grad():
                    out_eval = final_model(x_eval, eval_edge_index)
                    pred_val_scaled = out_eval[val_mask].cpu().numpy().reshape(-1, 1)
                    pred_val_raw = prod_fold_data.scaler_y.inverse_transform(pred_val_scaled).reshape(-1)

                    val_mae = mean_absolute_error(prod_fold_data.y_val_raw, pred_val_raw)

                    if val_mae < best_prod_mae:
                        best_prod_mae = val_mae
                        best_prod_state = copy.deepcopy(final_model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += eval_interval

                    if patience > 0 and epochs_no_improve >= patience:
                        logger.info(
                            f"Production Early Stopping triggered at epoch {epoch} "
                            f"(Best Holdout Val MAE: {best_prod_mae:.4f})"
                        )
                        break

        # Restore optimal checkpoint
        if best_prod_state is not None:
            final_model.load_state_dict(best_prod_state)
            logger.info(f"Loaded best production model weights with holdout MAE: {best_prod_mae:.4f}")

        # Phase 2: Decouple PyTorch weights from Scikit-Learn objects (CWE-502 remediation)
        scaler_params = {
            "scaler_x": {
                "mean": prod_fold_data.scaler_x.mean_.tolist(),
                "scale": prod_fold_data.scaler_x.scale_.tolist(),
                "var": prod_fold_data.scaler_x.var_.tolist(),
            },
            "scaler_y": {
                "mean": prod_fold_data.scaler_y.mean_.tolist(),
                "scale": prod_fold_data.scaler_y.scale_.tolist(),
                "var": prod_fold_data.scaler_y.var_.tolist(),
            },
        }
        scaler_params_path = os.path.join(model_dir, "scaler_params.json")
        with open(scaler_params_path, "w", encoding="utf-8") as f:
            json.dump(scaler_params, f, indent=2)
        logger.info(f"Saved decoupled scaler parameters to: {scaler_params_path}")

        best_bundle = {
            "config": best_combo,
            "model_state_dict": final_model.state_dict(),
            "scaler_x": prod_fold_data.scaler_x,
            "scaler_y": prod_fold_data.scaler_y,
            "x_train_ref": x_train_ref.detach().cpu(),
            "train_edge_index": full_train_edge_index.detach().cpu(),
            "oof_predictions": best_oof_preds,
        }

        # Save secure PyTorch model bundle (contains ONLY primitives & tensors - safe for weights_only=True)
        torch.save(
            {
                "model_state_dict": best_bundle["model_state_dict"],
                "hyperparameters": best_bundle["config"],
                "x_train_ref": best_bundle["x_train_ref"],
                "train_edge_index": best_bundle["train_edge_index"],
            },
            best_model_path,
        )
        logger.info(
            f"Successfully serialized secure production GAT model bundle with reference topology "
            f"({x_train_ref.size(0)} nodes, {full_train_edge_index.size(1)} edges) to: {best_model_path}"
        )

    return results_df, best_bundle
