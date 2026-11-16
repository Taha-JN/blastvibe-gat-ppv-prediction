import json
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch import nn

from blastvibe.data.graph import build_train_knn_graph
from blastvibe.models.gat import GATRegressor
from blastvibe.utils.logger import setup_logger
from blastvibe.utils.seed import set_seed

logger = setup_logger("blastvibe.training.artifacts")


def save_scaler_parameters(
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    model_dir: str,
) -> str:
    """
    Decouples PyTorch weights from Scikit-Learn pickle objects (CWE-502 remediation)
    by serializing normalization statistics as JSON primitives.
    """
    os.makedirs(model_dir, exist_ok=True)
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
    return scaler_params_path


def save_production_model_bundle(
    model_state_dict: dict[str, Any],
    hyperparameters: dict[str, Any],
    feature_names: list[str],
    target_name: str,
    x_train_ref: torch.Tensor,
    train_edge_index: torch.Tensor,
    model_path: str,
) -> str:
    """
    Serializes safe PyTorch model bundle containing model weights and topology metadata
    compatible with torch.load(..., weights_only=True).
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "hyperparameters": hyperparameters,
            "feature_names": feature_names,
            "target_name": target_name,
            "x_train_ref": x_train_ref.detach().cpu(),
            "train_edge_index": train_edge_index.detach().cpu(),
        },
        model_path,
    )
    logger.info(
        f"Successfully serialized secure production GAT model bundle with reference topology "
        f"({x_train_ref.size(0)} nodes, {train_edge_index.size(1)} edges) to: {model_path}"
    )
    return model_path


def save_metrics_log(results_df: pd.DataFrame, metrics_path: str) -> str:
    """Saves hyperparameter grid search metric log to CSV."""
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    results_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved complete grid search metric log to: {metrics_path}")
    return metrics_path


def fit_and_save_production_model(
    X: np.ndarray,
    y: np.ndarray,
    best_combo: dict[str, Any],
    avg_best_epoch: int,
    feature_names: list[str],
    target_name: str,
    output_dir: str,
    model_dir: str,
    device: torch.device,
    best_model_path: str | None = None,
    seed: int = 42,
    best_oof_preds: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Fits the final production model deterministically on 100% of the training partition
    for avg_best_epoch epochs and persists secure weights, scalers, and schema metadata.
    """
    # Enforce deterministic global seeding for model initialization & fitting
    set_seed(seed)

    logger.info("-" * 40)
    logger.info(
        f"Fitting Production Model on 100% of Training Partition ({len(X)} records) "
        f"for {avg_best_epoch} epochs (mean optimal epoch across CV folds, seed={seed}) "
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

    # Resolve target model artifact path, respecting configuration
    if best_model_path is None:
        best_model_path = os.path.join(model_dir, "best_gat_model.pt")

    scaler_dir = os.path.dirname(best_model_path) or model_dir

    # Decouple PyTorch weights from Scikit-Learn objects (CWE-502 remediation)
    save_scaler_parameters(scaler_x, scaler_y, scaler_dir)

    save_production_model_bundle(
        model_state_dict=final_model.state_dict(),
        hyperparameters=best_combo,
        feature_names=feature_names,
        target_name=target_name,
        x_train_ref=x_train_ref,
        train_edge_index=full_train_edge_index,
        model_path=best_model_path,
    )

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
    return best_bundle
