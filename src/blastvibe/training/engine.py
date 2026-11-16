import copy
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from blastvibe.data.loader import ProcessedFoldData
from blastvibe.data.graph import build_train_knn_graph, build_inductive_graph
from blastvibe.models.gat import GATRegressor
from blastvibe.utils.logger import setup_logger

logger = setup_logger("blastvibe.training.engine")


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
