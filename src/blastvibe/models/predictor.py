import os
import json
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import k_hop_subgraph

from blastvibe.models.gat import GATRegressor
from blastvibe.data.graph import build_inductive_graph
from blastvibe.data.loader import compute_scaled_distance
from blastvibe.utils.logger import setup_logger

logger = setup_logger("blastvibe.models.predictor")


class BlastVibePredictor:
    """
    Self-contained production inference engine for BlastVibe-GNN.
    
    Loads a serialized checkpoint bundle containing model weights, hyperparameters,
    and reference graph topology using secure weights_only=True deserialization (CWE-502 compliant),
    with decoupled feature normalization parameters loaded from JSON.
    """
    def __init__(
        self,
        bundle_path: Union[str, Path],
        scaler_params_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
    ):
        bundle_path = str(bundle_path)
        if scaler_params_path is not None:
            scaler_params_path = str(scaler_params_path)

        if not os.path.exists(bundle_path):
            raise RuntimeError(
                f"Model checkpoint not found at '{bundle_path}'. "
                "You must first run 'python main.py' to execute training and generate the production checkpoint."
            )

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading BlastVibe model bundle securely from: {bundle_path} to device: {self.device}")
        
        # CWE-502 Remediated: Secure weights_only=True loading prevents arbitrary code execution
        try:
            checkpoint = torch.load(bundle_path, map_location=self.device, weights_only=True)
        except FileNotFoundError as err:
            raise RuntimeError(
                f"Model checkpoint not found at '{bundle_path}'. "
                "You must first run 'python main.py' to execute training and generate the production checkpoint."
            ) from err

        # Validate required bundle components
        required_keys = [
            "model_state_dict",
            "hyperparameters",
            "x_train_ref",
            "train_edge_index",
        ]
        missing = [k for k in required_keys if k not in checkpoint]
        if missing:
            raise KeyError(
                f"Checkpoint is missing required production components: {missing}. "
                "Ensure model was saved using the updated production trainer."
            )

        self.hyperparameters = checkpoint["hyperparameters"]
        self.feature_names = list(checkpoint.get("feature_names", ["DIS", "CHARGE", "SD"]))
        self.target_name = str(checkpoint.get("target_name", "SUM"))

        # Reference training topology for inductive message passing
        self.x_train_ref = checkpoint["x_train_ref"].to(device=self.device, dtype=torch.float32)
        self.train_edge_index = checkpoint["train_edge_index"].to(device=self.device, dtype=torch.long)
        self.k_neighbors = int(self.hyperparameters.get("k_neighbors", 8))

        # Load decoupled feature normalization parameters from JSON
        if scaler_params_path is None:
            scaler_params_path = os.path.join(os.path.dirname(bundle_path), "scaler_params.json")

        if not os.path.exists(scaler_params_path):
            raise FileNotFoundError(
                f"Decoupled scaler parameters JSON not found at: {scaler_params_path}. "
                "Expected scaler_params.json alongside model bundle."
            )

        with open(scaler_params_path, "r", encoding="utf-8") as f:
            scaler_dict = json.load(f)

        self.x_mean = np.array(scaler_dict["scaler_x"]["mean"], dtype=np.float32)
        self.x_scale = np.array(scaler_dict["scaler_x"]["scale"], dtype=np.float32)
        self.y_mean = float(scaler_dict["scaler_y"]["mean"][0])
        self.y_scale = float(scaler_dict["scaler_y"]["scale"][0])

        # Instantiate model architecture
        ref_samples_count = self.x_train_ref.size(0)
        in_dim = self.x_train_ref.size(1)
        hidden_dim = int(self.hyperparameters.get("hidden_dim", 64))
        out_hidden_dim = self.hyperparameters.get("out_hidden_dim", None)
        heads = int(self.hyperparameters.get("heads", 4))

        self.model = GATRegressor(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_hidden_dim=out_hidden_dim,
            heads=heads,
            dropout=0.0,  # Zero dropout for deterministic production inference
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info(
            f"Successfully initialized BlastVibePredictor (k={self.k_neighbors}, "
            f"hidden_dim={hidden_dim}, heads={heads}, ref_nodes={ref_samples_count})"
        )

    def predict(self, raw_features: Union[np.ndarray, pd.DataFrame, List]) -> np.ndarray:
        """
        Executes inductive GNN inference on raw input features.
        
        Production Scaling Note:
        For massive reference datasets (e.g. N > 50,000), performing message passing over the entire
        historical graph creates an O(N) serving bottleneck. In high-throughput production environments,
        real-time inference should utilize a parametric student surrogate model (e.g., distilled MLP or
        calibrated gradient booster) or a localized KD-tree prototype cache that restricts message
        passing to the top-k nearest historical neighbors instead of the entire reference topology.
        
        Args:
            raw_features: Features corresponding to [DIS, CHARGE, SD].
                          Accepts 2D array of shape [N, 3] or 1D array [3].
                          
        Returns:
            np.ndarray of predicted Peak Particle Velocity (PPV) values [N].
        """
        if isinstance(raw_features, (list, tuple)):
            features = np.asarray(raw_features, dtype=np.float32)
        elif isinstance(raw_features, pd.DataFrame):
            if list(raw_features.columns) != self.feature_names:
                raise ValueError(
                    f"Feature schema mismatch! Expected columns {self.feature_names}, got {list(raw_features.columns)}"
                )
            features = raw_features[self.feature_names].values.astype(np.float32)
        elif isinstance(raw_features, np.ndarray):
            features = raw_features.astype(np.float32)
        else:
            raise TypeError(f"Unsupported features type: {type(raw_features)}")

        # Handle single-sample 1D input
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.ndim != 2 or features.shape[1] != self.x_train_ref.size(1):
            raise ValueError(
                f"Expected features shape [N, {self.x_train_ref.size(1)}], got {features.shape}"
            )

        # Scale features strictly using reference training statistics (vectorized)
        scaled_features = (features - self.x_mean) / self.x_scale
        x_query = torch.tensor(scaled_features, dtype=torch.float32, device=self.device)

        # Inductively connect query nodes to training reference graph
        with torch.no_grad():
            x_combined, eval_edge_index, query_mask = build_inductive_graph(
                x_train=self.x_train_ref,
                x_query=x_query,
                train_edge_index=self.train_edge_index,
                k=self.k_neighbors,
            )

            # Localize inference graph via 2-hop subgraph extraction around query nodes
            # Eliminates O(N_train) forward-pass memory and compute scaling
            n_train = self.x_train_ref.size(0)
            n_query = x_query.size(0)
            query_node_indices = torch.arange(n_train, n_train + n_query, device=self.device)

            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                node_idx=query_node_indices,
                num_hops=2,
                edge_index=eval_edge_index,
                relabel_nodes=True,
                flow="source_to_target",
            )
            sub_x = x_combined[subset]

            # Forward pass through multi-head GAT strictly on localized 2-hop neighborhood
            out = self.model(sub_x, sub_edge_index)
            pred_scaled = out[mapping].cpu().numpy().reshape(-1)

            # Inverse transform back to physical engineering units (mm/s) purely mathematically
            pred_raw = (pred_scaled * self.y_scale) + self.y_mean
            pred_raw = np.maximum(0.0, pred_raw)

        return pred_raw

    def predict_single(self, dis: float, charge: float, sd: Optional[float] = None) -> float:
        """
        Convenience method to predict PPV for a single blast shot.
        Calculates USBM Scaled Distance automatically if omitted.
        """
        if sd is None:
            sd = compute_scaled_distance(dis, charge)
        else:
            if charge <= 0:
                raise ValueError("Explosive charge must be strictly positive.")
            if dis < 0:
                raise ValueError("Distance must be non-negative.")

        raw = np.array([[dis, charge, sd]], dtype=np.float32)
        pred = self.predict(raw)
        return float(pred[0])
