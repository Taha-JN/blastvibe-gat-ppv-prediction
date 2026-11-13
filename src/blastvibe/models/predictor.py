import os
from typing import Union, List, Optional
import numpy as np
import pandas as pd
import torch

from blastvibe.models.gat import GATRegressor
from blastvibe.data.graph import build_inductive_graph
from blastvibe.utils.logger import setup_logger

logger = setup_logger("blastvibe.models.predictor")


class BlastVibePredictor:
    """
    Self-contained production inference engine for BlastVibe-GNN.
    
    Loads a serialized checkpoint bundle containing model weights, hyperparameters,
    scalers, and the reference training graph required for leak-free inductive inference.
    """
    def __init__(self, bundle_path: str, device: str = "auto"):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Model bundle not found at: {bundle_path}")

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading BlastVibe model bundle from: {bundle_path} to device: {self.device}")
        try:
            checkpoint = torch.load(bundle_path, map_location=self.device, weights_only=False)
        except TypeError:
            # Fallback for earlier PyTorch versions where weights_only argument is not present
            checkpoint = torch.load(bundle_path, map_location=self.device)

        # Validate required bundle components
        required_keys = [
            "model_state_dict",
            "hyperparameters",
            "scaler_x",
            "scaler_y",
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
        self.scaler_x = checkpoint["scaler_x"]
        self.scaler_y = checkpoint["scaler_y"]

        # Reference training topology for inductive message passing
        self.x_train_ref = checkpoint["x_train_ref"].to(device=self.device, dtype=torch.float32)
        self.train_edge_index = checkpoint["train_edge_index"].to(device=self.device, dtype=torch.long)
        self.k_neighbors = int(self.hyperparameters.get("k_neighbors", 8))

        # Instantiate model architecture
        in_dim = self.x_train_ref.size(1)
        hidden_dim = int(self.hyperparameters.get("hidden_dim", 64))
        heads = int(self.hyperparameters.get("heads", 4))

        self.model = GATRegressor(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=0.0,  # Zero dropout for deterministic production inference
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info(
            f"Successfully initialized BlastVibePredictor (k={self.k_neighbors}, "
            f"hidden_dim={hidden_dim}, heads={heads}, ref_nodes={self.x_train_ref.size(0)})"
        )

    def predict(self, raw_features: Union[np.ndarray, pd.DataFrame, List]) -> np.ndarray:
        """
        Executes inductive GNN inference on raw input features.
        
        Args:
            raw_features: Features corresponding to [DIS, CHARGE, SD].
                          Accepts 2D array of shape [N, 3] or 1D array [3].
                          
        Returns:
            np.ndarray of predicted Peak Particle Velocity (PPV) values [N].
        """
        if isinstance(raw_features, (list, tuple)):
            features = np.asarray(raw_features, dtype=np.float32)
        elif isinstance(raw_features, pd.DataFrame):
            features = raw_features.values.astype(np.float32)
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

        # Scale features strictly using reference training statistics
        scaled_features = self.scaler_x.transform(features)
        x_query = torch.tensor(scaled_features, dtype=torch.float32, device=self.device)

        # Inductively connect query nodes to training reference graph
        with torch.no_grad():
            x_combined, eval_edge_index, query_mask = build_inductive_graph(
                x_train=self.x_train_ref,
                x_query=x_query,
                train_edge_index=self.train_edge_index,
                k=self.k_neighbors,
            )

            # Forward pass through multi-head GAT
            out = self.model(x_combined, eval_edge_index)
            pred_scaled = out[query_mask].cpu().numpy().reshape(-1, 1)

            # Inverse transform back to physical engineering units (mm/s)
            pred_raw = self.scaler_y.inverse_transform(pred_scaled).reshape(-1)

        return pred_raw

    def predict_single(self, dis: float, charge: float, sd: Optional[float] = None) -> float:
        """
        Convenience method to predict PPV for a single blast shot.
        Calculates USBM Scaled Distance automatically if omitted.
        """
        if charge <= 0:
            raise ValueError("Explosive charge must be strictly positive.")
        if dis < 0:
            raise ValueError("Distance must be non-negative.")
        if sd is None:
            sd = dis / np.sqrt(charge)

        raw = np.array([[dis, charge, sd]], dtype=np.float32)
        pred = self.predict(raw)
        return float(pred[0])
