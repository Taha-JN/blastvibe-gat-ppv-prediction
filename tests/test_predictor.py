import pytest
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from blastvibe.models.gat import GATRegressor
from blastvibe.data.graph import build_train_knn_graph
from blastvibe.models.predictor import BlastVibePredictor


def test_predictor_mock_roundtrip(tmp_path):
    torch.manual_seed(42)
    n_nodes = 30
    in_dim = 3
    hidden_dim = 16
    heads = 2
    k_neighbors = 4

    # Create dummy reference tensors
    x_train_raw = np.random.randn(n_nodes, in_dim).astype(np.float32)
    y_train_raw = np.random.uniform(10.0, 100.0, size=(n_nodes, 1)).astype(np.float32)

    scaler_x = StandardScaler().fit(x_train_raw)
    scaler_y = StandardScaler().fit(y_train_raw)

    x_train_ref = torch.tensor(scaler_x.transform(x_train_raw), dtype=torch.float32)
    train_edge_index = build_train_knn_graph(x_train_ref, k=k_neighbors, loop=False)

    model = GATRegressor(in_dim=in_dim, hidden_dim=hidden_dim, heads=heads, dropout=0.0)
    model.eval()

    bundle_path = tmp_path / "mock_gat_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hyperparameters": {
                "k_neighbors": k_neighbors,
                "hidden_dim": hidden_dim,
                "heads": heads,
            },
            "scaler_x": scaler_x,
            "scaler_y": scaler_y,
            "x_train_ref": x_train_ref,
            "train_edge_index": train_edge_index,
        },
        str(bundle_path),
    )

    # Instantiate predictor from serialized bundle
    predictor = BlastVibePredictor(bundle_path=str(bundle_path), device="cpu")

    # 1. Test batch inference
    query_batch = np.array([
        [150.0, 50.0, 21.21],
        [300.0, 100.0, 30.0],
    ], dtype=np.float32)

    preds = predictor.predict(query_batch)
    assert isinstance(preds, np.ndarray), "Predictions must be a numpy ndarray"
    assert preds.shape == (2,), f"Expected shape (2,), got {preds.shape}"
    assert np.all(np.isfinite(preds)), "Predictions must be finite (no NaNs or Infs)"

    # 2. Test single-shot prediction
    single_pred = predictor.predict_single(dis=250.0, charge=75.0)
    assert isinstance(single_pred, float), "predict_single must return a float"
    assert np.isfinite(single_pred), "predict_single output must be finite"


def test_predictor_missing_keys_fails(tmp_path):
    incomplete_bundle = tmp_path / "incomplete.pt"
    torch.save({"model_state_dict": {}}, str(incomplete_bundle))

    with pytest.raises(KeyError, match="missing required production components"):
        BlastVibePredictor(str(incomplete_bundle))


def test_predictor_invalid_feature_dimension(tmp_path):
    torch.manual_seed(42)
    x_train_ref = torch.randn(20, 3)
    train_edge_index = build_train_knn_graph(x_train_ref, k=3, loop=False)
    model = GATRegressor(in_dim=3, hidden_dim=8, heads=1, dropout=0.0)

    scaler_x = StandardScaler().fit(x_train_ref.numpy())
    scaler_y = StandardScaler().fit(np.arange(20).reshape(-1, 1))

    bundle_path = tmp_path / "bundle.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hyperparameters": {"k_neighbors": 3, "hidden_dim": 8, "heads": 1},
            "scaler_x": scaler_x,
            "scaler_y": scaler_y,
            "x_train_ref": x_train_ref,
            "train_edge_index": train_edge_index,
        },
        str(bundle_path),
    )

    predictor = BlastVibePredictor(str(bundle_path), device="cpu")

    # Pass 4 features instead of 3
    invalid_batch = np.array([[10.0, 20.0, 30.0, 40.0]])
    with pytest.raises(ValueError):
        predictor.predict(invalid_batch)
