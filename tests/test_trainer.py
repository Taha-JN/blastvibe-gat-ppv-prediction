import os
import pytest
import numpy as np
import torch

from blastvibe.training.trainer import (
    run_cross_validation_grid_search,
    build_cv_folds,
    TopologyCache,
    train_single_fold,
)


def test_build_cv_folds_preserves_indices():
    """Asserts that build_cv_folds explicitly preserves train and val split indices."""
    X = np.random.randn(25, 3).astype(np.float32)
    y = np.random.uniform(10.0, 50.0, size=25).astype(np.float32)

    folds_data = build_cv_folds(X, y, n_splits=5, shuffle=True, seed=42)
    assert len(folds_data) == 5

    all_val_indices = []
    for fold in folds_data:
        assert hasattr(fold, "train_indices")
        assert hasattr(fold, "val_indices")
        assert len(fold.train_indices) == 20
        assert len(fold.val_indices) == 5
        # Intersection between train and val indices must be empty
        assert len(np.intersect1d(fold.train_indices, fold.val_indices)) == 0
        all_val_indices.extend(fold.val_indices.tolist())

    # Union of all val indices across folds must reconstruct all sample indices [0..24]
    assert sorted(all_val_indices) == list(range(25))


def test_evaluate_hyperparameter_grid_populates_oof_preds(tmp_path):
    """
    Core regression test verifying that evaluate_hyperparameter_grid properly
    populates out-of-fold predictions and that best_bundle['oof_predictions']
    does not contain all zeros.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 20
    X = np.random.uniform(10.0, 100.0, size=(n_samples, 3)).astype(np.float32)
    y = np.random.uniform(20.0, 80.0, size=n_samples).astype(np.float32)

    config = {
        "grid_search": {
            "k_neighbors": [3],
            "hidden_dim": [8],
            "heads": [1],
            "dropout": [0.0],
            "lr": [0.01],
            "weight_decay": [0.0],
        },
        "cv": {
            "n_splits": 2,
            "shuffle": True,
            "random_state": 42,
        },
        "training": {
            "epochs": 2,
            "patience": 0,
            "eval_interval": 1,
            "device": "cpu",
        },
        "artifacts": {
            "output_dir": str(tmp_path / "artifacts"),
            "model_dir": str(tmp_path / "artifacts" / "models"),
            "metrics_file": str(tmp_path / "artifacts" / "grid_search_results.csv"),
            "best_model_file": str(tmp_path / "artifacts" / "models" / "best_gat_model.pt"),
        },
        "data": {
            "feature_cols": ["f1", "f2", "f3"],
            "target_col": "target",
        },
    }

    results_df, best_bundle = run_cross_validation_grid_search(
        X=X,
        y=y,
        config=config,
        device=torch.device("cpu"),
    )

    assert results_df is not None
    assert len(results_df) == 1
    assert best_bundle is not None

    oof_preds = best_bundle.get("oof_predictions")
    assert oof_preds is not None, "best_bundle must contain 'oof_predictions' key."
    assert len(oof_preds) == n_samples, f"Expected {n_samples} OOF predictions, got {len(oof_preds)}"

    # Primary mathematical invariant: OOF predictions must NOT be all zeros
    assert not np.all(oof_preds == 0.0), (
        "Silent corruption detected: oof_predictions array contains all zeros!"
    )
    # Furthermore, since all ground-truth targets are > 20, predictions must be positive and finite
    assert np.all(oof_preds > 0.0), "All OOF predictions must be positive non-zero values."
    assert np.all(np.isfinite(oof_preds)), "All OOF predictions must be finite numbers."


def test_topology_cache_and_train_single_fold():
    """Validates that TopologyCache caches and train_single_fold executes properly."""
    torch.manual_seed(42)
    np.random.seed(42)

    X = np.random.randn(15, 3).astype(np.float32)
    y = np.random.uniform(10.0, 30.0, size=15).astype(np.float32)

    folds = build_cv_folds(X, y, n_splits=3, shuffle=False)
    device = torch.device("cpu")
    cache = TopologyCache(folds_data=folds, k_neighbors_list=[3], device=device)

    cached_topo = cache.get(k=3, fold_idx=0)
    assert cached_topo.train_edge_index.size(0) == 2
    assert cached_topo.x_train_t.size(0) == 10
    assert cached_topo.x_eval.size(0) == 15

    metrics, fold_preds, best_state = train_single_fold(
        fold_idx=0,
        fold_data=folds[0],
        cached_topo=cached_topo,
        hidden_dim=8,
        heads=1,
        dropout=0.0,
        lr=0.01,
        weight_decay=0.0,
        epochs=3,
        patience=0,
        eval_interval=1,
        device=device,
    )

    assert "mae" in metrics
    assert "r2" in metrics
    assert len(fold_preds) == 5
    assert not np.all(fold_preds == 0.0)
