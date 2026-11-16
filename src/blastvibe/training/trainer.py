"""
BlastVibe-GNN Unified Training Orchestration Facade.

Decoupled into dedicated Single Responsibility modules:
- blastvibe.training.engine: Single-fold model training, early stopping, and topology caching.
- blastvibe.training.grid: K-Fold cross-validation partitioning and hyperparameter grid search.
- blastvibe.training.artifacts: Production model re-fitting, serialization, and metric logging.
"""
from blastvibe.training.engine import (
    CachedFoldTopology,
    TopologyCache,
    fit_model_with_early_stopping,
    train_single_fold,
)
from blastvibe.training.grid import (
    build_cv_folds,
    evaluate_hyperparameter_grid,
    run_cross_validation_grid_search,
)
from blastvibe.training.artifacts import (
    save_scaler_parameters,
    save_production_model_bundle,
    save_metrics_log,
    fit_and_save_production_model,
)

__all__ = [
    "CachedFoldTopology",
    "TopologyCache",
    "fit_model_with_early_stopping",
    "train_single_fold",
    "build_cv_folds",
    "evaluate_hyperparameter_grid",
    "run_cross_validation_grid_search",
    "save_scaler_parameters",
    "save_production_model_bundle",
    "save_metrics_log",
    "fit_and_save_production_model",
]
