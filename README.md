# BlastVibe-GNN: Production Graph Attention Network for Ground Vibration Prediction

A production-grade Scientific Machine Learning (SciML) and Graph Neural Network (GNN) framework implemented in PyTorch Geometric to predict Peak Particle Velocity (PPV) resulting from blasting operations in mining and civil excavation.

---

## 🚀 Key Highlights & Scientific Innovations

- **Domain Physics Feature Engineering**: Integrates empirical blast physics (USBM formulation) by computing **Scaled Distance** ($SD = \frac{Distance}{\sqrt{Charge}}$), augmenting basic geometric features to capture non-linear energy dissipation dynamics.
- **Leak-Free Inductive Graph Construction**: Builds spatial/feature connectivity graphs using native, optimized $k$-Nearest Neighbors. Out-of-sample test and validation nodes are evaluated strictly inductively via directed bipartite message passing from reference training nodes, guaranteeing zero transductive data leakage.
- **Graph Attention Architecture (GATConv)**: Employs multi-head graph attention (`heads=4`, `dropout=0.2`, `ELU` activation) to dynamically weight neighborhood influences during spatial vibration propagation.
- **Strict Out-of-Fold Validation Checkpointing**: Checkpoint selection tracks true validation MAE and early stopping on out-of-fold partitions, preventing overfitted training memorization.
- **Empirical Baseline Benchmarking**: Includes an industrial gradient-boosted decision tree baseline (`xgboost.XGBRegressor`) evaluated on the same K-Fold splits.
- **Production-Ready Modularity**: Decoupled package architecture (`src/blastvibe`), declarative configuration (`configs/default_config.yaml`), automated test suite (`pytest`), and complete model serialization (`torch.save`).

---

## 📊 Benchmark Evaluation (5-Fold Cross-Validation)

| Model Architecture | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
|---|---|---|---|
| **XGBoost Regressor (Baseline)** | 0.6943 | 19.41 mm/s | 26.54 mm/s |
| **BlastVibe-GAT (Leak-Free Inductive)** | **0.7409** | **16.71 mm/s** | **24.35 mm/s** |

*Note: The GAT architecture demonstrates a ~14% reduction in MAE and significant $R^2$ improvement over standard gradient boosting, validating the efficacy of graph relational inductive bias combined with physics-informed Scaled Distance features.*

---

## 📁 Repository Architecture

```
02_blastvibe-gat-gnn/
├── .gitignore                     # Production exclusions (data, artifacts, envs)
├── README.md                      # Comprehensive project documentation
├── pyproject.toml                 # Package definition & pytest configuration
├── requirements.txt               # Pinned project dependencies
├── configs/
│   └── default_config.yaml        # Declarative hyperparameters and CV configs
├── src/
│   └── blastvibe/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py          # Validated ingestion & Scaled Distance engineering
│       │   └── graph.py           # Leak-free inductive KNN graph builder
│       ├── models/
│       │   ├── __init__.py
│       │   ├── gat.py             # PyG GATRegressor multi-head network
│       │   └── baseline.py        # XGBoost benchmark regressor
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py         # K-Fold grid search, early stopping, & serialization
│       └── utils/
│           ├── __init__.py
│           └── logger.py          # Centralized structured logging
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py        # Schema & feature engineering assertions
│   ├── test_graph_builder.py      # Duplicate edge prevention & zero-leakage tests
│   └── test_model_forward.py      # Output shapes & backward gradient flow tests
└── main.py                        # Production CLI entry point
```

---

## 🛠️ Quickstart & Usage

### 1. Installation
Clone the repository and install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Run Test Suite
Verify mathematical invariants, shape contracts, and zero-leakage assertions:

```bash
pytest -v
```

### 3. Run Pipeline (Baseline + GAT Grid Search)
Execute the complete end-to-end pipeline:

```bash
python main.py --config configs/default_config.yaml
```

Outputs:
- Full evaluation logs: `artifacts/grid_search_results.csv`
- Serialized optimal model bundle (weights, hyperparameters, scalers): `artifacts/models/best_gat_model.pt`
