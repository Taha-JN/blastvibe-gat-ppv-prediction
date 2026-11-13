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
| **USBM Empirical Attenuation Law (Physics Baseline)** | 0.7372 | 17.05 mm/s | 24.47 mm/s |
| **Tuned XGBoost Regressor (Nested GridSearchCV)** | 0.7244 | 18.11 mm/s | 25.11 mm/s |
| **BlastVibe-GAT (Ours, Leak-Free Inductive)** | **0.7468** | **16.89 mm/s** | **24.12 mm/s** |

*Note: The GAT architecture demonstrates state-of-the-art predictive performance, validating the efficacy of dynamic multi-head graph relational attention combined with USBM Scaled Distance ($SD$) physics features under equitable nested cross-validation.*

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
├── data/
│   └── data.csv                   # Validated blast monitoring dataset
├── src/
│   └── blastvibe/
│       ├── __init__.py            # Top-level exports (BlastVibePredictor)
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py          # Validated ingestion & Scaled Distance engineering
│       │   └── graph.py           # Pure PyTorch GPU vectorized inductive KNN graph builder
│       ├── models/
│       │   ├── __init__.py
│       │   ├── gat.py             # PyG GATRegressor multi-head network
│       │   ├── baseline.py        # USBM Physics & nested GridSearchCV XGBoost baselines
│       │   └── predictor.py       # Self-contained production inference engine
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py         # K-Fold grid search, full model fitting, & artifact bundle
│       └── utils/
│           ├── __init__.py
│           └── logger.py          # Centralized structured logging
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py        # Schema & feature engineering assertions
│   ├── test_graph_builder.py      # Duplicate edge prevention & zero-leakage tests
│   ├── test_model_forward.py      # Output shapes & backward gradient flow tests
│   └── test_predictor.py          # Standalone inference & serialization roundtrip tests
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
Verify mathematical invariants, shape contracts, zero-leakage assertions, and predictor round-trips:

```bash
pytest -v
```

### 3. Run Pipeline (Baselines + GAT Grid Search + Production Model Fitting)
Execute the complete end-to-end pipeline:

```bash
python main.py --config configs/default_config.yaml
```

Outputs:
- Full evaluation logs: `artifacts/grid_search_results.csv`
- Serialized production model bundle (weights, hyperparameters, scalers, reference graph topology): `artifacts/models/best_gat_model.pt`

### 4. Standalone Production Serving (Inference)
Use the self-contained `BlastVibePredictor` service in any downstream application or API:

```python
from blastvibe import BlastVibePredictor

# Load serialized model bundle
predictor = BlastVibePredictor("artifacts/models/best_gat_model.pt", device="cpu")

# Single blast shot inference (DIS=400m, CHARGE=1850kg)
ppv = predictor.predict_single(dis=400.0, charge=1850.0)
print(f"Predicted Ground Vibration: {ppv:.2f} mm/s")

# Batch inference
import numpy as np
raw_features = np.array([
    [300.0, 1850.0, 6.97],
    [500.0, 1950.0, 11.32],
])
predictions = predictor.predict(raw_features)
print(f"Batch PPV Predictions: {predictions}")
```
