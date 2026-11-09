# BlastVibe-GNN: Graph Attention Network for Blast Peak Particle Velocity (PPV) Prediction

A domain-informed Scientific Machine Learning (SciML) and Graph Neural Network (GNN) framework implemented in PyTorch Geometric to predict ground vibration and Peak Particle Velocity (PPV) resulting from blasting operations in mining and civil excavation.

## Technical Highlights
- **Domain Physics Feature Engineering**: Incorporates empirical explosion physics by engineering **Scaled Distance** ($SD = rac{Distance}{\sqrt{Charge}}$), augmenting basic geometric features to allow the network to learn non-linear energy dissipation dynamics.
- **Graph Topology Construction**: Dynamically converts tabular spatial blast monitoring points into a structured graph via $k$-Nearest Neighbors ($k$-NN graph with $k=12$).
- **Graph Attention Network (GATConv)**: Employs multi-head graph attention (`heads=4`, dropout=0.2) to assign dynamic attention weights to neighboring blast stations, significantly outperforming standard isotropic GCNs.
- **Rigorous Evaluation & Hyperparameter Tuning**:
  - 10-Fold Stratified Cross-Validation ensuring model generalization without spatial leakage.
  - Exhaustive Grid Search optimizing neighborhood connectivity ($K$), hidden dimension width, learning rate, and $L_2$ weight decay.
  - **Results**: Achieves $R^2 = 0.7673$ and $RMSE = 22.40$ on real-world industrial vibrational datasets.

## Repository Structure
```
02_blastvibe-gat-gnn/
├── gnn_regression_with_plots.py  # Main PyTorch Geometric training, validation, & plotting script
├── data.xlsx                     # Blast field measurement dataset (DIS, CHARGE, PPV/SUM)
├── grid_search_results.csv       # Complete evaluation metric logs across all hyperparameter runs
├── predictions_with_gnn.csv      # Out-of-fold ground truth vs. predicted values
├── report.md                     # Detailed experimental analysis report
├── requirements.txt              # Dependencies (torch, torch-geometric, scikit-learn, networkx)
├── graph_knn.png                 # Topological graph connectivity visualization
└── new_true_vs_pred.png          # Scatter plot: Measured vs. Predicted PPV
```

## Quickstart
```bash
pip install -r requirements.txt
python gnn_regression_with_plots.py
```
