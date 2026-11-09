# gnn_regression_with_plots.py
# Predict maximum particle speed (PPV) using GNN with visualization of graph and prediction scatter.

import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv
except Exception as e:
    raise ImportError("Please install torch_geometric. Error: {}".format(e))

import itertools

# ---------------- Config ----------------
# Robust dataset path resolution (supports both CSV and Excel)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'data.csv') if os.path.exists(os.path.join(SCRIPT_DIR, 'data.csv')) else os.path.join(SCRIPT_DIR, 'data.xlsx')
K_NEIGHBORS = 8
HIDDEN_DIM = 64
NUM_EPOCHS = 600
LR = 1e-3
WEIGHT_DECAY = 1e-5
TEST_SIZE = 0.15
VAL_SIZE = 0.15
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------- Utils ----------------


def load_csv(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    required = ['DIS', 'CHARGE', 'SUM']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"File must contain columns: {required}")
    # Feature engineering: add Scaled Distance (SD = DIS / sqrt(CHARGE))
    df['SD'] = df['DIS'] / np.sqrt(df['CHARGE'])
    return df[['DIS', 'CHARGE', 'SD', 'SUM']]


def build_knn_edge_index(X, k=8):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, indices = nbrs.kneighbors(X)
    N = X.shape[0]
    edge_index = []
    for i in range(N):
        for j in indices[i, 1:]:
            edge_index.append([i, j])
            edge_index.append([j, i])
    edge_index = np.array(edge_index).T
    return torch.tensor(edge_index, dtype=torch.long)

# ---------------- Model ----------------


class GATRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim // 2, heads=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.lin(x)
        return x.squeeze(-1)

# ---------------- Visualization ----------------


def plot_knn_graph(X, y_true, edge_index, save_path='graph_knn.png'):
    G = nx.Graph()
    for i in range(X.shape[0]):
        G.add_node(i, pos=(X[i, 0], X[i, 1]), label=y_true[i])
    edges = edge_index.cpu().numpy().T
    for (u, v) in edges:
        G.add_edge(u, v)
    pos = {i: (X[i, 0], X[i, 1]) for i in range(X.shape[0])}
    labels = [y for y in y_true]
    plt.figure(figsize=(8, 6))
    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=40, node_color=labels, cmap='viridis')
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.colorbar(nodes, label='True SUM')
    plt.title('k-NN Graph (colored by true SUM)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def plot_true_vs_pred(y_true, y_pred, fname='true_vs_pred.png'):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('True SUM')
    plt.ylabel('Predicted SUM')
    plt.title('True vs Predicted')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")

# ---------------- Main ----------------


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"File not found at {CSV_PATH}")
    df = load_csv(CSV_PATH)
    N = len(df)
    print(f"Loaded {N} samples from {CSV_PATH}")
    print("Sample dataset with Scaled Distance (SD) feature:")
    print(df.head())

    X_raw = df[['DIS', 'CHARGE', 'SD']].values.astype(float)
    y_raw = df['SUM'].values.astype(float)

    # Grid Search params (reduced space)
    K_NEIGHBORS_LIST = [4, 8, 12]
    HIDDEN_DIM_LIST = [64, 128]
    LR_LIST = [1e-2, 1e-3, 5e-4]
    WEIGHT_DECAY_LIST = [1e-4, 0]
    NUM_EPOCHS = 1000
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    best_score = float('inf')
    best_config = None
    best_results = None
    best_model_state = None
    best_scaler_x, best_scaler_y = None, None
    best_fold = -1
    results_table = []
    total_combos = len(K_NEIGHBORS_LIST)*len(HIDDEN_DIM_LIST)*len(LR_LIST)*len(WEIGHT_DECAY_LIST)
    combo_idx = 0
    for K_NEIGHBORS, HIDDEN_DIM, LR, WEIGHT_DECAY in itertools.product(K_NEIGHBORS_LIST, HIDDEN_DIM_LIST, LR_LIST, WEIGHT_DECAY_LIST):
        combo_idx += 1
        print(f"\n=== Grid Search Combo {combo_idx}/{total_combos} ===")
        print(f"K_NEIGHBORS={K_NEIGHBORS}, HIDDEN_DIM={HIDDEN_DIM}, LR={LR}, WEIGHT_DECAY={WEIGHT_DECAY}")
        r2_list, mae_list, mse_list, rmse_list = [], [], [], []
        all_true, all_pred = [], []
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_raw)):
            X_train, X_test = X_raw[train_idx], X_raw[test_idx]
            y_train, y_test = y_raw[train_idx], y_raw[test_idx]
            scaler_x = StandardScaler()
            X_train_scaled = scaler_x.fit_transform(X_train)
            X_test_scaled = scaler_x.transform(X_test)
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
            y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)
            X_all = np.concatenate([X_train_scaled, X_test_scaled], axis=0)
            edge_index = build_knn_edge_index(X_all, k=K_NEIGHBORS)
            N_fold = X_all.shape[0]
            data = Data(
                x=torch.tensor(X_all, dtype=torch.float),
                y=torch.tensor(np.concatenate([y_train_scaled, y_test_scaled]), dtype=torch.float),
                edge_index=edge_index
            )
            train_mask = torch.zeros(N_fold, dtype=torch.bool)
            test_mask = torch.zeros(N_fold, dtype=torch.bool)
            train_mask[:len(train_idx)] = True
            test_mask[len(train_idx):] = True
            data.train_mask, data.test_mask = train_mask, test_mask
            data = data.to(DEVICE)
            model = GATRegressor(in_dim=3, hidden_dim=HIDDEN_DIM).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            loss_fn = nn.MSELoss()
            best_state = None
            best_fold_val = float('inf')
            for epoch in range(1, NUM_EPOCHS+1):
                model.train()
                opt.zero_grad()
                out = model(data.x, data.edge_index)
                loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                opt.step()
                if epoch % 100 == 0:
                    model.eval()
                    with torch.no_grad():
                        pred = model(data.x, data.edge_index).cpu().numpy()
                        y_true = data.y.cpu().numpy()
                        pred_inv = scaler_y.inverse_transform(pred.reshape(-1, 1)).reshape(-1)
                        y_inv = scaler_y.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
                        val_mae = np.mean(
                            np.abs(pred_inv[train_mask.cpu().numpy()] - y_inv[train_mask.cpu().numpy()]))
                        if val_mae < best_fold_val:
                            best_fold_val = val_mae
                            best_state = model.state_dict()
            if best_state:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).cpu().numpy()
                y_true = data.y.cpu().numpy()
                pred_inv = scaler_y.inverse_transform(pred.reshape(-1, 1)).reshape(-1)
                y_inv = scaler_y.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
                y_test_true = y_inv[len(train_idx):]
                y_test_pred = pred_inv[len(train_idx):]
                all_true.extend(y_test_true)
                all_pred.extend(y_test_pred)
                r2 = r2_score(y_test_true, y_test_pred)
                mae = mean_absolute_error(y_test_true, y_test_pred)
                mse = mean_squared_error(y_test_true, y_test_pred)
                rmse = np.sqrt(mse)
                r2_list.append(r2)
                mae_list.append(mae)
                mse_list.append(mse)
                rmse_list.append(rmse)
        mean_mae = np.mean(mae_list)
        mean_r2 = np.mean(r2_list)
        mean_mse = np.mean(mse_list)
        mean_rmse = np.mean(rmse_list)
        results_table.append({
            'K_NEIGHBORS': K_NEIGHBORS,
            'HIDDEN_DIM': HIDDEN_DIM,
            'LR': LR,
            'WEIGHT_DECAY': WEIGHT_DECAY,
            'MAE': mean_mae,
            'R2': mean_r2,
            'MSE': mean_mse,
            'RMSE': mean_rmse
        })
        print(f"Grid Combo | R2: {mean_r2:.4f} | MAE: {mean_mae:.4f} | MSE: {mean_mse:.4f} | RMSE: {mean_rmse:.4f}")
        if mean_mae < best_score:
            best_score = mean_mae
            best_config = (K_NEIGHBORS, HIDDEN_DIM, LR, WEIGHT_DECAY)
            best_results = (mean_r2, mean_mae, mean_mse, mean_rmse)
            best_model_state = model.state_dict()
            best_scaler_x = scaler_x
            best_scaler_y = scaler_y
    print("\n=== Best Hyperparameters ===")
    print(f"K_NEIGHBORS={best_config[0]}, HIDDEN_DIM={best_config[1]}, LR={best_config[2]}, WEIGHT_DECAY={best_config[3]}")
    print(f"Best R2: {best_results[0]:.4f} | Best MAE: {best_results[1]:.4f} | Best MSE: {best_results[2]:.4f} | Best RMSE: {best_results[3]:.4f}")
    
    # Save full grid search evaluation results
    out_csv = os.path.join(SCRIPT_DIR, 'grid_search_results.csv')
    pd.DataFrame(results_table).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == '__main__':
    main()
