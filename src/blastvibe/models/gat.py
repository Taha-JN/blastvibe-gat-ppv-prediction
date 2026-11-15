from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATRegressor(nn.Module):
    """
    Graph Attention Network (GAT) for node-level vibration regression.
    
    Utilizes multi-head attention mechanism to dynamically weight neighborhood
    influences during spatial message aggregation.
    """
    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 64,
        out_hidden_dim: Optional[int] = None,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout_rate = dropout

        out_hidden = out_hidden_dim if out_hidden_dim is not None else (hidden_dim // 2)
        self.out_hidden_dim = out_hidden

        # Layer 1: Multi-head graph attention
        self.gat1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
        )

        # Layer 2: Single-head attention aggregation
        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=out_hidden,
            heads=1,
            dropout=dropout,
            concat=False,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.readout = nn.Linear(out_hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for node regression.
        
        Args:
            x: Node feature matrix of shape [N, in_dim].
            edge_index: Graph edge indices of shape [2, E].
            
        Returns:
            Continuous predictions tensor of shape [N].
        """
        # Block 1
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        # Block 2
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        # Readout head
        out = self.readout(x)
        return out.squeeze(-1)
