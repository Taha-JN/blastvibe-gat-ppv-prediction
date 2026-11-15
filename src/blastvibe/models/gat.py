from typing import Optional, List, Union, Tuple, Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATRegressor(nn.Module):
    """
    Flexible Graph Attention Network (GAT) for node-level vibration regression.
    
    Dynamically constructs an N-layer multi-head graph attention architecture
    using nn.ModuleList according to the Open/Closed Principle.
    
    Supports arbitrary hidden layer sequences (hidden_dims: List[int]) where preceding
    layers utilize multi-head attention (concat=True) and the final convolution layer
    aggregates into a single-head representation (heads=1, concat=False).
    """
    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: Optional[int] = 64,
        out_hidden_dim: Optional[int] = None,
        heads: int = 4,
        dropout: float = 0.2,
        hidden_dims: Optional[Union[List[int], Tuple[int, ...]]] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.heads = heads
        self.dropout_rate = dropout

        # Backward compatibility: resolve hidden_dims if not explicitly provided
        if hidden_dims is None:
            h_dim = hidden_dim if hidden_dim is not None else 64
            out_h = out_hidden_dim if out_hidden_dim is not None else (h_dim // 2)
            self.hidden_dims = [h_dim, out_h]
        else:
            self.hidden_dims = list(hidden_dims)

        if len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer dimension.")

        self.hidden_dim = self.hidden_dims[0]
        self.out_hidden_dim = self.hidden_dims[-1]

        # Dynamically build GAT layers using nn.ModuleList
        self.convs = nn.ModuleList()
        current_in = in_dim

        for i, h_dim in enumerate(self.hidden_dims):
            is_last = (i == len(self.hidden_dims) - 1)
            layer_heads = 1 if is_last else heads
            concat = not is_last

            conv = GATConv(
                in_channels=current_in,
                out_channels=h_dim,
                heads=layer_heads,
                dropout=dropout,
                concat=concat,
            )
            self.convs.append(conv)
            current_in = h_dim if is_last else (h_dim * layer_heads)

        self.dropout = nn.Dropout(p=dropout)
        self.readout = nn.Linear(self.hidden_dims[-1], 1)

    @property
    def gat1(self) -> GATConv:
        """Backward compatibility property accessing the first GAT layer."""
        return self.convs[0]

    @property
    def gat2(self) -> GATConv:
        """Backward compatibility property accessing the second GAT layer."""
        if len(self.convs) > 1:
            return self.convs[1]
        raise AttributeError("This GATRegressor instance has only 1 convolutional layer.")

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False):
        """
        Seamlessly remaps legacy checkpoints containing 'gat1.*' and 'gat2.*' keys
        into the modern ModuleList 'convs.0.*' and 'convs.1.*' structure.
        """
        remapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("gat1."):
                new_key = key.replace("gat1.", "convs.0.", 1)
                remapped_state_dict[new_key] = value
            elif key.startswith("gat2."):
                new_key = key.replace("gat2.", "convs.1.", 1)
                remapped_state_dict[new_key] = value
            else:
                remapped_state_dict[key] = value
        return super().load_state_dict(remapped_state_dict, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for node regression across dynamically assembled GAT layers.
        
        Args:
            x: Node feature matrix of shape [N, in_dim].
            edge_index: Graph edge indices of shape [2, E].
            
        Returns:
            Continuous predictions tensor of shape [N].
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)

        out = self.readout(x)
        return out.squeeze(-1)


# Semantic alias conforming to Senior Architect naming conventions
FlexibleGATRegressor = GATRegressor

__all__ = ["GATRegressor", "FlexibleGATRegressor"]
