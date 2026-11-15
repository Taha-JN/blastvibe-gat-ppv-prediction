from typing import Tuple, Optional
import torch

from blastvibe.utils.logger import setup_logger

logger = setup_logger("blastvibe.data.graph")


def _torch_native_knn_graph(x: torch.Tensor, k: int, loop: bool = False) -> torch.Tensor:
    """
    Pure PyTorch vectorized K-NN graph construction.
    
    Eliminates external C++ binary dependencies (such as pyg-lib) by leveraging
    hardware-accelerated torch.cdist and torch.topk.
    """
    n_nodes = x.size(0)
    dist = torch.cdist(x, x)
    if not loop:
        dist.fill_diagonal_(float("inf"))
    _, neighbors = torch.topk(dist, k=k, dim=1, largest=False)
    src_nodes = neighbors.flatten()
    dst_nodes = torch.arange(n_nodes, device=x.device).repeat_interleave(k)
    return torch.stack([src_nodes, dst_nodes], dim=0)


def build_train_knn_graph(
    x_train: torch.Tensor,
    k: int = 8,
    loop: bool = False,
) -> torch.Tensor:
    """
    Constructs a K-NN graph strictly using training features.
    
    Args:
        x_train: Training feature tensor of shape [N_train, num_features].
        k: Number of nearest neighbors per node.
        loop: Whether to include self-loops. Defaults to False.
        
    Returns:
        edge_index: Graph edge tensor of shape [2, num_edges] with no duplicates.
    """
    if x_train.dim() != 2:
        raise ValueError(f"x_train must be 2D tensor, got shape {x_train.shape}")

    n_nodes = x_train.size(0)
    if k >= n_nodes:
        raise ValueError(f"k ({k}) must be strictly less than number of training nodes ({n_nodes})")

    try:
        import torch_geometric.typing as pyg_typing
        if getattr(pyg_typing, "WITH_KNN", False):
            from torch_geometric.nn import knn_graph
            return knn_graph(x_train, k=k, loop=loop)
    except (ImportError, Exception):
        pass

    return _torch_native_knn_graph(x_train, k=k, loop=loop)


def build_inductive_graph(
    x_train: torch.Tensor,
    x_query: torch.Tensor,
    train_edge_index: torch.Tensor,
    k: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs an inductive inference graph for out-of-sample query/test nodes
    using pure PyTorch GPU-accelerated operations.
    
    Zero Data Leakage Guarantee:
    1. The reference graph (train_edge_index) remains purely internal to x_train.
    2. Directed edges point strictly from reference train nodes -> query nodes.
    3. No backward edges point from query nodes into train nodes.
    4. No edges exist between query nodes.
    
    Args:
        x_train: Reference training feature tensor of shape [N_train, D].
        x_query: Out-of-sample query feature tensor of shape [N_query, D].
        train_edge_index: Precomputed training edges of shape [2, E_train].
        k: Number of nearest neighbors to connect each query node to.
        
    Returns:
        x_combined: Combined node feature tensor of shape [N_train + N_query, D].
        combined_edge_index: Leak-free edge index of shape [2, E_train + (N_query * k)].
        query_mask: Boolean tensor of shape [N_train + N_query] where True indicates query nodes.
    """
    n_train = x_train.size(0)
    n_query = x_query.size(0)
    device = x_train.device

    if k >= n_train:
        raise ValueError(f"k ({k}) must be strictly less than reference nodes ({n_train})")

    # Ensure tensors are on the same device and float type
    x_query = x_query.to(device=device, dtype=x_train.dtype)
    train_edge_index = train_edge_index.to(device)

    # Attempt PyG accelerated knn if available with compiled pyg-lib; fall back to pure PyTorch cdist
    used_pyg = False
    try:
        import torch_geometric.typing as pyg_typing
        if getattr(pyg_typing, "WITH_KNN", False):
            from torch_geometric.nn import knn
            assign_index = knn(x=x_train, y=x_query, k=k)
            src_nodes = assign_index[1]
            dst_nodes = assign_index[0] + n_train
            used_pyg = True
    except (ImportError, Exception):
        pass

    if not used_pyg:
        # High-performance, pure PyTorch GPU/CPU fallback without C++ binary dependency
        # dist: [N_query, N_train]
        dist = torch.cdist(x_query, x_train)
        _, neighbor_indices = torch.topk(dist, k=k, dim=1, largest=False)
        src_nodes = neighbor_indices.flatten()
        dst_nodes = torch.arange(n_train, n_train + n_query, device=device).repeat_interleave(k)

    query_edge_index = torch.stack([src_nodes, dst_nodes], dim=0)

    # Concatenate internal train edges and directed train->query edges
    combined_edge_index = torch.cat([train_edge_index, query_edge_index], dim=1)
    x_combined = torch.cat([x_train, x_query], dim=0)

    query_mask = torch.zeros(n_train + n_query, dtype=torch.bool, device=device)
    query_mask[n_train:] = True

    return x_combined, combined_edge_index, query_mask
