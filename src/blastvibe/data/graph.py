import torch
from scipy.spatial import cKDTree

from blastvibe.utils.logger import setup_logger

logger = setup_logger("blastvibe.data.graph")


def build_scalable_knn_graph(x: torch.Tensor, k: int = 8, loop: bool = False) -> torch.Tensor:
    """
    O(N log N) scalable spatial indexing using scipy.spatial.cKDTree.

    Eliminates O(N^2) memory and compute bottlenecks of pairwise distance matrices
    (torch.cdist), enabling scalable neighbor construction for large-scale datasets.

    Args:
        x: Feature tensor of shape [N, D].
        k: Number of nearest neighbors per node.
        loop: Whether to include self-loops. Defaults to False.

    Returns:
        edge_index: Graph edge tensor of shape [2, N * k] with no duplicate edges.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D tensor, got shape {x.shape}")

    n_nodes = x.size(0)
    if k >= n_nodes:
        raise ValueError(f"k ({k}) must be strictly less than number of nodes ({n_nodes})")

    x_np = x.detach().cpu().numpy()
    tree = cKDTree(x_np)

    if not loop:
        # Query k + 1 to drop the self-loop (first neighbor at distance 0)
        _, indices = tree.query(x_np, k=k + 1)
        if (k + 1) == 1:
            indices = indices.reshape(-1, 1)
        neighbors = indices[:, 1:]
    else:
        _, indices = tree.query(x_np, k=k)
        if k == 1:
            indices = indices.reshape(-1, 1)
        neighbors = indices

    src_nodes = torch.tensor(neighbors.flatten(), dtype=torch.long, device=x.device)
    dst_nodes = torch.arange(n_nodes, device=x.device).repeat_interleave(k)
    return torch.stack([src_nodes, dst_nodes], dim=0)


# Backward compatibility alias
_torch_native_knn_graph = build_scalable_knn_graph


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
    except (ImportError, RuntimeError, AttributeError):
        pass

    return build_scalable_knn_graph(x_train, k=k, loop=loop)


def build_inductive_graph(
    x_train: torch.Tensor,
    x_query: torch.Tensor,
    train_edge_index: torch.Tensor,
    k: int = 8,
    spatial_tree: cKDTree | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs an inductive inference graph for out-of-sample query/test nodes
    using scalable O(N_query * log(N_train)) spatial indexing via cKDTree.

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
        spatial_tree: Optional precomputed scipy.spatial.cKDTree of x_train to eliminate
                      redundant tree construction overhead during inference.

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

    # Attempt PyG accelerated knn if available with compiled pyg-lib; fall back to scalable cKDTree
    used_pyg = False
    if spatial_tree is None:
        try:
            import torch_geometric.typing as pyg_typing

            if getattr(pyg_typing, "WITH_KNN", False):
                from torch_geometric.nn import knn

                assign_index = knn(x=x_train, y=x_query, k=k)
                src_nodes = assign_index[1]
                dst_nodes = assign_index[0] + n_train
                used_pyg = True
        except (ImportError, RuntimeError, AttributeError):
            pass

    if not used_pyg:
        # High-performance, O(N_query * log(N_train)) scalable cKDTree spatial indexing
        x_query_np = x_query.detach().cpu().numpy()
        tree = spatial_tree if spatial_tree is not None else cKDTree(x_train.detach().cpu().numpy())
        _, neighbor_indices = tree.query(x_query_np, k=k)
        if k == 1:
            neighbor_indices = neighbor_indices.reshape(-1, 1)
        src_nodes = torch.tensor(neighbor_indices.flatten(), dtype=torch.long, device=device)
        dst_nodes = torch.arange(n_train, n_train + n_query, device=device).repeat_interleave(k)

    query_edge_index = torch.stack([src_nodes, dst_nodes], dim=0)

    # Concatenate internal train edges and directed train->query edges
    combined_edge_index = torch.cat([train_edge_index, query_edge_index], dim=1)
    x_combined = torch.cat([x_train, x_query], dim=0)

    query_mask = torch.zeros(n_train + n_query, dtype=torch.bool, device=device)
    query_mask[n_train:] = True

    return x_combined, combined_edge_index, query_mask
