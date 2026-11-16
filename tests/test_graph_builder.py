import torch
import pytest
from blastvibe.data.graph import (
    build_train_knn_graph,
    build_inductive_graph,
    build_scalable_knn_graph,
)


def test_build_scalable_knn_graph_ckdtree():
    """Asserts that cKDTree-based build_scalable_knn_graph drops self-loops and scales properly."""
    torch.manual_seed(42)
    n_nodes = 25
    k = 4
    x = torch.randn(n_nodes, 3)

    edge_index = build_scalable_knn_graph(x, k=k, loop=False)

    assert edge_index.shape == (2, n_nodes * k)
    # Check that no self-loops exist
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        assert src != dst, f"Self-loop detected: {src} -> {dst}"

    # Check in-degrees
    dst_nodes = edge_index[1].tolist()
    for node_idx in range(n_nodes):
        assert dst_nodes.count(node_idx) == k


def test_train_knn_graph_no_duplicate_edges():
    torch.manual_seed(42)
    n_nodes = 30
    k = 5
    x = torch.randn(n_nodes, 3)

    edge_index = build_train_knn_graph(x, k=k, loop=False)

    # Assert edge_index has shape [2, E]
    assert edge_index.dim() == 2
    assert edge_index.size(0) == 2

    # Check for duplicate edges
    edge_pairs = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    unique_pairs = set(edge_pairs)
    assert len(edge_pairs) == len(unique_pairs), "Duplicate edges detected in training graph!"

    # In knn_graph(flow='source_to_target'), each target node receives exactly k incoming edges
    dst_nodes = edge_index[1].tolist()
    for node_idx in range(n_nodes):
        in_degree = dst_nodes.count(node_idx)
        assert in_degree == k, f"Node {node_idx} expected in-degree {k}, got {in_degree}"


def test_inductive_graph_zero_leakage():
    torch.manual_seed(42)
    n_train = 40
    n_query = 10
    k = 4

    x_train = torch.randn(n_train, 3)
    x_query = torch.randn(n_query, 3)
    train_edges = build_train_knn_graph(x_train, k=k, loop=False)

    x_combined, eval_edges, query_mask = build_inductive_graph(
        x_train=x_train,
        x_query=x_query,
        train_edge_index=train_edges,
        k=k,
    )

    # 1. Assert combined node features shape
    assert x_combined.shape == (n_train + n_query, 3)
    assert query_mask.sum().item() == n_query
    assert not query_mask[:n_train].any()
    assert query_mask[n_train:].all()

    # 2. Check no duplicate edges across the entire inductive graph
    eval_pairs = list(zip(eval_edges[0].tolist(), eval_edges[1].tolist()))
    assert len(eval_pairs) == len(set(eval_pairs)), "Duplicate edges found in inductive graph!"

    # 3. CRITICAL: Zero Leakage Assertion
    # No edge should have a source in query nodes (>= n_train) pointing into train nodes (< n_train)
    sources = eval_edges[0].tolist()
    destinations = eval_edges[1].tolist()

    for src, dst in zip(sources, destinations):
        if src >= n_train:
            pytest.fail(f"Data leakage detected! Query node {src} is acting as a message source.")
        if dst < n_train and src >= n_train:
            pytest.fail(f"Data leakage detected! Query node {src} points into train node {dst}.")

    # 4. Each query node must receive exactly k edges from reference nodes
    for query_idx in range(n_train, n_train + n_query):
        in_degree = destinations.count(query_idx)
        assert in_degree == k, f"Query node {query_idx} expected in-degree {k}, got {in_degree}"
