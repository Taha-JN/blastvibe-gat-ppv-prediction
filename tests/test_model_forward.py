import torch
import torch.nn as nn
from blastvibe.models.gat import GATRegressor
from blastvibe.data.graph import build_train_knn_graph


def test_gat_forward_shape():
    torch.manual_seed(42)
    n_nodes = 25
    in_dim = 3
    hidden_dim = 32
    heads = 4

    x = torch.randn(n_nodes, in_dim)
    edge_index = build_train_knn_graph(x, k=4, loop=False)

    model = GATRegressor(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        heads=heads,
        dropout=0.2,
    )
    model.eval()

    with torch.no_grad():
        out = model(x, edge_index)

    assert out.dim() == 1, f"Expected 1D output tensor for regression, got {out.dim()}D"
    assert out.shape == torch.Size([n_nodes]), f"Expected shape [{n_nodes}], got {out.shape}"


def test_gat_gradient_flow():
    torch.manual_seed(42)
    n_nodes = 20
    in_dim = 3

    x = torch.randn(n_nodes, in_dim)
    y = torch.randn(n_nodes)
    edge_index = build_train_knn_graph(x, k=4, loop=False)

    model = GATRegressor(in_dim=in_dim, hidden_dim=16, heads=2, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out, y)
    loss.backward()

    # Assert all learnable parameters received gradients
    has_grads = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} contains NaN gradient!"
            has_grads = True

    assert has_grads, "No gradients were computed!"

    # Test optimizer step updates weights
    initial_weight = model.readout.weight.clone()
    optimizer.step()
    assert not torch.equal(initial_weight, model.readout.weight), "Model weights did not update after step!"
