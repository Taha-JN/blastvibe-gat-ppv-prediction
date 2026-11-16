import torch
from torch import nn

from blastvibe.data.graph import build_train_knn_graph
from blastvibe.models.gat import FlexibleGATRegressor, GATRegressor


def test_flexible_gat_dynamic_layers():
    """Asserts that GATRegressor / FlexibleGATRegressor dynamically constructs N layers."""
    torch.manual_seed(42)
    n_nodes = 20
    in_dim = 3
    hidden_dims = [48, 24, 12]
    heads = 3

    x = torch.randn(n_nodes, in_dim)
    edge_index = build_train_knn_graph(x, k=4, loop=False)

    model = FlexibleGATRegressor(
        in_dim=in_dim,
        hidden_dims=hidden_dims,
        heads=heads,
        dropout=0.1,
    )
    assert len(model.convs) == 3
    # First 2 layers must have heads=heads and concat=True
    assert model.convs[0].heads == heads
    assert model.convs[0].concat is True
    assert model.convs[1].heads == heads
    assert model.convs[1].concat is True
    # Final layer must have heads=1 and concat=False
    assert model.convs[2].heads == 1
    assert model.convs[2].concat is False
    assert model.readout.in_features == 12

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)

    assert out.shape == torch.Size([n_nodes])


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
