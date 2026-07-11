import torch

from gnn_molecules.layers import GCNLayer, MPNNConv, scatter_mean, scatter_sum


def test_scatter_sum_matches_manual():
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 0, 1])
    out = scatter_sum(src, index, dim_size=2)
    assert torch.allclose(out, torch.tensor([[4.0, 6.0], [5.0, 6.0]]))


def test_scatter_mean_matches_manual():
    src = torch.tensor([[2.0], [4.0], [10.0]])
    index = torch.tensor([0, 0, 1])
    out = scatter_mean(src, index, dim_size=2)
    assert torch.allclose(out, torch.tensor([[3.0], [10.0]]))


def test_scatter_mean_empty_bucket_is_zero():
    src = torch.tensor([[1.0], [1.0]])
    index = torch.tensor([0, 0])
    out = scatter_mean(src, index, dim_size=3)
    assert torch.allclose(out[2], torch.zeros(1))


def test_mpnn_conv_preserves_node_count():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_attr = torch.randn(3, 5)
    conv = MPNNConv(node_dim=8, edge_dim=5, hidden_dim=16)
    out = conv(x, edge_index, edge_attr)
    assert out.shape == (4, 8)


def test_mpnn_conv_handles_no_edges():
    x = torch.randn(2, 8)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, 5))
    conv = MPNNConv(node_dim=8, edge_dim=5, hidden_dim=16)
    out = conv(x, edge_index, edge_attr)
    assert out.shape == (2, 8)


def test_gcn_layer_output_dim():
    x = torch.randn(5, 6)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    layer = GCNLayer(in_dim=6, out_dim=10)
    out = layer(x, edge_index)
    assert out.shape == (5, 10)


def test_gcn_layer_is_permutation_reasonable():
    # A GCN layer must run even when the graph has only self loops.
    x = torch.randn(3, 4)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    layer = GCNLayer(in_dim=4, out_dim=4)
    out = layer(x, edge_index)
    assert out.shape == (3, 4)
    assert torch.isfinite(out).all()
