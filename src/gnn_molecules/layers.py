"""Graph neural network building blocks written from scratch.

Nothing here depends on a graph-learning framework such as PyTorch Geometric.
The scatter reductions, the message-passing convolution, and the normalized
graph convolution are all implemented directly on ``edge_index`` tensors so the
mechanics are visible and unit-testable.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum rows of ``src`` into buckets given by ``index``.

    Args:
        src: Values to aggregate, shape ``(E, F)``.
        index: Destination bucket for each row, shape ``(E,)``.
        dim_size: Number of output buckets (rows of the result).

    Returns:
        Tensor of shape ``(dim_size, F)`` where row ``k`` is the sum of every
        ``src`` row whose index equals ``k``.
    """
    out = src.new_zeros((dim_size, src.shape[1]))
    idx = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean-reduce rows of ``src`` into buckets given by ``index``.

    Empty buckets are returned as zeros rather than NaN.
    """
    summed = scatter_sum(src, index, dim_size)
    count = src.new_zeros((dim_size, 1))
    ones = src.new_ones((src.shape[0], 1))
    count.scatter_add_(0, index.unsqueeze(-1), ones)
    return summed / count.clamp(min=1.0)


class MPNNConv(nn.Module):
    """One round of edge-conditioned message passing.

    For every directed edge ``j -> i`` a message is computed from the source
    node state and the bond features, then summed over the incoming edges of
    each node and used to update that node's state with a GRU cell. This is the
    core operator of the message passing neural network family.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.update = nn.GRUCell(node_dim, node_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Return updated node states of shape ``(num_nodes, node_dim)``."""
        num_nodes = x.shape[0]
        if edge_index.shape[1] == 0:
            # No bonds (e.g. a lone atom): pass the state through the GRU with a
            # zero message so parameters still receive gradient.
            zero_msg = x.new_zeros(x.shape)
            return self.update(zero_msg, x)
        src, dst = edge_index[0], edge_index[1]
        msg_input = torch.cat([x[src], edge_attr], dim=-1)
        messages = self.message_mlp(msg_input)
        aggregated = scatter_sum(messages, dst, num_nodes)
        return self.update(aggregated, x)


class GCNLayer(nn.Module):
    """A graph convolution with symmetric normalization and added self loops.

    Implements ``H' = D^{-1/2} (A + I) D^{-1/2} H W`` using the edge list rather
    than a dense adjacency matrix, so it scales to sparse molecular graphs.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return convolved node states of shape ``(num_nodes, out_dim)``."""
        num_nodes = x.shape[0]
        device = x.device
        self_loops = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        if edge_index.shape[1] > 0:
            full_edges = torch.cat([edge_index, self_loops], dim=1)
        else:
            full_edges = self_loops

        src, dst = full_edges[0], full_edges[1]
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        transformed = self.linear(x)
        messages = transformed[src] * norm.unsqueeze(-1)
        return scatter_sum(messages, dst, num_nodes)
