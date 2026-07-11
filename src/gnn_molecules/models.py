"""Molecular property models: MPNN, GCN, and a fingerprint MLP baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from gnn_molecules.data import GraphBatch
from gnn_molecules.layers import GCNLayer, MPNNConv, scatter_mean


class MPNN(nn.Module):
    """A message passing neural network for graph-level regression.

    Atom features are embedded, refined by several shared message-passing
    rounds, mean-pooled to a molecule vector, and mapped to a scalar.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList(
            MPNNConv(hidden_dim, edge_dim, hidden_dim) for _ in range(num_layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        """Return one prediction per graph, shape ``(num_graphs,)``."""
        x = torch.relu(self.embed(batch.node_feats))
        for conv in self.convs:
            x = conv(x, batch.edge_index, batch.edge_feats)
        pooled = scatter_mean(x, batch.batch, batch.num_graphs)
        return self.readout(pooled).squeeze(-1)


class GCN(nn.Module):
    """A graph convolutional network for graph-level regression."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        dims = [node_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            GCNLayer(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        """Return one prediction per graph, shape ``(num_graphs,)``."""
        x = batch.node_feats
        for layer in self.layers:
            x = torch.relu(layer(x, batch.edge_index))
        pooled = scatter_mean(x, batch.batch, batch.num_graphs)
        return self.readout(pooled).squeeze(-1)


class FingerprintMLP(nn.Module):
    """A plain MLP over Morgan fingerprints, used as a non-graph baseline."""

    def __init__(self, n_bits: int = 2048, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fingerprints: torch.Tensor) -> torch.Tensor:
        """Return one prediction per fingerprint, shape ``(batch,)``."""
        return self.net(fingerprints).squeeze(-1)
