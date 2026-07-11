"""Message passing neural networks for molecular property prediction.

A small, dependency-light PyTorch library that implements graph neural
networks for molecules from first principles: an explicit message-passing
convolution with a hand-written scatter aggregation, a graph convolutional
network with symmetric adjacency normalization, and a Morgan-fingerprint MLP
baseline. Graphs are built from SMILES with RDKit.
"""

from gnn_molecules.data import (
    MolGraphDataset,
    collate_graphs,
    load_esol,
    random_split,
    rdkit_property_dataset,
)
from gnn_molecules.featurize import atom_features, bond_features, mol_to_graph
from gnn_molecules.layers import GCNLayer, MPNNConv, scatter_mean, scatter_sum
from gnn_molecules.models import GCN, MPNN, FingerprintMLP
from gnn_molecules.train import Trainer, regression_metrics, set_seed

__all__ = [
    "mol_to_graph",
    "atom_features",
    "bond_features",
    "MolGraphDataset",
    "collate_graphs",
    "load_esol",
    "rdkit_property_dataset",
    "random_split",
    "scatter_sum",
    "scatter_mean",
    "MPNNConv",
    "GCNLayer",
    "MPNN",
    "GCN",
    "FingerprintMLP",
    "Trainer",
    "regression_metrics",
    "set_seed",
]

__version__ = "0.1.0"
