import numpy as np
import pytest

from gnn_molecules.featurize import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    mol_to_graph,
    morgan_fingerprint,
    smiles_to_graph,
)


def test_ethanol_graph_shapes():
    graph = smiles_to_graph("CCO")
    assert graph.num_nodes == 3
    assert graph.node_feats.shape == (3, ATOM_FEATURE_DIM)
    # 2 bonds -> 4 directed edges.
    assert graph.edge_index.shape == (2, 4)
    assert graph.edge_feats.shape == (4, BOND_FEATURE_DIM)


def test_edges_are_symmetric():
    graph = smiles_to_graph("CCO")
    edges = {tuple(e) for e in graph.edge_index.T.tolist()}
    for i, j in list(edges):
        assert (j, i) in edges


def test_single_atom_has_no_edges():
    graph = smiles_to_graph("[Na+]")
    assert graph.num_nodes == 1
    assert graph.edge_index.shape == (2, 0)


def test_invalid_smiles_raises():
    with pytest.raises(ValueError):
        smiles_to_graph("not_a_molecule")


def test_none_mol_raises():
    with pytest.raises(ValueError):
        mol_to_graph(None)


def test_fingerprint_is_binary():
    fp = morgan_fingerprint("c1ccccc1", n_bits=256)
    assert fp.shape == (256,)
    assert set(np.unique(fp)).issubset({0.0, 1.0})


def test_edge_index_within_bounds():
    graph = smiles_to_graph("c1ccc2ccccc2c1")
    assert graph.edge_index.max() < graph.num_nodes
