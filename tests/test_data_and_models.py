import numpy as np
import torch

from gnn_molecules.data import (
    collate_graphs,
    random_split,
    rdkit_property_dataset,
)
from gnn_molecules.featurize import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from gnn_molecules.models import GCN, MPNN
from gnn_molecules.train import Trainer, regression_metrics


def test_collate_offsets_edges():
    dataset = rdkit_property_dataset(["CCO", "c1ccccc1"], target="logp")
    batch = collate_graphs([dataset[0], dataset[1]])
    total_nodes = dataset.graphs[0].num_nodes + dataset.graphs[1].num_nodes
    assert batch.node_feats.shape[0] == total_nodes
    assert batch.batch.shape[0] == total_nodes
    assert batch.num_graphs == 2
    assert int(batch.edge_index.max()) < total_nodes
    # The second graph's nodes must carry batch id 1.
    assert batch.batch.max() == 1


def test_random_split_is_disjoint_and_complete():
    dataset = rdkit_property_dataset(target="logp")
    tr, va, te = random_split(dataset, (0.6, 0.2, 0.2), seed=1)
    assert len(tr) + len(va) + len(te) == len(dataset)
    all_smiles = set(tr.smiles) | set(va.smiles) | set(te.smiles)
    assert len(all_smiles) == len(set(dataset.smiles))


def test_models_forward_on_batch():
    dataset = rdkit_property_dataset(["CCO", "CCN", "c1ccccc1"], target="logp")
    batch = collate_graphs([dataset[i] for i in range(len(dataset))])
    mpnn = MPNN(ATOM_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim=16, num_layers=2)
    gcn = GCN(ATOM_FEATURE_DIM, hidden_dim=16, num_layers=2)
    assert mpnn(batch).shape == (3,)
    assert gcn(batch).shape == (3,)


def test_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0


def test_trainer_reduces_loss():
    torch.manual_seed(0)
    dataset = rdkit_property_dataset(target="logp")
    tr, va, te = random_split(dataset, seed=0)
    trainer = Trainer(MPNN(ATOM_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim=32, num_layers=2))
    trainer.fit(tr, va, epochs=15, batch_size=8, verbose=False)
    assert trainer.history["train"][-1] < trainer.history["train"][0]
