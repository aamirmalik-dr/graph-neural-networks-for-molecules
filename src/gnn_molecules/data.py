"""Datasets, batching, and data sources for molecular graphs.

Two data sources are supported:

* :func:`load_esol` reads the public ESOL (Delaney) aqueous-solubility set from
  a local CSV produced by ``scripts/download_data.py``.
* :func:`rdkit_property_dataset` builds a fully offline, license-free dataset by
  computing an RDKit descriptor (logP or TPSA) as the regression target for a
  list of SMILES. This path needs no download and is what the tests and CI use.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

from gnn_molecules.featurize import MolGraph, smiles_to_graph

# A small, well-known set of valid SMILES used to build an offline dataset.
# These are literal molecule strings, not a redistributed dataset.
SAMPLE_SMILES: list[str] = [
    "CCO", "CCN", "CCC", "CCCC", "CCCCC", "c1ccccc1", "c1ccccc1O", "c1ccccc1N",
    "CC(=O)O", "CC(=O)N", "CCOC(=O)C", "CC(C)O", "CC(C)C", "CCOCC", "CNC",
    "C1CCCCC1", "C1CCCC1", "c1ccncc1", "c1ccc2ccccc2c1", "CC(=O)Nc1ccccc1",
    "CCN(CC)CC", "OCC(O)CO", "CC(N)C(=O)O", "Cc1ccccc1", "Clc1ccccc1",
    "Brc1ccccc1", "Fc1ccccc1", "Oc1ccc(O)cc1", "Nc1ccc(N)cc1", "CC#N",
    "C=CC=C", "C#CC", "CCS", "CSC", "CC(=O)C", "CCC(=O)O", "CCCO", "CCCCO",
    "COC", "Cc1ccc(C)cc1", "c1ccc(cc1)C(=O)O", "NCCO", "OCCO", "CC(O)CO",
    "Cc1ccncc1", "Cn1cccc1", "c1cc[nH]c1", "c1ccoc1", "c1ccsc1", "CC(C)(C)O",
]

TARGET_FUNCTIONS = {
    "logp": Crippen.MolLogP,
    "tpsa": Descriptors.TPSA,
    "mw": Descriptors.MolWt,
}


@dataclass
class MolGraphDataset(torch.utils.data.Dataset):
    """A list of molecular graphs paired with scalar regression targets."""

    graphs: list[MolGraph]
    targets: np.ndarray
    smiles: list[str]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple[MolGraph, float]:
        return self.graphs[idx], float(self.targets[idx])


@dataclass
class GraphBatch:
    """A batch of graphs merged into one disjoint graph.

    Attributes:
        node_feats: All atoms stacked, shape ``(total_nodes, atom_dim)``.
        edge_index: Edges with node indices offset per graph, shape ``(2, E)``.
        edge_feats: Bond features aligned with ``edge_index``.
        batch: Graph id for each node, shape ``(total_nodes,)``.
        targets: One target per graph, shape ``(num_graphs,)``.
        num_graphs: Number of graphs in the batch.
    """

    node_feats: torch.Tensor
    edge_index: torch.Tensor
    edge_feats: torch.Tensor
    batch: torch.Tensor
    targets: torch.Tensor
    num_graphs: int

    def to(self, device: torch.device | str) -> GraphBatch:
        return GraphBatch(
            self.node_feats.to(device),
            self.edge_index.to(device),
            self.edge_feats.to(device),
            self.batch.to(device),
            self.targets.to(device),
            self.num_graphs,
        )


def collate_graphs(items: list[tuple[MolGraph, float]]) -> GraphBatch:
    """Merge a list of ``(graph, target)`` pairs into one :class:`GraphBatch`.

    Node indices in each graph's ``edge_index`` are shifted by the running node
    count so the merged graph stays disjoint. A ``batch`` vector records which
    graph every node belongs to, which the pooling layers use to read out one
    vector per molecule.
    """
    node_feats: list[np.ndarray] = []
    edge_index: list[np.ndarray] = []
    edge_feats: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    targets: list[float] = []

    node_offset = 0
    for gi, (graph, target) in enumerate(items):
        node_feats.append(graph.node_feats)
        if graph.edge_index.shape[1] > 0:
            edge_index.append(graph.edge_index + node_offset)
            edge_feats.append(graph.edge_feats)
        batch.append(np.full((graph.num_nodes,), gi, dtype=np.int64))
        targets.append(target)
        node_offset += graph.num_nodes

    ei = (
        np.concatenate(edge_index, axis=1)
        if edge_index
        else np.zeros((2, 0), dtype=np.int64)
    )
    ef = (
        np.concatenate(edge_feats, axis=0)
        if edge_feats
        else np.zeros((0, items[0][0].edge_feats.shape[1]), dtype=np.float32)
    )
    return GraphBatch(
        node_feats=torch.from_numpy(np.concatenate(node_feats, axis=0)),
        edge_index=torch.from_numpy(ei),
        edge_feats=torch.from_numpy(ef),
        batch=torch.from_numpy(np.concatenate(batch, axis=0)),
        targets=torch.tensor(targets, dtype=torch.float32),
        num_graphs=len(items),
    )


def _build_dataset(smiles: list[str], targets: list[float]) -> MolGraphDataset:
    graphs: list[MolGraph] = []
    kept_smiles: list[str] = []
    kept_targets: list[float] = []
    for smi, y in zip(smiles, targets, strict=False):
        try:
            graphs.append(smiles_to_graph(smi))
        except ValueError:
            continue
        kept_smiles.append(smi)
        kept_targets.append(y)
    return MolGraphDataset(
        graphs=graphs,
        targets=np.asarray(kept_targets, dtype=np.float32),
        smiles=kept_smiles,
    )


def rdkit_property_dataset(
    smiles: list[str] | None = None, target: str = "logp"
) -> MolGraphDataset:
    """Build an offline dataset whose target is an RDKit descriptor.

    Args:
        smiles: SMILES strings. Defaults to :data:`SAMPLE_SMILES`.
        target: One of ``"logp"``, ``"tpsa"``, or ``"mw"``.

    Returns:
        A dataset with the descriptor computed for each molecule.

    Raises:
        ValueError: If ``target`` is not a known descriptor.
    """
    if target not in TARGET_FUNCTIONS:
        raise ValueError(f"unknown target {target!r}; choose from {list(TARGET_FUNCTIONS)}")
    smiles = list(smiles) if smiles is not None else list(SAMPLE_SMILES)
    fn = TARGET_FUNCTIONS[target]
    values = [float(fn(Chem.MolFromSmiles(s))) for s in smiles]
    return _build_dataset(smiles, values)


def load_esol(csv_path: str) -> MolGraphDataset:
    """Load the ESOL (Delaney) aqueous-solubility dataset from a CSV.

    The CSV is the public ``delaney-processed.csv`` fetched by
    ``scripts/download_data.py``. The target is measured log solubility
    (``measured log solubility in mols per litre``).

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        KeyError: If the expected columns are missing.
    """
    smiles: list[str] = []
    targets: list[float] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        smi_col = _find_column(reader.fieldnames, ["smiles"])
        y_col = _find_column(
            reader.fieldnames,
            ["measured log solubility in mols per litre", "measured log", "logs"],
        )
        for row in reader:
            smiles.append(row[smi_col])
            targets.append(float(row[y_col]))
    return _build_dataset(smiles, targets)


def _find_column(fieldnames: list[str] | None, candidates: list[str]) -> str:
    if not fieldnames:
        raise KeyError("CSV has no header row")
    lowered = {name.lower(): name for name in fieldnames}
    for cand in candidates:
        for low, original in lowered.items():
            if cand in low:
                return original
    raise KeyError(f"could not find any of {candidates} in {fieldnames}")


def random_split(
    dataset: MolGraphDataset,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[MolGraphDataset, MolGraphDataset, MolGraphDataset]:
    """Split a dataset into train/validation/test subsets by random shuffle.

    Raises:
        ValueError: If ``fractions`` does not have three entries summing to 1.
    """
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must be three values summing to 1.0")
    n = len(dataset)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    idx_splits = (
        perm[:n_train],
        perm[n_train : n_train + n_val],
        perm[n_train + n_val :],
    )
    subsets = []
    for idx in idx_splits:
        subsets.append(
            MolGraphDataset(
                graphs=[dataset.graphs[i] for i in idx],
                targets=dataset.targets[idx],
                smiles=[dataset.smiles[i] for i in idx],
            )
        )
    return subsets[0], subsets[1], subsets[2]
