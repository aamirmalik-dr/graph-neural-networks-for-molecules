"""Turn RDKit molecules into numeric graph tensors.

The featurizers are deliberately explicit: every atom and bond descriptor is
listed here so the feature vector is easy to audit and extend. Graphs are
returned as plain NumPy arrays and converted to tensors later, which keeps this
module free of any deep-learning dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

# Elements common in small drug-like molecules. Anything outside this list maps
# to the trailing "other" slot so featurization never fails on rare atoms.
ATOM_ELEMENTS: list[str] = [
    "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "other",
]

HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def _one_hot(value, choices: list) -> list[int]:
    """One-hot encode ``value`` against ``choices`` with a trailing catch-all."""
    vec = [0] * (len(choices) + 1)
    try:
        vec[choices.index(value)] = 1
    except ValueError:
        vec[-1] = 1
    return vec


def atom_features(atom: Chem.Atom) -> np.ndarray:
    """Return the feature vector for a single atom.

    Args:
        atom: An RDKit atom.

    Returns:
        A 1D float32 array concatenating element, degree, formal charge,
        hydrogen count, hybridization, aromaticity, and ring membership.
    """
    feats: list[float] = []
    feats += _one_hot(atom.GetSymbol(), ATOM_ELEMENTS[:-1])
    feats += _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    feats += _one_hot(atom.GetFormalCharge(), [-1, 0, 1])
    feats += _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    feats += _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
    feats.append(float(atom.GetIsAromatic()))
    feats.append(float(atom.IsInRing()))
    return np.asarray(feats, dtype=np.float32)


def bond_features(bond: Chem.Bond) -> np.ndarray:
    """Return the feature vector for a single bond."""
    feats: list[float] = []
    feats += _one_hot(bond.GetBondType(), BOND_TYPES)
    feats.append(float(bond.GetIsConjugated()))
    feats.append(float(bond.IsInRing()))
    return np.asarray(feats, dtype=np.float32)


# Feature dimensions, computed once from a probe molecule so callers can size
# their networks without constructing a graph first.
_PROBE = Chem.MolFromSmiles("CCO")
ATOM_FEATURE_DIM: int = len(atom_features(_PROBE.GetAtomWithIdx(0)))
BOND_FEATURE_DIM: int = len(bond_features(_PROBE.GetBondWithIdx(0)))


@dataclass
class MolGraph:
    """A molecule as arrays ready for batching.

    Attributes:
        node_feats: Atom features, shape ``(num_atoms, ATOM_FEATURE_DIM)``.
        edge_index: Directed edges, shape ``(2, num_edges)``; each chemical bond
            appears twice, once in each direction.
        edge_feats: Bond features aligned with ``edge_index``.
        num_nodes: Number of atoms.
    """

    node_feats: np.ndarray
    edge_index: np.ndarray
    edge_feats: np.ndarray
    num_nodes: int


def mol_to_graph(mol: Chem.Mol) -> MolGraph:
    """Convert an RDKit molecule to a :class:`MolGraph`.

    Bonds are expanded into two directed edges so message passing can move
    information both ways. Molecules with a single atom yield an empty edge set,
    which the downstream layers handle gracefully.

    Args:
        mol: A parsed RDKit molecule.

    Returns:
        The molecule as node and edge arrays.

    Raises:
        ValueError: If ``mol`` is ``None``.
    """
    if mol is None:
        raise ValueError("mol_to_graph received None; check SMILES parsing")

    node_feats = np.stack(
        [atom_features(a) for a in mol.GetAtoms()], axis=0
    ).astype(np.float32)

    src: list[int] = []
    dst: list[int] = []
    edge_feats: list[np.ndarray] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        edge_feats += [bf, bf]

    if edge_feats:
        edge_index = np.asarray([src, dst], dtype=np.int64)
        edge_arr = np.stack(edge_feats, axis=0).astype(np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_arr = np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)

    return MolGraph(node_feats, edge_index, edge_arr, mol.GetNumAtoms())


def smiles_to_graph(smiles: str) -> MolGraph:
    """Parse a SMILES string and return its graph.

    Raises:
        ValueError: If the SMILES cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol_to_graph(mol)


def morgan_fingerprint(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Compute a binary Morgan (ECFP-like) fingerprint for a SMILES string.

    Used only by the fingerprint-MLP baseline, not by the graph models.

    Raises:
        ValueError: If the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
