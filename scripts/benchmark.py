"""Benchmark MPNN vs GCN vs a fingerprint-MLP baseline on a molecular dataset.

Runs all three models under one training budget and writes a metrics table and
learning-curve / parity figures to the output directory.

Usage:
    # Offline, no download needed (RDKit logP target on a built-in SMILES set):
    python scripts/benchmark.py --dataset synthetic --epochs 40

    # On the ESOL solubility CSV produced by download_data.py:
    python scripts/benchmark.py --dataset esol --csv data/esol.csv --epochs 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gnn_molecules.data import (
    MolGraphDataset,
    load_esol,
    random_split,
    rdkit_property_dataset,
)
from gnn_molecules.featurize import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    morgan_fingerprint,
)
from gnn_molecules.models import GCN, MPNN, FingerprintMLP
from gnn_molecules.train import Trainer, regression_metrics, set_seed


def _fingerprint_matrix(dataset: MolGraphDataset, n_bits: int) -> torch.Tensor:
    return torch.from_numpy(
        np.stack([morgan_fingerprint(s, n_bits=n_bits) for s in dataset.smiles])
    )


def train_fingerprint_mlp(
    train_set: MolGraphDataset,
    val_set: MolGraphDataset,
    test_set: MolGraphDataset,
    epochs: int,
    n_bits: int = 1024,
) -> tuple[dict[str, float], list[float]]:
    """Train the fingerprint baseline and return test metrics and loss curve."""
    set_seed(0)
    model = FingerprintMLP(n_bits=n_bits)
    x_train = _fingerprint_matrix(train_set, n_bits)
    x_test = _fingerprint_matrix(test_set, n_bits)
    y_train = torch.from_numpy(train_set.targets)
    mean, std = float(y_train.mean()), float(y_train.std()) or 1.0

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    curve: list[float] = []
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, (y_train - mean) / std)
        loss.backward()
        opt.step()
        curve.append(loss.item())
    model.eval()
    with torch.no_grad():
        test_pred = (model(x_test) * std + mean).numpy()
    return regression_metrics(test_set.targets, test_pred), curve


def build_dataset(args: argparse.Namespace) -> MolGraphDataset:
    if args.dataset == "esol":
        return load_esol(args.csv)
    return rdkit_property_dataset(target=args.target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synthetic", "esol"], default="synthetic")
    parser.add_argument("--csv", default="data/esol.csv")
    parser.add_argument("--target", default="logp", help="target for the synthetic set")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(0)
    dataset = build_dataset(args)
    train_set, val_set, test_set = random_split(dataset, seed=0)
    print(
        f"Dataset: {len(dataset)} molecules "
        f"(train={len(train_set)}, val={len(val_set)}, test={len(test_set)})"
    )

    results: dict[str, dict[str, float]] = {}
    curves: dict[str, list[float]] = {}

    set_seed(0)
    mpnn = Trainer(
        MPNN(ATOM_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim=64, num_layers=3),
        lr=1e-3,
    )
    mpnn.fit(train_set, val_set, epochs=args.epochs, verbose=False)
    results["MPNN"] = mpnn.evaluate(test_set)
    curves["MPNN"] = mpnn.history["train"]

    set_seed(0)
    gcn = Trainer(GCN(ATOM_FEATURE_DIM, hidden_dim=64, num_layers=3), lr=1e-3)
    gcn.fit(train_set, val_set, epochs=args.epochs, verbose=False)
    results["GCN"] = gcn.evaluate(test_set)
    curves["GCN"] = gcn.history["train"]

    fp_metrics, fp_curve = train_fingerprint_mlp(
        train_set, val_set, test_set, epochs=args.epochs
    )
    results["FingerprintMLP"] = fp_metrics
    curves["FingerprintMLP"] = fp_curve

    # Metrics table.
    print("\nTest-set results:")
    header = f"{'model':<16}{'RMSE':>10}{'MAE':>10}{'R2':>10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<16}{m['rmse']:>10.4f}{m['mae']:>10.4f}{m['r2']:>10.4f}")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    # Learning-curve figure.
    plt.figure(figsize=(7, 4.5))
    for name, curve in curves.items():
        plt.plot(curve, label=name)
    plt.xlabel("epoch")
    plt.ylabel("training loss (standardized MSE)")
    plt.title("Training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close()

    # Parity plot for the best graph model.
    best_graph = "MPNN" if results["MPNN"]["rmse"] <= results["GCN"]["rmse"] else "GCN"
    trainer = mpnn if best_graph == "MPNN" else gcn
    preds = trainer.predict(test_set)
    plt.figure(figsize=(5, 5))
    plt.scatter(test_set.targets, preds, alpha=0.6, s=18)
    lo = min(test_set.targets.min(), preds.min())
    hi = max(test_set.targets.max(), preds.max())
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("true")
    plt.ylabel("predicted")
    plt.title(f"{best_graph} parity (test)")
    plt.tight_layout()
    plt.savefig(out_dir / "parity.png", dpi=120)
    plt.close()

    print(f"\nWrote metrics and figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
