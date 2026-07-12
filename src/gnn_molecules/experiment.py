"""Run a single configured experiment or a layer-depth ablation.

This is the glue between :mod:`gnn_molecules.config` and the models and
trainer. It builds the dataset and model a config asks for, trains with early
stopping, and returns test metrics plus the trainer so callers can save weights
or draw a parity plot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gnn_molecules.config import AblationConfig, ExperimentConfig
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
from gnn_molecules.train import (
    MetricsLogger,
    Trainer,
    regression_metrics,
    set_seed,
)


@dataclass
class ExperimentResult:
    """The outcome of one run.

    Attributes:
        config: The config that produced this result.
        metrics: Test-set RMSE, MAE, and R^2.
        train_curve: Per-epoch training loss.
        val_curve: Per-epoch validation RMSE.
        best_epoch: The epoch whose weights were restored (0 if no early stop).
        y_true: Test targets, for a parity plot.
        y_pred: Test predictions, for a parity plot.
        trainer: The fitted trainer (``None`` for the fingerprint baseline).
    """

    config: ExperimentConfig
    metrics: dict[str, float]
    train_curve: list[float]
    val_curve: list[float]
    best_epoch: int
    y_true: np.ndarray
    y_pred: np.ndarray
    trainer: Trainer | None


def build_dataset(cfg: ExperimentConfig) -> MolGraphDataset:
    """Build the dataset named by ``cfg`` (ESOL CSV or offline RDKit target)."""
    if cfg.dataset == "esol":
        return load_esol(cfg.csv)
    return rdkit_property_dataset(target=cfg.target)


def build_model(cfg: ExperimentConfig) -> torch.nn.Module:
    """Instantiate the model named by ``cfg``."""
    if cfg.model == "mpnn":
        return MPNN(
            ATOM_FEATURE_DIM,
            BOND_FEATURE_DIM,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
        )
    if cfg.model == "gcn":
        return GCN(ATOM_FEATURE_DIM, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers)
    return FingerprintMLP(n_bits=cfg.n_bits, hidden_dim=cfg.hidden_dim)


def _fingerprint_matrix(dataset: MolGraphDataset, n_bits: int) -> torch.Tensor:
    return torch.from_numpy(
        np.stack([morgan_fingerprint(s, n_bits=n_bits) for s in dataset.smiles])
    )


def _run_fingerprint(
    cfg: ExperimentConfig,
    train_set: MolGraphDataset,
    val_set: MolGraphDataset,
    test_set: MolGraphDataset,
) -> ExperimentResult:
    set_seed(cfg.seed)
    model = FingerprintMLP(n_bits=cfg.n_bits, hidden_dim=cfg.hidden_dim)
    x_train = _fingerprint_matrix(train_set, cfg.n_bits)
    x_val = _fingerprint_matrix(val_set, cfg.n_bits)
    x_test = _fingerprint_matrix(test_set, cfg.n_bits)
    y_train = torch.from_numpy(train_set.targets)
    mean, std = float(y_train.mean()), float(y_train.std()) or 1.0

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    logger = MetricsLogger()
    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x_train), (y_train - mean) / std)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = (model(x_val) * std + mean).numpy()
        val_rmse = regression_metrics(val_set.targets, val_pred)["rmse"]
        logger.log(epoch + 1, loss.item(), val_rmse)

    model.eval()
    with torch.no_grad():
        test_pred = (model(x_test) * std + mean).numpy()
    return ExperimentResult(
        config=cfg,
        metrics=regression_metrics(test_set.targets, test_pred),
        train_curve=logger.train_curve,
        val_curve=logger.val_curve,
        best_epoch=0,
        y_true=test_set.targets,
        y_pred=test_pred,
        trainer=None,
    )


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """Train one model from ``cfg`` and return its test-set result.

    The split is seeded from ``cfg.seed`` so runs are reproducible. Graph models
    train with early stopping when ``cfg.patience > 0``.
    """
    set_seed(cfg.seed)
    dataset = build_dataset(cfg)
    train_set, val_set, test_set = random_split(dataset, cfg.split, seed=cfg.seed)

    if cfg.model == "fingerprint":
        return _run_fingerprint(cfg, train_set, val_set, test_set)

    set_seed(cfg.seed)
    trainer = Trainer(
        build_model(cfg),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        seed=cfg.seed,
    )
    trainer.fit(
        train_set,
        val_set,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=False,
    )
    preds = trainer.predict(test_set)
    return ExperimentResult(
        config=cfg,
        metrics=regression_metrics(test_set.targets, preds),
        train_curve=trainer.history["train"],
        val_curve=trainer.history["val"],
        best_epoch=trainer.best_epoch,
        y_true=test_set.targets,
        y_pred=preds,
        trainer=trainer,
    )


def run_ablation(ablation: AblationConfig) -> list[ExperimentResult]:
    """Run one experiment per layer count in ``ablation`` and return all results."""
    return [run_experiment(cfg) for cfg in ablation.configs()]
