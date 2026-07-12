"""Training loop, metrics, and target standardization for the graph models."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gnn_molecules.data import MolGraphDataset, collate_graphs


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch and pin deterministic CPU execution.

    Beyond seeding the three RNGs, this pins PyTorch to a single intra-op
    thread. Multi-threaded CPU reductions use a nondeterministic summation
    order, which compounds over many training epochs into visibly different
    final weights; single-threaded execution makes a run bit-for-bit
    reproducible, which is the point of a fixed-seed benchmark.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


@dataclass
class EpochRecord:
    """One row of the training log."""

    epoch: int
    train_loss: float
    val_rmse: float


class MetricsLogger:
    """Collects per-epoch training records and can serialize them to JSON."""

    def __init__(self) -> None:
        self.records: list[EpochRecord] = []

    def log(self, epoch: int, train_loss: float, val_rmse: float) -> None:
        """Append one epoch's numbers."""
        self.records.append(EpochRecord(epoch, float(train_loss), float(val_rmse)))

    @property
    def train_curve(self) -> list[float]:
        """The training-loss value at each epoch."""
        return [r.train_loss for r in self.records]

    @property
    def val_curve(self) -> list[float]:
        """The validation RMSE at each epoch."""
        return [r.val_rmse for r in self.records]

    def to_json(self, path: str | Path) -> None:
        """Write the full log to ``path`` as a JSON list of records."""
        payload = [
            {"epoch": r.epoch, "train_loss": r.train_loss, "val_rmse": r.val_rmse}
            for r in self.records
        ]
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE, and R^2 for a set of predictions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}


@dataclass
class Trainer:
    """Trains a graph model with standardized targets and Adam.

    Targets are standardized using the training set statistics so the loss is
    well scaled; predictions are mapped back to the original units for metrics.
    """

    model: torch.nn.Module
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    patience: int = 0
    seed: int = 0
    target_mean: float = 0.0
    target_std: float = 1.0
    history: dict[str, list[float]] = field(default_factory=lambda: {"train": [], "val": []})
    logger: MetricsLogger = field(default_factory=MetricsLogger)
    best_val: float = float("inf")
    best_epoch: int = 0

    def _loader(self, dataset: MolGraphDataset, batch_size: int, shuffle: bool) -> DataLoader:
        # A dedicated generator makes minibatch shuffling reproducible across
        # runs; the default DataLoader sampler draws a fresh, unseeded seed.
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_graphs,
            generator=generator,
        )

    def fit(
        self,
        train_set: MolGraphDataset,
        val_set: MolGraphDataset | None = None,
        epochs: int = 30,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> Trainer:
        """Train the model, tracking loss per epoch with optional early stopping.

        When ``patience > 0`` and a validation set is given, training stops after
        ``patience`` epochs without validation-RMSE improvement, and the best
        weights seen are restored before returning.
        """
        self.model.to(self.device)
        self.target_mean = float(train_set.targets.mean())
        self.target_std = float(train_set.targets.std()) or 1.0
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.MSELoss()
        loader = self._loader(train_set, batch_size, shuffle=True)

        use_es = self.patience > 0 and val_set is not None and len(val_set) > 0
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n = 0
            for batch in loader:
                batch = batch.to(self.device)
                target = (batch.targets - self.target_mean) / self.target_std
                opt.zero_grad()
                pred = self.model(batch)
                loss = loss_fn(pred, target)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * batch.num_graphs
                n += batch.num_graphs
            train_loss = epoch_loss / max(n, 1)
            self.history["train"].append(train_loss)
            val_loss = float("nan")
            if val_set is not None and len(val_set) > 0:
                val_loss = self.evaluate(val_set, batch_size)["rmse"]
            self.history["val"].append(val_loss)
            self.logger.log(epoch + 1, train_loss, val_loss)
            if verbose:
                print(
                    f"epoch {epoch + 1:3d}  train_mse={train_loss:.4f}  val_rmse={val_loss:.4f}"
                )

            improved = not np.isnan(val_loss) and val_loss < self.best_val - 1e-6
            if improved:
                self.best_val = val_loss
                self.best_epoch = epoch + 1
                if use_es:
                    best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
            if use_es and stale >= self.patience:
                if verbose:
                    print(f"early stop at epoch {epoch + 1} (best {self.best_epoch})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def save(self, path: str | Path) -> None:
        """Save model weights plus target standardization to a checkpoint.

        The checkpoint stores enough to run inference: the state dict and the
        target mean/std used to invert standardized predictions.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "target_mean": self.target_mean,
                "target_std": self.target_std,
            },
            path,
        )

    def load(self, path: str | Path) -> Trainer:
        """Load weights and standardization written by :meth:`save`."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["state_dict"])
        self.target_mean = float(ckpt["target_mean"])
        self.target_std = float(ckpt["target_std"])
        self.model.to(self.device)
        return self

    @torch.no_grad()
    def predict(self, dataset: MolGraphDataset, batch_size: int = 64) -> np.ndarray:
        """Return predictions in the original target units."""
        self.model.eval()
        preds: list[np.ndarray] = []
        for batch in self._loader(dataset, batch_size, shuffle=False):
            batch = batch.to(self.device)
            out = self.model(batch) * self.target_std + self.target_mean
            preds.append(out.cpu().numpy())
        return np.concatenate(preds, axis=0) if preds else np.zeros((0,))

    def evaluate(self, dataset: MolGraphDataset, batch_size: int = 64) -> dict[str, float]:
        """Return regression metrics on ``dataset``."""
        preds = self.predict(dataset, batch_size)
        return regression_metrics(dataset.targets, preds)
