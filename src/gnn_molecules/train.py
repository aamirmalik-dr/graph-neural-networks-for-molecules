"""Training loop, metrics, and target standardization for the graph models."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from gnn_molecules.data import MolGraphDataset, collate_graphs


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    target_mean: float = 0.0
    target_std: float = 1.0
    history: dict[str, list[float]] = field(default_factory=lambda: {"train": [], "val": []})

    def _loader(self, dataset: MolGraphDataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_graphs
        )

    def fit(
        self,
        train_set: MolGraphDataset,
        val_set: MolGraphDataset | None = None,
        epochs: int = 30,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> Trainer:
        """Train the model, tracking train and validation loss per epoch."""
        self.model.to(self.device)
        self.target_mean = float(train_set.targets.mean())
        self.target_std = float(train_set.targets.std()) or 1.0
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.MSELoss()
        loader = self._loader(train_set, batch_size, shuffle=True)

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
                val_metrics = self.evaluate(val_set, batch_size)
                val_loss = val_metrics["rmse"]
            self.history["val"].append(val_loss)
            if verbose:
                print(
                    f"epoch {epoch + 1:3d}  train_mse={train_loss:.4f}  val_rmse={val_loss:.4f}"
                )
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
