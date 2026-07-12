"""Typed experiment configuration loaded from YAML.

A run is fully described by a small YAML file so the benchmark and ablation
entry points stay declarative. Every field has a default, so a minimal config
only needs to name a model and a dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

VALID_MODELS = ("mpnn", "gcn", "fingerprint")
VALID_DATASETS = ("esol", "synthetic")


@dataclass
class ExperimentConfig:
    """One training run, model plus data plus optimization settings.

    Attributes:
        name: Display name used in tables and figures.
        model: One of ``"mpnn"``, ``"gcn"``, ``"fingerprint"``.
        dataset: ``"esol"`` (reads ``csv``) or ``"synthetic"`` (RDKit target).
        csv: Path to the ESOL CSV when ``dataset == "esol"``.
        target: RDKit descriptor name when ``dataset == "synthetic"``.
        epochs: Maximum number of training epochs.
        batch_size: Minibatch size for the graph models.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        hidden_dim: Hidden width of the graph models and the MLP.
        num_layers: Number of message-passing or convolution layers.
        n_bits: Morgan fingerprint length for the baseline.
        patience: Early-stopping patience in epochs; ``0`` disables it.
        seed: Random seed for splitting and initialization.
        split: Train/validation/test fractions.
    """

    name: str = "run"
    model: str = "mpnn"
    dataset: str = "synthetic"
    csv: str = "data/esol_sample.csv"
    target: str = "logp"
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 64
    num_layers: int = 3
    n_bits: int = 1024
    patience: int = 15
    seed: int = 0
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)

    def __post_init__(self) -> None:
        if self.model not in VALID_MODELS:
            raise ValueError(f"model must be one of {VALID_MODELS}, got {self.model!r}")
        if self.dataset not in VALID_DATASETS:
            raise ValueError(
                f"dataset must be one of {VALID_DATASETS}, got {self.dataset!r}"
            )
        self.split = tuple(float(x) for x in self.split)  # type: ignore[assignment]
        if len(self.split) != 3 or abs(sum(self.split) - 1.0) > 1e-6:
            raise ValueError("split must be three fractions summing to 1.0")

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        """Build a config from a dict, ignoring unknown keys with a clear error.

        Raises:
            ValueError: If ``raw`` contains keys that are not config fields.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)


@dataclass
class AblationConfig:
    """A sweep over the number of message-passing or convolution layers.

    Attributes:
        base: The shared config every point in the sweep starts from.
        num_layers: The layer counts to try, one training run each.
    """

    base: ExperimentConfig = field(default_factory=ExperimentConfig)
    num_layers: list[int] = field(default_factory=lambda: [1, 2, 3, 4])

    def configs(self) -> list[ExperimentConfig]:
        """Materialize one :class:`ExperimentConfig` per layer count."""
        out: list[ExperimentConfig] = []
        for k in self.num_layers:
            raw = self.base.to_dict()
            raw["num_layers"] = int(k)
            raw["name"] = f"{self.base.model}-L{k}"
            out.append(ExperimentConfig.from_dict(raw))
        return out


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a single :class:`ExperimentConfig` from a YAML file.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    raw = _read_yaml(path)
    return ExperimentConfig.from_dict(raw)


def load_ablation(path: str | Path) -> AblationConfig:
    """Load an :class:`AblationConfig` from a YAML file.

    The YAML must have a ``num_layers`` list; all other keys are treated as the
    shared base config.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``num_layers`` is missing or empty.
    """
    raw = _read_yaml(path)
    layers = raw.pop("num_layers", None)
    if not layers:
        raise ValueError("ablation config must define a non-empty 'num_layers' list")
    base = ExperimentConfig.from_dict(raw)
    return AblationConfig(base=base, num_layers=[int(x) for x in layers])


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
    return raw
